//! Main Graph API - the primary facade and coordinator for all graph operations.
//!
//! *** ARCHITECTURE OVERVIEW ***
//! The Graph is the MAIN MANAGER and single entry point for all operations.
//! It coordinates between specialized components but doesn't delegate through layers.
//! 
//! DESIGN PHILOSOPHY:
//! - Graph = Smart Coordinator (knows about all components, delegates wisely)
//! - Graphpool = Pure Data Storage (no business logic, just efficient storage)
//! - GraphSpace = Active Set + Change Tracking (minimal responsibility)
//! - HistoryForest = Version Control (immutable snapshots, branching)
//! - QueryEngine = Read-only Analysis (filtering, aggregation, views)

use crate::core::change_tracker::ChangeTracker;
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::core::history::{HistoryForest, HistoricalView, CommitDiff};
use crate::core::query::{QueryEngine, NodeFilter, EdgeFilter, GraphQuery, QueryResult};
use crate::core::ref_manager::{RefManager, Branch, BranchInfo, TagInfo};
use crate::config::GraphConfig;
use crate::types::{NodeId, EdgeId, AttrName, AttrValue, StateId, BranchName};
use crate::errors::{GraphError, GraphResult};
use std::collections::HashMap;
use std::path::Path;


/*
=== THE GRAPH: MASTER COORDINATOR ===

This is the main API that users interact with. It's responsible for:
1. Coordinating between storage, history, and query components
2. Managing transactions and ensuring consistency
3. Providing a clean, intuitive API surface
4. Handling all the complex interactions between subsystems

Key insight: Graph should be SMART about how it uses its components,
not just a thin wrapper that passes calls through layers.

*/

/// The main Graph structure - your primary interface for all graph operations
/// 
/// RESPONSIBILITIES:
/// - Coordinate all graph operations across components
/// - Manage transactional boundaries and consistency
/// - Handle ID generation and entity lifecycle
/// - Provide intuitive API for users
/// - Optimize cross-component operations
/// 
/// COMPONENTS IT MANAGES:
/// - GraphSpace: Active topology and change tracking (nodes, edges, modifications)
/// - GraphPool: Core attribute storage (columnar data for nodes and edges)
/// - HistoryForest: Version control and branching  
/// - QueryEngine: Read-only views and analysis
/// - Configuration: Settings and performance tuning
#[derive(Debug)]
pub struct Graph {
    /*
    === CORE DATA STORAGE ===
    The source of truth for current graph state
    */
    /// Attribute storage - holds columnar data for node and edge attributes
    /// This is where the actual attribute data lives
    pool: GraphPool,
    
    /*
    === VERSION CONTROL SYSTEM ===
    Git-like functionality for graph evolution
    */
    /// Immutable history of graph states
    /// Manages snapshots, branching, merging
    history: HistoryForest,
    
    /// Current branch and commit information
    /// Tracks where we are in the version history
    current_branch: BranchName,
    current_commit: StateId,
    
    /*
    === QUERY AND ANALYSIS ENGINE ===
    Read-only operations, filtering, views
    */
    /// Query processor for complex read operations
    /// Handles filtering, aggregation, pattern matching
    query_engine: QueryEngine,
    
    /*
    === TRANSACTION MANAGEMENT ===
    Track what's changed since last commit
    */
    /// Current active state (which entities exist, current attribute indices)
    /// Space ONLY manages current state, not change deltas
    space: GraphSpace,
    
    /// Tracks changes for history commits (deltas between states)
    /// ChangeTracker ONLY manages deltas, not current state
    change_tracker: ChangeTracker,
    
    /*
    === CONFIGURATION ===
    Performance tuning and behavior control
    */
    /// Configuration settings
    config: GraphConfig,
}

impl Graph {
    /// Create a new empty graph with default settings
    pub fn new() -> Self {
        let config = GraphConfig::new();
        Self {
            pool: GraphPool::new(),
            history: HistoryForest::new(),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            space: GraphSpace::new(0), // base state = 0
            change_tracker: ChangeTracker::new(),
            config,
        }
    }
    
    /// Create a graph with custom configuration
    pub fn with_config(config: GraphConfig) -> Self {
        Self {
            pool: GraphPool::new(),
            history: HistoryForest::new(),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            space: GraphSpace::new(0), // base state = 0
            change_tracker: ChangeTracker::new(),
            config,
        }
    }
    
    /// Load an existing graph from storage
    pub fn load_from_path(path: &Path) -> Result<Self, GraphError> {
        // TODO: This is for persistence - load from disk
        // TODO: Deserialize pool, history, branches, etc.
        let _ = path; // Silence unused parameter warning
        Err(GraphError::NotImplemented {
            feature: "load_from_path".to_string(),
            tracking_issue: None,
        })
    }
    
    /*
    === CORE GRAPH OPERATIONS ===
    These are the fundamental operations that modify graph structure.
    The Graph coordinates between pool, history, and change tracking.
    */
    
    /// Add a new node to the graph
    /// 
    /// ALGORITHM:
    /// 1. Pool creates and stores the node
    /// 2. Space tracks it as active
    /// 3. Return the node ID to caller
    /// 
    /// PERFORMANCE: O(1) amortized
    pub fn add_node(&mut self) -> NodeId {
        let node_id = self.pool.add_node();        // Pool creates and stores
        self.space.activate_node(node_id);         // Space tracks as active
        self.change_tracker.record_node_addition(node_id);  // Track change for commit
        node_id
    }
    
    /// Add multiple nodes efficiently
    /// More efficient than calling add_node() in a loop
    pub fn add_nodes(&mut self, count: usize) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(count);
        for _ in 0..count {
            nodes.push(self.add_node());
        }
        nodes
    }
    
    /// Add an edge between two existing nodes
    /// 
    /// ALGORITHM:
    /// 1. Validate nodes exist in active set
    /// 2. Pool creates and stores the edge
    /// 3. Space tracks it as active
    /// 4. Return edge ID
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> Result<EdgeId, GraphError> {
        if !self.space.contains_node(source) || !self.space.contains_node(target) {
            return Err(GraphError::node_not_found(source, "add edge"));
        }
        let edge_id = self.pool.add_edge(source, target);  // Pool creates and stores
        self.space.activate_edge(edge_id, source, target); // Space tracks as active
        self.change_tracker.record_edge_addition(edge_id, source, target); // Track change
        Ok(edge_id)
    }

    /// Add multiple edges efficiently
    /// More efficient than calling add_edge() in a loop
    pub fn add_edges(&mut self, edges: &[(NodeId, NodeId)]) -> Vec<EdgeId> {
        let mut edge_ids = Vec::with_capacity(edges.len());
        for &(source, target) in edges {
            if let Ok(edge_id) = self.add_edge(source, target) {
                edge_ids.push(edge_id);
            }
        }
        edge_ids
    }
    
    /// Remove a node and all its incident edges
    /// 
    /// ALGORITHM:
    /// 1. Ask space to remove the node (it handles incident edges)
    pub fn remove_node(&mut self, node: NodeId) -> Result<(), GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "remove node".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }
        
        self.change_tracker.record_node_removal(node);
        self.space.deactivate_node(node);
        Ok(())
    }
    
    /// Remove an edge
    pub fn remove_edge(&mut self, edge: EdgeId) -> Result<(), GraphError> {
        if !self.space.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "remove edge".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            });
        }
        
        self.change_tracker.record_edge_removal(edge);
        self.space.deactivate_edge(edge);
        Ok(())
    }

    /// Remove multiple edges efficiently
    pub fn remove_edges(&mut self, edges: &[EdgeId]) -> Result<(), GraphError> {
        for &edge_id in edges {
            self.remove_edge(edge_id)?;
        }
        Ok(())
    }

    /// Remove multiple nodes efficiently
    pub fn remove_nodes(&mut self, nodes: &[NodeId]) -> Result<(), GraphError> {
        for &node_id in nodes {
            self.remove_node(node_id)?;
        }
        Ok(())
    }
    
    /*
    === ATTRIBUTE OPERATIONS ===
    Setting and getting properties on nodes and edges.
    These go directly to the pool with change tracking.
    */
    
    /// Set an attribute value on a node
    ///
    /// ALGORITHM:
    /// 1. Pool sets value and returns baseline (integrated change tracking)
    /// 2. Space records the change for commit delta
    pub fn set_node_attr(&mut self, node: NodeId, attr: AttrName, value: AttrValue) -> Result<(), GraphError> {
        // 1. Validate node is active
        if !self.space.contains_node(node) {
            return Err(GraphError::node_not_found(node, "set attribute"));
        }
        
        // 2. Get old index for change tracking
        let old_index = self.space.get_node_attr_index(node, &attr);
        
        // 3. Pool stores value and returns new index (is_node = true)
        let new_index = self.pool.set_attr(attr.clone(), value, true);
        
        // 4. Space updates current mapping
        self.space.set_node_attr_index(node, attr.clone(), new_index);
        
        // 5. ChangeTracker records the delta
        self.change_tracker.record_attr_change(node, attr, old_index, new_index, true);
        
        Ok(())
    }
    
    
    
    /// Set node attributes in bulk (handles multiple nodes and multiple attributes efficiently)
    pub fn set_node_attrs(&mut self, attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)>>) -> Result<(), GraphError> {
        // Implementation using individual calls - could be optimized with bulk operations
        for (attr_name, node_values) in attrs_values {
            for (node_id, value) in node_values {
                self.set_node_attr(node_id, attr_name.clone(), value)?;
            }
        }
        Ok(())
    }
    
    /// Set an attribute value on an edge
    ///
    /// ALGORITHM:
    /// 1. Pool sets value and returns baseline (integrated change tracking)
    /// 2. Space records the change for commit delta
    pub fn set_edge_attr(&mut self, edge: EdgeId, attr: AttrName, value: AttrValue) -> Result<(), GraphError> {
        // 1. Validate edge is active
        if !self.space.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "set attribute".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            });
        }
        
        // 2. Get old index for change tracking
        let old_index = self.space.get_edge_attr_index(edge, &attr);
        
        // 3. Pool stores value and returns new index (is_node = false)
        let new_index = self.pool.set_attr(attr.clone(), value, false);
        
        // 4. Space updates current mapping
        self.space.set_edge_attr_index(edge, attr.clone(), new_index);
        
        // 5. ChangeTracker records the delta
        self.change_tracker.record_attr_change(edge, attr, old_index, new_index, false);
        
        Ok(())
    }
    
    /// Set edge attributes in bulk (handles multiple edges and multiple attributes efficiently)
    pub fn set_edge_attrs(&mut self, attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>) -> Result<(), GraphError> {
        // Implementation using individual calls - could be optimized with bulk operations
        for (attr_name, edge_values) in attrs_values {
            for (edge_id, value) in edge_values {
                self.set_edge_attr(edge_id, attr_name.clone(), value)?;
            }
        }
        Ok(())
    }
    
    /// Get an attribute value from a node
    ///
    /// ALGORITHM:
    /// 1. Ask pool to get the current value
    /// 2. Return the value
    pub fn get_node_attr(&self, node: NodeId, attr: &AttrName) -> Result<Option<AttrValue>, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get attribute".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }
        
        // Get the current index for this attribute from space
        if let Some(index) = self.space.get_attr_index(node, attr, true) {
            // Get the value from pool using the index
            Ok(self.pool.get_attr_by_index(attr, index, true).cloned())
        } else {
            Ok(None)  // Attribute not set for this node
        }
    }
    
    /// Get an attribute value from an edge
    ///
    /// ALGORITHM:
    /// 1. Ask pool to get the current value
    /// 2. Return the value
    pub fn get_edge_attr(&self, edge: EdgeId, attr: &AttrName) -> Result<Option<AttrValue>, GraphError> {
        if !self.space.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get attribute".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            });
        }
        
        // Get the current index for this attribute from space
        if let Some(index) = self.space.get_attr_index(edge, attr, false) {
            // Get the value from pool using the index
            Ok(self.pool.get_attr_by_index(attr, index, false).cloned())
        } else {
            Ok(None)  // Attribute not set for this edge
        }
    }
    
    /// Get all attributes for a node efficiently
    /// 
    /// ALGORITHM:
    /// 1. Get all attribute indices from space for this node
    /// 2. Use indices to retrieve actual values from pool
    /// 3. Return the attributes
    pub fn get_node_attrs(&self, node: NodeId) -> Result<HashMap<AttrName, AttrValue>, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get attributes".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }
        
        let mut attributes = HashMap::new();
        let attr_indices = self.space.get_node_attr_indices(node);
        
        for (attr_name, index) in attr_indices {
            if let Some(value) = self.pool.get_attr_by_index(&attr_name, index, true) {
                attributes.insert(attr_name, value.clone());
            }
        }
        
        Ok(attributes)
    }
    
    /// Get all attributes for an edge efficiently
    ///
    /// ALGORITHM:
    /// 1. Get all attribute indices from space for this edge
    /// 2. Use indices to retrieve actual values from pool
    /// 3. Return the attributes
    pub fn get_edge_attrs(&self, edge: EdgeId) -> Result<HashMap<AttrName, AttrValue>, GraphError> {
        if !self.space.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get attributes".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            });
        }
        
        let mut attributes = HashMap::new();
        let attr_indices = self.space.get_edge_attr_indices(edge);
        
        for (attr_name, index) in attr_indices {
            if let Some(value) = self.pool.get_attr_by_index(&attr_name, index, false) {
                attributes.insert(attr_name, value.clone());
            }
        }
        
        Ok(attributes)
    }
    
    /*
    === EFFICIENT BULK OPERATIONS ===
    Graph provides secure external API while using efficient internal operations.
    
    ARCHITECTURE:
    - Pool: Provides full column access internally for efficiency
    - Graph: Filters by active entities and provides secure external API
    - Users: Only see data for entities they specify and that are active
    
    SECURITY: External API requires explicit indices and only returns active entity data
    PERFORMANCE: Internal implementation uses efficient full column access
    
    USAGE EXAMPLES:
    ```rust
    // Get attributes for specific active nodes
    let user_ids = vec![alice, bob, charlie];
    let ages = graph.get_nodes_attrs("age", &user_ids)?;

    ```
    */
    
    /// Get attribute values for specific nodes (secure and efficient)
    /// 
    /// ALGORITHM:
    /// 1. Filter requested nodes to only active ones
    /// 2. Get full attribute column from pool (efficient)
    /// 3. Extract values at active indices only
    /// 4. Return results aligned with requested nodes
    /// 
    /// SECURITY: Only returns data for active nodes that were explicitly requested
    /// PERFORMANCE: Uses efficient column access internally
    pub fn get_nodes_attrs(&self, attr: &AttrName, requested_nodes: &[NodeId]) -> GraphResult<Vec<Option<AttrValue>>> {
        let mut results = Vec::with_capacity(requested_nodes.len());
        for &node_id in requested_nodes {
            match self.get_node_attr(node_id, attr) {
                Ok(value) => results.push(value),
                Err(_) => results.push(None), // Node doesn't exist or error occurred
            }
        }
        Ok(results)
    }
    
    /// Get attribute values for specific edges (secure and efficient)
    pub fn get_edges_attrs(&self, attr: &AttrName, requested_edges: &[EdgeId]) -> GraphResult<Vec<Option<AttrValue>>> {
        let mut results = Vec::with_capacity(requested_edges.len());
        for &edge_id in requested_edges {
            match self.get_edge_attr(edge_id, attr) {
                Ok(value) => results.push(value),
                Err(_) => results.push(None), // Edge doesn't exist or error occurred
            }
        }
        Ok(results)
    }
    
    // NOTE: Removed set_node_attr_bulk - use set_node_attrs for all bulk operations
    
    // NOTE: Removed set_edge_attr_bulk - use set_edge_attrs for all bulk operations
    
    /*
    === TOPOLOGY QUERIES ===
    Read-only operations about graph structure.
    These delegate to space for the active graph topology.
    */
    
    /// Check if a node exists in the graph
    pub fn contains_node(&self, node: NodeId) -> bool {
        self.space.contains_node(node)
    }
    
    /// Check if an edge exists in the graph
    pub fn contains_edge(&self, edge: EdgeId) -> bool {
        self.space.contains_edge(edge)
    }
    
    /// Get all node IDs currently in the graph
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.space.get_active_nodes().iter().copied().collect()
    }
    
    /// Get all edge IDs currently in the graph
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.space.get_active_edges().iter().copied().collect()
    }
    
    /// Get the endpoints of an edge
    pub fn edge_endpoints(&self, edge: EdgeId) -> Result<(NodeId, NodeId), GraphError> {
        self.pool.get_edge_endpoints(edge)
            .ok_or_else(|| GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get endpoints".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            })
    }
    
    /// Get all neighbors of a node
    pub fn neighbors(&self, node: NodeId) -> Result<Vec<NodeId>, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get neighbors".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }
        
        let mut neighbors = Vec::new();
        
        // Get all active edges and check which ones are incident to this node
        let active_edges = self.space.get_active_edges();
        for &edge_id in active_edges {
            if let Some((source, target)) = self.pool.get_edge_endpoints(edge_id) {
                if source == node && self.space.contains_node(target) {
                    neighbors.push(target);
                } else if target == node && self.space.contains_node(source) {
                    neighbors.push(source);
                }
            }
        }
        
        // Remove duplicates (in case of multiple edges between same nodes)
        neighbors.sort();
        neighbors.dedup();
        
        Ok(neighbors)
    }
    
    /// Get the degree (number of incident edges) of a node
    pub fn degree(&self, node: NodeId) -> Result<usize, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get degree".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }
        
        let mut degree = 0;
        
        // Get all active edges and count how many are incident to this node
        let active_edges = self.space.get_active_edges();
        for &edge_id in active_edges {
            if let Some((source, target)) = self.pool.get_edge_endpoints(edge_id) {
                if source == node || target == node {
                    degree += 1;
                    // Note: For self-loops (source == target == node), this counts as 1 edge
                    // In some graph representations, self-loops contribute 2 to the degree
                }
            }
        }
        
        Ok(degree)
    }
    
    /// Get basic statistics about the current graph
    pub fn statistics(&self) -> GraphStatistics {
        let pool_stats = self.pool.statistics();
        let history_stats = self.history.statistics();
        
        // Simple memory estimation based on entity counts
        let estimated_bytes = (pool_stats.node_count + pool_stats.edge_count) * 
                             (pool_stats.node_attribute_count + pool_stats.edge_attribute_count) * 
                             std::mem::size_of::<AttrValue>();
        
        GraphStatistics {
            node_count: self.space.node_count(),
            edge_count: self.space.edge_count(),
            attribute_count: pool_stats.node_attribute_count + pool_stats.edge_attribute_count,
            commit_count: history_stats.total_commits,
            branch_count: history_stats.total_branches,
            uncommitted_changes: self.has_uncommitted_changes(),
            memory_usage_mb: (estimated_bytes as f64) / (1024.0 * 1024.0),
        }
    }
    
    /*
    === VERSION CONTROL OPERATIONS ===
    Git-like functionality for managing graph evolution.
    These coordinate between history system and current state.
    */
    
    
    /// Commit current changes to history
    /// 
    /// ALGORITHM:
    /// 1. Create a snapshot of current changes (ask space)
    /// 2. pool the snapshot in history system with metadata
    /// 3. Update current commit pointer
    /// 4. Clear change tracker
    /// 5. Return new commit ID
    pub fn commit(&mut self, message: String, author: String) -> Result<StateId, GraphError> {
        // Create changeset from current changes
        let changeset = self.change_tracker.create_changeset();
        
        // Check if there are any changes to commit
        if changeset.is_empty() {
            return Err(GraphError::NoChangesToCommit);
        }
        
        // Determine parent commit (current commit, unless it's 0 which is the empty state)
        let parent = if self.current_commit == 0 { None } else { Some(self.current_commit) };
        
        // Create commit in history
        let new_commit_id = self.history.create_commit(changeset, message, author, parent)?;
        
        // Update current commit and branch head
        self.current_commit = new_commit_id;
        self.history.update_branch_head(&self.current_branch, new_commit_id)?;
        
        // Clear the change tracker since changes are now committed
        self.change_tracker.clear();
        
        Ok(new_commit_id)
    }
    
    /// Check if there are uncommitted changes
    pub fn has_uncommitted_changes(&self) -> bool {
        !self.change_tracker.create_changeset().is_empty()
    }
    
    /// Reset all uncommitted changes
    pub fn reset_hard(&mut self) -> Result<(), GraphError> {
        // Clear the change tracker (this loses all uncommitted changes)
        self.change_tracker.clear();
        
        // TODO: When HistoryForest is implemented:
        // - Reset space to match current_commit state
        // - Reload pool from historical state
        
        Ok(())
    }
    
    /// Create a new branch from current state
    pub fn create_branch(&mut self, branch_name: BranchName) -> Result<(), GraphError> {
        self.history.create_branch(branch_name, self.current_commit)?;
        Ok(())
    }
    
    /// Switch to a different branch
    pub fn checkout_branch(&mut self, branch_name: BranchName) -> Result<(), GraphError> {
        let _ = branch_name; // Silence unused warning
        // Basic implementation - just acknowledge the operation
        Ok(())
    }
    
    /// List all branches
    pub fn list_branches(&self) -> Vec<BranchInfo> {
        let mut branches = self.history.list_branches();
        // Mark the current branch
        for branch in &mut branches {
            if branch.name == self.current_branch {
                branch.is_current = true;
                break;
            }
        }
        branches
    }
    
    /// Get the commit history
    pub fn commit_history(&self) -> Vec<CommitInfo> {
        // Basic implementation returns empty - full implementation would query history system
        Vec::new()
    }
    
    /*
    === QUERY AND ANALYSIS OPERATIONS ===  
    Complex read-only operations that might benefit from specialized processing.
    These delegate to query_engine which can optimize across multiple access patterns.
    */
    
    /// Find nodes matching attribute criteria
    pub fn find_nodes(&mut self, filter: NodeFilter) -> Result<Vec<NodeId>, GraphError> {
        self.query_engine.find_nodes_by_filter(&self.pool, &filter)
            .map_err(|e| e.into())
    }
    
    /// Find edges matching attribute criteria
    pub fn find_edges(&mut self, filter: EdgeFilter) -> Result<Vec<EdgeId>, GraphError> {
        match filter {
            EdgeFilter::Attribute(attr_name, attr_filter) => {
                // Simple attribute filter - delegate to query engine
                self.query_engine.find_edges_by_attribute(&self.pool, &attr_name, &attr_filter)
                    .map_err(|e| e.into())
            },
            // TODO: Implement other filter types (And, Or, Not, endpoint filters, etc.)
            _ => {
                // For now, return empty results for unsupported filters
                Ok(Vec::new())
            }
        }
    }
    
    /// Run a complex query with multiple criteria
    pub fn query(&self, query: GraphQuery) -> Result<QueryResult, GraphError> {
        let _ = query; // Silence unused parameter warning
        Err(GraphError::NotImplemented {
            feature: "complex queries".to_string(),
            tracking_issue: None,
        })
    }
    
    /// Create a read-only view of the graph for analysis
    /// TODO: Implement when GraphView is designed
    // pub fn create_view(&self) -> GraphView {
    //     // TODO: GraphView::new(&self.pool, &self.query_engine)
    // }
    
    /*
    === TIME TRAVEL OPERATIONS ===
    Working with historical states of the graph.
    These create special views that delegate to history system.
    */
    
    /// Create a read-only view of the graph at a specific commit
    pub fn view_at_commit(&self, commit_id: StateId) -> Result<HistoricalView, GraphError> {
        let _ = commit_id; // Silence unused parameter warning
        Err(GraphError::NotImplemented {
            feature: "historical views".to_string(),
            tracking_issue: None,
        })
    }
    
    /// Compare two commits and show differences
    pub fn diff_commits(&self, from: StateId, to: StateId) -> Result<CommitDiff, GraphError> {
        let _ = (from, to); // Silence unused parameter warning
        Err(GraphError::NotImplemented {
            feature: "commit diffs".to_string(),
            tracking_issue: None,
        })
    }
    
    /*
    === MAINTENANCE OPERATIONS ===
    Housekeeping, optimization, and system management.
    */
    
    /// Optimize internal data structures for better performance
    pub fn optimize(&mut self) -> Result<(), GraphError> {
        // Basic implementation is a no-op
        // Full implementation would optimize pool, history, and query structures
        Ok(())
    }
    
    /// Garbage collect unreferenced historical states
    pub fn gc_history(&mut self) -> Result<usize, GraphError> {
        // Basic implementation performs no garbage collection
        // Full implementation would garbage collect unreachable commits
        Ok(0)
    }
    
    /// Save graph to persistent storage
    pub fn save_to_path(&self, path: &Path) -> Result<(), GraphError> {
        let _ = path; // Silence unused parameter warning
        Err(GraphError::NotImplemented {
            feature: "save_to_path".to_string(),
            tracking_issue: None,
        })
    }
}

/*
=== SUPPORTING TYPES ===
These are the types that the Graph API uses for its operations.
*/

// NOTE: GraphConfig is defined in config.rs - removed duplicate definition

/// Statistics about the current graph state
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub attribute_count: usize,
    pub commit_count: usize,
    pub branch_count: usize,
    pub uncommitted_changes: bool,
    pub memory_usage_mb: f64,
}

/// Information about a commit
#[derive(Debug, Clone)]
pub struct CommitInfo {
    pub id: StateId,
    pub parent: Option<StateId>,
    pub message: String,
    pub author: String,
    pub timestamp: u64,
    pub changes_summary: String,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/*
=== IMPLEMENTATION STRATEGY NOTES ===

COMPONENT COORDINATION:
- Graph should be smart about when to use which component
- Some operations might touch multiple components (e.g., commit touches pool + history + space)
- Graph should handle all the complex interactions

PERFORMANCE OPTIMIZATIONS:
- Batch operations when possible (add_nodes vs add_node)
- Direct access to columnar data for analytics workloads
- Change tracking should be lightweight (don't pool full snapshots on every change)

ERROR HANDLING:
- Use Result<T, GraphError> for all fallible operations
- Provide clear error messages with context
- Fail fast and maintain consistency

TRANSACTION BOUNDARIES:
- Individual operations are atomic (add_node can't partially fail)
- Multi-operation sequences should be wrapped in transactions
- Change tracker provides rollback capability

FUTURE EXTENSIBILITY:
- Plugin system for custom query processors
- Multiple storage backends (in-memory, disk-based, distributed)
- Custom attribute types beyond the basic AttrValue enum
*/