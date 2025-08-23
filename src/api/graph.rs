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

use crate::config::GraphConfig;
use crate::core::adjacency::{AdjacencyMatrix, AdjacencyMatrixBuilder, MatrixFormat, MatrixType};
use crate::core::change_tracker::ChangeTracker;
use crate::core::history::{CommitDiff, HistoricalView, HistoryForest};
use crate::core::neighborhood::NeighborhoodSampler;
use crate::core::pool::GraphPool;
use crate::core::query::{EdgeFilter, NodeFilter, QueryEngine};
use crate::core::ref_manager::BranchInfo;
use crate::core::space::GraphSpace;
use crate::core::traversal::TraversalEngine;
use crate::errors::{GraphError, GraphResult};
use crate::types::{
    AttrName, AttrValue, BranchName, CompressionStatistics, EdgeId, MemoryEfficiency,
    MemoryStatistics, NodeId, StateId,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

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
    /// Wrapped in Rc<RefCell<>> for interior mutability access from GraphSpace
    pool: Rc<RefCell<GraphPool>>,

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

    /// Graph traversal engine for BFS, DFS, pathfinding
    /// Handles connectivity analysis and traversal algorithms
    traversal_engine: TraversalEngine,

    /// Neighborhood subgraph sampler for generating neighborhood views
    /// Handles 1-hop, k-hop, and multi-node neighborhood generation
    neighborhood_sampler: NeighborhoodSampler,

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
    #[allow(dead_code)]
    config: GraphConfig,
}

impl Graph {
    /// Create a new empty graph with default settings (undirected by default)
    pub fn new() -> Self {
        Self::new_with_type(crate::types::GraphType::default())
    }

    /// Create a new empty graph with specified directionality
    ///
    /// # Arguments
    /// * `graph_type` - Whether the graph is directed or undirected
    ///
    /// # Examples
    /// ```
    /// use groggy::Graph;
    /// use groggy::types::GraphType;
    ///
    /// // Create a directed graph (like NetworkX DiGraph)
    /// let directed_graph = Graph::new_with_type(GraphType::Directed);
    ///
    /// // Create an undirected graph (like NetworkX Graph)  
    /// let undirected_graph = Graph::new_with_type(GraphType::Undirected);
    /// ```
    pub fn new_with_type(graph_type: crate::types::GraphType) -> Self {
        let mut config = GraphConfig::new();
        config.graph_type = graph_type;
        let pool = Rc::new(RefCell::new(GraphPool::new_with_type(graph_type)));
        Self {
            space: GraphSpace::new(pool.clone(), 0), // base state = 0
            pool,
            history: HistoryForest::new(),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            traversal_engine: TraversalEngine::new(),
            neighborhood_sampler: NeighborhoodSampler::new(),
            change_tracker: ChangeTracker::new(),
            config,
        }
    }

    /// Create a new directed graph (like NetworkX DiGraph)
    ///
    /// This is a convenience method equivalent to `new_with_type(GraphType::Directed)`
    pub fn new_directed() -> Self {
        Self::new_with_type(crate::types::GraphType::Directed)
    }

    /// Create a new undirected graph (like NetworkX Graph)
    ///
    /// This is a convenience method equivalent to `new_with_type(GraphType::Undirected)`
    pub fn new_undirected() -> Self {
        Self::new_with_type(crate::types::GraphType::Undirected)
    }

    /// Create a graph with custom configuration
    pub fn with_config(config: GraphConfig) -> Self {
        let pool = Rc::new(RefCell::new(GraphPool::new_with_type(config.graph_type)));
        Self {
            space: GraphSpace::new(pool.clone(), 0), // base state = 0
            pool,
            history: HistoryForest::new(),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            traversal_engine: TraversalEngine::new(),
            neighborhood_sampler: NeighborhoodSampler::new(),
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

    /// Get the graph type (directed or undirected)
    pub fn graph_type(&self) -> crate::types::GraphType {
        self.config.graph_type
    }

    /// Check if this graph is directed
    pub fn is_directed(&self) -> bool {
        matches!(self.config.graph_type, crate::types::GraphType::Directed)
    }

    /// Check if this graph is undirected
    pub fn is_undirected(&self) -> bool {
        matches!(self.config.graph_type, crate::types::GraphType::Undirected)
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
        let node_id = self.pool.borrow_mut().add_node(); // Pool creates and stores
        self.space.activate_node(node_id); // Space tracks as active
        self.change_tracker.record_node_addition(node_id); // Track change for commit
        node_id
    }

    /// Add multiple nodes efficiently
    /// OPTIMIZED: True bulk operation using vectorized pool operations
    pub fn add_nodes(&mut self, count: usize) -> Vec<NodeId> {
        // Use optimized bulk pool operation
        let (_start_id, _end_id, node_ids) = self.pool.borrow_mut().add_nodes_bulk(count);

        // Use optimized bulk space activation
        self.space.activate_nodes(node_ids.clone());

        // Single bulk change tracking update
        self.change_tracker.record_nodes_addition(&node_ids);

        node_ids
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
        let edge_id = self.pool.borrow_mut().add_edge(source, target); // Pool creates and stores
        self.space.activate_edge(edge_id, source, target); // Space tracks as active
        self.change_tracker
            .record_edge_addition(edge_id, source, target); // Track change
        Ok(edge_id)
    }

    /// Add multiple edges efficiently
    /// OPTIMIZED: True bulk operation with batch validation and vectorized operations
    pub fn add_edges(&mut self, edges: &[(NodeId, NodeId)]) -> Vec<EdgeId> {
        // Pre-filter valid edges in single pass
        let valid_edges: Vec<_> = edges
            .iter()
            .filter(|&&(source, target)| {
                self.space.contains_node(source) && self.space.contains_node(target)
            })
            .cloned()
            .collect();

        if valid_edges.is_empty() {
            return Vec::new();
        }

        // Use optimized bulk pool operation
        let edge_ids = self.pool.borrow_mut().add_edges(&valid_edges);

        // Use optimized bulk space activation
        self.space.activate_edges(edge_ids.clone());

        // Prepare data for change tracking
        let change_data: Vec<_> = edge_ids
            .iter()
            .zip(valid_edges.iter())
            .map(|(&edge_id, &(source, target))| (edge_id, source, target))
            .collect();

        // Single bulk change tracking update
        self.change_tracker.record_edges_addition(&change_data);

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
    pub fn set_node_attr(
        &mut self,
        node: NodeId,
        attr: AttrName,
        value: AttrValue,
    ) -> Result<(), GraphError> {
        // 1. Validate node is active
        if !self.space.contains_node(node) {
            return Err(GraphError::node_not_found(node, "set attribute"));
        }

        // 2. Get old index for change tracking
        let old_index = self.space.get_node_attr_index(node, &attr);

        // 3. Pool stores value and returns new index (is_node = true)
        let new_index = self.pool.borrow_mut().set_attr(attr.clone(), value, true);

        // 4. Space updates current mapping
        self.space
            .set_node_attr_index(node, attr.clone(), new_index);

        // 5. ChangeTracker records the delta
        self.change_tracker
            .record_attr_change(node, attr, old_index, new_index, true);

        Ok(())
    }

    /// Set node attributes in bulk (OPTIMIZED: True vectorized bulk operation)
    pub fn set_node_attrs(
        &mut self,
        attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> Result<(), GraphError> {
        // Batch validation - check all nodes exist upfront
        for node_values in attrs_values.values() {
            for &(node_id, _) in node_values {
                if !self.space.contains_node(node_id) {
                    return Err(GraphError::node_not_found(
                        node_id,
                        "set bulk node attributes",
                    ));
                }
            }
        }

        // Use optimized vectorized pool operation
        let index_changes = self.pool.borrow_mut().set_bulk_attrs(attrs_values, true);

        // Update space attribute indices in bulk
        for (attr_name, entity_indices) in index_changes {
            for (node_id, new_index) in entity_indices {
                self.space
                    .set_node_attr_index(node_id, attr_name.clone(), new_index);
            }
        }

        // TODO: Bulk change tracking for attributes
        // For now, we skip individual change tracking for bulk operations
        // This could be optimized further with bulk change recording

        Ok(())
    }

    /// Set an attribute value on an edge
    ///
    /// ALGORITHM:
    /// 1. Pool sets value and returns baseline (integrated change tracking)
    /// 2. Space records the change for commit delta
    pub fn set_edge_attr(
        &mut self,
        edge: EdgeId,
        attr: AttrName,
        value: AttrValue,
    ) -> Result<(), GraphError> {
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
        let new_index = self.pool.borrow_mut().set_attr(attr.clone(), value, false);

        // 4. Space updates current mapping
        self.space
            .set_edge_attr_index(edge, attr.clone(), new_index);

        // 5. ChangeTracker records the delta
        self.change_tracker
            .record_attr_change(edge, attr, old_index, new_index, false);

        Ok(())
    }

    /// Set edge attributes in bulk (OPTIMIZED: True vectorized bulk operation)
    pub fn set_edge_attrs(
        &mut self,
        attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> Result<(), GraphError> {
        // Batch validation - check all edges exist upfront
        for edge_values in attrs_values.values() {
            for &(edge_id, _) in edge_values {
                if !self.space.contains_edge(edge_id) {
                    return Err(GraphError::edge_not_found(
                        edge_id,
                        "set bulk edge attributes",
                    ));
                }
            }
        }

        // Use optimized vectorized pool operation
        let index_changes = self.pool.borrow_mut().set_bulk_attrs(attrs_values, false);

        // Update space attribute indices in bulk
        for (attr_name, entity_indices) in index_changes {
            for (edge_id, new_index) in entity_indices {
                self.space
                    .set_edge_attr_index(edge_id, attr_name.clone(), new_index);
            }
        }

        // TODO: Bulk change tracking for attributes
        // For now, we skip individual change tracking for bulk operations
        // This could be optimized further with bulk change recording

        Ok(())
    }

    /// Get an attribute value from a node
    ///
    /// ALGORITHM:
    /// 1. Ask pool to get the current value
    /// 2. Return the value
    pub fn get_node_attr(
        &self,
        node: NodeId,
        attr: &AttrName,
    ) -> Result<Option<AttrValue>, GraphError> {
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
            Ok(self
                .pool
                .borrow()
                .get_attr_by_index(attr, index, true)
                .cloned())
        } else {
            Ok(None) // Attribute not set for this node
        }
    }

    /// Get an attribute value from an edge
    ///
    /// ALGORITHM:
    /// 1. Ask pool to get the current value
    /// 2. Return the value
    pub fn get_edge_attr(
        &self,
        edge: EdgeId,
        attr: &AttrName,
    ) -> Result<Option<AttrValue>, GraphError> {
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
            Ok(self
                .pool
                .borrow()
                .get_attr_by_index(attr, index, false)
                .cloned())
        } else {
            Ok(None) // Attribute not set for this edge
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
            if let Some(value) = self
                .pool
                .borrow()
                .get_attr_by_index(&attr_name, index, true)
            {
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
            if let Some(value) = self
                .pool
                .borrow()
                .get_attr_by_index(&attr_name, index, false)
            {
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
    /// 1. Get full attribute column from pool (O(1) HashMap lookup)
    /// 2. For each requested node: get its attribute index and extract value
    /// 3. Return results aligned with requested nodes
    ///
    /// SECURITY: Only returns data for active nodes that were explicitly requested
    /// PERFORMANCE: Uses bulk columnar access - much faster than individual get_node_attr calls
    pub fn get_nodes_attrs(
        &self,
        attr: &AttrName,
        requested_nodes: &[NodeId],
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        let mut results = Vec::with_capacity(requested_nodes.len());

        // Get the attribute column once (O(1) HashMap lookup)
        if let Some(attr_column) = self.pool.borrow().get_node_attribute_column(attr) {
            // Bulk process all nodes using direct column access
            for &node_id in requested_nodes {
                if !self.space.contains_node(node_id) {
                    results.push(None); // Node doesn't exist
                } else if let Some(index) = self.space.get_node_attr_index(node_id, attr) {
                    // Direct O(1) access to column value
                    results.push(attr_column.get(index).cloned());
                } else {
                    results.push(None); // Attribute not set for this node
                }
            }
        } else {
            // Attribute doesn't exist in the graph
            results.resize(requested_nodes.len(), None);
        }

        Ok(results)
    }

    /// Get attribute values for specific edges (secure and efficient)
    pub fn get_edges_attrs(
        &self,
        attr: &AttrName,
        requested_edges: &[EdgeId],
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        let mut results = Vec::with_capacity(requested_edges.len());

        // Get the attribute column once (O(1) HashMap lookup)
        if let Some(attr_column) = self.pool.borrow().get_edge_attribute_column(attr) {
            // Bulk process all edges using direct column access
            for &edge_id in requested_edges {
                if !self.space.contains_edge(edge_id) {
                    results.push(None); // Edge doesn't exist
                } else if let Some(index) = self.space.get_edge_attr_index(edge_id, attr) {
                    // Direct O(1) access to column value
                    results.push(attr_column.get(index).cloned());
                } else {
                    results.push(None); // Attribute not set for this edge
                }
            }
        } else {
            // Attribute doesn't exist in the graph
            results.resize(requested_edges.len(), None);
        }

        Ok(results)
    }

    /// INTERNAL: Get attribute column for ALL active nodes (optimized for GraphTable)
    ///
    /// This is the key optimization for GraphTable - instead of O(n*m) individual calls,
    /// we make O(m) bulk calls to get complete attribute columns.
    ///
    /// PERFORMANCE: Single column access + filtering by active nodes only
    /// USAGE: Internal use by table(), DataFrame creation, bulk data export
    pub fn _get_node_attribute_column(
        &self,
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        let node_ids = self.space.node_ids(); // Get all active node IDs
        let mut results = Vec::with_capacity(node_ids.len());

        // Get the attribute column once (O(1) HashMap lookup)
        if let Some(attr_column) = self.pool.borrow().get_node_attribute_column(attr) {
            // Process all active nodes using direct column access
            for node_id in node_ids {
                if let Some(index) = self.space.get_node_attr_index(node_id, attr) {
                    results.push(attr_column.get(index).cloned());
                } else {
                    results.push(None); // Attribute not set for this node
                }
            }
        } else {
            // Attribute doesn't exist in the graph
            results.resize(node_ids.len(), None);
        }

        Ok(results)
    }

    /// INTERNAL: Get attribute column for ALL active edges (optimized for GraphTable)
    pub fn _get_edge_attribute_column(
        &self,
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        let edge_ids = self.space.edge_ids(); // Get all active edge IDs
        let mut results = Vec::with_capacity(edge_ids.len());

        // Get the attribute column once (O(1) HashMap lookup)
        if let Some(attr_column) = self.pool.borrow().get_edge_attribute_column(attr) {
            // Process all active edges using direct column access
            for edge_id in edge_ids {
                if let Some(index) = self.space.get_edge_attr_index(edge_id, attr) {
                    results.push(attr_column.get(index).cloned());
                } else {
                    results.push(None); // Attribute not set for this edge
                }
            }
        } else {
            // Attribute doesn't exist in the graph
            results.resize(edge_ids.len(), None);
        }

        Ok(results)
    }

    /// Get attribute column for specific nodes (optimized for subgraph tables)
    ///
    /// INTERNAL: This enables subgraph.table() to be as efficient as graph.table()
    /// by using bulk column access instead of individual attribute calls.
    pub fn _get_node_attributes_for_nodes(
        &self,
        node_ids: &[NodeId],
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        // This is the same as get_nodes_attrs but with a more descriptive name
        // for use in subgraph table creation
        self.get_nodes_attrs(attr, node_ids)
    }

    /// INTERNAL: Get attribute column for specific edges (optimized for subgraph edge tables)
    pub fn _get_edge_attributes_for_edges(
        &self,
        edge_ids: &[EdgeId],
        attr: &AttrName,
    ) -> GraphResult<Vec<Option<AttrValue>>> {
        // This is the same as get_edges_attrs but with a more descriptive name
        // for use in subgraph edge table creation
        self.get_edges_attrs(attr, edge_ids)
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

    /// Get the degree of a node (number of incident edges)
    pub fn degree(&self, node: NodeId) -> Result<usize, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get degree".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }

        // Get fresh topology snapshot
        let (_, sources, targets, _) = self.space.snapshot(&self.pool.borrow());
        let mut count = 0;
        for i in 0..sources.len() {
            if sources[i] == node || targets[i] == node {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get the neighbors of a node
    pub fn neighbors(&self, node: NodeId) -> Result<Vec<NodeId>, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get neighbors".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }

        // Get fresh adjacency snapshot - much more efficient than columnar scan
        let (_, _, _, neighbors_map) = self.space.snapshot(&self.pool.borrow());

        if let Some(neighbors) = neighbors_map.get(&node) {
            // Extract just the neighbor nodes (not the edge IDs)
            let neighbor_nodes: Vec<NodeId> =
                neighbors.iter().map(|(neighbor, _)| *neighbor).collect();
            Ok(neighbor_nodes)
        } else {
            // Node exists but has no neighbors
            Ok(Vec::new())
        }
    }

    /// Get columnar topology vectors for efficient subgraph operations
    ///
    /// Returns (edge_ids, sources, targets) as parallel vectors where:
    /// - edge_ids[i] is the EdgeId
    /// - sources[i] is the source NodeId of that edge
    /// - targets[i] is the target NodeId of that edge
    ///
    /// This is used internally for optimized operations like subgraph edge calculation.
    pub fn get_columnar_topology(&self) -> (Arc<Vec<EdgeId>>, Arc<Vec<NodeId>>, Arc<Vec<NodeId>>) {
        let (edge_ids, sources, targets, _) = self.space.snapshot(&self.pool.borrow());
        (edge_ids, sources, targets)
    }

    /// Get the endpoints of an edge
    pub fn edge_endpoints(&self, edge: EdgeId) -> Result<(NodeId, NodeId), GraphError> {
        self.pool
            .borrow()
            .get_edge_endpoints(edge)
            .ok_or_else(|| GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get endpoints".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            })
    }

    /// Get basic statistics about the current graph
    pub fn statistics(&self) -> GraphStatistics {
        let _pool_stats = self.pool.borrow().statistics();
        let _history_stats = self.history.statistics();

        // Accurate memory calculation using new memory monitoring
        // let memory_stats = self.memory_statistics();

        // Simple memory calculation to avoid stack overflow
        let pool_stats = self.pool.borrow().statistics();
        let history_stats = self.history.statistics();

        // Basic memory estimate
        let base_memory = 1024 * 1024; // 1MB base
        let entity_memory = (pool_stats.node_count + pool_stats.edge_count) * 100; // rough estimate
        let total_memory_mb = (base_memory + entity_memory) as f64 / (1024.0 * 1024.0);

        GraphStatistics {
            node_count: self.space.node_count(),
            edge_count: self.space.edge_count(),
            attribute_count: pool_stats.node_attribute_count + pool_stats.edge_attribute_count,
            commit_count: history_stats.total_commits,
            branch_count: history_stats.total_branches,
            uncommitted_changes: self.has_uncommitted_changes(),
            memory_usage_mb: total_memory_mb,
        }
    }

    /// Get comprehensive memory statistics (Memory Optimization 4)
    pub fn memory_statistics(&self) -> MemoryStatistics {
        // Calculate component memory usage
        let pool_memory = self.calculate_pool_memory();
        let space_memory = self.calculate_space_memory();
        let history_memory = self.calculate_history_memory();
        let change_tracker_memory = self.change_tracker.memory_usage();

        let total_bytes = pool_memory + space_memory + history_memory + change_tracker_memory;

        MemoryStatistics {
            pool_memory_bytes: pool_memory,
            space_memory_bytes: space_memory,
            history_memory_bytes: history_memory,
            change_tracker_memory_bytes: change_tracker_memory,
            total_memory_bytes: total_bytes,
            total_memory_mb: total_bytes as f64 / (1024.0 * 1024.0),
            memory_efficiency: self.calculate_memory_efficiency(total_bytes),
            compression_stats: self.calculate_compression_stats(),
        }
    }

    /// Calculate pool memory usage with detailed breakdown
    fn calculate_pool_memory(&self) -> usize {
        // Basic structure overhead
        let base_size = std::mem::size_of::<crate::core::pool::GraphPool>();

        // Node and edge storage
        let topology_size = std::mem::size_of::<
            std::collections::HashMap<
                crate::types::EdgeId,
                (crate::types::NodeId, crate::types::NodeId),
            >,
        >();

        // Attribute storage (this is where the main memory is)
        let node_attrs_size = self.estimate_attribute_memory(true);
        let edge_attrs_size = self.estimate_attribute_memory(false);

        base_size + topology_size + node_attrs_size + edge_attrs_size
    }

    /// Estimate attribute memory usage (with access to pool internals)
    fn estimate_attribute_memory(&self, is_node: bool) -> usize {
        // This is a simplified estimate - in a real implementation,
        // we'd need access to pool internals or expose memory_usage() methods
        let attr_count = if is_node {
            self.pool.borrow().statistics().node_attribute_count
        } else {
            self.pool.borrow().statistics().edge_attribute_count
        };

        // Rough estimate: assume average of 100 bytes per attribute value
        attr_count * 100
    }

    /// Calculate space memory usage
    fn calculate_space_memory(&self) -> usize {
        let base_size = std::mem::size_of::<crate::core::space::GraphSpace>();

        // Active sets
        let active_nodes_size =
            self.space.node_count() * std::mem::size_of::<crate::types::NodeId>();
        let active_edges_size =
            self.space.edge_count() * std::mem::size_of::<crate::types::EdgeId>();

        // Topology cache
        let topology_cache_size = self.space.edge_count() * 3 * std::mem::size_of::<usize>(); // edge_ids, sources, targets

        // Attribute index maps (simplified estimate)
        let index_maps_size = (self.space.node_count() + self.space.edge_count()) * 50; // rough estimate

        base_size + active_nodes_size + active_edges_size + topology_cache_size + index_maps_size
    }

    /// Calculate history memory usage
    fn calculate_history_memory(&self) -> usize {
        // Simplified estimate - in real implementation, would query history component
        let base_size = std::mem::size_of::<crate::core::history::HistoryForest>();
        let commits = self.statistics().commit_count;
        let estimated_commit_size = 1000; // bytes per commit (rough estimate)

        base_size + commits * estimated_commit_size
    }

    /// Calculate memory efficiency metrics
    fn calculate_memory_efficiency(&self, total_memory_bytes: usize) -> MemoryEfficiency {
        let total_entities = self.space.node_count() + self.space.edge_count();

        let bytes_per_entity = if total_entities > 0 {
            total_memory_bytes as f64 / total_entities as f64
        } else {
            0.0
        };

        // Memory overhead ratio (structure overhead vs actual data)
        let estimated_data_size = total_entities * 32; // rough estimate of minimal entity data
        let overhead_ratio = if estimated_data_size > 0 {
            (total_memory_bytes as f64 - estimated_data_size as f64) / estimated_data_size as f64
        } else {
            0.0
        };

        MemoryEfficiency {
            bytes_per_node: if self.space.node_count() > 0 {
                total_memory_bytes as f64 / self.space.node_count() as f64
            } else {
                0.0
            },
            bytes_per_edge: if self.space.edge_count() > 0 {
                total_memory_bytes as f64 / self.space.edge_count() as f64
            } else {
                0.0
            },
            bytes_per_entity,
            overhead_ratio,
            cache_efficiency: 0.95, // Placeholder - would be calculated from actual cache hit rates
        }
    }
    /// Calculate compression statistics
    fn calculate_compression_stats(&self) -> CompressionStatistics {
        // This would require querying the pool for compression ratios
        // For now, provide placeholder statistics
        let pool_stats = self.pool.borrow().statistics();
        let total_attributes = pool_stats.node_attribute_count + pool_stats.edge_attribute_count;

        CompressionStatistics {
            compressed_attributes: 0,
            total_attributes,
            average_compression_ratio: 1.0,
            memory_saved_bytes: 0,
            memory_saved_percentage: 0.0,
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
        let parent = if self.current_commit == 0 {
            None
        } else {
            Some(self.current_commit)
        };

        // Create commit in history
        let new_commit_id = self
            .history
            .create_commit(changeset, message, author, parent)?;

        // Update current commit and branch head
        self.current_commit = new_commit_id;
        self.history
            .update_branch_head(&self.current_branch, new_commit_id)?;

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
        self.history
            .create_branch(branch_name, self.current_commit)?;
        Ok(())
    }

    /// Switch to a different branch
    pub fn checkout_branch(&mut self, branch_name: BranchName) -> Result<(), GraphError> {
        // 1. Validate the branch exists
        let target_head = self.history.get_branch_head(&branch_name)?;

        // 2. Check for uncommitted changes
        if self.has_uncommitted_changes() {
            return Err(GraphError::InvalidInput(
                "Cannot switch branches with uncommitted changes. Please commit or reset first."
                    .to_string(),
            ));
        }

        // 3. Reconstruct the graph state at the target branch's head
        if target_head != self.current_commit {
            let target_snapshot = self.history.reconstruct_state_at(target_head)?;

            // 4. Reset the current graph state to match the target snapshot
            self.reset_to_snapshot(target_snapshot)?;
        }

        // 5. Update current branch and commit pointers
        self.current_branch = branch_name;
        self.current_commit = target_head;

        // 6. Clear any change tracking since we're at a clean state
        self.change_tracker.clear();

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
        self.query_engine
            .find_nodes_by_filter_with_space(&self.pool.borrow(), &self.space, &filter)
            .map_err(|e| e.into())
    }

    /// Find edges matching attribute criteria
    pub fn find_edges(&mut self, filter: EdgeFilter) -> Result<Vec<EdgeId>, GraphError> {
        self.query_engine
            .find_edges_by_filter_with_space(&self.pool.borrow(), &self.space, &filter)
            .map_err(|e| e.into())
    }

    /*
    === GRAPH TRAVERSAL OPERATIONS ===
    Advanced graph algorithms for traversal, pathfinding, and connectivity analysis.
    These delegate to the traversal_engine for optimized traversal algorithms.
    */

    /// Perform Breadth-First Search from a starting node
    pub fn bfs(
        &mut self,
        start: NodeId,
        options: crate::core::traversal::TraversalOptions,
    ) -> Result<crate::core::traversal::TraversalResult, GraphError> {
        self.traversal_engine
            .bfs(&self.pool.borrow(), &mut self.space, start, options)
            .map_err(|e| e.into())
    }

    /// Perform Depth-First Search from a starting node
    pub fn dfs(
        &mut self,
        start: NodeId,
        options: crate::core::traversal::TraversalOptions,
    ) -> Result<crate::core::traversal::TraversalResult, GraphError> {
        self.traversal_engine
            .dfs(&self.pool.borrow(), &mut self.space, start, options)
            .map_err(|e| e.into())
    }

    /// Find shortest path between two nodes
    pub fn shortest_path(
        &mut self,
        start: NodeId,
        end: NodeId,
        options: crate::core::traversal::PathFindingOptions,
    ) -> Result<Option<crate::core::traversal::Path>, GraphError> {
        self.traversal_engine
            .shortest_path(&self.pool.borrow(), &mut self.space, start, end, options)
            .map_err(|e| e.into())
    }

    /// Find all simple paths between two nodes
    pub fn all_paths(
        &mut self,
        start: NodeId,
        end: NodeId,
        max_length: usize,
    ) -> Result<Vec<crate::core::traversal::Path>, GraphError> {
        self.traversal_engine
            .all_paths(&self.pool.borrow(), &mut self.space, start, end, max_length)
            .map_err(|e| e.into())
    }

    /// Find all connected components in the graph
    pub fn connected_components(
        &mut self,
        options: crate::core::traversal::TraversalOptions,
    ) -> Result<crate::core::traversal::ConnectedComponentsResult, GraphError> {
        self.traversal_engine
            .connected_components(&self.pool.borrow(), &mut self.space, options)
            .map_err(|e| e.into())
    }

    /// Get traversal performance statistics
    pub fn traversal_statistics(&self) -> &crate::core::traversal::TraversalStats {
        self.traversal_engine.statistics()
    }

    // ===== NEIGHBORHOOD SUBGRAPH SAMPLING =====

    /// Generate 1-hop neighborhood subgraph for a single node
    /// Returns a Subgraph containing the central node and all its direct neighbors
    pub fn neighborhood(
        &mut self,
        node_id: NodeId,
    ) -> Result<crate::core::neighborhood::NeighborhoodSubgraph, GraphError> {
        self.neighborhood_sampler
            .single_neighborhood(&self.pool.borrow(), &self.space, node_id)
            .map_err(|e| e.into())
    }

    /// Generate 1-hop neighborhoods for multiple nodes
    /// Returns separate neighborhood subgraphs for each central node
    pub fn multi_neighborhood(
        &mut self,
        node_ids: &[NodeId],
    ) -> Result<crate::core::neighborhood::NeighborhoodResult, GraphError> {
        self.neighborhood_sampler
            .multi_neighborhood(&self.pool.borrow(), &self.space, node_ids)
            .map_err(|e| e.into())
    }

    /// Generate k-hop neighborhood subgraph for a single node
    /// Returns a Subgraph containing all nodes within k hops of the central node
    pub fn k_hop_neighborhood(
        &mut self,
        node_id: NodeId,
        k: usize,
    ) -> Result<crate::core::neighborhood::NeighborhoodSubgraph, GraphError> {
        self.neighborhood_sampler
            .k_hop_neighborhood(&self.pool.borrow(), &self.space, node_id, k)
            .map_err(|e| e.into())
    }

    /// Generate unified neighborhood for multiple nodes
    /// Returns a single subgraph containing all nodes and their combined k-hop neighborhoods
    pub fn unified_neighborhood(
        &mut self,
        node_ids: &[NodeId],
        k: usize,
    ) -> Result<crate::core::neighborhood::NeighborhoodSubgraph, GraphError> {
        self.neighborhood_sampler
            .unified_neighborhood(&self.pool.borrow(), &self.space, node_ids, k)
            .map_err(|e| e.into())
    }

    /// Get neighborhood sampling performance statistics
    pub fn neighborhood_statistics(&self) -> &crate::core::neighborhood::NeighborhoodStats {
        self.neighborhood_sampler.stats()
    }

    // TODO: Implement complex query composition when needed

    /// Create a new complex query builder
    pub fn query(&self) -> Result<(), GraphError> {
        Err(GraphError::NotImplemented {
            feature: "complex query builder".to_string(),
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
    pub fn view_at_commit(&self, commit_id: StateId) -> Result<HistoricalView<'_>, GraphError> {
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
    === INTERNAL STATE MANAGEMENT ===
    Helper methods for branch switching and state reconstruction
    */

    /// Reset the current graph state (pool + space) to match a historical snapshot
    /// This is used during branch switching to restore the graph to a specific state
    fn reset_to_snapshot(
        &mut self,
        snapshot: crate::core::state::GraphSnapshot,
    ) -> Result<(), GraphError> {
        // 1. Clear current state
        self.pool = Rc::new(RefCell::new(crate::core::pool::GraphPool::new()));
        self.space = crate::core::space::GraphSpace::new(self.pool.clone(), snapshot.state_id);

        // 2. Restore nodes
        for &node_id in &snapshot.active_nodes {
            // Ensure node ID exists in pool
            self.pool.borrow_mut().ensure_node_id_exists(node_id);
            // Activate in space
            self.space.activate_node(node_id);

            // Restore node attributes
            if let Some(attrs) = snapshot.node_attributes.get(&node_id) {
                for (attr_name, attr_value) in attrs {
                    let index = self.pool.borrow_mut().set_attr(
                        attr_name.clone(),
                        attr_value.clone(),
                        true,
                    );
                    self.space
                        .set_node_attr_index(node_id, attr_name.clone(), index);
                }
            }
        }

        // 3. Restore edges
        for (&edge_id, &(source, target)) in &snapshot.edges {
            // Store topology in pool with specific ID
            self.pool
                .borrow_mut()
                .add_edge_with_id(edge_id, source, target);
            // Activate in space
            self.space.activate_edge(edge_id, source, target);

            // Restore edge attributes
            if let Some(attrs) = snapshot.edge_attributes.get(&edge_id) {
                for (attr_name, attr_value) in attrs {
                    let index = self.pool.borrow_mut().set_attr(
                        attr_name.clone(),
                        attr_value.clone(),
                        false,
                    );
                    self.space
                        .set_edge_attr_index(edge_id, attr_name.clone(), index);
                }
            }
        }

        Ok(())
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

    /*
    === AGGREGATION AND ANALYTICS ===
    Statistical operations and data analysis functionality.
    */

    /// Compute aggregate statistics for a node attribute
    pub fn aggregate_node_attribute(
        &self,
        attr_name: &AttrName,
        operation: &str,
    ) -> Result<AggregationResult, GraphError> {
        // Get all active nodes
        let node_ids: Vec<NodeId> = self.space.get_active_nodes().iter().copied().collect();

        // Use bulk attribute retrieval for much better performance (10-100x faster than individual lookups)
        let bulk_attributes = self._get_node_attributes_for_nodes(&node_ids, attr_name)?;
        let mut values = Vec::new();

        // Extract values from bulk result
        for value in bulk_attributes.into_iter().flatten() {
            values.push(value);
        }

        if values.is_empty() {
            return Ok(AggregationResult::new(0.0));
        }

        // Perform the requested aggregation
        let result = match operation {
            "count" => values.len() as f64,
            "average" | "mean" => {
                let sum = values
                    .iter()
                    .fold(0.0, |acc, val| acc + extract_numeric(val));
                sum / values.len() as f64
            }
            "sum" => values
                .iter()
                .fold(0.0, |acc, val| acc + extract_numeric(val)),
            "min" => values
                .iter()
                .fold(f64::INFINITY, |acc, val| acc.min(extract_numeric(val))),
            "max" => values
                .iter()
                .fold(f64::NEG_INFINITY, |acc, val| acc.max(extract_numeric(val))),
            "stddev" => {
                let mean = values
                    .iter()
                    .fold(0.0, |acc, val| acc + extract_numeric(val))
                    / values.len() as f64;
                let variance = values.iter().fold(0.0, |acc, val| {
                    let diff = extract_numeric(val) - mean;
                    acc + diff * diff
                }) / values.len() as f64;
                variance.sqrt()
            }
            "median" => {
                let mut numeric_values: Vec<f64> = values.iter().map(extract_numeric).collect();
                numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = numeric_values.len() / 2;
                if numeric_values.len() % 2 == 0 {
                    (numeric_values[mid - 1] + numeric_values[mid]) / 2.0
                } else {
                    numeric_values[mid]
                }
            }
            "percentile_95" => {
                let mut numeric_values: Vec<f64> = values.iter().map(extract_numeric).collect();
                numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let index = ((numeric_values.len() as f64 - 1.0) * 0.95).round() as usize;
                numeric_values[index.min(numeric_values.len() - 1)]
            }
            "unique_count" => {
                let mut unique_values = std::collections::HashSet::new();
                for value in values {
                    unique_values.insert(value);
                }
                unique_values.len() as f64
            }
            _ => {
                return Err(GraphError::InvalidInput(format!(
                    "Unsupported aggregation operation: {}",
                    operation
                )))
            }
        };

        Ok(AggregationResult::new(result))
    }

    /// Compute aggregate statistics for an edge attribute  
    pub fn aggregate_edge_attribute(
        &self,
        attr_name: &AttrName,
        operation: &str,
    ) -> Result<AggregationResult, GraphError> {
        // Get all active edges
        let edge_ids: Vec<EdgeId> = self.space.get_active_edges().iter().copied().collect();

        // Use bulk attribute retrieval for much better performance (10-100x faster than individual lookups)
        let bulk_attributes = self._get_edge_attributes_for_edges(&edge_ids, attr_name)?;
        let mut values = Vec::new();

        // Extract values from bulk result
        for value in bulk_attributes.into_iter().flatten() {
            values.push(value);
        }

        if values.is_empty() {
            return Ok(AggregationResult::new(0.0));
        }

        // Perform the requested aggregation (same logic as node aggregation)
        let result = match operation {
            "count" => values.len() as f64,
            "average" | "mean" => {
                let sum = values
                    .iter()
                    .fold(0.0, |acc, val| acc + extract_numeric(val));
                sum / values.len() as f64
            }
            "sum" => values
                .iter()
                .fold(0.0, |acc, val| acc + extract_numeric(val)),
            "min" => values
                .iter()
                .fold(f64::INFINITY, |acc, val| acc.min(extract_numeric(val))),
            "max" => values
                .iter()
                .fold(f64::NEG_INFINITY, |acc, val| acc.max(extract_numeric(val))),
            "stddev" => {
                let mean = values
                    .iter()
                    .fold(0.0, |acc, val| acc + extract_numeric(val))
                    / values.len() as f64;
                let variance = values.iter().fold(0.0, |acc, val| {
                    let diff = extract_numeric(val) - mean;
                    acc + diff * diff
                }) / values.len() as f64;
                variance.sqrt()
            }
            _ => {
                return Err(GraphError::InvalidInput(format!(
                    "Unsupported aggregation operation: {}",
                    operation
                )))
            }
        };

        Ok(AggregationResult::new(result))
    }

    /// Group nodes by attribute value and compute aggregates for each group
    /// OPTIMIZED: O(N) algorithm using bulk attribute retrieval instead of O(N²) individual lookups
    pub fn group_nodes_by_attribute(
        &self,
        group_by_attr: &AttrName,
        aggregate_attr: &AttrName,
        operation: &str,
    ) -> Result<std::collections::HashMap<AttrValue, AggregationResult>, GraphError> {
        // Get all active nodes
        let node_ids: Vec<NodeId> = self.space.get_active_nodes().iter().copied().collect();

        if node_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        // BULK OPERATION 1: Get group_by attribute for all nodes at once (O(N))
        let pool_ref = self.pool.borrow();
        let group_by_values =
            self.space
                .get_attributes_for_nodes(&pool_ref, group_by_attr, &node_ids);

        // BULK OPERATION 2: Get aggregate attribute for all nodes at once (O(N))
        let aggregate_values =
            self.space
                .get_attributes_for_nodes(&pool_ref, aggregate_attr, &node_ids);

        // Create lookup maps for efficient access
        let group_by_map: std::collections::HashMap<NodeId, &AttrValue> = group_by_values
            .iter()
            .filter_map(|(node_id, opt_val)| opt_val.map(|val| (*node_id, val)))
            .collect();

        let aggregate_map: std::collections::HashMap<NodeId, &AttrValue> = aggregate_values
            .iter()
            .filter_map(|(node_id, opt_val)| opt_val.map(|val| (*node_id, val)))
            .collect();

        // Group nodes by attribute value and collect aggregate values (O(N))
        let mut groups: std::collections::HashMap<AttrValue, Vec<AttrValue>> =
            std::collections::HashMap::new();

        for &node_id in &node_ids {
            if let (Some(&group_val), Some(&agg_val)) =
                (group_by_map.get(&node_id), aggregate_map.get(&node_id))
            {
                groups
                    .entry(group_val.clone())
                    .or_default()
                    .push(agg_val.clone());
            }
        }

        // Compute aggregations for each group (O(N) total across all groups)
        let mut results = std::collections::HashMap::new();

        for (group_value, values) in groups {
            if !values.is_empty() {
                let result = match operation {
                    "count" => values.len() as f64,
                    "average" | "mean" => {
                        let sum = values
                            .iter()
                            .fold(0.0, |acc, val| acc + extract_numeric(val));
                        sum / values.len() as f64
                    }
                    "sum" => values
                        .iter()
                        .fold(0.0, |acc, val| acc + extract_numeric(val)),
                    "min" => values
                        .iter()
                        .fold(f64::INFINITY, |acc, val| acc.min(extract_numeric(val))),
                    "max" => values
                        .iter()
                        .fold(f64::NEG_INFINITY, |acc, val| acc.max(extract_numeric(val))),
                    _ => {
                        return Err(GraphError::InvalidInput(format!(
                            "Unsupported aggregation operation: {}",
                            operation
                        )))
                    }
                };

                results.insert(group_value, AggregationResult::new(result));
            }
        }

        Ok(results)
    }

    // ===== ADJACENCY MATRIX OPERATIONS =====

    /// Generate adjacency matrix for the entire graph
    pub fn adjacency_matrix(&mut self) -> GraphResult<AdjacencyMatrix> {
        AdjacencyMatrixBuilder::new().build_full_graph(&self.pool.borrow(), &mut self.space)
    }

    /// Generate weighted adjacency matrix using specified edge attribute
    pub fn weighted_adjacency_matrix(&mut self, weight_attr: &str) -> GraphResult<AdjacencyMatrix> {
        AdjacencyMatrixBuilder::new()
            .matrix_type(MatrixType::Weighted {
                weight_attr: Some(weight_attr.to_string()),
            })
            .build_full_graph(&self.pool.borrow(), &mut self.space)
    }

    /// Generate dense adjacency matrix
    pub fn dense_adjacency_matrix(&mut self) -> GraphResult<AdjacencyMatrix> {
        AdjacencyMatrixBuilder::new()
            .format(MatrixFormat::Dense)
            .build_full_graph(&self.pool.borrow(), &mut self.space)
    }

    /// Generate sparse adjacency matrix
    pub fn sparse_adjacency_matrix(&mut self) -> GraphResult<AdjacencyMatrix> {
        AdjacencyMatrixBuilder::new()
            .format(MatrixFormat::Sparse)
            .build_full_graph(&self.pool.borrow(), &mut self.space)
    }

    /// Generate Laplacian matrix
    pub fn laplacian_matrix(&mut self, normalized: bool) -> GraphResult<AdjacencyMatrix> {
        AdjacencyMatrixBuilder::new()
            .matrix_type(MatrixType::Laplacian { normalized })
            .build_full_graph(&self.pool.borrow(), &mut self.space)
    }

    /// Generate adjacency matrix for a subgraph with specific nodes
    pub fn subgraph_adjacency_matrix(
        &mut self,
        node_ids: &[NodeId],
    ) -> GraphResult<AdjacencyMatrix> {
        AdjacencyMatrixBuilder::new().build_subgraph(&self.pool.borrow(), &mut self.space, node_ids)
    }

    /// Generate custom adjacency matrix with full control
    pub fn custom_adjacency_matrix(
        &mut self,
        format: MatrixFormat,
        matrix_type: MatrixType,
        compact_indexing: bool,
        node_ids: Option<&[NodeId]>,
    ) -> GraphResult<AdjacencyMatrix> {
        let builder = AdjacencyMatrixBuilder::new()
            .format(format)
            .matrix_type(matrix_type)
            .compact_indexing(compact_indexing);

        if let Some(nodes) = node_ids {
            builder.build_subgraph(&self.pool.borrow(), &mut self.space, nodes)
        } else {
            builder.build_full_graph(&self.pool.borrow(), &mut self.space)
        }
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

/// Result of an aggregation operation
#[derive(Debug, Clone)]
pub struct AggregationResult {
    pub value: f64,
}

impl AggregationResult {
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

/// Extract numeric value from AttrValue for aggregation
fn extract_numeric(attr_value: &AttrValue) -> f64 {
    match attr_value {
        AttrValue::Int(i) => *i as f64,
        AttrValue::Float(f) => *f as f64,
        AttrValue::SmallInt(i) => *i as f64,
        _ => 0.0, // Non-numeric values default to 0
    }
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
