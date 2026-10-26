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

mod checkpoints;
pub(crate) mod composition;
mod effective_read;
mod identity;
pub(crate) mod kinds;
mod matrices;
mod ordinary;
mod persistence;
mod property_unset;
mod publication;
mod read_attributes;
pub(crate) mod retention;
mod schema;
pub(crate) mod snapshot;
mod tables;
mod view_updates;
use identity::ExternalIdentityIndex;
pub use identity::{EdgeUpsert, ExternalEntityRef, NodeUpsert, EXTERNAL_ID};
pub use kinds::{LABELS, RELATIONSHIP_TYPE};
pub use schema::{GraphSchema, PropertyConstraint, PropertyKind, SchemaViolation};
pub use view_updates::ViewBatchResult;

use crate::api::batch::{BatchResult, GraphBatch};
use crate::errors::{GraphError, GraphResult};
use crate::query::query::{EdgeFilter, NodeFilter, QueryEngine};
use crate::query::traversal::TraversalEngine;
use crate::state::change_tracker::ChangeTracker;
use crate::state::history::{CommitDiff, HistoricalView, HistoryForest};
use crate::state::ref_manager::BranchInfo;
use crate::state::space::GraphSpace;
use crate::storage::matrix::adjacency::AdjacencyMatrix;
use crate::storage::pool::GraphPool;
use crate::subgraphs::neighborhood::NeighborhoodSampler;
use crate::temporal::TemporalSnapshot;
use crate::types::SubgraphId;
use crate::types::{
    AttrName, AttrValue, BranchName, CompressionStatistics, EdgeId, MemoryEfficiency,
    MemoryStatistics, NodeId, StateId,
};
use crate::utils::config::GraphConfig;
use crate::viz::VizModule;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::rc::{Rc, Weak};
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

    /// Allocation-lifetime external identities, independent of active graph state.
    identities: ExternalIdentityIndex,

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

    /// Weak self-reference so components can access the live graph instance
    self_reference: Option<Weak<RefCell<Graph>>>,

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
    automatic_publication: bool,
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
    /// ```ignore
    /// use groggy::Graph;
    /// use groggy::types::GraphType;
    ///
    /// // Create a directed graph (like NetworkX DiGraph)
    /// let directed_graph = Graph::new_with_type(GraphType::Directed);
    ///
    /// // Create an undirected graph (like NetworkX Graph)  
    /// let undirected_graph = Graph::new_with_type(GraphType::Undirected);
    /// ```ignore
    pub fn new_with_type(graph_type: crate::types::GraphType) -> Self {
        let mut config = GraphConfig::new();
        config.graph_type = graph_type;
        let pool = Rc::new(RefCell::new(GraphPool::new_with_type(graph_type)));
        Self {
            space: GraphSpace::new(pool.clone(), 0), // base state = 0
            pool,
            history: HistoryForest::new_with_type(graph_type),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            traversal_engine: TraversalEngine::new(),
            neighborhood_sampler: NeighborhoodSampler::new(),
            self_reference: None,
            identities: ExternalIdentityIndex::default(),
            change_tracker: ChangeTracker::new(),
            automatic_publication: false,
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
            history: HistoryForest::new_with_type(config.graph_type),
            current_branch: "main".to_string(),
            current_commit: 0,
            query_engine: QueryEngine::new(),
            traversal_engine: TraversalEngine::new(),
            neighborhood_sampler: NeighborhoodSampler::new(),
            self_reference: None,
            identities: ExternalIdentityIndex::default(),
            change_tracker: ChangeTracker::new(),
            automatic_publication: false,
            config,
        }
    }

    /// Convert this graph into an `Rc<RefCell<Graph>>`, wiring internal back-references.
    ///
    /// This is the safest way to obtain a shared handle required by entities and subgraphs.
    pub fn into_shared(self) -> Rc<RefCell<Self>> {
        let graph_rc = Rc::new(RefCell::new(self));
        {
            let mut graph_mut = graph_rc.borrow_mut();
            graph_mut.attach_self_reference(&graph_rc);
        }
        graph_rc
    }

    /// Attach a self reference so helpers like `NeighborhoodSampler` can access the live graph.
    pub fn attach_self_reference(&mut self, graph_ref: &Rc<RefCell<Graph>>) {
        self.self_reference = Some(Rc::downgrade(graph_ref));
        self.neighborhood_sampler.set_graph_ref(graph_ref);
    }

    /// Retrieve a shared reference to this graph if one has been registered.
    pub fn shared_reference(&self) -> Option<Rc<RefCell<Graph>>> {
        self.self_reference.as_ref().and_then(|weak| weak.upgrade())
    }

    /// Load an active graph export, accepting typed v2 and legacy JSON.
    pub fn load_from_path(path: &Path) -> Result<Self, GraphError> {
        persistence::load(path)
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

    /// Get the number of active nodes in the graph
    pub fn node_count(&self) -> usize {
        self.space.node_count()
    }

    /// Get the number of active edges in the graph
    pub fn edge_count(&self) -> usize {
        self.space.edge_count()
    }

    /// Check whether the graph has no nodes or edges
    pub fn is_empty(&self) -> bool {
        self.node_count() == 0 && self.edge_count() == 0
    }

    /*
    === COMPONENT ACCESS METHODS ===
    These provide access to the internal storage and state components
    for trait implementations and advanced operations.
    */

    /// Get read-only access to the GraphPool storage component
    ///
    /// This provides access to the columnar attribute storage and
    /// is used by traits and advanced operations that need direct
    /// access to the storage layer.
    pub fn pool(&self) -> std::cell::Ref<'_, GraphPool> {
        self.pool.borrow()
    }

    /// Get mutable access to the GraphPool storage component
    ///
    /// This provides mutable access to the columnar attribute storage
    /// for operations that need to modify the storage directly.
    /// Requires an exclusive graph borrow so read handles cannot bypass publication.
    ///
    /// ```compile_fail
    /// fn bypass(graph: &groggy::Graph) {
    ///     graph.pool_mut();
    /// }
    /// ```
    pub fn pool_mut(&mut self) -> std::cell::RefMut<'_, GraphPool> {
        self.pool.borrow_mut()
    }

    /// Get read-only access to the GraphSpace active state component
    ///
    /// This provides access to the active node/edge sets and change tracking
    /// for operations that need to query the current graph state.
    pub fn space(&self) -> &GraphSpace {
        &self.space
    }

    /// Get mutable access to the GraphSpace for operations that need to modify space state
    pub fn space_mut(&mut self) -> &mut GraphSpace {
        &mut self.space
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

        // WORF SAFETY: Set entity_type efficiently using direct pool access
        // This bypasses the expensive validation in set_node_attr_internal
        let entity_type_attr: AttrName = "entity_type".into();
        let index = self.pool.borrow_mut().set_attr(
            entity_type_attr.clone(),
            AttrValue::Text("base".to_string()),
            true,
        );

        // Register the index in GraphSpace so the attribute is queryable
        self.space
            .set_node_attr_index(node_id, entity_type_attr.clone(), index);
        self.change_tracker
            .record_attr_change(node_id, entity_type_attr, None, index, true);

        node_id
    }

    /// Add multiple nodes efficiently
    /// OPTIMIZED: True bulk operation using vectorized pool operations
    pub fn add_nodes(&mut self, count: usize) -> Vec<NodeId> {
        if count == 0 {
            return Vec::new();
        }
        // Use optimized bulk pool operation
        let (_start_id, _end_id, node_ids) = self.pool.borrow_mut().add_nodes_bulk(count);

        // Use optimized bulk space activation
        self.space.activate_nodes(node_ids.clone());

        // Single bulk change tracking update
        self.change_tracker.record_nodes_addition(&node_ids);

        // WORF SAFETY: Set entity_type for all new nodes efficiently using bulk operation
        // Build bulk attribute data structure
        let entity_type_attr: AttrName = "entity_type".into();
        let base_value = AttrValue::Text("base".to_string());
        let mut attrs_values = HashMap::new();
        let entity_type_values: Vec<(NodeId, AttrValue)> = node_ids
            .iter()
            .map(|&node_id| (node_id, base_value.clone()))
            .collect();
        attrs_values.insert(entity_type_attr.clone(), entity_type_values);

        // Use vectorized bulk pool operation
        let index_changes = self.pool.borrow_mut().set_bulk_attrs(attrs_values, true);

        // Register all indices in GraphSpace in bulk
        if let Some(entity_indices) = index_changes.get(&entity_type_attr) {
            for &(node_id, new_index) in entity_indices {
                self.space
                    .set_node_attr_index(node_id, entity_type_attr.clone(), new_index);
            }
            let changes: Vec<_> = entity_indices
                .iter()
                .map(|&(node_id, new_index)| (node_id, entity_type_attr.clone(), None, new_index))
                .collect();
            self.change_tracker.record_attr_changes(&changes, true);
        }

        node_ids
    }

    /// Validate properties for new node rows without allocating IDs or writing values.
    pub fn validate_new_node_attrs(
        &self,
        rows: &[HashMap<AttrName, AttrValue>],
    ) -> GraphResult<()> {
        let mut errors = Vec::new();
        for (position, attrs) in rows.iter().enumerate() {
            for (name, value) in attrs {
                if let Err(error) = kinds::validate_kind_property(name, value, true) {
                    errors.push(format!("{position}.{name}: {error}"));
                }
            }
            for name in ["entity_type", EXTERNAL_ID] {
                if attrs.contains_key(name) {
                    errors.push(format!(
                        "{position}.{name}: system-managed node attribute cannot be changed"
                    ));
                }
            }
        }
        if !errors.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "Invalid node attributes at batch positions {}",
                errors.join("; ")
            )));
        }

        Ok(())
    }

    /// Validate new edge properties before allocating any edge ID.
    pub fn validate_new_edge_attrs(
        &self,
        rows: &[HashMap<AttrName, AttrValue>],
    ) -> GraphResult<()> {
        self.validate_new_edge_attr_rows(rows.iter())
    }

    fn validate_new_edge_attr_rows<'a>(
        &self,
        rows: impl IntoIterator<Item = &'a HashMap<AttrName, AttrValue>>,
    ) -> GraphResult<()> {
        let mut errors = Vec::new();
        for (position, attrs) in rows.into_iter().enumerate() {
            for (name, value) in attrs {
                if name == EXTERNAL_ID {
                    errors.push(format!("{position}.external_id: use the external-ID API"));
                }
                if let Err(error) = kinds::validate_kind_property(name, value, false) {
                    errors.push(format!("{position}.{name}: {error}"));
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(GraphError::InvalidInput(errors.join("; ")))
        }
    }

    /// Add nodes with properties, validating all rows before allocating IDs.
    /// Rows correspond to returned IDs; properties are written column by column.
    pub fn add_nodes_with_attrs(
        &mut self,
        rows: Vec<HashMap<AttrName, AttrValue>>,
    ) -> GraphResult<Vec<NodeId>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if self.automatic_publication {
            return self.writer("groggy").add_nodes_with_attrs(rows);
        }

        self.validate_new_node_attrs(&rows)?;
        let nodes = self.add_nodes(rows.len());
        let mut columns: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
        for (node, attrs) in nodes.iter().copied().zip(rows) {
            for (name, value) in attrs {
                columns.entry(name).or_default().push((node, value));
            }
        }
        // All IDs are newly allocated and unique; the reserved column was checked above.
        self.set_node_attrs(columns)?;
        Ok(nodes)
    }

    /// Add an edge between two existing nodes
    ///
    /// ALGORITHM:
    /// 1. Validate nodes exist in active set
    /// 2. Pool creates and stores the edge
    /// 3. Space tracks it as active
    /// 4. Return edge ID
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> Result<EdgeId, GraphError> {
        if self.automatic_publication {
            return self
                .writer("groggy")
                .add_edge(source, target, HashMap::new());
        }

        if !self.space.contains_node(source) || !self.space.contains_node(target) {
            if !self.space.contains_node(source) {
                return Err(GraphError::node_not_found(source, "add edge"));
            } else {
                return Err(GraphError::node_not_found(target, "add edge"));
            }
        }
        let edge_id = self.pool.borrow_mut().add_edge(source, target); // Pool creates and stores
        self.space.activate_edge(edge_id, source, target); // Space tracks as active
        self.change_tracker
            .record_edge_addition(edge_id, source, target); // Track change
        Ok(edge_id)
    }

    /// Add multiple edges efficiently
    /// OPTIMIZED: True bulk operation with batch validation and vectorized operations
    pub fn add_edges(&mut self, edges: &[(NodeId, NodeId)]) -> GraphResult<Vec<EdgeId>> {
        if self.automatic_publication {
            if edges.is_empty() {
                return Ok(Vec::new());
            }
            return self.writer("groggy").add_edges_with_attrs(
                edges.iter().map(|&(s, t)| (s, t, HashMap::new())).collect(),
            );
        }

        // Validate the entire batch before allocating any edge ID or changing state.
        let invalid: Vec<_> = edges
            .iter()
            .enumerate()
            .filter(|(_, (source, target))| {
                !self.space.contains_node(*source) || !self.space.contains_node(*target)
            })
            .map(|(index, &(source, target))| format!("{index}: ({source}, {target})"))
            .collect();
        if !invalid.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "Invalid edge endpoints at batch positions {}",
                invalid.join(", ")
            )));
        }
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        // Use optimized bulk pool operation
        let edge_ids = self.pool.borrow_mut().add_edges(edges);

        // Use optimized bulk space activation
        self.space.activate_edges(edge_ids.clone());

        // Prepare data for change tracking
        let change_data: Vec<_> = edge_ids
            .iter()
            .zip(edges.iter())
            .map(|(&edge_id, &(source, target))| (edge_id, source, target))
            .collect();

        // Single bulk change tracking update
        self.change_tracker.record_edges_addition(&change_data);

        Ok(edge_ids)
    }

    /// Copy another graph's active base entities through one prepared operation.
    /// Typed entity references are remapped; unsupported pooled contents and
    /// managed identities reject before allocating destination IDs.
    pub fn add_graph(&mut self, other: &Graph) -> Result<(), GraphError> {
        self.copy_graph_active(other)
    }

    /// Remove a node and all its incident edges
    ///
    /// ALGORITHM:
    /// 1. Ask space to remove the node (it handles incident edges)
    pub fn remove_node(&mut self, node: NodeId) -> Result<(), GraphError> {
        if self.automatic_publication {
            return self.writer("groggy").remove_nodes(vec![node]).map(|_| ());
        }

        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "remove node".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }

        self.remove_nodes(&[node])
    }

    /// Remove an edge
    pub fn remove_edge(&mut self, edge: EdgeId) -> Result<(), GraphError> {
        if self.automatic_publication {
            return self.writer("groggy").remove_edges(vec![edge]).map(|_| ());
        }

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

    /// Remove an edge if it exists, otherwise silently succeed
    pub fn try_remove_edge(&mut self, edge: EdgeId) -> Result<(), GraphError> {
        if self.automatic_publication {
            return if self.has_edge(edge) {
                self.remove_edge(edge)
            } else {
                Ok(())
            };
        }

        if self.space.contains_edge(edge) {
            self.change_tracker.record_edge_removal(edge);
            self.space.deactivate_edge(edge);
        }
        Ok(())
    }

    /// Remove multiple edges after validating the entire input, including duplicates.
    pub fn remove_edges(&mut self, edges: &[EdgeId]) -> Result<(), GraphError> {
        if self.automatic_publication {
            if edges.is_empty() {
                return Ok(());
            }
            self.validate_removal_targets(edges, false)?;
            return self
                .writer("groggy")
                .remove_edges(edges.to_vec())
                .map(|_| ());
        }

        self.validate_removal_targets(edges, false)?;
        if !edges.is_empty() {
            for &edge_id in edges {
                self.change_tracker.record_edge_removal(edge_id);
            }
            self.space.deactivate_edges(edges);
        }
        Ok(())
    }

    /// Remove multiple nodes and their incident edges after validating the entire input.
    pub fn remove_nodes(&mut self, nodes: &[NodeId]) -> Result<(), GraphError> {
        if self.automatic_publication {
            if nodes.is_empty() {
                return Ok(());
            }
            self.validate_removal_targets(nodes, true)?;
            return self
                .writer("groggy")
                .remove_nodes(nodes.to_vec())
                .map(|_| ());
        }

        self.validate_removal_targets(nodes, true)?;
        if nodes.is_empty() {
            return Ok(());
        }
        // Prepare the complete cascade before writing. Scan active edges once,
        // independent of the number of selected nodes or retained tombstones.
        let selected = nodes.iter().copied().collect();
        let incident = self.incident_edges_for_nodes(&selected)?;
        let edges: Vec<_> = incident.into_iter().map(|(edge, _, _)| edge).collect();
        self.remove_edges(&edges)?;
        for &node in nodes {
            self.change_tracker.record_node_removal(node);
        }
        self.space.deactivate_nodes(nodes);
        Ok(())
    }

    // Shared by mutation execution and publication-clock preparation. Only active
    // edges matter; each loop, parallel instance and shared edge is returned once.
    fn incident_edges_for_nodes(
        &self,
        nodes: &HashSet<NodeId>,
    ) -> GraphResult<Vec<(EdgeId, NodeId, NodeId)>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.borrow();
        let mut incident = Vec::new();
        for edge in self.space.edge_ids() {
            let (source, target) = pool
                .get_edge_endpoints(edge)
                .ok_or_else(|| GraphError::edge_not_found(edge, "prepare node removal cascade"))?;
            if nodes.contains(&source) || nodes.contains(&target) {
                incident.push((edge, source, target));
            }
        }
        Ok(incident)
    }

    fn validate_removal_targets(&self, ids: &[usize], is_nodes: bool) -> GraphResult<()> {
        let kind = if is_nodes { "node" } else { "edge" };
        let mut seen = HashMap::new();
        let mut errors = Vec::new();
        for (position, &id) in ids.iter().enumerate() {
            let active = if is_nodes {
                self.space.contains_node(id)
            } else {
                self.space.contains_edge(id)
            };
            if !active {
                errors.push(format!("{position}: {kind} {id} is not active"));
            }
            if let Some(first) = seen.get(&id) {
                errors.push(format!(
                    "{position}: duplicate {kind} {id} (first at {first})"
                ));
            } else {
                seen.insert(id, position);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(GraphError::InvalidInput(format!(
                "Invalid {kind} removals at batch positions {}",
                errors.join("; ")
            )))
        }
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
        if self.automatic_publication {
            return self
                .writer("groggy")
                .set_node_attr(node, attr, value)
                .map(|_| ());
        }

        if attr == EXTERNAL_ID {
            return Err(GraphError::InvalidInput(
                "external_id is system-managed; use set_node_external_id".into(),
            ));
        }
        // WORF SAFETY: Prevent direct entity_type modification
        if attr == "entity_type" {
            return Err(GraphError::InvalidInput(
                "entity_type is immutable and managed by the system. Use create_meta_node() for meta-nodes or add_node() for base nodes.".to_string()
            ));
        }

        self.set_node_attr_internal(node, attr, value)
    }

    /// Internal method for setting node attributes (bypasses entity_type safety)
    /// SAFETY: Only used by system methods that need to set entity_type
    pub(crate) fn set_node_attr_internal(
        &mut self,
        node: NodeId,
        attr: AttrName,
        value: AttrValue,
    ) -> Result<(), GraphError> {
        // 1. Validate node is active
        if !self.space.contains_node(node) {
            return Err(GraphError::node_not_found(node, "set attribute"));
        }

        let value = kinds::normalize_kind_property(&attr, value, true)?;

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

    // NOTE: Bulk attribute operations have been moved to trait system
    // Use NodeOperations::set_bulk_node_attrs() instead

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
        if self.automatic_publication {
            return self
                .writer("groggy")
                .set_edge_attr(edge, attr, value)
                .map(|_| ());
        }

        if attr == EXTERNAL_ID {
            return Err(GraphError::InvalidInput(
                "external_id is system-managed; use set_edge_external_id".into(),
            ));
        }
        // 1. Validate edge is active
        if !self.space.contains_edge(edge) {
            return Err(GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "set attribute".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            });
        }

        let value = kinds::normalize_kind_property(&attr, value, false)?;

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

    // NOTE: Bulk attribute operations have been moved to trait system
    // Use EdgeOperations::set_bulk_edge_attrs() instead

    /*
    === WORF'S AIRTIGHT ENTITY TYPE SYSTEM ===
    Safe, efficient querying of entity types using columnar storage
    */

    /// Check if a node is a meta-node (type-safe query)
    /// PERFORMANCE: O(1) using efficient columnar attribute lookup
    pub fn is_meta_node(&self, node_id: NodeId) -> bool {
        match self.get_node_attr(node_id, &"entity_type".into()) {
            Ok(Some(value)) => value.get_text().as_deref() == Some("meta"),
            _ => false,
        }
    }

    /// Check if a node is a base node (type-safe query)
    /// PERFORMANCE: O(1) using efficient columnar attribute lookup
    pub fn is_base_node(&self, node_id: NodeId) -> bool {
        match self.get_node_attr(node_id, &"entity_type".into()) {
            Ok(Some(value)) => value.get_text().as_deref() == Some("base"),
            Ok(None) => true, // Default to base for nodes without entity_type (migration compatibility)
            _ => false,
        }
    }

    /// Get all validated meta-nodes in the graph
    /// SAFETY: Only returns nodes that pass validation checks
    pub fn get_meta_nodes(&self) -> Vec<NodeId> {
        self.space
            .get_active_nodes()
            .into_iter()
            .filter(|&node_id| {
                self.is_meta_node(node_id) && self.validate_meta_node(node_id).is_ok()
            })
            .collect()
    }

    /// Get all validated base nodes in the graph
    /// SAFETY: Only returns nodes that pass validation checks
    pub fn get_base_nodes(&self) -> Vec<NodeId> {
        self.space
            .get_active_nodes()
            .into_iter()
            .filter(|&node_id| {
                self.is_base_node(node_id) && self.validate_base_node(node_id).is_ok()
            })
            .collect()
    }

    /// Validate a meta-node has all required attributes and structure
    /// SAFETY: Ensures meta-node integrity
    fn validate_meta_node(&self, node_id: NodeId) -> Result<(), GraphError> {
        // REQUIREMENT 1: Must have entity_type = "meta"
        match self.get_node_attr(node_id, &"entity_type".into())? {
            Some(value) if value.get_text().as_deref() == Some("meta") => {}
            _ => {
                return Err(GraphError::InvalidInput(format!(
                    "Node {} does not have entity_type='meta'",
                    node_id
                )))
            }
        }

        // REQUIREMENT 2: Must have contains_subgraph reference
        match self.get_node_attr(node_id, &"contains_subgraph".into())? {
            Some(AttrValue::SubgraphRef(subgraph_id)) => {
                self.pool.borrow().validate_stored_subgraph(subgraph_id)
            }
            _ => Err(GraphError::InvalidInput(format!(
                "Meta-node {} missing required contains_subgraph attribute",
                node_id
            ))),
        }
    }

    /// Validate a base node structure
    /// SAFETY: Ensures base node integrity
    fn validate_base_node(&self, node_id: NodeId) -> Result<(), GraphError> {
        // REQUIREMENT: Must have entity_type = "base" or None (default)
        match self.get_node_attr(node_id, &"entity_type".into())? {
            Some(value) if value.get_text().as_deref() == Some("base") => Ok(()),
            None => Ok(()), // Default to base for migration compatibility
            _ => Err(GraphError::InvalidInput(format!(
                "Node {} has invalid entity_type for base node",
                node_id
            ))),
        }
    }

    /// Create a meta-node with its required system properties through prepared
    /// execution. Publishing accepts one state; manual mode leaves pending changes.
    /// Invalid retained references reject before allocating an ID.
    pub fn create_meta_node(&mut self, subgraph_id: SubgraphId) -> Result<NodeId, GraphError> {
        self.create_meta_node_prepared(subgraph_id)
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

    /// Get all node IDs as a HashSet (zero-copy clone)
    /// PERFORMANCE: Use this when you need a HashSet to avoid Vec intermediate
    pub fn node_ids_set(&self) -> HashSet<NodeId> {
        self.space.get_active_nodes().clone()
    }

    /// Get all edge IDs as a HashSet (zero-copy clone)
    /// PERFORMANCE: Use this when you need a HashSet to avoid Vec intermediate
    pub fn edge_ids_set(&self) -> HashSet<EdgeId> {
        self.space.get_active_edges().clone()
    }

    /// Get all source node IDs for edges currently in the graph
    /// Returns a vector parallel to edge_ids() where each element is the source of the corresponding edge
    pub fn edge_sources(&self) -> Vec<NodeId> {
        let (_, sources, _, _) = self.space.snapshot(&self.pool.borrow());
        sources.to_vec()
    }

    /// Get all target node IDs for edges currently in the graph
    /// Returns a vector parallel to edge_ids() where each element is the target of the corresponding edge
    pub fn edge_targets(&self) -> Vec<NodeId> {
        let (_, _, targets, _) = self.space.snapshot(&self.pool.borrow());
        targets.to_vec()
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

    /// Get the in-degree of a node (number of incoming edges)
    pub fn in_degree(&self, node: NodeId) -> Result<usize, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get in-degree".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }

        // Get fresh topology snapshot
        let (_, _, targets, _) = self.space.snapshot(&self.pool.borrow());
        let mut count = 0;
        for &target in targets.iter() {
            if target == node {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get the out-degree of a node (number of outgoing edges)
    pub fn out_degree(&self, node: NodeId) -> Result<usize, GraphError> {
        if !self.space.contains_node(node) {
            return Err(GraphError::NodeNotFound {
                node_id: node,
                operation: "get out-degree".to_string(),
                suggestion: "Check if node exists with contains_node()".to_string(),
            });
        }

        // Get fresh topology snapshot
        let (_, sources, _, _) = self.space.snapshot(&self.pool.borrow());
        let mut count = 0;
        for &source in sources.iter() {
            if source == node {
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

    /// Get neighbors for multiple nodes in bulk - returns columnar BaseArray
    ///
    /// This is the optimized bulk operation for neighbor queries.
    /// Instead of returning a dict, returns a BaseArray with columns:
    /// - "node_id": the queried node
    /// - "neighbor_id": each neighbor (one row per neighbor)
    ///
    /// PERFORMANCE: O(1) snapshot retrieval + O(N*avg_degree) neighbor extraction
    pub fn neighbors_bulk(
        &self,
        nodes: &[NodeId],
    ) -> Result<crate::storage::table::BaseTable, GraphError> {
        use crate::storage::array::BaseArray;
        use crate::storage::table::BaseTable;

        // Validate all nodes exist first
        for &node in nodes {
            if !self.space.contains_node(node) {
                return Err(GraphError::NodeNotFound {
                    node_id: node,
                    operation: "get neighbors bulk".to_string(),
                    suggestion: "Check if all nodes exist with contains_node()".to_string(),
                });
            }
        }

        // Get fresh adjacency snapshot once for all queries
        let (_, _, _, neighbors_map) = self.space.snapshot(&self.pool.borrow());

        // Build columnar data: node_id, neighbor_id pairs
        let mut node_ids = Vec::new();
        let mut neighbor_ids = Vec::new();

        for &node in nodes {
            if let Some(neighbors) = neighbors_map.get(&node) {
                for (neighbor, _) in neighbors {
                    node_ids.push(node);
                    neighbor_ids.push(*neighbor);
                }
            }
            // If node has no neighbors, we don't add any rows for it
        }

        // Create BaseArrays
        let mut columns = HashMap::new();
        columns.insert("node_id".to_string(), BaseArray::from_node_ids(node_ids));
        columns.insert(
            "neighbor_id".to_string(),
            BaseArray::from_node_ids(neighbor_ids),
        );

        BaseTable::from_columns(columns)
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
        if !self.space.contains_edge(edge) {
            return Err(GraphError::edge_not_found(edge, "get endpoints"));
        }
        self.pool
            .borrow()
            .get_edge_endpoints(edge)
            .ok_or_else(|| GraphError::EdgeNotFound {
                edge_id: edge,
                operation: "get endpoints".to_string(),
                suggestion: "Check if edge exists with contains_edge()".to_string(),
            })
    }

    /// Check if there's an edge between two nodes
    pub fn has_edge_between(&self, source: NodeId, target: NodeId) -> GraphResult<bool> {
        if !self.space.contains_node(source) || !self.space.contains_node(target) {
            return Ok(false);
        }
        let pool = self.pool.borrow();
        for edge_id in self.space.get_active_edges() {
            if let Some((edge_source, edge_target)) = pool.get_edge_endpoints(edge_id) {
                if edge_source == source && edge_target == target
                    || (self.is_undirected() && edge_source == target && edge_target == source)
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Get all edges connected to a node (only active edges)
    pub fn incident_edges(&self, node: NodeId) -> GraphResult<Vec<EdgeId>> {
        if !self.space.contains_node(node) {
            return Err(GraphError::node_not_found(node, "get incident edges"));
        }

        let pool = self.pool.borrow();
        let all_incident_edges = pool.get_incident_edges(node)?;

        // CONSISTENCY FIX: Filter to only return edges that are still active in the space
        let active_incident_edges: Vec<EdgeId> = all_incident_edges
            .into_iter()
            .filter(|&edge_id| self.space.contains_edge(edge_id))
            .collect();

        Ok(active_incident_edges)
    }

    /// Check if a node exists
    pub fn has_node(&self, node: NodeId) -> bool {
        self.space.has_node(node)
    }

    /// Check if an edge exists
    pub fn has_edge(&self, edge: EdgeId) -> bool {
        self.space.has_edge(edge)
    }

    /// Get neighbors filtered to a specific node set
    pub fn neighbors_filtered(
        &self,
        node: NodeId,
        filter_nodes: &std::collections::HashSet<NodeId>,
    ) -> GraphResult<Vec<NodeId>> {
        let all_neighbors = self.neighbors(node)?;
        Ok(all_neighbors
            .into_iter()
            .filter(|&n| filter_nodes.contains(&n))
            .collect())
    }

    /// Get degree filtered to a specific node set
    pub fn degree_filtered(
        &self,
        node: NodeId,
        filter_nodes: &std::collections::HashSet<NodeId>,
    ) -> GraphResult<usize> {
        let filtered_neighbors = self.neighbors_filtered(node, filter_nodes)?;
        Ok(filtered_neighbors.len())
    }

    /// Check if there's an edge between two nodes, filtered to a specific edge set
    pub fn has_edge_between_filtered(
        &self,
        source: NodeId,
        target: NodeId,
        filter_edges: &std::collections::HashSet<EdgeId>,
    ) -> GraphResult<bool> {
        let incident = self.incident_edges(source)?;
        for edge_id in incident {
            if filter_edges.contains(&edge_id) {
                if let Ok((edge_source, edge_target)) = self.edge_endpoints(edge_id) {
                    if (edge_source == source && edge_target == target)
                        || (self.is_undirected() && edge_source == target && edge_target == source)
                    {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
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
        let base_size = std::mem::size_of::<crate::storage::pool::GraphPool>();

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
        let base_size = std::mem::size_of::<crate::state::space::GraphSpace>();

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
        let base_size = std::mem::size_of::<crate::state::history::HistoryForest>();
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

    /// The published state selected as this graph's working head.
    pub fn current_state_id(&self) -> StateId {
        self.current_commit
    }

    /// Apply a validated mixed write and publish one new state.
    ///
    /// The existing bulk storage paths remain responsible for node, edge, and
    /// property writes. The expected state rejects writes prepared against an
    /// older head. This first batch API requires a clean working state while
    /// legacy mutations still use explicit `commit`.
    pub fn apply_batch(
        &mut self,
        batch: GraphBatch,
        expected_state: StateId,
        message: String,
        author: String,
    ) -> GraphResult<BatchResult> {
        self.apply_batch_internal(batch, expected_state, message, author, None)
    }

    /// Publish one batch only if its complete resulting graph satisfies schema.
    /// This policy applies to this call; it is not installed on future writes.
    pub fn apply_batch_with_schema(
        &mut self,
        batch: GraphBatch,
        expected_state: StateId,
        message: String,
        author: String,
        schema: &GraphSchema,
    ) -> GraphResult<BatchResult> {
        self.apply_batch_internal(batch, expected_state, message, author, Some(schema))
    }

    pub(crate) fn validate_write_head(&self, expected_state: StateId) -> GraphResult<()> {
        if expected_state != self.current_commit {
            return Err(GraphError::InvalidInput(format!(
                "Stale graph batch: expected state {expected_state}, current state {}",
                self.current_commit
            )));
        }
        if self.has_uncommitted_changes() {
            return Err(GraphError::InvalidInput(
                "Cannot publish a batch while legacy writes are uncommitted".into(),
            ));
        }
        self.history.get_branch_head(&self.current_branch)?;

        Ok(())
    }

    fn apply_batch_internal(
        &mut self,
        batch: GraphBatch,
        expected_state: StateId,
        message: String,
        author: String,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<BatchResult> {
        let prepared = self.prepare_publication(batch, expected_state, schema)?;
        self.execute_publication(prepared, message, author)
    }

    pub(crate) fn prepare_publication(
        &self,
        batch: GraphBatch,
        expected_state: StateId,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<crate::api::batch::PreparedPublication> {
        self.prepare_publication_with_additional_schema(batch, expected_state, schema, None)
    }

    pub(crate) fn prepare_publication_with_additional_schema(
        &self,
        batch: GraphBatch,
        expected_state: StateId,
        schema: Option<&GraphSchema>,
        additional_schema: Option<&GraphSchema>,
    ) -> GraphResult<crate::api::batch::PreparedPublication> {
        self.prepare_publication_with_additional_schema_at(
            batch,
            expected_state,
            schema,
            additional_schema,
            None,
        )
    }

    pub(crate) fn prepare_publication_with_additional_schema_at(
        &self,
        batch: GraphBatch,
        expected_state: StateId,
        schema: Option<&GraphSchema>,
        additional_schema: Option<&GraphSchema>,
        recorded_at: Option<i64>,
    ) -> GraphResult<crate::api::batch::PreparedPublication> {
        self.validate_write_head(expected_state)?;

        let allocated = self.pool.borrow().statistics();
        let prepared = batch.prepare(
            |id| self.space.contains_node(id),
            |id| self.space.contains_edge(id),
            |id| self.pool.borrow().get_edge_endpoints(id),
            allocated.node_count,
            allocated.edge_count,
        )?;

        self.validate_unsets(&prepared)?;

        // Every checked pooled reference must already exist. Compound creation
        // stages its trusted managed reference after preparation and installs its
        // validated source before execution.
        let references: HashSet<_> = prepared
            .node_attrs
            .values()
            .flatten()
            .map(|(_, value)| value)
            .chain(
                prepared
                    .edge_attrs
                    .values()
                    .flatten()
                    .map(|(_, value)| value),
            )
            .filter_map(|value| {
                if let AttrValue::SubgraphRef(id) = value {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        for id in references {
            self.pool.borrow().validate_stored_subgraph(id)?;
        }

        let mut identity_errors = Vec::new();
        for (position, (name, entity)) in prepared.identity_bindings.iter().enumerate() {
            if let Err(error) = self.identities.validate_binding(name, *entity) {
                identity_errors.push(format!("identity_bindings[{position}]: {error}"));
            }
        }
        if !identity_errors.is_empty() {
            return Err(GraphError::InvalidInput(identity_errors.join("; ")));
        }

        for schema in [schema, additional_schema].into_iter().flatten() {
            let violations = schema.validate_proposed(
                self,
                &prepared,
                allocated.node_count,
                allocated.edge_count,
            )?;
            schema::reject_violations(violations)?;
        }

        let publication = match recorded_at {
            Some(at) => self.batch_publication_at(
                &prepared,
                allocated.node_count,
                allocated.edge_count,
                at,
            )?,
            None => {
                self.batch_publication(&prepared, allocated.node_count, allocated.edge_count)?
            }
        };
        self.validate_publication_topology(&prepared, &publication, allocated.edge_count)?;
        Ok(crate::api::batch::PreparedPublication {
            batch: prepared,
            metadata: publication,
        })
    }

    pub(crate) fn execute_publication(
        &mut self,
        publication: crate::api::batch::PreparedPublication,
        message: String,
        author: String,
    ) -> GraphResult<BatchResult> {
        let policy = self.automatic_publication;
        self.automatic_publication = false;
        let result = self.execute_prepared_publication(publication, message, author);
        self.automatic_publication = policy;
        result
    }

    fn execute_prepared_publication(
        &mut self,
        publication: crate::api::batch::PreparedPublication,
        message: String,
        author: String,
    ) -> GraphResult<BatchResult> {
        let (added_nodes, added_edges) = self.execute_prepared_mutation(publication.batch)?;
        let state_id = self.commit_internal(message, author, Some(publication.metadata))?;
        Ok(BatchResult {
            state_id,
            added_nodes,
            added_edges,
        })
    }

    // Storage-only executor, shared by checked publication and manual compatibility.
    // Callers must prepare all inputs first and disable ordinary dispatch.
    fn execute_prepared_mutation(
        &mut self,
        prepared: crate::api::batch::PreparedBatch,
    ) -> GraphResult<(Vec<NodeId>, Vec<EdgeId>)> {
        let added_nodes = if prepared.add_nodes > 0 {
            self.add_nodes(prepared.add_nodes)
        } else {
            Vec::new()
        };
        let added_edges = self.add_edges(&prepared.add_edges)?;
        self.write_node_attrs(prepared.node_attrs)?;
        self.write_edge_attrs(prepared.edge_attrs)?;
        self.execute_unsets(prepared.unset_node_attrs, true);
        self.execute_unsets(prepared.unset_edge_attrs, false);
        self.remove_edges(&prepared.remove_edges)?;
        self.remove_nodes(&prepared.remove_nodes)?;
        self.identities.reserve_bindings(prepared.identity_bindings);
        Ok((added_nodes, added_edges))
    }

    /// Explicitly checkpoint the current published state, preserving its version ID.
    /// For compatibility, pending legacy/manual writes are first published once;
    /// checked writer changes are already published and do not create a new delta.
    pub fn commit(&mut self, message: String, author: String) -> Result<StateId, GraphError> {
        if !self.has_uncommitted_changes() {
            return self.checkpoint(message, author);
        }
        // Compatibility: publish pending manual writes once, then mark that state.
        // New checked writes are already published and take the clean path above.
        self.history.get_branch_head(&self.current_branch)?;
        let recorded_at = crate::state::publication::now_micros()?;
        self.history.reserve_checkpoint()?;
        let state_id = self.commit_internal(message.clone(), author.clone(), None)?;
        self.history.record_checkpoint(crate::Checkpoint {
            state_id,
            branch: self.current_branch.clone(),
            message,
            author,
            recorded_at,
        });
        Ok(state_id)
    }

    fn commit_internal(
        &mut self,
        message: String,
        author: String,
        publication: Option<crate::PublicationMetadata>,
    ) -> GraphResult<StateId> {
        // Create changeset from current changes
        let changeset = self
            .change_tracker
            .create_changeset_resolved(&self.pool.borrow())?;

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
        let new_commit_id = self.history.create_commit_with_publication(
            changeset,
            message,
            author,
            parent,
            publication,
        )?;

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
        self.change_tracker.has_changes()
    }

    /// Reset all uncommitted changes
    pub fn reset_hard(&mut self) -> Result<(), GraphError> {
        let snapshot = self.history.reconstruct_state_at(self.current_commit)?;
        self.reset_to_snapshot(snapshot)?;
        self.change_tracker.clear();
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

    /// The selected configuration (persisted by the durable journal).
    pub fn configuration(&self) -> &GraphConfig {
        &self.config
    }

    /// Read engine-recorded and effective times for an automatically published batch.
    /// Legacy manual commits have no publication record and return None.
    pub fn publication_metadata(
        &self,
        state: StateId,
    ) -> GraphResult<Option<crate::PublicationMetadata>> {
        Ok(self.history.get_commit(state)?.publication.clone())
    }

    /// Read immutable native commit records without synthesizing metadata.
    pub fn history_records(&self) -> Vec<Arc<crate::state::history::Commit>> {
        self.history.get_commit_history()
    }

    /// Get the commit history
    pub fn commit_history(&self) -> Vec<CommitInfo> {
        self.history
            .get_commit_history()
            .into_iter()
            .map(|commit| CommitInfo {
                id: commit.id,
                message: commit.message.clone(),
                author: commit.author.clone(),
                timestamp: commit.timestamp,
                parent: commit.parents.first().copied(),
                changes_summary: commit.delta.summary(),
            })
            .collect()
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
        options: crate::query::traversal::TraversalOptions,
    ) -> Result<crate::query::traversal::TraversalResult, GraphError> {
        self.traversal_engine
            .bfs(&self.pool.borrow(), &mut self.space, start, options)
            .map_err(|e| e.into())
    }

    /// Perform Depth-First Search from a starting node
    pub fn dfs(
        &mut self,
        start: NodeId,
        options: crate::query::traversal::TraversalOptions,
    ) -> Result<crate::query::traversal::TraversalResult, GraphError> {
        self.traversal_engine
            .dfs(&self.pool.borrow(), &mut self.space, start, options)
            .map_err(|e| e.into())
    }

    /// Find shortest path between two nodes
    pub fn shortest_path(
        &mut self,
        start: NodeId,
        end: NodeId,
        options: crate::query::traversal::PathFindingOptions,
    ) -> Result<Option<crate::query::traversal::Path>, GraphError> {
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
    ) -> Result<Vec<crate::query::traversal::Path>, GraphError> {
        self.traversal_engine
            .all_paths(&self.pool.borrow(), &mut self.space, start, end, max_length)
            .map_err(|e| e.into())
    }

    /// Find all connected components in the graph
    pub fn connected_components(
        &mut self,
        options: crate::query::traversal::TraversalOptions,
    ) -> Result<crate::query::traversal::ConnectedComponentsResult, GraphError> {
        self.traversal_engine
            .connected_components(&self.pool.borrow(), &self.space, options)
            .map_err(|e| e.into())
    }

    /// Get traversal performance statistics
    pub fn traversal_statistics(&self) -> &crate::query::traversal::TraversalStats {
        self.traversal_engine.statistics()
    }

    // ===== NEIGHBORHOOD SUBGRAPH SAMPLING =====

    /// Generate 1-hop neighborhood subgraph for a single node
    /// Returns a Subgraph containing the central node and all its direct neighbors
    pub fn neighborhood(
        &mut self,
        node_id: NodeId,
    ) -> Result<crate::subgraphs::neighborhood::NeighborhoodSubgraph, GraphError> {
        self.neighborhood_sampler
            .single_neighborhood(&self.pool.borrow(), &self.space, node_id)
            .map_err(|e| e.into())
    }

    /// Generate 1-hop neighborhoods for multiple nodes
    /// Returns separate neighborhood subgraphs for each central node
    pub fn multi_neighborhood(
        &mut self,
        node_ids: &[NodeId],
    ) -> Result<crate::subgraphs::neighborhood::NeighborhoodResult, GraphError> {
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
    ) -> Result<crate::subgraphs::neighborhood::NeighborhoodSubgraph, GraphError> {
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
    ) -> Result<crate::subgraphs::neighborhood::NeighborhoodSubgraph, GraphError> {
        self.neighborhood_sampler
            .unified_neighborhood(&self.pool.borrow(), &self.space, node_ids, k)
            .map_err(|e| e.into())
    }

    /// Get neighborhood sampling performance statistics
    pub fn neighborhood_statistics(&self) -> &crate::subgraphs::neighborhood::NeighborhoodStats {
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
        HistoricalView::new(&self.history, commit_id)
    }

    pub(crate) fn owns_temporal_snapshot(&self, snapshot: &TemporalSnapshot) -> bool {
        snapshot.belongs_to_history(&self.history)
    }

    /// Build an immutable snapshot of the graph at a given commit.
    pub fn snapshot_at_commit(&self, commit_id: StateId) -> Result<TemporalSnapshot, GraphError> {
        TemporalSnapshot::at_commit(&self.history, commit_id).map(|snapshot| {
            snapshot
                .with_groups(self.pool.borrow().retained_groups())
                .with_materialization(self.snapshot_materialization(false))
        })
    }

    /// Select a recorded snapshot on the current branch's first-parent lineage
    /// at or before the timestamp (Unix seconds). Explicit state-ID reads remain global.
    pub fn snapshot_at_timestamp(&self, timestamp: u64) -> Result<TemporalSnapshot, GraphError> {
        TemporalSnapshot::at_timestamp_on_lineage(&self.history, self.current_commit, timestamp)
            .map(|snapshot| {
                snapshot
                    .with_groups(self.pool.borrow().retained_groups())
                    .with_materialization(self.snapshot_materialization(false))
            })
    }

    /// Build a temporal index from the history for efficient temporal queries.
    ///
    /// The temporal index provides O(log n) or better lookups for:
    /// - When nodes/edges were created/deleted
    /// - Which nodes/edges existed at any commit
    /// - Attribute value timelines
    /// - Neighbors at specific commits or within time windows
    ///
    /// This is a one-time build operation that processes all commits.
    /// The resulting index can then be used for many temporal queries.
    pub fn build_temporal_index(&self) -> Result<crate::temporal::TemporalIndex, GraphError> {
        self.build_temporal_index_at(self.current_state_id())
    }

    /// Build a temporal index on the retained first-parent lineage of one state.
    pub fn build_temporal_index_at(
        &self,
        state: StateId,
    ) -> Result<crate::temporal::TemporalIndex, GraphError> {
        crate::temporal::TemporalIndex::from_history_at(&self.history, state)
    }

    pub(crate) fn temporal_index_for_run(
        &self,
        state: StateId,
        cancelled: impl Fn() -> bool,
    ) -> Result<crate::temporal::TemporalIndex, GraphError> {
        crate::temporal::TemporalIndex::from_history_at_cancellable(&self.history, state, cancelled)
    }

    pub(crate) fn owns_temporal_index(
        &self,
        index: &crate::temporal::TemporalIndex,
        state: StateId,
    ) -> bool {
        index.matches(&self.history, state)
    }

    /// Get neighbors of nodes as they existed at a specific commit.
    ///
    /// This uses a temporal index for efficient lookups. If you need to make many
    /// temporal queries, build the index once with `build_temporal_index()` and
    /// use it directly.
    pub fn neighbors_at_commit(
        &self,
        node_ids: &[NodeId],
        commit_id: StateId,
    ) -> Result<HashMap<NodeId, Vec<NodeId>>, GraphError> {
        let index = self.build_temporal_index()?;
        Ok(index.neighbors_bulk_at_commit(node_ids, commit_id))
    }

    /// Get neighbors that existed at any point within a commit range.
    pub fn neighbors_in_window(
        &self,
        node_id: NodeId,
        start_commit: StateId,
        end_commit: StateId,
    ) -> Result<Vec<NodeId>, GraphError> {
        let index = self.build_temporal_index()?;
        Ok(index.neighbors_in_window(node_id, start_commit, end_commit))
    }

    /// Get attribute value history for a node within a commit range.
    ///
    /// Returns all attribute changes that occurred during [from_commit, to_commit].
    pub fn node_attr_history(
        &self,
        node_id: NodeId,
        attr: &AttrName,
        from_commit: StateId,
        to_commit: StateId,
    ) -> Result<Vec<(StateId, Option<AttrValue>)>, GraphError> {
        let index = self.build_temporal_index()?;
        Ok(index.node_attr_history(node_id, attr, from_commit, to_commit))
    }

    /// Compare two commits and show differences
    pub fn diff_commits(&self, from: StateId, to: StateId) -> Result<CommitDiff, GraphError> {
        self.history.diff_commits(from, to)
    }

    /*
    === INTERNAL STATE MANAGEMENT ===
    Helper methods for branch switching and state reconstruction
    */

    /// Reset the current graph state (pool + space) to match a historical snapshot
    /// This is used during branch switching to restore the graph to a specific state
    fn reset_to_snapshot(
        &mut self,
        snapshot: crate::state::state::GraphSnapshot,
    ) -> Result<(), GraphError> {
        let identity_bindings = self.snapshot_identity_bindings(&snapshot)?;
        // A standalone snapshot must not manufacture a dangling pooled reference.
        // State switches on this graph can resolve references in its retained pool.
        for attrs in snapshot
            .node_attributes
            .values()
            .chain(snapshot.edge_attributes.values())
        {
            for value in attrs.values() {
                if let AttrValue::SubgraphRef(id) = value {
                    self.pool.borrow().get_subgraph(*id)?;
                }
            }
        }
        // Reset selected membership and column mappings, not the retained pool.
        // Older branches, stored subgraphs and column indices still own data here.
        self.space = crate::state::space::GraphSpace::new(self.pool.clone(), snapshot.state_id);
        self.config.graph_type = snapshot.graph_type;

        // 2. Restore nodes
        for &node_id in &snapshot.active_nodes {
            // Ensure node ID exists in pool
            self.pool.borrow_mut().ensure_node_id_exists(node_id);
            // Activate in space
            self.space.activate_node(node_id);

            // Restore node attributes
            if let Some(attrs) = snapshot.node_attributes.get(&node_id) {
                for (attr_name, attr_value) in attrs {
                    let index =
                        self.pool
                            .borrow_mut()
                            .restore_attr(attr_name.clone(), attr_value, true);
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
                    let index =
                        self.pool
                            .borrow_mut()
                            .restore_attr(attr_name.clone(), attr_value, false);
                    self.space
                        .set_edge_attr_index(edge_id, attr_name.clone(), index);
                }
            }
        }

        self.retain_snapshot_identity_bindings(identity_bindings);
        Ok(())
    }

    /// Capture a full immutable read selection of the current working graph.
    pub fn read_view(&self) -> GraphResult<crate::subgraphs::GraphView> {
        Ok(crate::subgraphs::GraphView::all(self.pin_current_state()?))
    }

    /// Capture immutable current read state, including uncheckpointed property edits.
    /// No commit/delta is created; clones of the result share its frozen values.
    pub fn pin_current_state(&self) -> GraphResult<crate::state::PinnedReadState> {
        let snapshot = self
            .space
            .capture_read_snapshot(&self.pool.borrow(), self.current_commit)?;
        Ok(crate::state::PinnedReadState::from_snapshot(snapshot)?
            .with_groups(self.pool.borrow().retained_groups())
            .with_origin(
                Arc::clone(self.history.identity()),
                self.current_branch.clone(),
                self.current_commit,
                !self.has_uncommitted_changes() && (self.current_commit != 0 || self.is_empty()),
            ))
    }

    /// Construct a fresh graph from a historical snapshot.
    pub fn from_snapshot(snapshot: crate::state::state::GraphSnapshot) -> Result<Self, GraphError> {
        let mut graph = Graph::new_with_type(snapshot.graph_type);
        graph.reset_to_snapshot(snapshot)?;
        graph.seed_selected_import()?;
        Ok(graph)
    }

    /*
    === CONVERSION OPERATIONS ===
    Export graph to external formats like NetworkX.
    */

    /// Convert this graph to NetworkX format
    ///
    /// This creates a NetworkXGraph representation that can be easily
    /// converted to a Python NetworkX graph in the FFI layer.
    ///
    /// # Returns
    /// * `NetworkXGraph` - A representation compatible with NetworkX
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::Graph;
    /// let graph = Graph::new();
    /// let nx_graph = graph.to_networkx().unwrap();
    /// ```ignore
    pub fn to_networkx(&self) -> Result<crate::utils::convert::NetworkXGraph, GraphError> {
        crate::utils::convert::graph_to_networkx(self)
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
        let prepared = self.prepare_history_collection(None)?;
        self.execute_history_collection(prepared)
    }

    /// Export the active graph with typed property values (not a history backup).
    pub fn save_to_path(&self, path: &Path) -> Result<(), GraphError> {
        persistence::save(self, path)
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
                if numeric_values.len().is_multiple_of(2) {
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
        self.to_adjacency_matrix()
    }

    /// Simple adjacency matrix (alias for adjacency_matrix)
    pub fn adjacency(&mut self) -> GraphResult<AdjacencyMatrix> {
        self.adjacency_matrix()
    }

    // ===== GRAPHMATRIX CONVERSION OPERATIONS =====

    /// Convert to f64 matrix (most common use case)
    pub fn to_matrix_f64(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<f64>> {
        self.to_matrix::<f64>()
    }

    /// Convert to f32 matrix (memory-efficient for ML)
    pub fn to_matrix_f32(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<f32>> {
        self.to_matrix::<f32>()
    }

    /// Convert to integer matrix
    pub fn to_matrix_i64(&self) -> GraphResult<crate::storage::matrix::GraphMatrix<i64>> {
        self.to_matrix::<i64>()
    }

    // === BULK ATTRIBUTE OPERATIONS (FFI WRAPPERS) ===
    // These are thin wrappers around the trait system for FFI compatibility

    fn collect_invalid_attr_targets(
        &self,
        attr_name: &str,
        values: &[(usize, AttrValue)],
        is_node: bool,
        positions: Option<&[usize]>,
        errors: &mut Vec<String>,
    ) {
        let kind = if is_node { "node" } else { "edge" };
        let mut first_positions = HashMap::new();
        for (offset, (id, value)) in values.iter().enumerate() {
            let id = *id;
            let position = positions
                .and_then(|positions| positions.get(offset))
                .copied()
                .unwrap_or(offset);
            if let Err(error) = kinds::validate_kind_property(attr_name, value, is_node) {
                errors.push(format!("{attr_name}[{position}]: {error}"));
            }
            let exists = if is_node {
                self.space.contains_node(id)
            } else {
                self.space.contains_edge(id)
            };
            if !exists {
                errors.push(format!(
                    "{attr_name}[{position}]: {kind} {id} is not active"
                ));
            }
            match first_positions.entry(id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(position);
                }
                std::collections::hash_map::Entry::Occupied(entry) => errors.push(format!(
                    "{attr_name}[{position}]: duplicate {kind} {id} (first at {attr_name}[{}])",
                    entry.get()
                )),
            }
        }
    }

    fn validate_bulk_attr_targets(
        &self,
        attrs_values: &HashMap<AttrName, Vec<(usize, AttrValue)>>,
        is_node: bool,
        positions: Option<&HashMap<AttrName, Vec<usize>>>,
    ) -> GraphResult<()> {
        let mut attrs: Vec<_> = attrs_values.iter().collect();
        attrs.sort_unstable_by_key(|(name, _)| *name);
        let mut errors = Vec::new();
        for (attr_name, values) in attrs {
            if is_node && attr_name == "entity_type" && !values.is_empty() {
                errors.push("entity_type: system-managed node attribute cannot be changed".into());
            }
            let column_positions = positions
                .and_then(|positions| positions.get(attr_name))
                .map(Vec::as_slice);
            if attr_name == EXTERNAL_ID && !values.is_empty() {
                errors
                    .push("external_id: system-managed attribute; use the external-ID API".into());
            }
            self.collect_invalid_attr_targets(
                attr_name,
                values,
                is_node,
                column_positions,
                &mut errors,
            );
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(GraphError::InvalidInput(format!(
                "Invalid bulk attribute assignments: {}",
                errors.join("; ")
            )))
        }
    }

    /// Validate node property assignments without changing values or change tracking.
    /// Optional positions retain original input offsets when an adapter cannot decode some IDs.
    pub fn validate_node_attrs(
        &self,
        attrs_values: &HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
        positions: Option<&HashMap<AttrName, Vec<usize>>>,
    ) -> GraphResult<()> {
        self.validate_bulk_attr_targets(attrs_values, true, positions)
    }

    /// Validate edge property assignments without changing values or change tracking.
    /// Optional positions retain original input offsets when an adapter cannot decode some IDs.
    pub fn validate_edge_attrs(
        &self,
        attrs_values: &HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
        positions: Option<&HashMap<AttrName, Vec<usize>>>,
    ) -> GraphResult<()> {
        self.validate_bulk_attr_targets(attrs_values, false, positions)
    }

    /// Set node attributes in bulk (delegates to existing bulk operations)
    pub fn set_node_attrs(
        &mut self,
        attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> GraphResult<()> {
        self.validate_node_attrs(&attrs_values, None)?;
        if self.automatic_publication {
            if attrs_values.values().all(Vec::is_empty) {
                return Ok(());
            }
            return self
                .writer("groggy")
                .set_node_attrs(attrs_values)
                .map(|_| ());
        }
        self.write_node_attrs(attrs_values)
    }

    // Internal column writer: the coordinator must validate the complete operation first.
    fn write_node_attrs(
        &mut self,
        mut attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> GraphResult<()> {
        for (name, values) in &mut attrs_values {
            if name == LABELS || name == RELATIONSHIP_TYPE {
                for (_, value) in values {
                    kinds::normalize_kind_property_in_place(name, value, true)?;
                }
            }
        }
        // All targets were validated before collecting their old indices.
        let mut old_indices: HashMap<AttrName, HashMap<NodeId, Option<usize>>> = HashMap::new();
        for (attr_name, node_values) in &attrs_values {
            let mut attr_old_indices = HashMap::new();
            for &(node_id, _) in node_values {
                // Collect old index for change tracking.
                let old_index = self.space.get_node_attr_index(node_id, attr_name);
                attr_old_indices.insert(node_id, old_index);
            }
            old_indices.insert(attr_name.clone(), attr_old_indices);
        }

        // Use optimized vectorized pool operation
        let index_changes = self.pool.borrow_mut().set_bulk_attrs(attrs_values, true);

        // Update space attribute indices and record changes (optimized bulk recording)
        for (attr_name, entity_indices) in index_changes {
            let attr_old_indices = old_indices.get(&attr_name).unwrap();

            // Build change records for bulk recording
            let changes: Vec<_> = entity_indices
                .iter()
                .map(|&(node_id, new_index)| {
                    let old_index = attr_old_indices.get(&node_id).copied().flatten();
                    (node_id, attr_name.clone(), old_index, new_index)
                })
                .collect();

            // Use bulk recording API (strategy can optimize internally)
            self.change_tracker.record_attr_changes(&changes, true);

            // Update space indices
            for (node_id, new_index) in entity_indices {
                self.space
                    .set_node_attr_index(node_id, attr_name.clone(), new_index);
            }
        }

        Ok(())
    }

    /// Set a single node attribute column using a pre-built vector of values
    pub fn set_node_attr_column(
        &mut self,
        attr_name: AttrName,
        mut node_values: Vec<(NodeId, AttrValue)>,
    ) -> GraphResult<()> {
        if self.automatic_publication {
            return self.set_node_attrs(HashMap::from([(attr_name, node_values)]));
        }

        if node_values.is_empty() {
            return Ok(());
        }

        let mut errors = Vec::new();
        if attr_name == "entity_type" {
            errors.push("entity_type: system-managed node attribute cannot be changed".into());
        }
        if attr_name == EXTERNAL_ID {
            errors.push("external_id: system-managed attribute; use set_node_external_id".into());
        }
        self.collect_invalid_attr_targets(&attr_name, &node_values, true, None, &mut errors);
        if !errors.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "Invalid bulk attribute assignments: {}",
                errors.join("; ")
            )));
        }

        for (_, value) in &mut node_values {
            kinds::normalize_kind_property_in_place(&attr_name, value, true)?;
        }

        // Collect previous indices for change tracking after validation.
        let mut old_indices: HashMap<NodeId, Option<usize>> =
            HashMap::with_capacity(node_values.len());
        for (node_id, _) in &node_values {
            let previous = self.space.get_node_attr_index(*node_id, &attr_name);
            old_indices.insert(*node_id, previous);
        }

        // Preserve deterministic ordering by sorting by node id
        node_values.sort_unstable_by_key(|(node_id, _)| *node_id);

        // Vectorized column append
        let index_changes =
            self.pool
                .borrow_mut()
                .set_attr_pairs(attr_name.clone(), node_values, true);

        // Apply space index updates and record change tracker entries
        let mut change_records = Vec::with_capacity(index_changes.len());
        for (node_id, new_index) in index_changes {
            let old_index = old_indices.get(&node_id).copied().flatten();
            change_records.push((node_id, attr_name.clone(), old_index, new_index));
            self.space
                .set_node_attr_index(node_id, attr_name.clone(), new_index);
        }
        self.change_tracker
            .record_attr_changes(&change_records, true);

        Ok(())
    }

    /// Set edge attributes in bulk (delegates to existing bulk operations)
    pub fn set_edge_attrs(
        &mut self,
        attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> GraphResult<()> {
        self.validate_edge_attrs(&attrs_values, None)?;
        if self.automatic_publication {
            if attrs_values.values().all(Vec::is_empty) {
                return Ok(());
            }
            return self
                .writer("groggy")
                .set_edge_attrs(attrs_values)
                .map(|_| ());
        }
        self.write_edge_attrs(attrs_values)
    }

    // Internal column writer: the coordinator must validate the complete operation first.
    fn write_edge_attrs(
        &mut self,
        mut attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> GraphResult<()> {
        for (name, values) in &mut attrs_values {
            if name == LABELS || name == RELATIONSHIP_TYPE {
                for (_, value) in values {
                    kinds::normalize_kind_property_in_place(name, value, false)?;
                }
            }
        }
        // All targets were validated before collecting their old indices.
        let mut old_indices: HashMap<AttrName, HashMap<EdgeId, Option<usize>>> = HashMap::new();
        for (attr_name, edge_values) in &attrs_values {
            let mut attr_old_indices = HashMap::new();
            for &(edge_id, _) in edge_values {
                // Collect old index for change tracking.
                let old_index = self.space.get_edge_attr_index(edge_id, attr_name);
                attr_old_indices.insert(edge_id, old_index);
            }
            old_indices.insert(attr_name.clone(), attr_old_indices);
        }

        // Use optimized vectorized pool operation
        let index_changes = self.pool.borrow_mut().set_bulk_attrs(attrs_values, false);

        // Update space attribute indices and record changes (optimized bulk recording)
        for (attr_name, entity_indices) in index_changes {
            let attr_old_indices = old_indices.get(&attr_name).unwrap();

            // Build change records for bulk recording
            let changes: Vec<_> = entity_indices
                .iter()
                .map(|&(edge_id, new_index)| {
                    let old_index = attr_old_indices.get(&edge_id).copied().flatten();
                    (edge_id, attr_name.clone(), old_index, new_index)
                })
                .collect();

            // Use bulk recording API (strategy can optimize internally)
            self.change_tracker.record_attr_changes(&changes, false);

            // Update space indices
            for (edge_id, new_index) in entity_indices {
                self.space
                    .set_edge_attr_index(edge_id, attr_name.clone(), new_index);
            }
        }

        Ok(())
    }

    // =============================================================================
    // PHASE 5: Graph Integration & Equivalence - Table Access Methods
    // =============================================================================

    /// 🎨 Get visualization module for unified rendering
    ///
    /// This provides a single entry point for all visualization backends:
    /// - Jupyter widgets
    /// - WebSocket streaming servers
    /// - Static file export (HTML, SVG, PNG, PDF)
    /// - Local browser display
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use groggy::Graph;
    ///
    /// let mut graph = Graph::new();
    /// graph.add_node("A", [("label", "Node A")].into());
    /// graph.add_node("B", [("label", "Node B")].into());
    /// graph.add_edge("A", "B", [("weight", 1.0)].into());
    ///
    /// // Unified API - all use the same core engine
    /// let widget = graph.viz().widget()?;           // Jupyter widget
    /// let server = graph.viz().serve(Some(8080))?;  // Streaming server
    /// graph.viz().save("graph.html")?;              // Static HTML
    /// graph.viz().show()?;                          // Local browser
    /// ```
    pub fn viz(&self) -> VizModule {
        // Create a GraphDataSource adapter for this graph
        let graph_data_source = Arc::new(crate::viz::streaming::GraphDataSource::new(self));
        VizModule::new(graph_data_source)
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
        AttrValue::Float64(f) => *f,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node_entity_type_indexed() {
        // Test that entity_type is properly indexed after add_node
        let mut graph = Graph::new();
        let node_id = graph.add_node();

        // Should be able to retrieve entity_type attribute
        let entity_type = graph
            .get_node_attr(node_id, &"entity_type".into())
            .expect("Failed to get entity_type")
            .expect("entity_type should be set");

        assert_eq!(entity_type, AttrValue::Text("base".to_string()));
    }

    #[test]
    fn test_add_nodes_entity_type_indexed() {
        // Test that entity_type is properly indexed after add_nodes bulk operation
        let mut graph = Graph::new();
        let node_ids = graph.add_nodes(5);

        assert_eq!(node_ids.len(), 5);

        // All nodes should have entity_type attribute queryable
        for node_id in node_ids {
            let entity_type = graph
                .get_node_attr(node_id, &"entity_type".into())
                .expect("Failed to get entity_type")
                .expect("entity_type should be set");

            assert_eq!(entity_type, AttrValue::Text("base".to_string()));
        }
    }

    #[test]
    fn test_bulk_node_attrs_tracked() {
        // Test that bulk attribute updates are recorded in change tracker
        let mut graph = Graph::new();
        let node_ids = graph.add_nodes(3);

        // Commit initial state
        graph
            .commit("Initial commit".to_string(), "test".to_string())
            .expect("Commit should succeed");

        // Build bulk attribute update
        let mut attrs_values = HashMap::new();
        let label_values = vec![
            (node_ids[0], AttrValue::Text("Node A".to_string())),
            (node_ids[1], AttrValue::Text("Node B".to_string())),
            (node_ids[2], AttrValue::Text("Node C".to_string())),
        ];
        attrs_values.insert("label".into(), label_values);

        // Apply bulk update
        graph
            .set_node_attrs(attrs_values)
            .expect("Bulk update should succeed");

        // Should be able to commit (which means changes were tracked)
        // If changes weren't tracked, this would fail with NoChangesToCommit
        let result = graph.commit("Bulk attribute update".to_string(), "test".to_string());
        assert!(
            result.is_ok(),
            "Commit should succeed when changes are tracked"
        );

        // Verify attributes were actually set
        assert_eq!(
            graph.get_node_attr(node_ids[0], &"label".into()).unwrap(),
            Some(AttrValue::Text("Node A".to_string()))
        );
        assert_eq!(
            graph.get_node_attr(node_ids[1], &"label".into()).unwrap(),
            Some(AttrValue::Text("Node B".to_string()))
        );
    }

    #[test]
    fn test_bulk_edge_attrs_tracked() {
        // Test that bulk edge attribute updates are recorded in change tracker
        let mut graph = Graph::new();
        let node_ids = graph.add_nodes(4);

        // Add some edges
        let edge1 = graph
            .add_edge(node_ids[0], node_ids[1])
            .expect("Add edge should succeed");
        let edge2 = graph
            .add_edge(node_ids[1], node_ids[2])
            .expect("Add edge should succeed");
        let edge3 = graph
            .add_edge(node_ids[2], node_ids[3])
            .expect("Add edge should succeed");

        // Commit initial state
        graph
            .commit("Initial commit".to_string(), "test".to_string())
            .expect("Commit should succeed");

        // Build bulk edge attribute update
        let mut attrs_values = HashMap::new();
        let weight_values = vec![
            (edge1, AttrValue::Float(1.5)),
            (edge2, AttrValue::Float(2.0)),
            (edge3, AttrValue::Float(3.5)),
        ];
        attrs_values.insert("weight".into(), weight_values);

        // Apply bulk update
        graph
            .set_edge_attrs(attrs_values)
            .expect("Bulk update should succeed");

        // Should be able to commit (which means changes were tracked)
        // If changes weren't tracked, this would fail with NoChangesToCommit
        let result = graph.commit("Bulk edge attribute update".to_string(), "test".to_string());
        assert!(
            result.is_ok(),
            "Commit should succeed when changes are tracked"
        );

        // Verify attributes were actually set
        assert_eq!(
            graph.get_edge_attr(edge1, &"weight".into()).unwrap(),
            Some(AttrValue::Float(1.5))
        );
        assert_eq!(
            graph.get_edge_attr(edge2, &"weight".into()).unwrap(),
            Some(AttrValue::Float(2.0))
        );
    }

    #[test]
    fn test_add_graph_topology_preserved() {
        // Test that add_graph properly clones topology with remapped IDs
        let mut source_graph = Graph::new();

        // Build source graph with 3 nodes and 2 edges
        let src_nodes = source_graph.add_nodes(3);
        let src_edge1 = source_graph
            .add_edge(src_nodes[0], src_nodes[1])
            .expect("Add edge should succeed");
        let src_edge2 = source_graph
            .add_edge(src_nodes[1], src_nodes[2])
            .expect("Add edge should succeed");

        // Add node attributes
        source_graph
            .set_node_attr(
                src_nodes[0],
                "name".into(),
                AttrValue::Text("A".to_string()),
            )
            .unwrap();
        source_graph
            .set_node_attr(
                src_nodes[1],
                "name".into(),
                AttrValue::Text("B".to_string()),
            )
            .unwrap();
        source_graph
            .set_node_attr(
                src_nodes[2],
                "name".into(),
                AttrValue::Text("C".to_string()),
            )
            .unwrap();

        // Verify source graph has the attributes
        assert_eq!(
            source_graph
                .get_node_attr(src_nodes[0], &"name".into())
                .unwrap(),
            Some(AttrValue::Text("A".to_string())),
            "Source graph should have attribute on node 0"
        );

        // Add edge attributes
        source_graph
            .set_edge_attr(src_edge1, "weight".into(), AttrValue::Float(1.0))
            .unwrap();
        source_graph
            .set_edge_attr(src_edge2, "weight".into(), AttrValue::Float(2.0))
            .unwrap();

        // Create target graph and merge
        let mut target_graph = Graph::new();
        target_graph
            .add_graph(&source_graph)
            .expect("add_graph should succeed");

        // Verify topology
        assert_eq!(target_graph.node_count(), 3, "Should have 3 nodes");
        assert_eq!(target_graph.edge_count(), 2, "Should have 2 edges");

        // Verify node attributes were copied (check that we have nodes with names A, B, C)
        let target_nodes = target_graph.node_ids();
        let mut found_names = Vec::new();
        for node_id in target_nodes {
            let attr_result = target_graph.get_node_attr(node_id, &"name".into());
            if let Ok(Some(attr_val)) = attr_result {
                // Handle both Text and CompactText variants
                let name = match attr_val {
                    AttrValue::Text(s) => s,
                    AttrValue::CompactText(cs) => cs.as_str().to_string(),
                    _ => continue,
                };
                found_names.push(name);
            }
        }
        found_names.sort();
        assert_eq!(
            found_names,
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            "All node names should be copied"
        );

        // Verify edge attributes were copied (check that we have edges with weights 1.0 and 2.0)
        let target_edges = target_graph.edge_ids();
        let mut found_weights = Vec::new();
        for edge_id in target_edges {
            if let Ok(Some(AttrValue::Float(weight))) =
                target_graph.get_edge_attr(edge_id, &"weight".into())
            {
                found_weights.push(weight);
            }
        }
        found_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            found_weights,
            vec![1.0, 2.0],
            "All edge weights should be copied"
        );
    }

    #[test]
    fn test_neighbors_bulk() {
        use crate::storage::table::traits::Table;

        // Test bulk neighbors operation returns proper BaseTable
        let mut graph = Graph::new();

        // Create a small graph: 0->1, 0->2, 1->2, 2->3
        let nodes = graph.add_nodes(4);
        graph.add_edge(nodes[0], nodes[1]).unwrap();
        graph.add_edge(nodes[0], nodes[2]).unwrap();
        graph.add_edge(nodes[1], nodes[2]).unwrap();
        graph.add_edge(nodes[2], nodes[3]).unwrap();

        // Query neighbors for multiple nodes
        let query_nodes = vec![nodes[0], nodes[1], nodes[2]];
        let table = graph
            .neighbors_bulk(&query_nodes)
            .expect("neighbors_bulk should succeed");

        // Verify table has correct columns
        assert!(table.has_column("node_id"), "Should have node_id column");
        assert!(
            table.has_column("neighbor_id"),
            "Should have neighbor_id column"
        );

        // Verify we have neighbor relationships
        // (exact count depends on directed/undirected graph type)
        assert!(
            table.nrows() >= 3,
            "Should have at least 3 neighbor relationships"
        );
    }
}
