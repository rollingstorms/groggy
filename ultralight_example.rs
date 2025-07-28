use std::collections::{HashMap, HashSet};
use std::sync::Arc;

type NodeIndex = usize;
type EdgeIndex = usize;
type AttrName = String;
type StateId = u64;
type BranchName = String;

#[derive(Debug, Clone)]
pub enum AttrValue {
    Float(f32),
    Int(i64),
    Text(String),
    Vec(Vec<f32>),
    Bool(bool)
}

/// Columnar delta storage for efficient bulk operations
#[derive(Debug, Clone)]
pub struct ColumnDelta {
    /// Sorted indices where changes occurred
    indices: Vec<usize>,
    /// Corresponding values (parallel array to indices)
    values: Vec<AttrValue>,
}

/// Immutable delta object representing changes between states
#[derive(Debug, Clone)]
pub struct DeltaObject {
    /// Node attribute changes stored columnarly
    node_attrs: Arc<HashMap<AttrName, ColumnDelta>>,
    /// Edge attribute changes stored columnarly  
    edge_attrs: Arc<HashMap<AttrName, ColumnDelta>>,
    /// Nodes that became active/inactive
    node_active_changes: Arc<ColumnDelta>, // indices + bool values
    /// Edges that became active/inactive
    edge_active_changes: Arc<ColumnDelta>, // indices + bool values
}

/// Tracks what's changed since base_state (mutable)
pub struct ChangeTracker {
    /// Node attribute changes by attribute name
    node_attrs: HashMap<AttrName, ColumnDelta>,
    /// Edge attribute changes by attribute name
    edge_attrs: HashMap<AttrName, ColumnDelta>,
    /// Nodes that became active/inactive
    node_active_changes: ColumnDelta,
    /// Edges that became active/inactive
    edge_active_changes: ColumnDelta,
}

/// Immutable state object - a point in the graph's history
#[derive(Debug, Clone)]
pub struct StateObject {
    /// Parent state (None for root)
    parent: Option<StateId>,
    /// Changes from parent
    delta: Arc<DeltaObject>,
    /// Metadata
    metadata: Arc<StateMetadata>,
}

#[derive(Debug, Clone)]
pub struct StateMetadata {
    /// Human-readable label
    label: String,
    /// When this state was created
    timestamp: u64,
    /// Who created this state
    author: String,
    /// Content hash for verification/deduplication
    hash: [u8; 32],
}

/// Immutable append-only storage layer of states
pub struct GraphForest {
    /// All states indexed by ID
    states: HashMap<StateId, Arc<StateObject>>,
    /// State parent->children index for traversal
    children: HashMap<StateId, Vec<StateId>>,
    /// Content-addressed storage for deduplication
    deltas_by_hash: HashMap<[u8; 32], Arc<DeltaObject>>,
}

/// Branch pointer to a state
#[derive(Debug, Clone)]
pub struct Branch {
    name: BranchName,
    head: StateId,
}

/// Configuration for the graph
pub struct GraphConfig {
    /// How often to create snapshots
    snapshot_frequency: u32,
    /// Maximum delta chain length before forcing snapshot
    max_delta_chain: u32,
    /// Whether to compress old deltas
    enable_compression: bool,
}

/// The main graph structure
pub struct Graph {
    /// Mutable working copy
    space: GraphSpace,
    /// Immutable history
    states: GraphForest,
    /// Branch management
    refs: RefManager,
    /// Configuration
    config: GraphConfig,
}

/// Mutable workspace - the "working graph"
pub struct GraphSpace {
    /// Current graph structure
    pool: GraphPool,
    /// Which state we're based on
    base_state: StateId,
    /// Uncommitted changes tracking
    changes: ChangeTracker,
}

/// The actual mutable graph data flyweight pool
pub struct GraphPool {
    /// Graph dimensions
    node_count: usize,
    edge_count: usize,
    /// Topology
    edge_index: HashMap<EdgeIndex, (NodeIndex, NodeIndex)>,
    active_nodes: HashSet<NodeIndex>,
    active_edges: HashSet<EdgeIndex>,
    /// Columnar attribute storage
    node_attrs: HashMap<AttrName, Vec<AttrValue>>,
    edge_attrs: HashMap<AttrName, Vec<AttrValue>>,
}

/// Manages branches and refs
pub struct RefManager {
    /// All branches
    branches: HashMap<BranchName, Branch>,
    /// Currently checked out branch
    current_branch: BranchName,
    /// Tags (immutable refs)
    tags: HashMap<String, StateId>,
}

/// Read-only view at a specific state
pub struct StateView<'a> {
    /// Reference to the forest
    forest: &'a GraphForest,
    /// State we're viewing
    state_id: StateId,
}