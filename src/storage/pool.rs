//! Graph Pool - Pure Data Storage Tables
//!
//! ARCHITECTURE ROLE:
//! GraphPool is the "database" - it stores ALL the data tables but doesn't
//! know what's currently "active". It's just efficient storage for nodes,
//! edges, and attributes that can grow indefinitely.
//!
//! DESIGN PHILOSOPHY:
//! - GraphPool = Pure Storage (all the data tables, no business logic)
//! - GraphSpace = Active View (knows what's currently active)
//! - Pool provides raw storage, Space manages the active subset
//! - Pool can store "deleted" entities, Space decides what's visible

/*
=== POOL VS SPACE SEPARATION ===

GraphPool (this module):
- Stores ALL nodes/edges/attributes that have ever existed
- Grows indefinitely (append-only for performance)
- No concept of "active" vs "inactive"
- Pure storage with efficient access methods
- Can store soft-deleted entities

GraphSpace (space.rs):
- Knows which entities are currently "active"
- Manages the active subset of pool data
- Handles add/remove operations by updating active sets
- Provides the "current view" of the graph
- Tracks changes for history commits

This separation allows:
- Pool to be optimized for storage efficiency
- Space to be optimized for current state operations
- Better separation of concerns
- Easier testing and reasoning
*/

mod archive;
pub(crate) mod lifecycle;
use lifecycle::PurgedIds;
pub(crate) mod compaction;
pub(crate) use archive::PoolDocument;

use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, GraphType, NodeId, SubgraphId};
use std::collections::HashMap;

/// Columnar storage for attribute values with memory pooling
///
/// DESIGN: Store attribute values in a single vector for cache efficiency.
/// This is much faster for bulk operations and analytics workloads compared
/// to storing attributes per-entity.
///
/// USAGE: Each entity gets an index into this column. When the entity's
/// attribute changes, we append the new value and update the index.
///
/// MEMORY OPTIMIZATION: Uses memory pooling to reduce allocations
#[derive(Debug, Clone)]
pub struct AttributeColumn {
    /// All values ever stored for this attribute (append-only)
    values: Vec<AttrValue>,
    /// Memory pool for reused values (Memory Optimization 2)
    memory_pool: AttributeMemoryPool,
    /// Built only for state restoration; hashes retain indices rather than duplicate values.
    restore_indices: Option<RestorationIndex>,
}

/// One representative per distinct value; extra positions only for real hash collisions.
#[derive(Debug, Clone, Default)]
struct RestorationIndex {
    first: HashMap<u64, usize>,
    collisions: HashMap<u64, Vec<usize>>,
}
impl RestorationIndex {
    fn insert(&mut self, index: usize, values: &[AttrValue]) {
        let hash = AttributeColumn::value_hash(&values[index]);
        match self.first.entry(hash) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(index);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                if !AttributeColumn::restoration_equal(&values[*entry.get()], &values[index]) {
                    let candidates = self.collisions.entry(hash).or_default();
                    if !candidates.iter().any(|&old| {
                        AttributeColumn::restoration_equal(&values[old], &values[index])
                    }) {
                        candidates.push(index);
                    }
                }
            }
        }
    }
    fn find(&self, value: &AttrValue, values: &[AttrValue]) -> Option<usize> {
        let hash = AttributeColumn::value_hash(value);
        let &first = self.first.get(&hash)?;
        if AttributeColumn::restoration_equal(&values[first], value) {
            return Some(first);
        }
        self.collisions
            .get(&hash)?
            .iter()
            .copied()
            .find(|&index| AttributeColumn::restoration_equal(&values[index], value))
    }
    fn memory_usage(&self) -> usize {
        self.first.capacity() * std::mem::size_of::<(u64, usize)>()
            + self.collisions.capacity() * std::mem::size_of::<(u64, Vec<usize>)>()
            + self
                .collisions
                .values()
                .map(|rows| rows.capacity() * std::mem::size_of::<usize>())
                .sum::<usize>()
    }
}

/// Memory pool for reusing attribute values to reduce allocations
#[derive(Debug, Clone)]
pub struct AttributeMemoryPool {
    /// Pool of reusable string allocations
    string_pool: Vec<String>,
    /// Pool of reusable float vectors
    float_vec_pool: Vec<Vec<f32>>,
    /// Pool of reusable byte vectors  
    #[allow(dead_code)] // TODO: Implement byte pool reuse
    byte_pool: Vec<Vec<u8>>,
    /// Statistics
    reuse_count: usize,
    allocation_count: usize,
}

impl AttributeMemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            string_pool: Vec::new(),
            float_vec_pool: Vec::new(),
            byte_pool: Vec::new(),
            reuse_count: 0,
            allocation_count: 0,
        }
    }

    /// Get a reusable string or allocate a new one
    pub fn get_string(&mut self, content: &str) -> String {
        if let Some(mut reused) = self.string_pool.pop() {
            reused.clear();
            reused.push_str(content);
            self.reuse_count += 1;
            reused
        } else {
            self.allocation_count += 1;
            content.to_string()
        }
    }

    /// Return a string to the pool for reuse
    pub fn return_string(&mut self, mut string: String) {
        if string.capacity() <= 1024 {
            // Only pool reasonably sized strings
            string.clear();
            self.string_pool.push(string);
        }
    }

    /// Get a reusable float vector or allocate a new one
    pub fn get_float_vec(&mut self, capacity: usize) -> Vec<f32> {
        if let Some(mut reused) = self.float_vec_pool.pop() {
            reused.clear();
            reused.reserve(capacity);
            self.reuse_count += 1;
            reused
        } else {
            self.allocation_count += 1;
            Vec::with_capacity(capacity)
        }
    }

    /// Return a float vector to the pool for reuse
    pub fn return_float_vec(&mut self, mut vec: Vec<f32>) {
        if vec.capacity() <= 10000 {
            // Only pool reasonably sized vectors
            vec.clear();
            self.float_vec_pool.push(vec);
        }
    }

    /// Get memory pool statistics
    pub fn stats(&self) -> (usize, usize, f64) {
        let total = self.reuse_count + self.allocation_count;
        let reuse_rate = if total > 0 {
            self.reuse_count as f64 / total as f64
        } else {
            0.0
        };
        (self.reuse_count, self.allocation_count, reuse_rate)
    }
}

impl AttributeColumn {
    /// Create a new empty attribute column with memory pooling
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            memory_pool: AttributeMemoryPool::new(),
            restore_indices: None,
        }
    }

    /// Create a new attribute column with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            memory_pool: AttributeMemoryPool::new(),
            restore_indices: None,
        }
    }

    /// Append a new value and return its index
    ///
    /// PERFORMANCE: O(1) amortized append with memory optimization
    pub fn push(&mut self, value: AttrValue) -> usize {
        let index = self.values.len();
        // Optimize the value before storing (Memory Optimization 1)
        let optimized_value = value.optimize();
        self.values.push(optimized_value);
        if let Some(indices) = &mut self.restore_indices {
            indices.insert(index, &self.values);
        }
        index
    }

    // Query/property identity deliberately groups signed zeros and NaNs. State
    // restoration must preserve their stored bits, so its lookup uses a stricter
    // relation without changing public comparison or historical content hashes.
    fn restoration_f32_values(value: &AttrValue) -> Option<std::borrow::Cow<'_, [f32]>> {
        match value {
            AttrValue::FloatVec(values) => Some(std::borrow::Cow::Borrowed(values)),
            AttrValue::CompressedFloatVec(data) => data
                .decompress_float_vec()
                .ok()
                .map(std::borrow::Cow::Owned),
            _ => None,
        }
    }
    fn restoration_equal(left: &AttrValue, right: &AttrValue) -> bool {
        match (left, right) {
            (AttrValue::Float(a), AttrValue::Float(b)) => a.to_bits() == b.to_bits(),
            (AttrValue::Float64(a), AttrValue::Float64(b)) => a.to_bits() == b.to_bits(),
            (AttrValue::Float64Vec(a), AttrValue::Float64Vec(b)) => a
                .iter()
                .map(|v| v.to_bits())
                .eq(b.iter().map(|v| v.to_bits())),
            (
                AttrValue::FloatVec(_) | AttrValue::CompressedFloatVec(_),
                AttrValue::FloatVec(_) | AttrValue::CompressedFloatVec(_),
            ) => {
                match (
                    Self::restoration_f32_values(left),
                    Self::restoration_f32_values(right),
                ) {
                    (Some(a), Some(b)) => a
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(b.iter().map(|v| v.to_bits())),
                    _ => false,
                }
            }
            _ => left == right,
        }
    }
    fn value_hash(value: &AttrValue) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        match value {
            AttrValue::Float(v) => {
                0u8.hash(&mut hash);
                v.to_bits().hash(&mut hash);
            }
            AttrValue::Float64(v) => {
                1u8.hash(&mut hash);
                v.to_bits().hash(&mut hash);
            }
            AttrValue::Float64Vec(values) => {
                2u8.hash(&mut hash);
                values.len().hash(&mut hash);
                for value in values {
                    value.to_bits().hash(&mut hash);
                }
            }
            AttrValue::FloatVec(_) | AttrValue::CompressedFloatVec(_) => {
                if let Some(values) = Self::restoration_f32_values(value) {
                    3u8.hash(&mut hash);
                    values.len().hash(&mut hash);
                    for value in values.iter() {
                        value.to_bits().hash(&mut hash);
                    }
                } else {
                    4u8.hash(&mut hash);
                    value.hash(&mut hash);
                }
            }
            _ => {
                4u8.hash(&mut hash);
                value.hash(&mut hash);
            }
        }
        hash.finish()
    }

    /// Reuse an exactly represented retained logical value for reconstruction,
    /// preserving floating bits and checking hash collisions.
    /// Ordinary mutations continue to append their own retained column values.
    pub(crate) fn restore_value(&mut self, value: &AttrValue) -> usize {
        let indices = self.restore_indices.get_or_insert_with(|| {
            let mut indices = RestorationIndex::default();
            for index in 0..self.values.len() {
                indices.insert(index, &self.values);
            }
            indices
        });
        if let Some(index) = indices.find(value, &self.values) {
            return index;
        }
        self.push(value.clone())
    }

    /// Bulk append values using Vec::extend (VECTORIZED - much faster than push loop)
    ///
    /// PERFORMANCE: O(n) but with vectorized operations, single allocation, and memory optimization
    /// Returns (start_index, end_index) range
    pub fn extend_values(&mut self, values: Vec<AttrValue>) -> (usize, usize) {
        let start_index = self.values.len();

        // Optimize all values before bulk insertion (Memory Optimization 1)
        let optimized_values: Vec<_> = values.into_iter().map(|v| v.optimize()).collect();

        self.values.extend(optimized_values); // Single vectorized operation!
        if let Some(indices) = &mut self.restore_indices {
            for index in start_index..self.values.len() {
                indices.insert(index, &self.values);
            }
        }
        let end_index = self.values.len() - 1;
        (start_index, end_index)
    }

    /// Pre-allocate capacity for bulk operations (prevents reallocations)
    ///
    /// PERFORMANCE: Critical for large bulk operations
    pub fn reserve_capacity(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    /// Get value at a specific index
    ///
    /// PERFORMANCE: O(1) random access
    pub fn get(&self, index: usize) -> Option<&AttrValue> {
        self.values.get(index)
    }

    /// Get the number of values stored
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the column is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get all values as a slice (for bulk operations)
    pub fn as_slice(&self) -> &[AttrValue] {
        &self.values
    }

    /// Get memory pool statistics (Memory Optimization 2)
    pub fn memory_stats(&self) -> (usize, usize, f64) {
        self.memory_pool.stats()
    }

    /// Calculate total memory usage of this column (Memory Optimization 1)
    pub fn memory_usage(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let values_size = self.values.iter().map(|v| v.memory_size()).sum::<usize>();
        let pool_overhead = std::mem::size_of::<AttributeMemoryPool>();

        let restore_overhead = self
            .restore_indices
            .as_ref()
            .map_or(0, RestorationIndex::memory_usage);
        base_size + values_size + pool_overhead + restore_overhead
    }

    /// Ensure the column can address up to the given index (sparse indexing support)
    pub fn ensure_len(&mut self, min_len: usize) {
        if self.values.len() < min_len {
            self.values.resize(min_len, AttrValue::Null);
            self.restore_indices = None;
        }
    }

    /// Set value at a specific index (for node/edge ID indexing)
    pub fn set(&mut self, index: usize, value: AttrValue) {
        if index >= self.values.len() {
            self.ensure_len(index + 1);
        }
        let optimized_value = value.optimize();
        self.values[index] = optimized_value;
        self.restore_indices = None;
    }
}

/// Pure data storage for all graph entities and attributes
///
/// DESIGN: This is just efficient storage - it doesn't know or care
/// about what's "active". That's GraphSpace's responsibility.
///
/// RESPONSIBILITIES:
/// - Store all nodes, edges, and attributes efficiently
/// - Provide fast access to stored data
/// - Handle memory management and growth
/// - Support bulk operations on raw data
///
/// NOT RESPONSIBLE FOR:
/// - Determining what's "active" (that's GraphSpace)
/// - Business logic (that's Graph coordinator)
/// - Version control (that's HistoryForest)
/// - Change tracking (that's ChangeTracker)
#[derive(Debug, Clone)]
pub struct GraphPool {
    /*
    === GRAPH CONFIGURATION ===
    Fundamental graph properties that affect storage and retrieval
    */
    /// Graph directionality - affects how edges are interpreted
    /// DESIGN: Stored here for consistency and potential future optimizations
    graph_type: GraphType,

    /*
    === COLUMNAR ATTRIBUTE STORAGE ===
    Store attributes in columns (one Vec per attribute name) rather than
    rows (one HashMap per entity). This gives better cache locality for
    analytics workloads and bulk operations.
    */
    /// Node attributes: attr_name -> AttributeColumn (append-only columnar storage)
    /// DESIGN: All attribute values ever stored, indexed by position
    node_attributes: HashMap<AttrName, AttributeColumn>,

    /// Edge attributes: attr_name -> AttributeColumn (append-only columnar storage)
    edge_attributes: HashMap<AttrName, AttributeColumn>,

    /*
    === TOPOLOGY STORAGE ===
    Raw storage for all edge connectivity information
    */
    /// Dense storage for edge sources aligned by EdgeId
    edge_sources: Vec<NodeId>,
    /// Dense storage for edge targets aligned by EdgeId
    edge_targets: Vec<NodeId>,
    /// Tombstone bit indicating if an edge slot is currently active
    edge_alive: Vec<bool>,
    /// Compact endpoint storage after explicit purge, preserving lifetime IDs.
    sparse_edges: Option<HashMap<EdgeId, (NodeId, NodeId)>>,

    /*
    === SUBGRAPH STORAGE ===
    Storage for subgraph data used by meta-nodes
    */
    /// Stored subgraphs: subgraph_id -> (nodes, edges, type)
    /// STORAGE: Used by hierarchical meta-nodes to reconstruct contained subgraphs
    stored_subgraphs: HashMap<
        crate::types::SubgraphId,
        (
            std::collections::HashSet<NodeId>,
            std::collections::HashSet<EdgeId>,
            String,
        ),
    >,

    /// Frozen source revisions; membership-only legacy references remain separate.
    retained_groups:
        std::sync::Arc<HashMap<SubgraphId, crate::storage::retained_group::RetainedGroup>>,

    /*
    === ID MANAGEMENT ===
    Simple incrementing counters for new entities
    */
    /// Next available node ID - increment on each new node
    next_node_id: NodeId,
    purged_nodes: PurgedIds,
    purged_edges: PurgedIds,

    /// Next available edge ID - increment on each new edge
    next_edge_id: EdgeId,
}

/// A validated immutable group reference; installation cannot collide or allocate
/// after prepare has checked and reserved its slot under an exclusive pool borrow.
pub(crate) struct PreparedStoredSubgraph {
    pub id: crate::types::SubgraphId,
    retained: Option<crate::storage::retained_group::RetainedGroup>,
    content: (
        std::collections::HashSet<NodeId>,
        std::collections::HashSet<EdgeId>,
        String,
    ),
}

impl GraphPool {
    /// Create new empty graph store with default settings (undirected)
    pub fn new() -> Self {
        Self::new_with_type(GraphType::default())
    }

    /// Create new empty graph store with specified directionality
    pub fn new_with_type(graph_type: GraphType) -> Self {
        Self {
            graph_type,
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            edge_sources: Vec::new(),
            edge_targets: Vec::new(),
            edge_alive: Vec::new(),
            sparse_edges: None,
            stored_subgraphs: HashMap::new(),
            retained_groups: Default::default(),
            next_node_id: 0,
            purged_nodes: PurgedIds::default(),
            purged_edges: PurgedIds::default(),
            next_edge_id: 0,
        }
    }

    /// Get the graph type
    pub fn graph_type(&self) -> GraphType {
        self.graph_type
    }

    /// Commit changes (no-op for append-only storage)
    /// In append-only storage, committing just means the current indices become the new baseline
    pub fn commit_baseline(&mut self) {
        // ALGORITHM: No action needed for append-only storage
        // The current state is already the baseline - Space manages the index mappings
        // This method exists for API compatibility but is essentially a no-op

        // Future optimization: Could implement garbage collection of unreferenced indices here
    }

    /// Get attribute value by index (for Space to resolve indices)
    pub fn get_attr_by_index(
        &self,
        attr: &AttrName,
        index: usize,
        is_node: bool,
    ) -> Option<&AttrValue> {
        // ALGORITHM: Direct index lookup in columnar storage
        // 1. Get the appropriate attribute column
        // 2. Return value at the specified index

        let column_map = if is_node {
            &self.node_attributes
        } else {
            &self.edge_attributes
        };

        column_map.get(attr).and_then(|column| column.get(index))
    }

    /*
    === ENTITY CREATION ===
    Pool handles creating and storing all nodes/edges
    */

    /// Create a new node and return its ID
    /// DESIGN: Pool creates the node, Space tracks it as active
    pub fn add_node(&mut self) -> NodeId {
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        node_id
    }

    /// Create multiple nodes with single ID allocation (BULK OPTIMIZED)
    ///
    /// PERFORMANCE: O(1) instead of O(n) for ID allocation
    /// Returns an inclusive (start_id, end_id) range and individual node IDs.
    /// For zero count, both endpoints equal the next ID and the ID list is empty;
    /// the endpoints do not describe an allocated range in that case.
    pub fn add_nodes_bulk(&mut self, count: usize) -> (NodeId, NodeId, Vec<NodeId>) {
        let start_id = self.next_node_id;
        if count == 0 {
            return (start_id, start_id, Vec::new());
        }
        let end_id = start_id + count - 1;

        // Single ID counter update
        self.next_node_id += count;

        // Generate node IDs (much faster than individual calls)
        let node_ids: Vec<NodeId> = (start_id..=end_id).collect();

        (start_id, end_id, node_ids)
    }

    /// Ensure a specific node ID exists (used for state reconstruction)
    /// This is used when restoring historical states during branch switching
    pub fn ensure_node_id_exists(&mut self, node_id: NodeId) -> GraphResult<()> {
        if self.purged_nodes.contains(node_id) {
            return Err(GraphError::InvalidInput(format!(
                "Node {node_id} was explicitly purged"
            )));
        }
        // Update next_node_id if necessary to avoid ID collisions
        if node_id >= self.next_node_id {
            self.next_node_id = node_id
                .checked_add(1)
                .ok_or_else(|| GraphError::InvalidInput("Node ID exhausted".into()))?;
        }
        Ok(())
    }

    /// Ensure internal edge vectors can address the requested edge slot
    fn ensure_edge_slot(&mut self, edge_id: EdgeId) {
        if edge_id >= self.edge_sources.len() {
            let new_len = edge_id + 1;
            self.edge_sources.resize(new_len, NodeId::default());
            self.edge_targets.resize(new_len, NodeId::default());
            self.edge_alive.resize(new_len, false);
        }
    }

    /// Write edge endpoints into dense storage with tombstone tracking
    fn write_edge_slot(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        if let Some(edges) = &mut self.sparse_edges {
            edges.insert(edge_id, (source, target));
            return;
        }
        self.ensure_edge_slot(edge_id);
        self.edge_sources[edge_id] = source;
        self.edge_targets[edge_id] = target;
        self.edge_alive[edge_id] = true;
    }

    /// Create a new edge between two nodes
    /// DESIGN: Pool creates and stores the edge, Space tracks it as active
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> EdgeId {
        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;
        self.write_edge_slot(edge_id, source, target);
        edge_id
    }

    /// Create multiple edges with bulk vector insertion (BULK OPTIMIZED)
    ///
    /// PERFORMANCE: Batch validation + dense vector writes
    /// Returns Vec<EdgeId> for created edges
    pub fn add_edges(&mut self, edges: &[(NodeId, NodeId)]) -> Vec<EdgeId> {
        let start_id = self.next_edge_id;
        let edge_ids: Vec<EdgeId> = (start_id..start_id + edges.len()).collect();

        if let Some(sparse) = &mut self.sparse_edges {
            sparse.reserve(edges.len());
        } else {
            self.edge_sources.reserve(edges.len());
            self.edge_targets.reserve(edges.len());
            self.edge_alive.reserve(edges.len());
        }

        for (i, &(source, target)) in edges.iter().enumerate() {
            let edge_id = start_id + i;
            self.write_edge_slot(edge_id, source, target);
        }

        // Single counter update
        self.next_edge_id += edges.len();

        edge_ids
    }

    /// Create an edge with a specific ID (used for state reconstruction)
    /// This is used when restoring historical states during branch switching
    pub fn add_edge_with_id(
        &mut self,
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<()> {
        if self.purged_edges.contains(edge_id)
            || self.purged_nodes.contains(source)
            || self.purged_nodes.contains(target)
        {
            return Err(GraphError::InvalidInput(
                "Cannot reconstruct an explicitly purged entity".into(),
            ));
        }
        if self
            .get_edge_endpoints(edge_id)
            .is_some_and(|pair| pair != (source, target))
        {
            return Err(GraphError::InvalidInput(format!(
                "Raw restoration cannot rewire edge {edge_id}; endpoint changes require coordinated history"
            )));
        }
        let end = edge_id
            .checked_add(1)
            .ok_or_else(|| GraphError::InvalidInput("Edge ID exhausted".into()))?;
        self.write_edge_slot(edge_id, source, target);
        // Update next_edge_id if necessary to avoid ID collisions
        if edge_id >= self.next_edge_id {
            self.next_edge_id = end;
        }
        Ok(())
    }

    /// Get the endpoints of an edge from storage
    pub fn get_edge_endpoints(&self, edge_id: EdgeId) -> Option<(NodeId, NodeId)> {
        if let Some(edges) = &self.sparse_edges {
            return edges.get(&edge_id).copied();
        }
        let idx = edge_id;
        if idx < self.edge_alive.len() && self.edge_alive[idx] {
            return Some((self.edge_sources[idx], self.edge_targets[idx]));
        }
        None
    }

    /*
    === ATTRIBUTE OPERATIONS ===
    Generic attribute storage - Graph decides which column (node/edge)
    */

    /// Set single attribute value (appends to specified column and returns index)
    pub fn set_attr(&mut self, attr: AttrName, value: AttrValue, is_node: bool) -> usize {
        // ALGORITHM: Append-only storage with index allocation
        // 1. Get or create the appropriate attribute column
        // 2. Append the value and return the new index

        let column = if is_node {
            self.node_attributes
                .entry(attr)
                .or_insert_with(AttributeColumn::new)
        } else {
            self.edge_attributes
                .entry(attr)
                .or_insert_with(AttributeColumn::new)
        };
        column.push(value)
    }

    /// Resolve a reconstructed value without growing columns on repeated switches.
    pub(crate) fn restore_attr(
        &mut self,
        name: AttrName,
        value: &AttrValue,
        is_node: bool,
    ) -> usize {
        let columns = if is_node {
            &mut self.node_attributes
        } else {
            &mut self.edge_attributes
        };
        columns
            .entry(name)
            .or_insert_with(AttributeColumn::new)
            .restore_value(value)
    }

    /// Set multiple attributes on single entity (appends to columns and returns indices)
    pub fn set_attrs(
        &mut self,
        attrs: HashMap<AttrName, AttrValue>,
        is_node: bool,
    ) -> HashMap<AttrName, usize> {
        // ALGORITHM: Bulk append operations
        // For each attribute, append the value and collect the index

        let mut indices = HashMap::with_capacity(attrs.len());
        for (attr_name, value) in attrs {
            let index = self.set_attr(attr_name.clone(), value, is_node);
            indices.insert(attr_name, index);
        }
        indices
    }

    /// Set same attribute for multiple entities (appends to column and returns indices)
    pub fn set_bulk_attr(
        &mut self,
        attr: AttrName,
        values: Vec<AttrValue>,
        is_node: bool,
    ) -> Vec<usize> {
        // ALGORITHM: Bulk columnar append operation
        // 1. Get or create the appropriate attribute column
        // 2. Append all values and collect their indices

        let column = if is_node {
            self.node_attributes
                .entry(attr)
                .or_insert_with(AttributeColumn::new)
        } else {
            self.edge_attributes
                .entry(attr)
                .or_insert_with(AttributeColumn::new)
        };
        values.into_iter().map(|value| column.push(value)).collect()
    }

    /// Set multiple attributes on multiple entities (VECTORIZED BULK OPERATION)
    /// Returns the new indices for change tracking
    pub fn set_bulk_attrs<T>(
        &mut self,
        attrs_values: HashMap<AttrName, Vec<(T, AttrValue)>>,
        is_node: bool,
    ) -> HashMap<AttrName, Vec<(T, usize)>>
    where
        T: Copy,
    {
        // OPTIMIZED: Use vectorized column operations
        let mut all_index_changes = HashMap::new();

        for (attr_name, entity_values) in attrs_values {
            // Get or create column
            let column = if is_node {
                self.node_attributes
                    .entry(attr_name.clone())
                    .or_insert_with(AttributeColumn::new)
            } else {
                self.edge_attributes
                    .entry(attr_name.clone())
                    .or_insert_with(AttributeColumn::new)
            };

            // Pre-allocate capacity to prevent reallocations
            column.reserve_capacity(entity_values.len());

            // Extract values for vectorized append
            let values: Vec<_> = entity_values.iter().map(|(_, val)| val.clone()).collect();
            let (start_idx, _end_idx) = column.extend_values(values);

            // Generate entity->index mappings
            let index_changes: Vec<_> = entity_values
                .iter()
                .enumerate()
                .map(|(i, &(entity_id, _))| (entity_id, start_idx + i))
                .collect();

            all_index_changes.insert(attr_name, index_changes);
        }

        all_index_changes
    }

    /// Set a single attribute column with explicit entity/value pairs
    pub fn set_attr_pairs<T>(
        &mut self,
        attr_name: AttrName,
        entity_values: Vec<(T, AttrValue)>,
        is_node: bool,
    ) -> Vec<(T, usize)>
    where
        T: Copy,
    {
        if entity_values.is_empty() {
            return Vec::new();
        }

        let column = if is_node {
            self.node_attributes
                .entry(attr_name.clone())
                .or_insert_with(AttributeColumn::new)
        } else {
            self.edge_attributes
                .entry(attr_name.clone())
                .or_insert_with(AttributeColumn::new)
        };

        column.reserve_capacity(entity_values.len());
        let values: Vec<_> = entity_values
            .iter()
            .map(|(_, value)| value.clone())
            .collect();
        let (start_idx, _) = column.extend_values(values);

        entity_values
            .into_iter()
            .enumerate()
            .map(|(offset, (entity, _))| (entity, start_idx + offset))
            .collect()
    }

    /*
    === BULK ATTRIBUTE OPERATIONS ===
    Single unified method for all attribute retrieval needs
    */

    /// Get attribute values for multiple entities using their column indices
    ///
    /// PERFORMANCE: Single bulk operation replacing N individual lookups
    /// INPUT: Vec of (entity_id, optional_column_index) pairs  
    /// OUTPUT: Vec of (entity_id, optional_attribute_value) pairs
    pub fn get_attribute_values(
        &self,
        attr_name: &AttrName,
        entity_indices: &[(NodeId, Option<usize>)],
        is_node: bool,
    ) -> Vec<(NodeId, Option<&AttrValue>)> {
        // Get the columnar attribute storage once
        let attr_column = if is_node {
            self.node_attributes.get(attr_name)
        } else {
            self.edge_attributes.get(attr_name)
        };

        // Early return if column doesn't exist
        let column = match attr_column {
            Some(col) => col,
            None => return entity_indices.iter().map(|(id, _)| (*id, None)).collect(),
        };

        // Get direct access to values slice to avoid repeated column access
        let values_slice = column.as_slice();

        // Bulk retrieval with optimized access pattern
        entity_indices
            .iter()
            .map(|(entity_id, index_opt)| {
                let attr_value = index_opt.and_then(|index| values_slice.get(index));
                (*entity_id, attr_value)
            })
            .collect()
    }

    /// OPTIMIZED: Direct access to attribute column to avoid repeated HashMap lookups
    pub fn get_node_attribute_column(&self, attr_name: &AttrName) -> Option<&AttributeColumn> {
        self.node_attributes.get(attr_name)
    }

    /// OPTIMIZED: Direct access to edge attribute column
    pub fn get_edge_attribute_column(&self, attr_name: &AttrName) -> Option<&AttributeColumn> {
        self.edge_attributes.get(attr_name)
    }

    /// OPTIMIZED: Iterator-based attribute retrieval to eliminate intermediate allocations
    ///
    /// PERFORMANCE: Zero-allocation bulk operation for filtering  
    /// INPUT: Iterator of (entity_id, optional_column_index) pairs
    /// OUTPUT: Vec of (entity_id, optional_attribute_value) pairs
    pub fn get_attribute_values_iter<I>(
        &self,
        attr_name: &AttrName,
        entity_indices_iter: I,
        is_node: bool,
    ) -> Vec<(NodeId, Option<&AttrValue>)>
    where
        I: Iterator<Item = (NodeId, Option<usize>)>,
    {
        // Get the columnar attribute storage
        let attr_column = if is_node {
            self.node_attributes.get(attr_name)
        } else {
            self.edge_attributes.get(attr_name)
        };

        // OPTIMIZED: Direct iterator processing without intermediate collections
        entity_indices_iter
            .map(|(entity_id, index_opt)| {
                let attr_value = index_opt.and_then(|index| attr_column?.values.get(index));
                (entity_id, attr_value)
            })
            .collect()
    }

    /*
    === STATISTICS & INTROSPECTION ===
    Information about the current state of the store
    */

    /// Get basic statistics about the graph
    pub fn statistics(&self) -> PoolStatistics {
        // Calculate memory usage approximations
        let _node_attrs_size = self
            .node_attributes
            .iter()
            .map(|(_, column)| column.len() * std::mem::size_of::<AttrValue>())
            .sum::<usize>();
        let _edge_attrs_size = self
            .edge_attributes
            .iter()
            .map(|(_, column)| column.len() * std::mem::size_of::<AttrValue>())
            .sum::<usize>();

        PoolStatistics {
            node_count: self.next_node_id,
            edge_count: self.next_edge_id,
            node_attribute_count: self.node_attributes.len(),
            edge_attribute_count: self.edge_attributes.len(),
        }
    }

    /// List all attribute names currently in use
    pub fn attribute_names(&self) -> (Vec<AttrName>, Vec<AttrName>) {
        let node_attrs = self.node_attributes.keys().cloned().collect();
        let edge_attrs = self.edge_attributes.keys().cloned().collect();
        (node_attrs, edge_attrs)
    }

    /// Get a specific attribute for a node
    pub fn get_node_attribute(
        &self,
        node_id: NodeId,
        attr_name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        if let Some(column) = self.node_attributes.get(attr_name) {
            Ok(column.get(node_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Set a specific attribute for a node
    pub fn set_node_attribute(
        &mut self,
        node_id: NodeId,
        attr_name: AttrName,
        value: AttrValue,
    ) -> GraphResult<()> {
        // Ensure the column exists and can address node_id
        let column = self
            .node_attributes
            .entry(attr_name.clone())
            .or_insert_with(|| AttributeColumn::with_capacity(self.next_node_id));
        column.ensure_len(self.next_node_id); // Grow to current size
        column.set(node_id, value); // Write per-node value
        Ok(())
    }

    /// Get all attributes for a node
    pub fn get_all_node_attributes(
        &self,
        node_id: NodeId,
    ) -> GraphResult<std::collections::HashMap<AttrName, AttrValue>> {
        let mut attributes = std::collections::HashMap::new();

        // Iterate through all attribute columns to find this node's attributes
        for (attr_name, column) in &self.node_attributes {
            if let Some(value) = column.get(node_id) {
                attributes.insert(attr_name.clone(), value.clone());
            }
        }

        Ok(attributes)
    }

    /// Get a specific attribute for an edge
    pub fn get_edge_attribute(
        &self,
        edge_id: EdgeId,
        attr_name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        if let Some(column) = self.edge_attributes.get(attr_name) {
            Ok(column.get(edge_id).cloned())
        } else {
            Ok(None)
        }
    }

    /// Set a specific attribute for an edge
    pub fn set_edge_attribute(
        &mut self,
        edge_id: EdgeId,
        attr_name: AttrName,
        value: AttrValue,
    ) -> GraphResult<()> {
        // Ensure the column exists and can address edge_id
        let column = self
            .edge_attributes
            .entry(attr_name.clone())
            .or_insert_with(|| AttributeColumn::with_capacity(self.next_edge_id));
        column.ensure_len(self.next_edge_id); // Grow to current size
        column.set(edge_id, value); // Write per-edge value
        Ok(())
    }

    /// Get all attributes for an edge
    pub fn get_all_edge_attributes(
        &self,
        edge_id: EdgeId,
    ) -> GraphResult<std::collections::HashMap<AttrName, AttrValue>> {
        let mut attributes = std::collections::HashMap::new();

        // Iterate through all attribute columns to find this edge's attributes
        for (attr_name, column) in &self.edge_attributes {
            if let Some(value) = column.get(edge_id) {
                attributes.insert(attr_name.clone(), value.clone());
            }
        }

        Ok(attributes)
    }

    /// Check if there's an edge between two nodes
    pub fn has_edge_between(&self, source: NodeId, target: NodeId) -> bool {
        if let Some(edges) = &self.sparse_edges {
            return edges
                .values()
                .any(|&(u, v)| (u == source && v == target) || (u == target && v == source));
        }
        // Check if any edge connects these two nodes
        for idx in 0..self.edge_alive.len() {
            if !self.edge_alive[idx] {
                continue;
            }
            let edge_source = self.edge_sources[idx];
            let edge_target = self.edge_targets[idx];
            if (edge_source == source && edge_target == target)
                || (edge_source == target && edge_target == source)
            {
                return true;
            }
        }
        false
    }

    /// Get all edges connected to a node
    pub fn get_incident_edges(&self, node_id: NodeId) -> GraphResult<Vec<EdgeId>> {
        if let Some(edges) = &self.sparse_edges {
            let mut ids: Vec<_> = edges
                .iter()
                .filter_map(|(&id, &(u, v))| (u == node_id || v == node_id).then_some(id))
                .collect();
            ids.sort_unstable();
            return Ok(ids);
        }
        let mut incident_edges = Vec::new();

        // Check all edges to find ones connected to this node
        // BUT only return edges that are still active (edge_alive tombstone check)
        for edge_id in 0..self.edge_sources.len() {
            if !self.edge_alive[edge_id] {
                continue;
            }
            let edge_source = self.edge_sources[edge_id];
            let edge_target = self.edge_targets[edge_id];
            if edge_source == node_id || edge_target == node_id {
                // CONSISTENCY FIX: Only return edges that are still active
                // Dense storage contains tombstoned edges as well, so rely on edge_alive
                incident_edges.push(edge_id);
            }
        }

        Ok(incident_edges)
    }

    /// Store immutable subgraph membership. Equal selections share a reference.
    /// Removing an entity from a working state does not rewrite retained selections.
    pub fn store_subgraph(
        &mut self,
        nodes: std::collections::HashSet<NodeId>,
        edges: std::collections::HashSet<EdgeId>,
        subgraph_type: String,
    ) -> GraphResult<crate::types::SubgraphId> {
        let prepared = self.prepare_stored_subgraph(nodes, edges, subgraph_type)?;
        let id = prepared.id;
        self.install_stored_subgraph(prepared);
        Ok(id)
    }

    /// Number of immutable stored group references, independent of active nodes.
    pub fn subgraph_count(&self) -> usize {
        self.stored_subgraphs.len()
    }

    pub(crate) fn prepare_stored_subgraph(
        &mut self,
        nodes: std::collections::HashSet<NodeId>,
        edges: std::collections::HashSet<EdgeId>,
        subgraph_type: String,
    ) -> GraphResult<PreparedStoredSubgraph> {
        // Generate a unique subgraph ID using a hash of the content
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        // Hash the actual node and edge IDs for uniqueness
        let mut sorted_nodes: Vec<NodeId> = nodes.iter().copied().collect();
        sorted_nodes.sort();
        sorted_nodes.hash(&mut hasher);

        let mut sorted_edges: Vec<EdgeId> = edges.iter().copied().collect();
        sorted_edges.sort();
        sorted_edges.hash(&mut hasher);

        subgraph_type.hash(&mut hasher);
        let subgraph_id = (hasher.finish() as usize) as crate::types::SubgraphId;

        let content = (nodes, edges, subgraph_type);
        match self.stored_subgraphs.get(&subgraph_id) {
            Some(old) if old != &content => {
                return Err(GraphError::InvalidInput(
                    "Stored subgraph reference collision; retained content was not replaced".into(),
                ))
            }
            Some(_) => {}
            None => self.stored_subgraphs.try_reserve(1).map_err(|error| {
                GraphError::InvalidInput(format!("Cannot reserve stored subgraph: {error}"))
            })?,
        }
        Ok(PreparedStoredSubgraph {
            id: subgraph_id,
            content,
            retained: None,
        })
    }

    pub(crate) fn install_stored_subgraph(&mut self, prepared: PreparedStoredSubgraph) {
        if let Some(group) = prepared.retained {
            std::sync::Arc::make_mut(&mut self.retained_groups)
                .entry(prepared.id)
                .or_insert(group);
        }
        self.stored_subgraphs
            .entry(prepared.id)
            .or_insert(prepared.content);
    }

    /// Share immutable group revisions without copying source graphs or values.
    pub(crate) fn retained_groups(
        &self,
    ) -> std::sync::Arc<HashMap<SubgraphId, crate::storage::retained_group::RetainedGroup>> {
        self.retained_groups.clone()
    }

    pub(crate) fn install_snapshot_groups(
        &mut self,
        groups: std::sync::Arc<HashMap<SubgraphId, crate::storage::retained_group::RetainedGroup>>,
    ) {
        for (&id, group) in groups.iter() {
            self.stored_subgraphs.entry(id).or_insert_with(|| {
                (
                    group.view.nodes().clone(),
                    group.view.edges().clone(),
                    group.kind.clone(),
                )
            });
        }
        self.retained_groups = groups;
    }

    pub(crate) fn preserve_allocation(&mut self, nodes: usize, edges: usize) {
        self.next_node_id = self.next_node_id.max(nodes);
        self.next_edge_id = self.next_edge_id.max(edges);
    }

    pub(crate) fn prepare_frozen_group(
        &mut self,
        view: crate::GraphView,
        kind: String,
    ) -> GraphResult<PreparedStoredSubgraph> {
        let group = crate::storage::retained_group::RetainedGroup::new(view, kind)?;
        self.prepare_group_revision(group)
    }
    pub(crate) fn prepare_group_revision(
        &mut self,
        group: crate::storage::retained_group::RetainedGroup,
    ) -> GraphResult<PreparedStoredSubgraph> {
        let id = group.id();
        let content = (
            group.view.nodes().clone(),
            group.view.edges().clone(),
            group.kind.clone(),
        );
        if let Some(old) = self.stored_subgraphs.get(&id) {
            if old != &content || !self.retained_groups.contains_key(&id) {
                return Err(GraphError::InvalidInput(
                    "Frozen group reference collision".into(),
                ));
            }
            if !self.retained_groups[&id].same_content(&group)? {
                return Err(GraphError::InvalidInput(
                    "Frozen group value collision".into(),
                ));
            }
            return Ok(PreparedStoredSubgraph {
                id,
                content,
                retained: None,
            });
        }
        self.stored_subgraphs
            .try_reserve(1)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        std::sync::Arc::make_mut(&mut self.retained_groups)
            .try_reserve(1)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        Ok(PreparedStoredSubgraph {
            id,
            content,
            retained: Some(group),
        })
    }

    pub(crate) fn prepare_group_documents(
        &mut self,
        documents: Vec<(SubgraphId, crate::storage::retained_group::GroupDocument)>,
    ) -> GraphResult<Vec<PreparedStoredSubgraph>> {
        let count = documents.len();
        self.stored_subgraphs
            .try_reserve(count)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        std::sync::Arc::make_mut(&mut self.retained_groups)
            .try_reserve(count)
            .map_err(|e| GraphError::InvalidInput(e.to_string()))?;
        let mut available = self.retained_groups.as_ref().clone();
        let mut prepared = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (id, document) in documents {
            if !seen.insert(id) {
                return Err(GraphError::InvalidInput(
                    "Duplicate retained group document".into(),
                ));
            }
            let group = document.restore(&available)?;
            if group.id() != id {
                return Err(GraphError::InvalidInput(
                    "Retained group content hash mismatch".into(),
                ));
            }
            let staged = self.prepare_group_revision(group.clone())?;
            available.insert(id, group);
            prepared.push(staged);
        }
        Ok(prepared)
    }
    /// Check a retained reference without cloning its member sets.
    pub(crate) fn validate_stored_subgraph(&self, id: SubgraphId) -> GraphResult<()> {
        if self.stored_subgraphs.contains_key(&id) {
            Ok(())
        } else {
            Err(crate::errors::GraphError::InvalidInput(format!(
                "Subgraph with ID {id} not found in storage"
            )))
        }
    }

    /// Get stored subgraph data
    pub fn get_subgraph(
        &self,
        subgraph_id: crate::types::SubgraphId,
    ) -> GraphResult<(
        std::collections::HashSet<NodeId>,
        std::collections::HashSet<EdgeId>,
        String,
    )> {
        match self.stored_subgraphs.get(&subgraph_id) {
            Some((nodes, edges, subgraph_type)) => {
                Ok((nodes.clone(), edges.clone(), subgraph_type.clone()))
            }
            None => Err(crate::errors::GraphError::InvalidInput(format!(
                "Subgraph with ID {} not found in storage",
                subgraph_id
            ))),
        }
    }

    /// Group handles have no independent mutable attribute store.
    pub fn get_subgraph_attribute(
        &self,
        subgraph_id: crate::types::SubgraphId,
        name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        let _ = (subgraph_id, name);
        Err(GraphError::NotImplemented {feature:"Pooled group attributes; use meta-node properties or an explicit frozen source revision".into(),tracking_issue:None})
    }

    /// Reject unsupported group attribute writes instead of silently losing data.
    pub fn set_subgraph_attribute(
        &mut self,
        subgraph_id: crate::types::SubgraphId,
        name: AttrName,
        value: AttrValue,
    ) -> GraphResult<()> {
        let _ = (subgraph_id, name, value);
        Err(GraphError::NotImplemented {feature:"Pooled group attributes; use meta-node properties or an explicit frozen source revision".into(),tracking_issue:None})
    }

    /// Compatibility no-op: working-edge deletion cannot alter retained selections.
    #[deprecated(
        note = "Deactivate edges through Graph; retained subgraph membership is immutable"
    )]
    pub fn update_stored_subgraphs_remove_edge(&mut self, _removed_edge: EdgeId) {}

    /// Compatibility no-op: working-node deletion cannot alter retained selections.
    #[deprecated(
        note = "Deactivate nodes through Graph; retained subgraph membership is immutable"
    )]
    pub fn update_stored_subgraphs_remove_node(&mut self, _removed_node: NodeId) {}
}

/// Statistics about the current state of the graph store
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_attribute_count: usize,
    pub edge_attribute_count: usize,
    // TODO: Add memory usage, load factors, etc.
}

impl Default for GraphPool {
    fn default() -> Self {
        Self::new()
    }
}

/*
=== IMPLEMENTATION NOTES ===

MEMORY LAYOUT:
- Columnar storage means attributes of same type are stored together
- Better cache locality for bulk operations (ML workloads)
- Slightly more complex than row-based storage but worth it

ID MANAGEMENT:
- Simple incrementing counters for now
- Could optimize later with free lists, compaction
- NodeId/EdgeId reuse after deletion is possible but complex

SPARSE vs DENSE:
- Current design is dense (all entities have slots in all attribute vectors)
- Wastes memory but gives O(1) access
- Could optimize with sparse storage (HashMap<EntityId, AttrValue>) later

PERFORMANCE CHARACTERISTICS:
- Add node/edge: O(1) amortized (may need to grow vectors)
- Remove node/edge: O(1) for edges, O(degree) for nodes
- Attribute access: O(1)
- Neighbor queries: O(total edges) - could optimize with adjacency lists
- Bulk operations: O(n) where n is number of entities processed

ERROR HANDLING:
- Use Result<T, GraphError> for operations that can fail
- Validate entity existence before operations
- Fail fast and clear error messages
*/

#[cfg(test)]
mod restoration_tests {
    use super::*;
    #[test]
    fn restoration_lookup_tracks_appends_bulk_extends_and_explicit_overwrites() {
        let mut column = AttributeColumn::new();
        let first = column.push(AttrValue::Int(1));
        assert_eq!(column.restore_value(&AttrValue::Int(1)), first);
        let second = column.push(AttrValue::Int(2));
        column.extend_values(vec![AttrValue::Int(3), AttrValue::Int(4)]);
        assert_eq!(column.restore_value(&AttrValue::Int(2)), second);
        assert_eq!(column.restore_value(&AttrValue::Int(4)), 3);
        column.set(first, AttrValue::Bool(true));
        assert_eq!(column.restore_value(&AttrValue::Bool(true)), first);
        assert_eq!(column.restore_value(&AttrValue::Int(1)), 4);
        column.ensure_len(7);
        assert_eq!(column.restore_value(&AttrValue::Null), 5);
        assert_eq!(column.len(), 7);
    }
    #[test]
    fn restoration_preserves_float_representation_without_changing_property_equality() {
        for (left, right) in [
            (AttrValue::Float(0.0), AttrValue::Float(-0.0)),
            (AttrValue::Float64(0.0), AttrValue::Float64(-0.0)),
            (
                AttrValue::FloatVec(vec![0.0, f32::from_bits(0x7fc00001)]),
                AttrValue::FloatVec(vec![-0.0, f32::from_bits(0x7fc0007b)]),
            ),
            (
                AttrValue::Float64Vec(vec![0.0, f64::from_bits(0x7ff8000000000001)]),
                AttrValue::Float64Vec(vec![-0.0, f64::from_bits(0x7ff800000000007b)]),
            ),
        ] {
            assert_eq!(left, right); // Public property identity remains canonical.
            let mut column = AttributeColumn::new();
            let a = column.push(left.clone());
            let b = column.push(right.clone());
            for _ in 0..3 {
                assert_eq!(column.restore_value(&left), a);
                assert_eq!(column.restore_value(&right), b);
            }
            assert_eq!(column.len(), 2);
        }
    }
    #[test]
    fn restoration_hash_collisions_compare_values_before_reusing_indices() {
        let mut column = AttributeColumn::new();
        column.push(AttrValue::Bool(true));
        column.restore_indices = Some(RestorationIndex {
            first: HashMap::from([(AttributeColumn::value_hash(&AttrValue::Int(42)), 0)]),
            collisions: HashMap::new(),
        });
        assert_eq!(column.restore_value(&AttrValue::Int(42)), 1);
        assert_eq!(column.restore_value(&AttrValue::Int(42)), 1);
        assert_eq!(column.len(), 2);
    }
}
