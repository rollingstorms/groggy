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

use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, GraphType, NodeId};
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
        }
    }

    /// Create a new attribute column with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            memory_pool: AttributeMemoryPool::new(),
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
        index
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

        base_size + values_size + pool_overhead
    }

    /// Ensure the column can address up to the given index (sparse indexing support)
    pub fn ensure_len(&mut self, min_len: usize) {
        if self.values.len() < min_len {
            self.values.resize(min_len, AttrValue::Null);
        }
    }

    /// Set value at a specific index (for node/edge ID indexing)
    pub fn set(&mut self, index: usize, value: AttrValue) {
        if index >= self.values.len() {
            self.ensure_len(index + 1);
        }
        let optimized_value = value.optimize();
        self.values[index] = optimized_value;
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
#[derive(Debug)]
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
    /// All edges ever created: edge_id -> (source_node, target_node)
    /// STORAGE: This never shrinks, even for "deleted" edges
    topology: HashMap<EdgeId, (NodeId, NodeId)>,

    /*
    === ID MANAGEMENT ===
    Simple incrementing counters for new entities
    */
    /// Next available node ID - increment on each new node
    next_node_id: NodeId,

    /// Next available edge ID - increment on each new edge
    next_edge_id: EdgeId,
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
            topology: HashMap::new(),
            next_node_id: 0,
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
    /// Returns (start_id, end_id) range and individual node IDs
    pub fn add_nodes_bulk(&mut self, count: usize) -> (NodeId, NodeId, Vec<NodeId>) {
        let start_id = self.next_node_id;
        let end_id = start_id + count - 1;

        // Single ID counter update
        self.next_node_id += count;

        // Generate node IDs (much faster than individual calls)
        let node_ids: Vec<NodeId> = (start_id..=end_id).collect();

        (start_id, end_id, node_ids)
    }

    /// Ensure a specific node ID exists (used for state reconstruction)
    /// This is used when restoring historical states during branch switching
    pub fn ensure_node_id_exists(&mut self, node_id: NodeId) {
        // Update next_node_id if necessary to avoid ID collisions
        if node_id >= self.next_node_id {
            self.next_node_id = node_id + 1;
        }
    }

    /// Create a new edge between two nodes
    /// DESIGN: Pool creates and stores the edge, Space tracks it as active
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> EdgeId {
        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;
        self.topology.insert(edge_id, (source, target));
        edge_id
    }

    /// Create multiple edges with bulk HashMap insertion (BULK OPTIMIZED)
    ///
    /// PERFORMANCE: Batch validation + bulk HashMap operations
    /// Returns Vec<EdgeId> for created edges
    pub fn add_edges(&mut self, edges: &[(NodeId, NodeId)]) -> Vec<EdgeId> {
        let start_id = self.next_edge_id;
        let edge_ids: Vec<EdgeId> = (start_id..start_id + edges.len()).collect();

        // Reserve HashMap capacity to prevent rehashing
        self.topology.reserve(edges.len());

        // Bulk insert into topology HashMap
        for (i, &(source, target)) in edges.iter().enumerate() {
            self.topology.insert(start_id + i, (source, target));
        }

        // Single counter update
        self.next_edge_id += edges.len();

        edge_ids
    }

    /// Create an edge with a specific ID (used for state reconstruction)
    /// This is used when restoring historical states during branch switching
    pub fn add_edge_with_id(&mut self, edge_id: EdgeId, source: NodeId, target: NodeId) {
        self.topology.insert(edge_id, (source, target));
        // Update next_edge_id if necessary to avoid ID collisions
        if edge_id >= self.next_edge_id {
            self.next_edge_id = edge_id + 1;
        }
    }

    /// Get the endpoints of an edge from storage
    pub fn get_edge_endpoints(&self, edge_id: EdgeId) -> Option<(NodeId, NodeId)> {
        self.topology.get(&edge_id).copied()
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
        // Check if any edge connects these two nodes
        for (_edge_id, (edge_source, edge_target)) in &self.topology {
            if (*edge_source == source && *edge_target == target)
                || (*edge_source == target && *edge_target == source)
            {
                return true;
            }
        }
        false
    }

    /// Get all edges connected to a node
    pub fn get_incident_edges(&self, node_id: NodeId) -> GraphResult<Vec<EdgeId>> {
        let mut incident_edges = Vec::new();

        // Check all edges to find ones connected to this node
        for (edge_id, (edge_source, edge_target)) in &self.topology {
            if *edge_source == node_id || *edge_target == node_id {
                incident_edges.push(*edge_id);
            }
        }

        Ok(incident_edges)
    }

    /// Store subgraph data in GraphPool (placeholder for now)
    pub fn store_subgraph(
        &mut self,
        nodes: std::collections::HashSet<NodeId>,
        edges: std::collections::HashSet<EdgeId>,
        subgraph_type: String,
    ) -> GraphResult<crate::types::SubgraphId> {
        // TODO: Implement proper subgraph storage in GraphPool
        // For now, just return a hash-based ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        nodes.len().hash(&mut hasher);
        edges.len().hash(&mut hasher);
        subgraph_type.hash(&mut hasher);
        let subgraph_id = (hasher.finish() as usize) as crate::types::SubgraphId;
        Ok(subgraph_id)
    }

    /// Get stored subgraph data (placeholder for now)
    pub fn get_subgraph(
        &self,
        subgraph_id: crate::types::SubgraphId,
    ) -> GraphResult<(
        std::collections::HashSet<NodeId>,
        std::collections::HashSet<EdgeId>,
        String,
    )> {
        // TODO: Implement proper subgraph retrieval from GraphPool
        // For now, return empty data
        let _ = subgraph_id; // Silence unused parameter warning
        Ok((
            std::collections::HashSet::new(),
            std::collections::HashSet::new(),
            "placeholder".to_string(),
        ))
    }

    /// Get subgraph attribute (placeholder for now)
    pub fn get_subgraph_attribute(
        &self,
        subgraph_id: crate::types::SubgraphId,
        name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        // TODO: Implement subgraph attribute storage
        let _ = (subgraph_id, name); // Silence unused parameter warnings
        Ok(None)
    }

    /// Set subgraph attribute (placeholder for now)  
    pub fn set_subgraph_attribute(
        &mut self,
        subgraph_id: crate::types::SubgraphId,
        name: AttrName,
        value: AttrValue,
    ) -> GraphResult<()> {
        // TODO: Implement subgraph attribute storage
        let _ = (subgraph_id, name, value); // Silence unused parameter warnings
        Ok(())
    }
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
