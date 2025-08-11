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

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, AttrName, AttrValue};
use crate::errors::{GraphError, GraphResult};

/// Columnar storage for attribute values
/// 
/// DESIGN: Store attribute values in a single vector for cache efficiency.
/// This is much faster for bulk operations and analytics workloads compared
/// to storing attributes per-entity.
/// 
/// USAGE: Each entity gets an index into this column. When the entity's
/// attribute changes, we append the new value and update the index.
#[derive(Debug, Clone)]
pub struct AttributeColumn {
    /// All values ever stored for this attribute (append-only)
    values: Vec<AttrValue>,
}

impl AttributeColumn {
    /// Create a new empty attribute column
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
        }
    }
    
    /// Append a new value and return its index
    /// 
    /// PERFORMANCE: O(1) amortized append
    pub fn push(&mut self, value: AttrValue) -> usize {
        let index = self.values.len();
        self.values.push(value);
        index
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
    /// Create new empty graph store
    pub fn new() -> Self {
        Self {
            node_attributes: HashMap::new(),
            edge_attributes: HashMap::new(),
            topology: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
        }
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
    pub fn get_attr_by_index(&self, attr: &AttrName, index: usize, is_node: bool) -> Option<&AttrValue> {
        // ALGORITHM: Direct index lookup in columnar storage
        // 1. Get the appropriate attribute column
        // 2. Return value at the specified index
        
        let column_map = if is_node {
            &self.node_attributes
        } else {
            &self.edge_attributes
        };
        
        column_map
            .get(attr)
            .and_then(|column| column.get(index))
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
    
    /// Create a new edge between two nodes
    /// DESIGN: Pool creates and stores the edge, Space tracks it as active
    pub fn add_edge(&mut self, source: NodeId, target: NodeId) -> EdgeId {
        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;
        self.topology.insert(edge_id, (source, target));
        edge_id
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
            self.node_attributes.entry(attr).or_insert_with(AttributeColumn::new)
        } else {
            self.edge_attributes.entry(attr).or_insert_with(AttributeColumn::new)
        };
        column.push(value)
    }
    
    /// Set multiple attributes on single entity (appends to columns and returns indices)
    pub fn set_attrs(&mut self, attrs: HashMap<AttrName, AttrValue>, is_node: bool) -> HashMap<AttrName, usize> {
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
    pub fn set_bulk_attr(&mut self, attr: AttrName, values: Vec<AttrValue>, is_node: bool) -> Vec<usize> {
        // ALGORITHM: Bulk columnar append operation
        // 1. Get or create the appropriate attribute column
        // 2. Append all values and collect their indices
        
        let column = if is_node {
            self.node_attributes.entry(attr).or_insert_with(AttributeColumn::new)
        } else {
            self.edge_attributes.entry(attr).or_insert_with(AttributeColumn::new)
        };
        values.into_iter().map(|value| column.push(value)).collect()
    }
    
    /// Set multiple attributes on multiple entities (bulk operation)
    /// Returns the new indices for change tracking
    pub fn set_bulk_attrs<T>(&mut self, attrs_values: HashMap<AttrName, Vec<(T, AttrValue)>>, is_node: bool) -> HashMap<AttrName, Vec<(T, usize)>> 
    where T: Copy {
        // ALGORITHM: Append values and return indices (entity-agnostic)
        let mut all_index_changes = HashMap::new();
        for (attr_name, entity_values) in attrs_values {
            let mut index_changes = Vec::new();
            for (entity_id, value) in entity_values {
                // Use existing set_attr method
                let new_index = self.set_attr(attr_name.clone(), value, is_node);
                index_changes.push((entity_id, new_index));
            }
            all_index_changes.insert(attr_name, index_changes);
        }
        all_index_changes
    }
    
    /*
    === INTERNAL BULK OPERATIONS ===
    Efficient operations for internal use by Graph coordinator.
    Pool provides full column access - Graph handles filtering and security.
    */
    
    /// Get full attribute column (internal use only)
    /// 
    /// INTERNAL: This exposes the full column - Graph coordinator handles filtering
    /// PERFORMANCE: Direct access to columnar data for maximum efficiency
    /// RETURNS: Reference to the entire attribute vector
    pub fn get_attr_column(&self, attr: &AttrName, is_node: bool) -> Option<&[AttrValue]> {
        let column_map = if is_node {
            &self.node_attributes
        } else {
            &self.edge_attributes
        };
        column_map.get(attr).map(|column| column.as_slice())
    }
    
    /// Get attribute values for specific indices (internal bulk operation)
    /// 
    /// INTERNAL: Used by Graph when it has already determined which indices to access
    /// PERFORMANCE: More efficient than individual lookups for known valid indices
    pub fn get_attrs_at_indices(&self, attr: &AttrName, indices: &[usize], is_node: bool) -> GraphResult<Vec<Option<AttrValue>>> {
        let column_map = if is_node {
            &self.node_attributes
        } else {
            &self.edge_attributes
        };
        
        if let Some(attr_column) = column_map.get(attr) {
            let mut results = Vec::with_capacity(indices.len());
            for &index in indices {
                if index < attr_column.values.len() {
                    results.push(Some(attr_column.values[index].clone()));
                } else {
                    results.push(None);
                }
            }
            Ok(results)
        } else {
            Ok(vec![None; indices.len()])
        }
    }
    
    
    
    
    /*
    === STATISTICS & INTROSPECTION ===
    Information about the current state of the store
    */
    
    /// Get basic statistics about the graph
    pub fn statistics(&self) -> PoolStatistics {
        // Calculate memory usage approximations
        let node_attrs_size = self.node_attributes.iter()
            .map(|(_, column)| column.len() * std::mem::size_of::<AttrValue>())
            .sum::<usize>();
        let edge_attrs_size = self.edge_attributes.iter()
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
