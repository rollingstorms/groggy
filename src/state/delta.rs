//! Delta Storage System - Efficient representation of changes between graph states.
//!
//! ARCHITECTURE ROLE:
//! This module provides the core data structures for representing changes
//! (deltas) between different graph states. It's the foundation of the
//! version control system, enabling efficient storage and application of changes.
//!
//! DESIGN PHILOSOPHY:
//! - Columnar storage for bulk operations and cache efficiency
//! - Content-addressed deltas for automatic deduplication
//! - Immutable delta objects for safe sharing
//! - Sparse representation (only store what changed)

/*
=== DELTA SYSTEM OVERVIEW ===

The delta system is responsible for:
1. EFFICIENT REPRESENTATION: Store only what changed, not full snapshots
2. COLUMNAR LAYOUT: Group changes by attribute for bulk operations
3. CONTENT ADDRESSING: Hash-based deduplication of identical changes
4. FAST APPLICATION: Apply changes efficiently to reconstruct states
5. MERGING: Combine deltas from different sources (branch merging)

KEY INSIGHTS:
- Sparse storage: Most entities don't change between states
- Columnar layout: Better cache locality for bulk attribute operations
- Sorted indices: Enable binary search and efficient merging
- Immutable design: Safe for concurrent access and sharing
*/

use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Index-based columnar delta for efficient temporal versioning
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnIndexDelta {
    /// Entity indices that changed (sorted)
    pub entity_indices: Vec<usize>,
    /// Previous column indices (None = attribute was new)
    pub old_column_indices: Vec<Option<usize>>,
    /// New column indices
    pub new_column_indices: Vec<Option<usize>>,
}

impl ColumnIndexDelta {
    /// Create a new empty index delta
    pub fn new() -> Self {
        Self {
            entity_indices: Vec::new(),
            old_column_indices: Vec::new(),
            new_column_indices: Vec::new(),
        }
    }

    /// Add an index change at the specified entity index
    pub fn add_index_change(
        &mut self,
        entity_index: usize,
        old_idx: Option<usize>,
        new_idx: Option<usize>,
    ) {
        // Find insertion point to maintain sorted order by entity_index
        let pos = self
            .entity_indices
            .binary_search(&entity_index)
            .unwrap_or_else(|e| e);

        if pos < self.entity_indices.len() && self.entity_indices[pos] == entity_index {
            // Update existing change - keep original old_idx, update to latest new_idx
            self.new_column_indices[pos] = new_idx;
        } else {
            // Insert new change
            self.entity_indices.insert(pos, entity_index);
            self.old_column_indices.insert(pos, old_idx);
            self.new_column_indices.insert(pos, new_idx);
        }
    }

    /// Get the change at a specific entity index
    pub fn get_change(&self, entity_index: usize) -> Option<(Option<usize>, Option<usize>)> {
        self.entity_indices
            .binary_search(&entity_index)
            .ok()
            .map(|pos| (self.old_column_indices[pos], self.new_column_indices[pos]))
    }

    /// Check if this delta has changes at the given entity index
    pub fn has_change(&self, entity_index: usize) -> bool {
        self.entity_indices.binary_search(&entity_index).is_ok()
    }

    /// Get the number of changes in this delta
    pub fn len(&self) -> usize {
        self.entity_indices.len()
    }

    /// Check if this delta is empty
    pub fn is_empty(&self) -> bool {
        self.entity_indices.is_empty()
    }
}

impl Hash for ColumnIndexDelta {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entity_indices.hash(state);
        self.old_column_indices.hash(state);
        self.new_column_indices.hash(state);
    }
}

/// Columnar delta storage for efficient bulk operations (legacy - stores values)
#[derive(Debug, Clone)]
pub struct ColumnDelta {
    /// Sorted indices where changes occurred
    pub indices: Vec<usize>,
    /// Corresponding values (parallel array to indices)
    pub values: Vec<AttrValue>,
}

impl ColumnDelta {
    /// Create a new empty column delta
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create a column delta with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            indices: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Add a change at the specified index
    pub fn add_change(&mut self, index: usize, value: AttrValue) {
        // Find insertion point to maintain sorted order
        let pos = self.indices.binary_search(&index).unwrap_or_else(|e| e);

        if pos < self.indices.len() && self.indices[pos] == index {
            // Update existing value
            self.values[pos] = value;
        } else {
            // Insert new value
            self.indices.insert(pos, index);
            self.values.insert(pos, value);
        }
    }

    /// Get the value at a specific index, if it exists
    pub fn get(&self, index: usize) -> Option<&AttrValue> {
        self.indices
            .binary_search(&index)
            .ok()
            .map(|pos| &self.values[pos])
    }

    /// Check if this delta has changes at the given index
    pub fn has_change(&self, index: usize) -> bool {
        self.indices.binary_search(&index).is_ok()
    }

    /// Get the number of changes in this delta
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if this delta is empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Merge another column delta into this one
    pub fn merge(&mut self, other: &ColumnDelta) {
        for (i, &index) in other.indices.iter().enumerate() {
            self.add_change(index, other.values[i].clone());
        }
    }
}

impl Hash for ColumnDelta {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.indices.hash(state);
        for value in &self.values {
            value.hash(state);
        }
    }
}

/// Owned physical column-reference delta. It is not a portable logical history record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDeltaObject {
    /// Node property physical references, deterministically ordered by name.
    pub node_attrs: BTreeMap<AttrName, ColumnIndexDelta>,
    /// Edge property physical references, deterministically ordered by name.
    pub edge_attrs: BTreeMap<AttrName, ColumnIndexDelta>,
    /// Node additions retained by stable identity.
    pub nodes_added: Vec<NodeId>,
    /// Node removals retained by stable identity.
    pub nodes_removed: Vec<NodeId>,
    /// Edge additions retained with ordered endpoints.
    pub edges_added: Vec<(EdgeId, NodeId, NodeId)>,
    /// Edge removals retained by stable identity.
    pub edges_removed: Vec<EdgeId>,
}
impl IndexDeltaObject {
    fn change_count(&self) -> usize {
        self.node_attrs
            .values()
            .map(ColumnIndexDelta::len)
            .sum::<usize>()
            + self
                .edge_attrs
                .values()
                .map(ColumnIndexDelta::len)
                .sum::<usize>()
            + self.nodes_added.len()
            + self.nodes_removed.len()
            + self.edges_added.len()
            + self.edges_removed.len()
    }
}

/// Immutable delta object representing changes between states
#[derive(Debug, Clone)]
pub struct DeltaObject {
    /// Explicit physical index payload, when constructed from pending index changes.
    /// Never interpret its offsets as logical property values.
    pub index_changes: Option<Arc<IndexDeltaObject>>,
    /// Node attribute changes stored columnarly
    pub node_attrs: Arc<HashMap<AttrName, ColumnDelta>>,
    /// Edge attribute changes stored columnarly  
    pub edge_attrs: Arc<HashMap<AttrName, ColumnDelta>>,
    /// Nodes that became active/inactive
    pub node_active_changes: Arc<ColumnDelta>,
    /// Edges that became active/inactive
    pub edge_active_changes: Arc<ColumnDelta>,
    /// Typed endpoint changes; retained separately from liveness.
    pub edge_endpoint_changes: Arc<Vec<crate::EdgeEndpointChange>>,
    /// Content hash for deduplication
    pub content_hash: [u8; 32],
}

impl DeltaObject {
    /// Create a new delta object
    pub fn new(
        node_attrs: HashMap<AttrName, ColumnDelta>,
        edge_attrs: HashMap<AttrName, ColumnDelta>,
        node_active_changes: ColumnDelta,
        edge_active_changes: ColumnDelta,
    ) -> Self {
        let delta = Self {
            index_changes: None,
            node_attrs: Arc::new(node_attrs),
            edge_attrs: Arc::new(edge_attrs),
            node_active_changes: Arc::new(node_active_changes),
            edge_active_changes: Arc::new(edge_active_changes),
            edge_endpoint_changes: Arc::new(Vec::new()),
            content_hash: [0; 32], // Will be computed below
        };

        // Compute content hash by converting HashMaps to sorted vectors
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash node attributes in sorted order
        let mut node_attrs_sorted: Vec<_> = delta.node_attrs.iter().collect();
        node_attrs_sorted.sort_by_key(|(k, _)| *k);
        for (key, value) in node_attrs_sorted {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }

        // Hash edge attributes in sorted order
        let mut edge_attrs_sorted: Vec<_> = delta.edge_attrs.iter().collect();
        edge_attrs_sorted.sort_by_key(|(k, _)| *k);
        for (key, value) in edge_attrs_sorted {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }

        // Hash active changes
        delta.node_active_changes.hash(&mut hasher);
        delta.edge_active_changes.hash(&mut hasher);

        let hash_value = hasher.finish();
        let mut hash = [0u8; 32];
        hash[..8].copy_from_slice(&hash_value.to_le_bytes());

        Self {
            content_hash: hash,
            ..delta
        }
    }

    /// Attach canonical endpoint changes without altering old empty-change hashes.
    pub(crate) fn with_endpoint_changes(
        mut self,
        mut changes: Vec<crate::EdgeEndpointChange>,
    ) -> Self {
        changes.sort_unstable_by_key(|c| c.edge_id);
        if !changes.is_empty() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.content_hash.hash(&mut hasher);
            "groggy:endpoint:v1".hash(&mut hasher);
            changes.hash(&mut hasher);
            self.content_hash[..8].copy_from_slice(&hasher.finish().to_le_bytes());
        }
        self.edge_endpoint_changes = Arc::new(changes);
        self
    }

    /// Create a validated, complete physical index delta. Pool-independent fields
    /// remain in `index_changes`; resolving values requires a matching pool.
    pub fn new_with_indices(
        node_attrs: HashMap<AttrName, ColumnIndexDelta>,
        edge_attrs: HashMap<AttrName, ColumnIndexDelta>,
        mut nodes_added: Vec<NodeId>,
        mut nodes_removed: Vec<NodeId>,
        mut edges_added: Vec<(EdgeId, NodeId, NodeId)>,
        mut edges_removed: Vec<EdgeId>,
    ) -> crate::GraphResult<Self> {
        for columns in [&node_attrs, &edge_attrs] {
            for (name, column) in columns {
                if column.entity_indices.len() != column.old_column_indices.len()
                    || column.entity_indices.len() != column.new_column_indices.len()
                    || column.entity_indices.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(crate::GraphError::InvalidInput(format!(
                        "Malformed indexed delta column {name}"
                    )));
                }
            }
        }
        nodes_added.sort_unstable();
        nodes_removed.sort_unstable();
        edges_added.sort_unstable();
        edges_removed.sort_unstable();
        let indices = IndexDeltaObject {
            node_attrs: node_attrs.into_iter().collect(),
            edge_attrs: edge_attrs.into_iter().collect(),
            nodes_added,
            nodes_removed,
            edges_added,
            edges_removed,
        };
        let mut result = Self::empty();
        if indices.change_count() == 0 {
            return Ok(result);
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "groggy:index-delta:v1".hash(&mut hasher);
        for columns in [&indices.node_attrs, &indices.edge_attrs] {
            columns.len().hash(&mut hasher);
            for (name, column) in columns {
                name.hash(&mut hasher);
                column.hash(&mut hasher);
            }
        }
        indices.nodes_added.hash(&mut hasher);
        indices.nodes_removed.hash(&mut hasher);
        indices.edges_added.hash(&mut hasher);
        indices.edges_removed.hash(&mut hasher);
        result.content_hash[..8].copy_from_slice(&hasher.finish().to_le_bytes());
        result.index_changes = Some(Arc::new(indices));
        Ok(result)
    }

    /// Create an empty delta object
    pub fn empty() -> Self {
        Self::new(
            HashMap::new(),
            HashMap::new(),
            ColumnDelta::new(),
            ColumnDelta::new(),
        )
    }

    /// Check if this delta is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.index_changes
            .as_ref()
            .is_none_or(|changes| changes.change_count() == 0)
            && self.node_attrs.is_empty()
            && self.edge_attrs.is_empty()
            && self.node_active_changes.is_empty()
            && self.edge_active_changes.is_empty()
            && self.edge_endpoint_changes.is_empty()
    }

    /// Get the total number of changes in this delta
    pub fn change_count(&self) -> usize {
        let node_attr_changes: usize = self.node_attrs.values().map(|d| d.len()).sum();
        let edge_attr_changes: usize = self.edge_attrs.values().map(|d| d.len()).sum();
        self.index_changes
            .as_ref()
            .map_or(0, |changes| changes.change_count())
            + node_attr_changes
            + edge_attr_changes
            + self.node_active_changes.len()
            + self.edge_active_changes.len()
            + self.edge_endpoint_changes.len()
    }
}

impl Hash for DeltaObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.content_hash.hash(state);
    }
}

impl PartialEq for DeltaObject {
    fn eq(&self, other: &Self) -> bool {
        fn column(a: &ColumnDelta, b: &ColumnDelta) -> bool {
            a.indices == b.indices
                && a.values.len() == b.values.len()
                && a.values.iter().zip(&b.values).all(|(a, b)| {
                    match super::change::same_value(a, b) {
                        Ok(same) => same,
                        // Raw legacy objects may contain unencodable values. Keep
                        // Eq reflexive by comparing their exact derived representation;
                        // checked history resolution still rejects these payloads.
                        Err(_) => match (serde_json::to_value(a), serde_json::to_value(b)) {
                            (Ok(a), Ok(b)) => a == b,
                            _ => std::ptr::eq(a, b),
                        },
                    }
                })
        }
        fn columns(a: &HashMap<AttrName, ColumnDelta>, b: &HashMap<AttrName, ColumnDelta>) -> bool {
            a.len() == b.len()
                && a.iter()
                    .all(|(name, a)| b.get(name).is_some_and(|b| column(a, b)))
        }
        self.content_hash == other.content_hash
            && self.index_changes == other.index_changes
            && self.edge_endpoint_changes == other.edge_endpoint_changes
            && columns(&self.node_attrs, &other.node_attrs)
            && columns(&self.edge_attrs, &other.edge_attrs)
            && column(&self.node_active_changes, &other.node_active_changes)
            && column(&self.edge_active_changes, &other.edge_active_changes)
    }
}

impl Eq for DeltaObject {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_delta_basic_operations() {
        let mut delta = ColumnDelta::new();

        delta.add_change(5, AttrValue::Int(42));
        delta.add_change(2, AttrValue::Text("hello".to_string()));
        delta.add_change(8, AttrValue::Bool(true));

        assert_eq!(delta.len(), 3);
        assert_eq!(delta.get(5), Some(&AttrValue::Int(42)));
        assert_eq!(delta.get(2), Some(&AttrValue::Text("hello".to_string())));
        assert!(delta.has_change(8));
        assert!(!delta.has_change(10));
    }

    #[test]
    fn test_column_delta_maintains_order() {
        let mut delta = ColumnDelta::new();

        delta.add_change(5, AttrValue::Int(1));
        delta.add_change(2, AttrValue::Int(2));
        delta.add_change(8, AttrValue::Int(3));
        delta.add_change(1, AttrValue::Int(4));

        assert_eq!(delta.indices, vec![1, 2, 5, 8]);
    }
}
