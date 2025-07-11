#![allow(non_local_definitions)]
use bitvec::prelude::*;
use dashmap::DashMap;
use pyo3::prelude::*;
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};

/// Unique identifier for attributes across the entire graph
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct AttrUID(pub u64);

/// Simplified columnar storage for graph attributes with on-demand bitmap indexing
#[pyclass]
pub struct ColumnarStore {
    /// Maps attribute names to their unique IDs
    pub attr_name_to_uid: DashMap<String, AttrUID>,
    /// Maps attribute UIDs back to names
    pub attr_uid_to_name: DashMap<AttrUID, String>,
    /// Next available attribute UID
    pub next_attr_uid: std::sync::atomic::AtomicU64,

    /// Sparse columnar storage: attr_uid -> HashMap<entity_index, attr_value>
    /// This is the single source of truth for attribute values
    pub node_attributes: DashMap<AttrUID, HashMap<usize, JsonValue>>,
    pub edge_attributes: DashMap<AttrUID, HashMap<usize, JsonValue>>,

    /// Bitmap indices for fast filtering (built on-demand)
    /// Maps (attr_uid, attr_value) -> BitVec indicating which nodes/edges have this value
    pub node_value_bitmaps: DashMap<(AttrUID, JsonValue), BitVec>,
    pub edge_value_bitmaps: DashMap<(AttrUID, JsonValue), BitVec>,

    /// Track if bitmaps are up-to-date
    pub bitmaps_dirty: std::sync::atomic::AtomicBool,
    
    /// Track maximum entity indices for bitmap sizing
    pub max_node_index: std::sync::atomic::AtomicUsize,
    pub max_edge_index: std::sync::atomic::AtomicUsize,
}

impl Default for ColumnarStore {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl ColumnarStore {
    #[new]
    pub fn new() -> Self {
        Self {
            attr_name_to_uid: DashMap::new(),
            attr_uid_to_name: DashMap::new(),
            next_attr_uid: std::sync::atomic::AtomicU64::new(0),
            node_attributes: DashMap::new(),
            edge_attributes: DashMap::new(),
            node_value_bitmaps: DashMap::new(),
            edge_value_bitmaps: DashMap::new(),
            bitmaps_dirty: std::sync::atomic::AtomicBool::new(false),
            max_node_index: std::sync::atomic::AtomicUsize::new(0),
            max_edge_index: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get storage statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert(
            "attributes_registered".to_string(),
            self.attr_name_to_uid.len(),
        );
        stats.insert("node_attributes".to_string(), self.node_attributes.len());
        stats.insert("edge_attributes".to_string(), self.edge_attributes.len());
        stats.insert("node_bitmaps".to_string(), self.node_value_bitmaps.len());
        stats.insert("edge_bitmaps".to_string(), self.edge_value_bitmaps.len());
        stats.insert(
            "max_node_index".to_string(),
            self.max_node_index.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "max_edge_index".to_string(),
            self.max_edge_index.load(std::sync::atomic::Ordering::Relaxed),
        );
        stats.insert(
            "bitmaps_dirty".to_string(),
            if self.bitmaps_dirty.load(std::sync::atomic::Ordering::Relaxed) { 1 } else { 0 },
        );
        stats
    }
}

impl ColumnarStore {
    /// Get or create an attribute UID for a given attribute name
    pub fn get_or_create_attr_uid(&self, attr_name: &str) -> AttrUID {
        if let Some(uid) = self.attr_name_to_uid.get(attr_name) {
            uid.clone()
        } else {
            let uid = AttrUID(
                self.next_attr_uid
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            );
            self.attr_name_to_uid
                .insert(attr_name.to_string(), uid.clone());
            self.attr_uid_to_name
                .insert(uid.clone(), attr_name.to_string());
            uid
        }
    }

    /// Get attribute name from UID
    pub fn get_attr_name(&self, uid: &AttrUID) -> Option<String> {
        self.attr_uid_to_name.get(uid).map(|name| name.clone())
    }

    /// Set node attribute (simplified approach)
    pub fn set_node_attribute(&self, node_index: usize, attr_name: &str, value: JsonValue) -> AttrUID {
        let attr_uid = self.get_or_create_attr_uid(attr_name);

        // Update max node index
        let current_max = self.max_node_index.load(std::sync::atomic::Ordering::Relaxed);
        if node_index > current_max {
            self.max_node_index.store(node_index, std::sync::atomic::Ordering::Relaxed);
        }

        // Store in sparse columnar storage (single source of truth)
        let mut attr_map = self.node_attributes.entry(attr_uid.clone()).or_default();
        
        // Remove old value from bitmap if it exists
        if let Some(old_value) = attr_map.get(&node_index) {
            if let Some(mut bitmap) = self.node_value_bitmaps.get_mut(&(attr_uid.clone(), old_value.clone())) {
                if node_index < bitmap.len() {
                    bitmap.set(node_index, false);
                }
            }
        }
        
        attr_map.insert(node_index, value.clone());

        // Mark bitmaps as dirty for lazy rebuilding
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);

        attr_uid
    }

    /// Get node attribute
    pub fn get_node_attribute(&self, node_index: usize, attr_name: &str) -> Option<JsonValue> {
        let attr_uid = self.attr_name_to_uid.get(attr_name)?;
        
        // Look up in sparse storage (single source of truth)
        self.node_attributes
            .get(&attr_uid)?
            .get(&node_index)
            .cloned()
    }

    /// Fast attribute-based filtering using bitmaps (builds on-demand)
    pub fn filter_nodes_by_attribute(
        &self,
        attr_name: &str,
        value: &JsonValue,
    ) -> Option<Vec<usize>> {
        let attr_uid = self.attr_name_to_uid.get(attr_name)?;
        let bitmap_key = (attr_uid.clone(), value.clone());

        // Build bitmap if needed
        self.ensure_bitmap_exists(&bitmap_key, true);

        if let Some(bitmap) = self.node_value_bitmaps.get(&bitmap_key) {
            let mut result = Vec::new();
            for (index, bit) in bitmap.iter().enumerate() {
                if *bit {
                    result.push(index);
                }
            }
            Some(result)
        } else {
            Some(Vec::new()) // No nodes have this attribute value
        }
    }

    /// Ensure a bitmap exists for the given key
    fn ensure_bitmap_exists(&self, bitmap_key: &(AttrUID, JsonValue), is_node: bool) {
        let (attr_uid, value) = bitmap_key;
        
        // Check if bitmap already exists
        if is_node && self.node_value_bitmaps.contains_key(bitmap_key) {
            return;
        }
        if !is_node && self.edge_value_bitmaps.contains_key(bitmap_key) {
            return;
        }

        // Build bitmap from sparse data
        let max_index = if is_node {
            self.max_node_index.load(std::sync::atomic::Ordering::Relaxed)
        } else {
            self.max_edge_index.load(std::sync::atomic::Ordering::Relaxed)
        };

        let mut bitmap = BitVec::with_capacity(max_index + 1);
        bitmap.resize(max_index + 1, false);

        // Find all entities with this attribute value
        let attr_map = if is_node {
            self.node_attributes.get(attr_uid)
        } else {
            self.edge_attributes.get(attr_uid)
        };

        if let Some(attr_map) = attr_map {
            for (&entity_index, entity_value) in attr_map.iter() {
                if entity_value == value && entity_index <= max_index {
                    bitmap.set(entity_index, true);
                }
            }
        }

        // Store the bitmap
        if is_node {
            self.node_value_bitmaps.insert(bitmap_key.clone(), bitmap);
        } else {
            self.edge_value_bitmaps.insert(bitmap_key.clone(), bitmap);
        }
    }

    /// Multi-attribute filtering with bitmap intersection (builds bitmaps on-demand)
    pub fn filter_nodes_by_attributes(&self, filters: &HashMap<String, JsonValue>) -> Vec<usize> {
        if filters.is_empty() {
            return Vec::new();
        }

        let mut result_bitmap: Option<BitVec> = None;

        for (attr_name, expected_value) in filters {
            if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
                let bitmap_key = (attr_uid.clone(), expected_value.clone());

                // Build bitmap if needed
                self.ensure_bitmap_exists(&bitmap_key, true);

                if let Some(bitmap) = self.node_value_bitmaps.get(&bitmap_key) {
                    if let Some(ref mut result) = result_bitmap {
                        // Intersection: result = result & bitmap
                        *result &= bitmap.as_bitslice();
                    } else {
                        // First bitmap - clone it
                        result_bitmap = Some(bitmap.clone());
                    }
                } else {
                    // No nodes have this attribute value - return empty
                    return Vec::new();
                }
            } else {
                // Attribute doesn't exist - return empty
                return Vec::new();
            }
        }

        // Convert bitmap to indices
        if let Some(bitmap) = result_bitmap {
            let mut result = Vec::new();
            for (index, bit) in bitmap.iter().enumerate() {
                if *bit {
                    result.push(index);
                }
            }
            result
        } else {
            Vec::new()
        }
    }

    /// Get all attributes for a node using attr_uids from NodeData
    pub fn get_node_attributes_by_uids(&self, node_index: usize, attr_uids: &HashSet<AttrUID>) -> HashMap<String, JsonValue> {
        let mut attributes = HashMap::new();

        for attr_uid in attr_uids {
            if let Some(attr_name) = self.get_attr_name(attr_uid) {
                if let Some(attr_map) = self.node_attributes.get(attr_uid) {
                    if let Some(value) = attr_map.get(&node_index) {
                        attributes.insert(attr_name, value.clone());
                    }
                }
            }
        }

        attributes
    }

    /// Get all attributes for an edge using attr_uids from EdgeData  
    pub fn get_edge_attributes_by_uids(&self, edge_index: usize, attr_uids: &HashSet<AttrUID>) -> HashMap<String, JsonValue> {
        let mut attributes = HashMap::new();

        for attr_uid in attr_uids {
            if let Some(attr_name) = self.get_attr_name(attr_uid) {
                if let Some(attr_map) = self.edge_attributes.get(attr_uid) {
                    if let Some(value) = attr_map.get(&edge_index) {
                        attributes.insert(attr_name, value.clone());
                    }
                }
            }
        }

        attributes
    }

    /// Get all unique values for an attribute (for analytics)
    pub fn get_attribute_values(&self, attr_name: &str) -> Vec<JsonValue> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut values = HashSet::new();

            // Collect from node attributes
            if let Some(node_attrs) = self.node_attributes.get(&attr_uid) {
                for value in node_attrs.values() {
                    values.insert(value.clone());
                }
            }

            // Collect from edge attributes
            if let Some(edge_attrs) = self.edge_attributes.get(&attr_uid) {
                for value in edge_attrs.values() {
                    values.insert(value.clone());
                }
            }

            values.into_iter().collect()
        } else {
            Vec::new()
        }
    }

    /// Get attribute statistics (for analytics)
    pub fn get_attribute_stats(&self, attr_name: &str) -> Option<HashMap<String, usize>> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut stats = HashMap::new();
            let mut value_counts: HashMap<JsonValue, usize> = HashMap::new();

            // Count from node attributes
            if let Some(node_attrs) = self.node_attributes.get(&attr_uid) {
                for value in node_attrs.values() {
                    *value_counts.entry(value.clone()).or_insert(0) += 1;
                }
            }

            // Count from edge attributes  
            if let Some(edge_attrs) = self.edge_attributes.get(&attr_uid) {
                for value in edge_attrs.values() {
                    *value_counts.entry(value.clone()).or_insert(0) += 1;
                }
            }

            for (value, count) in value_counts {
                stats.insert(format!("count_{}", value), count);
            }

            Some(stats)
        } else {
            None
        }
    }

    /// Remove node and clean up attribute storage
    pub fn remove_node(&self, node_index: usize, attr_uids: &HashSet<AttrUID>) {
        for attr_uid in attr_uids {
            if let Some(mut attr_map) = self.node_attributes.get_mut(attr_uid) {
                if let Some(old_value) = attr_map.remove(&node_index) {
                    // Remove from bitmap if it exists
                    let bitmap_key = (attr_uid.clone(), old_value);
                    if let Some(mut bitmap) = self.node_value_bitmaps.get_mut(&bitmap_key) {
                        if node_index < bitmap.len() {
                            bitmap.set(node_index, false);
                        }
                    }
                }
            }
        }
        
        // Mark bitmaps as dirty
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Edge storage methods (simplified, same pattern as nodes)
impl ColumnarStore {
    /// Set edge attribute (simplified approach)
    pub fn set_edge_attribute(&self, edge_index: usize, attr_name: &str, value: JsonValue) -> AttrUID {
        let attr_uid = self.get_or_create_attr_uid(attr_name);

        // Update max edge index
        let current_max = self.max_edge_index.load(std::sync::atomic::Ordering::Relaxed);
        if edge_index > current_max {
            self.max_edge_index.store(edge_index, std::sync::atomic::Ordering::Relaxed);
        }

        // Store in sparse columnar storage (single source of truth)
        let mut attr_map = self.edge_attributes.entry(attr_uid.clone()).or_default();
        
        // Remove old value from bitmap if it exists
        if let Some(old_value) = attr_map.get(&edge_index) {
            if let Some(mut bitmap) = self.edge_value_bitmaps.get_mut(&(attr_uid.clone(), old_value.clone())) {
                if edge_index < bitmap.len() {
                    bitmap.set(edge_index, false);
                }
            }
        }
        
        attr_map.insert(edge_index, value.clone());

        // Mark bitmaps as dirty for lazy rebuilding
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);

        attr_uid
    }

    /// Get edge attribute
    pub fn get_edge_attribute(&self, edge_index: usize, attr_name: &str) -> Option<JsonValue> {
        let attr_uid = self.attr_name_to_uid.get(attr_name)?;
        
        // Look up in sparse storage (single source of truth)
        self.edge_attributes
            .get(&attr_uid)?
            .get(&edge_index)
            .cloned()
    }

    /// Fast edge filtering using bitmaps (builds on-demand)
    pub fn filter_edges_by_attributes(&self, filters: &HashMap<String, JsonValue>) -> Vec<usize> {
        if filters.is_empty() {
            return Vec::new();
        }

        let mut result_bitmap: Option<BitVec> = None;

        for (attr_name, expected_value) in filters {
            if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
                let bitmap_key = (attr_uid.clone(), expected_value.clone());

                // Build bitmap if needed
                self.ensure_bitmap_exists(&bitmap_key, false);

                if let Some(bitmap) = self.edge_value_bitmaps.get(&bitmap_key) {
                    if let Some(ref mut result) = result_bitmap {
                        // Intersection: result = result & bitmap
                        *result &= bitmap.as_bitslice();
                    } else {
                        // First bitmap - clone it
                        result_bitmap = Some(bitmap.clone());
                    }
                } else {
                    // No edges have this attribute value - return empty
                    return Vec::new();
                }
            } else {
                // Attribute doesn't exist - return empty
                return Vec::new();
            }
        }

        // Convert bitmap to indices
        if let Some(bitmap) = result_bitmap {
            let mut result = Vec::new();
            for (index, bit) in bitmap.iter().enumerate() {
                if *bit {
                    result.push(index);
                }
            }
            result
        } else {
            Vec::new()
        }
    }

    /// Remove edge and clean up attribute storage
    pub fn remove_edge(&self, edge_index: usize, attr_uids: &HashSet<AttrUID>) {
        for attr_uid in attr_uids {
            if let Some(mut attr_map) = self.edge_attributes.get_mut(attr_uid) {
                if let Some(old_value) = attr_map.remove(&edge_index) {
                    // Remove from bitmap if it exists
                    let bitmap_key = (attr_uid.clone(), old_value);
                    if let Some(mut bitmap) = self.edge_value_bitmaps.get_mut(&bitmap_key) {
                        if edge_index < bitmap.len() {
                            bitmap.set(edge_index, false);
                        }
                    }
                }
            }
        }
        
        // Mark bitmaps as dirty
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    }
impl ColumnarStore {
    /// Set edge attribute
    pub fn set_edge_attribute(&self, edge_index: usize, attr_name: &str, value: JsonValue) {
        let attr_uid = self.get_or_create_attr_uid(attr_name);

        // Ensure capacity
        self.ensure_edge_capacity(edge_index + 1);

        // Update columnar storage
        if let Some(mut column) = self.edge_columns.get_mut(&attr_uid) {
            // Remove old value from bitmap if it exists
            if let Some(old_value) = &column[edge_index] {
                if let Some(mut bitmap) = self
                    .edge_value_bitmaps
                    .get_mut(&(attr_uid.clone(), old_value.clone()))
                {
                    bitmap.set(edge_index, false);
                }
            }
            column[edge_index] = Some(value.clone());
        } else {
            // Create new column
            let current_capacity = self
                .edge_capacity
                .load(std::sync::atomic::Ordering::Relaxed);
            let mut new_column = vec![None; current_capacity];
            new_column[edge_index] = Some(value.clone());
            self.edge_columns.insert(attr_uid.clone(), new_column);
        }

        // Update bitmap index
        let bitmap_key = (attr_uid.clone(), value.clone());
        if let Some(mut bitmap) = self.edge_value_bitmaps.get_mut(&bitmap_key) {
            bitmap.set(edge_index, true);
        } else {
            let current_capacity = self
                .edge_capacity
                .load(std::sync::atomic::Ordering::Relaxed);
            let mut new_bitmap = bitvec![0; current_capacity];
            new_bitmap.set(edge_index, true);
            self.edge_value_bitmaps.insert(bitmap_key, new_bitmap);
        }

        // Update sparse storage
        let mut sparse_entry = self
            .sparse_edge_storage
            .entry(attr_uid.clone())
            .or_default();
        sparse_entry.insert(edge_index, value.clone());

        // Update attribute set for this edge
        let mut attr_set = self.edge_attr_sets.entry(edge_index).or_default();
        attr_set.insert(attr_uid);
    }

    /// Get edge attribute
    pub fn get_edge_attribute(&self, edge_index: usize, attr_name: &str) -> Option<JsonValue> {
        let attr_uid = self.attr_name_to_uid.get(attr_name)?;

        // Try sparse storage first (faster)
        if let Some(sparse_map) = self.sparse_edge_storage.get(&attr_uid) {
            if let Some(value) = sparse_map.get(&edge_index) {
                return Some(value.clone());
            }
        }

        // Fall back to columnar storage
        if let Some(column) = self.edge_columns.get(&attr_uid) {
            if edge_index < column.len() {
                return column[edge_index].clone();
            }
        }

        None
    }

    /// Fast edge filtering using bitmaps
    pub fn filter_edges_by_attributes(&self, filters: &HashMap<String, JsonValue>) -> Vec<usize> {
        if filters.is_empty() {
            return Vec::new();
        }

        let mut result_bitmap: Option<BitVec> = None;

        for (attr_name, expected_value) in filters {
            if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
                let bitmap_key = (attr_uid.clone(), expected_value.clone());

                if let Some(bitmap) = self.edge_value_bitmaps.get(&bitmap_key) {
                    if let Some(ref mut result) = result_bitmap {
                        // Intersection: result = result & bitmap
                        *result &= bitmap.as_bitslice();
                    } else {
                        // First bitmap - clone it
                        result_bitmap = Some(bitmap.clone());
                    }
                } else {
                    // No edges have this attribute value - return empty
                    return Vec::new();
                }
            } else {
                // Attribute doesn't exist - return empty
                return Vec::new();
            }
        }

        // Convert bitmap to indices
        if let Some(bitmap) = result_bitmap {
            let mut result = Vec::new();
            for (index, bit) in bitmap.iter().enumerate() {
                if *bit {
                    result.push(index);
                }
            }
            result
        } else {
            Vec::new()
        }
    }

    /// Filter nodes by numeric comparison using sparse storage
    pub fn filter_nodes_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut result = Vec::new();

            if let Some(attr_map) = self.node_attributes.get(&attr_uid) {
                for (&node_index, json_value) in attr_map.iter() {
                    // Try to get numeric value
                    let node_value = match json_value {
                        JsonValue::Number(n) => n.as_f64().unwrap_or(0.0),
                        JsonValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => continue, // Skip non-numeric values
                    };

                    let matches = match operator {
                        ">" => node_value > value,
                        ">=" => node_value >= value,
                        "<" => node_value < value,
                        "<=" => node_value <= value,
                        "==" => (node_value - value).abs() < f64::EPSILON,
                        "!=" => (node_value - value).abs() >= f64::EPSILON,
                        _ => false,
                    };

                    if matches {
                        result.push(node_index);
                    }
                }
            }

            result
        } else {
            Vec::new()
        }
    }

    /// Filter nodes by string comparison - optimized for performance
    pub fn filter_nodes_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut result = Vec::new();

            // Use sparse storage for efficient iteration
            if let Some(sparse_data) = self.sparse_node_storage.get(&attr_uid) {
                for (&node_index, json_value) in sparse_data.iter() {
                    if let Some(node_value) = json_value.as_str() {
                        let matches = match operator {
                            "==" => node_value == value,
                            "!=" => node_value != value,
                            "contains" => node_value.contains(value),
                            "startswith" => node_value.starts_with(value),
                            "endswith" => node_value.ends_with(value),
                            _ => false,
                        };

                        if matches {
                            result.push(node_index);
                        }
                    }
                }
            }

            result
        } else {
            Vec::new()
        }
    }

    /// Filter edges by numeric comparison - optimized for performance
    pub fn filter_edges_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut result = Vec::new();

            // Use sparse storage for efficient iteration
            if let Some(sparse_data) = self.sparse_edge_storage.get(&attr_uid) {
                for (&edge_index, json_value) in sparse_data.iter() {
                    if let Some(edge_value) = json_value.as_f64() {
                        let matches = match operator {
                            ">" => edge_value > value,
                            ">=" => edge_value >= value,
                            "<" => edge_value < value,
                            "<=" => edge_value <= value,
                            "==" => (edge_value - value).abs() < f64::EPSILON,
                            "!=" => (edge_value - value).abs() >= f64::EPSILON,
                            _ => false,
                        };

                        if matches {
                            result.push(edge_index);
                        }
                    }
                }
            }

            result
        } else {
            Vec::new()
        }
    }

    /// Filter edges by string comparison - optimized for performance
    pub fn filter_edges_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut result = Vec::new();

            // Use sparse storage for efficient iteration
            if let Some(sparse_data) = self.sparse_edge_storage.get(&attr_uid) {
                for (&edge_index, json_value) in sparse_data.iter() {
                    if let Some(edge_value) = json_value.as_str() {
                        let matches = match operator {
                            "==" => edge_value == value,
                            "!=" => edge_value != value,
                            "contains" => edge_value.contains(value),
                            "startswith" => edge_value.starts_with(value),
                            "endswith" => edge_value.ends_with(value),
                            _ => false,
                        };

                        if matches {
                            result.push(edge_index);
                        }
                    }
                }
            }

            result
        } else {
            Vec::new()
        }
    }

    // ...existing code...
}

impl Clone for ColumnarStore {
    fn clone(&self) -> Self {
        Self {
            attr_name_to_uid: self.attr_name_to_uid.clone(),
            attr_uid_to_name: self.attr_uid_to_name.clone(),
            next_attr_uid: std::sync::atomic::AtomicU64::new(
                self.next_attr_uid
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            node_columns: self.node_columns.clone(),
            edge_columns: self.edge_columns.clone(),
            node_value_bitmaps: self.node_value_bitmaps.clone(),
            edge_value_bitmaps: self.edge_value_bitmaps.clone(),
            sparse_node_storage: self.sparse_node_storage.clone(),
            sparse_edge_storage: self.sparse_edge_storage.clone(),
            node_attr_sets: self.node_attr_sets.clone(),
            edge_attr_sets: self.edge_attr_sets.clone(),
            node_capacity: std::sync::atomic::AtomicUsize::new(
                self.node_capacity
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            edge_capacity: std::sync::atomic::AtomicUsize::new(
                self.edge_capacity
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}
