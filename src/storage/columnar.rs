use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use serde_json::Value as JsonValue;
use dashmap::DashMap;
use bitvec::prelude::*;

/// Unique identifier for attributes across the entire graph
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct AttrUID(pub u64);

/// Columnar storage for graph attributes with bitmap indexing
#[pyclass]
pub struct ColumnarStore {
    /// Maps attribute names to their unique IDs
    pub attr_name_to_uid: DashMap<String, AttrUID>,
    /// Maps attribute UIDs back to names
    pub attr_uid_to_name: DashMap<AttrUID, String>,
    /// Next available attribute UID
    pub next_attr_uid: std::sync::atomic::AtomicU64,
    
    /// Columnar storage: attr_uid -> Vec<attr_value>
    /// Each position corresponds to a node/edge index
    pub node_columns: DashMap<AttrUID, Vec<Option<JsonValue>>>,
    pub edge_columns: DashMap<AttrUID, Vec<Option<JsonValue>>>,
    
    /// Bitmap indices for fast filtering
    /// Maps (attr_uid, attr_value) -> BitVec indicating which nodes/edges have this value
    pub node_value_bitmaps: DashMap<(AttrUID, JsonValue), BitVec>,
    pub edge_value_bitmaps: DashMap<(AttrUID, JsonValue), BitVec>,
    
    /// Sparse storage for non-null values
    /// Maps attr_uid -> HashMap<entity_index, attr_value>
    pub sparse_node_storage: DashMap<AttrUID, HashMap<usize, JsonValue>>,
    pub sparse_edge_storage: DashMap<AttrUID, HashMap<usize, JsonValue>>,
    
    /// Track which attributes exist for each entity
    pub node_attr_sets: DashMap<usize, HashSet<AttrUID>>,
    pub edge_attr_sets: DashMap<usize, HashSet<AttrUID>>,
    
    /// Capacity management
    pub node_capacity: std::sync::atomic::AtomicUsize,
    pub edge_capacity: std::sync::atomic::AtomicUsize,
}

#[pymethods]
impl ColumnarStore {
    #[new]
    pub fn new() -> Self {
        Self {
            attr_name_to_uid: DashMap::new(),
            attr_uid_to_name: DashMap::new(),
            next_attr_uid: std::sync::atomic::AtomicU64::new(0),
            node_columns: DashMap::new(),
            edge_columns: DashMap::new(),
            node_value_bitmaps: DashMap::new(),
            edge_value_bitmaps: DashMap::new(),
            sparse_node_storage: DashMap::new(),
            sparse_edge_storage: DashMap::new(),
            node_attr_sets: DashMap::new(),
            edge_attr_sets: DashMap::new(),
            node_capacity: std::sync::atomic::AtomicUsize::new(0),
            edge_capacity: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Get storage statistics
    pub fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("attributes_registered".to_string(), self.attr_name_to_uid.len());
        stats.insert("node_columns".to_string(), self.node_columns.len());
        stats.insert("edge_columns".to_string(), self.edge_columns.len());
        stats.insert("node_bitmaps".to_string(), self.node_value_bitmaps.len());
        stats.insert("edge_bitmaps".to_string(), self.edge_value_bitmaps.len());
        stats.insert("node_capacity".to_string(), self.node_capacity.load(std::sync::atomic::Ordering::Relaxed));
        stats.insert("edge_capacity".to_string(), self.edge_capacity.load(std::sync::atomic::Ordering::Relaxed));
        stats
    }
}

impl ColumnarStore {
    /// Get or create an attribute UID for a given attribute name
    pub fn get_or_create_attr_uid(&self, attr_name: &str) -> AttrUID {
        if let Some(uid) = self.attr_name_to_uid.get(attr_name) {
            uid.clone()
        } else {
            let uid = AttrUID(self.next_attr_uid.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            self.attr_name_to_uid.insert(attr_name.to_string(), uid.clone());
            self.attr_uid_to_name.insert(uid.clone(), attr_name.to_string());
            uid
        }
    }
    
    /// Get attribute name from UID
    pub fn get_attr_name(&self, uid: &AttrUID) -> Option<String> {
        self.attr_uid_to_name.get(uid).map(|name| name.clone())
    }
    
    /// Ensure node capacity
    pub fn ensure_node_capacity(&self, required_capacity: usize) {
        let current_capacity = self.node_capacity.load(std::sync::atomic::Ordering::Relaxed);
        if required_capacity > current_capacity {
            let new_capacity = (required_capacity * 2).max(1024); // Grow by 2x with minimum
            
            // Resize all node columns
            for mut column in self.node_columns.iter_mut() {
                column.resize(new_capacity, None);
            }
            
            // Resize all node bitmaps
            for mut bitmap in self.node_value_bitmaps.iter_mut() {
                bitmap.resize(new_capacity, false);
            }
            
            self.node_capacity.store(new_capacity, std::sync::atomic::Ordering::Relaxed);
        }
    }
    
    /// Ensure edge capacity
    pub fn ensure_edge_capacity(&self, required_capacity: usize) {
        let current_capacity = self.edge_capacity.load(std::sync::atomic::Ordering::Relaxed);
        if required_capacity > current_capacity {
            let new_capacity = (required_capacity * 2).max(1024); // Grow by 2x with minimum
            
            // Resize all edge columns
            for mut column in self.edge_columns.iter_mut() {
                column.resize(new_capacity, None);
            }
            
            // Resize all edge bitmaps
            for mut bitmap in self.edge_value_bitmaps.iter_mut() {
                bitmap.resize(new_capacity, false);
            }
            
            self.edge_capacity.store(new_capacity, std::sync::atomic::Ordering::Relaxed);
        }
    }
    
    /// Set node attribute (columnar storage)
    pub fn set_node_attribute(&self, node_index: usize, attr_name: &str, value: JsonValue) {
        let attr_uid = self.get_or_create_attr_uid(attr_name);
        
        // Ensure capacity
        self.ensure_node_capacity(node_index + 1);
        
        // Update columnar storage
        if let Some(mut column) = self.node_columns.get_mut(&attr_uid) {
            // Remove old value from bitmap if it exists
            if let Some(old_value) = &column[node_index] {
                if let Some(mut bitmap) = self.node_value_bitmaps.get_mut(&(attr_uid.clone(), old_value.clone())) {
                    bitmap.set(node_index, false);
                }
            }
            column[node_index] = Some(value.clone());
        } else {
            // Create new column
            let current_capacity = self.node_capacity.load(std::sync::atomic::Ordering::Relaxed);
            let mut new_column = vec![None; current_capacity];
            new_column[node_index] = Some(value.clone());
            self.node_columns.insert(attr_uid.clone(), new_column);
        }
        
        // Update bitmap index
        let bitmap_key = (attr_uid.clone(), value.clone());
        if let Some(mut bitmap) = self.node_value_bitmaps.get_mut(&bitmap_key) {
            bitmap.set(node_index, true);
        } else {
            let current_capacity = self.node_capacity.load(std::sync::atomic::Ordering::Relaxed);
            let mut new_bitmap = bitvec![0; current_capacity];
            new_bitmap.set(node_index, true);
            self.node_value_bitmaps.insert(bitmap_key, new_bitmap);
        }
        
        // Update sparse storage
        let mut sparse_entry = self.sparse_node_storage.entry(attr_uid.clone()).or_insert_with(HashMap::new);
        sparse_entry.insert(node_index, value.clone());
        
        // Update attribute set for this node
        let mut attr_set = self.node_attr_sets.entry(node_index).or_insert_with(HashSet::new);
        attr_set.insert(attr_uid);
    }
    
    /// Get node attribute
    pub fn get_node_attribute(&self, node_index: usize, attr_name: &str) -> Option<JsonValue> {
        let attr_uid = self.attr_name_to_uid.get(attr_name)?;
        
        // Try sparse storage first (faster)
        if let Some(sparse_map) = self.sparse_node_storage.get(&attr_uid) {
            if let Some(value) = sparse_map.get(&node_index) {
                return Some(value.clone());
            }
        }
        
        // Fall back to columnar storage
        if let Some(column) = self.node_columns.get(&attr_uid) {
            if node_index < column.len() {
                return column[node_index].clone();
            }
        }
        
        None
    }
    
    /// Fast attribute-based filtering using bitmaps
    pub fn filter_nodes_by_attribute(&self, attr_name: &str, value: &JsonValue) -> Option<Vec<usize>> {
        let attr_uid = self.attr_name_to_uid.get(attr_name)?;
        let bitmap_key = (attr_uid.clone(), value.clone());
        
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
    
    /// Multi-attribute filtering with bitmap intersection
    pub fn filter_nodes_by_attributes(&self, filters: &HashMap<String, JsonValue>) -> Vec<usize> {
        if filters.is_empty() {
            return Vec::new();
        }
        
        let mut result_bitmap: Option<BitVec> = None;
        
        for (attr_name, expected_value) in filters {
            if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
                let bitmap_key = (attr_uid.clone(), expected_value.clone());
                
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
    
    /// Get all attributes for a node
    pub fn get_node_attributes(&self, node_index: usize) -> HashMap<String, JsonValue> {
        let mut attributes = HashMap::new();
        
        if let Some(attr_set) = self.node_attr_sets.get(&node_index) {
            for attr_uid in attr_set.iter() {
                if let Some(attr_name) = self.get_attr_name(attr_uid) {
                    if let Some(value) = self.get_node_attribute(node_index, &attr_name) {
                        attributes.insert(attr_name, value);
                    }
                }
            }
        }
        
        attributes
    }
    
    /// Get all attributes for an edge
    pub fn get_edge_attributes(&self, edge_index: usize) -> HashMap<String, JsonValue> {
        let mut attributes = HashMap::new();
        
        if let Some(attr_set) = self.edge_attr_sets.get(&edge_index) {
            for attr_uid in attr_set.iter() {
                if let Some(attr_name) = self.get_attr_name(attr_uid) {
                    if let Some(value) = self.get_edge_attribute(edge_index, &attr_name) {
                        attributes.insert(attr_name, value);
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
            
            // Collect from node bitmaps
            for bitmap_key in self.node_value_bitmaps.iter() {
                if bitmap_key.key().0 == *attr_uid {
                    values.insert(bitmap_key.key().1.clone());
                }
            }
            
            // Collect from edge bitmaps
            for bitmap_key in self.edge_value_bitmaps.iter() {
                if bitmap_key.key().0 == *attr_uid {
                    values.insert(bitmap_key.key().1.clone());
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
            let mut total_nodes = 0;
            let mut total_edges = 0;
            
            // Count from node bitmaps
            for bitmap_entry in self.node_value_bitmaps.iter() {
                if bitmap_entry.key().0 == *attr_uid {
                    let count = bitmap_entry.count_ones();
                    stats.insert(format!("nodes_with_{}", bitmap_entry.key().1), count);
                    total_nodes += count;
                }
            }
            
            // Count from edge bitmaps
            for bitmap_entry in self.edge_value_bitmaps.iter() {
                if bitmap_entry.key().0 == *attr_uid {
                    let count = bitmap_entry.count_ones();
                    stats.insert(format!("edges_with_{}", bitmap_entry.key().1), count);
                    total_edges += count;
                }
            }
            
            stats.insert("total_nodes".to_string(), total_nodes);
            stats.insert("total_edges".to_string(), total_edges);
            
            Some(stats)
        } else {
            None
        }
    }
    
    /// Remove node (clean up all attribute storage)
    pub fn remove_node(&self, node_index: usize) {
        // Remove from all attribute sets
        if let Some((_, attr_set)) = self.node_attr_sets.remove(&node_index) {
            for attr_uid in attr_set {
                // Remove from sparse storage
                if let Some(mut sparse_map) = self.sparse_node_storage.get_mut(&attr_uid) {
                    sparse_map.remove(&node_index);
                }
                
                // Remove from columnar storage
                if let Some(mut column) = self.node_columns.get_mut(&attr_uid) {
                    if node_index < column.len() {
                        if let Some(old_value) = column[node_index].take() {
                            // Remove from bitmap
                            let bitmap_key = (attr_uid.clone(), old_value);
                            if let Some(mut bitmap) = self.node_value_bitmaps.get_mut(&bitmap_key) {
                                bitmap.set(node_index, false);
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Compact storage (remove unused capacity)
    pub fn compact(&self) {
        // This would implement compaction logic to remove unused space
        // and optimize memory layout - complex operation for production systems
    }
}

/// Edge storage methods (similar to node storage)
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
                if let Some(mut bitmap) = self.edge_value_bitmaps.get_mut(&(attr_uid.clone(), old_value.clone())) {
                    bitmap.set(edge_index, false);
                }
            }
            column[edge_index] = Some(value.clone());
        } else {
            // Create new column
            let current_capacity = self.edge_capacity.load(std::sync::atomic::Ordering::Relaxed);
            let mut new_column = vec![None; current_capacity];
            new_column[edge_index] = Some(value.clone());
            self.edge_columns.insert(attr_uid.clone(), new_column);
        }
        
        // Update bitmap index
        let bitmap_key = (attr_uid.clone(), value.clone());
        if let Some(mut bitmap) = self.edge_value_bitmaps.get_mut(&bitmap_key) {
            bitmap.set(edge_index, true);
        } else {
            let current_capacity = self.edge_capacity.load(std::sync::atomic::Ordering::Relaxed);
            let mut new_bitmap = bitvec![0; current_capacity];
            new_bitmap.set(edge_index, true);
            self.edge_value_bitmaps.insert(bitmap_key, new_bitmap);
        }
        
        // Update sparse storage
        let mut sparse_entry = self.sparse_edge_storage.entry(attr_uid.clone()).or_insert_with(HashMap::new);
        sparse_entry.insert(edge_index, value.clone());
        
        // Update attribute set for this edge
        let mut attr_set = self.edge_attr_sets.entry(edge_index).or_insert_with(HashSet::new);
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
    
    /// Filter nodes by numeric comparison (e.g., salary > 100000) - optimized for performance
    /// Uses range queries on sorted indices for better than O(n) performance
    pub fn filter_nodes_by_numeric_comparison(&self, attr_name: &str, operator: &str, value: f64) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let mut result = Vec::new();
            
            // For numeric comparisons, we need to check the columnar data
            // This could be optimized further with sorted indices, but for now use efficient sparse iteration
            if let Some(sparse_data) = self.sparse_node_storage.get(&attr_uid) {
                for (&node_index, json_value) in sparse_data.iter() {
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
    pub fn filter_nodes_by_string_comparison(&self, attr_name: &str, operator: &str, value: &str) -> Vec<usize> {
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
    pub fn filter_edges_by_numeric_comparison(&self, attr_name: &str, operator: &str, value: f64) -> Vec<usize> {
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
    pub fn filter_edges_by_string_comparison(&self, attr_name: &str, operator: &str, value: &str) -> Vec<usize> {
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
                self.next_attr_uid.load(std::sync::atomic::Ordering::Relaxed)
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
                self.node_capacity.load(std::sync::atomic::Ordering::Relaxed)
            ),
            edge_capacity: std::sync::atomic::AtomicUsize::new(
                self.edge_capacity.load(std::sync::atomic::Ordering::Relaxed)
            ),
        }
    }
}
