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

    /// Get all attributes for a node (legacy interface - looks up attr_uids from NodeData)
    /// This method will need to be called with the actual node data to get attr_uids
    pub fn get_node_attributes(&self, node_index: usize) -> HashMap<String, JsonValue> {
        let mut attributes = HashMap::new();

        // Iterate through all attributes and check if this node has them
        // This is less efficient than the new approach but maintains compatibility
        for attr_entry in self.node_attributes.iter() {
            let attr_uid = attr_entry.key();
            let attr_map = attr_entry.value();
            
            if let Some(value) = attr_map.get(&node_index) {
                if let Some(attr_name) = self.get_attr_name(attr_uid) {
                    attributes.insert(attr_name, value.clone());
                }
            }
        }

        attributes
    }

    /// Get all attributes for an edge (legacy interface)
    pub fn get_edge_attributes(&self, edge_index: usize) -> HashMap<String, JsonValue> {
        let mut attributes = HashMap::new();

        // Iterate through all attributes and check if this edge has them
        for attr_entry in self.edge_attributes.iter() {
            let attr_uid = attr_entry.key();
            let attr_map = attr_entry.value();
            
            if let Some(value) = attr_map.get(&edge_index) {
                if let Some(attr_name) = self.get_attr_name(attr_uid) {
                    attributes.insert(attr_name, value.clone());
                }
            }
        }

        attributes
    }

    /// Filter nodes by numeric comparison (optimized implementation)
    pub fn filter_nodes_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            if let Some(attr_map) = self.node_attributes.get(&attr_uid) {
                let mut result = Vec::with_capacity(attr_map.len() / 4); // Estimate capacity
                
                // Vectorized processing with efficient iterator
                result.extend(
                    attr_map
                        .iter()
                        .filter_map(|(&node_index, json_value)| {
                            let node_value = match json_value {
                                JsonValue::Number(n) => n.as_f64()?,
                                JsonValue::String(s) => s.parse::<f64>().ok()?,
                                _ => return None,
                            };

                            let matches = match operator {
                                ">" => node_value > value,
                                ">=" => node_value >= value,
                                "<" => node_value < value,
                                "<=" => node_value <= value,
                                "==" => (node_value - value).abs() < f64::EPSILON,
                                "!=" => (node_value - value).abs() >= f64::EPSILON,
                                _ => return None,
                            };

                            if matches { Some(node_index) } else { None }
                        })
                );
                
                result
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Filter nodes by string comparison (optimized implementation)
    pub fn filter_nodes_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            if let Some(attr_map) = self.node_attributes.get(&attr_uid) {
                let mut result = Vec::with_capacity(attr_map.len() / 4); // Estimate capacity
                
                // Vectorized processing with efficient iterator
                result.extend(
                    attr_map
                        .iter()
                        .filter_map(|(&node_index, json_value)| {
                            let node_value = json_value.as_str()?;

                            let matches = match operator {
                                "==" => node_value == value,
                                "!=" => node_value != value,
                                "contains" => node_value.contains(value),
                                "startswith" => node_value.starts_with(value),
                                "endswith" => node_value.ends_with(value),
                                _ => return None,
                            };

                            if matches { Some(node_index) } else { None }
                        })
                );
                
                result
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
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

    /// Remove node and clean up attribute storage (legacy interface - finds attr_uids automatically)
    pub fn remove_node_legacy(&self, node_index: usize) {
        // Find all attributes this node has by scanning all attribute maps
        let mut node_attr_uids = HashSet::new();
        
        for attr_entry in self.node_attributes.iter() {
            let attr_uid = attr_entry.key();
            let attr_map = attr_entry.value();
            
            if attr_map.contains_key(&node_index) {
                node_attr_uids.insert(attr_uid.clone());
            }
        }
        
        // Use the new method with the found attr_uids
        self.remove_node(node_index, &node_attr_uids);
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

    /// Filter edges by numeric comparison (optimized implementation)
    pub fn filter_edges_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            if let Some(attr_map) = self.edge_attributes.get(&attr_uid) {
                let mut result = Vec::with_capacity(attr_map.len() / 4); // Estimate capacity
                
                // Vectorized processing with efficient iterator
                result.extend(
                    attr_map
                        .iter()
                        .filter_map(|(&edge_index, json_value)| {
                            let edge_value = match json_value {
                                JsonValue::Number(n) => n.as_f64()?,
                                JsonValue::String(s) => s.parse::<f64>().ok()?,
                                _ => return None,
                            };

                            let matches = match operator {
                                ">" => edge_value > value,
                                ">=" => edge_value >= value,
                                "<" => edge_value < value,
                                "<=" => edge_value <= value,
                                "==" => (edge_value - value).abs() < f64::EPSILON,
                                "!=" => (edge_value - value).abs() >= f64::EPSILON,
                                _ => return None,
                            };

                            if matches { Some(edge_index) } else { None }
                        })
                );
                
                result
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Filter edges by string comparison (optimized implementation)
    pub fn filter_edges_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> Vec<usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            if let Some(attr_map) = self.edge_attributes.get(&attr_uid) {
                let mut result = Vec::with_capacity(attr_map.len() / 4); // Estimate capacity
                
                // Vectorized processing with efficient iterator
                result.extend(
                    attr_map
                        .iter()
                        .filter_map(|(&edge_index, json_value)| {
                            let edge_value = json_value.as_str()?;

                            let matches = match operator {
                                "==" => edge_value == value,
                                "!=" => edge_value != value,
                                "contains" => edge_value.contains(value),
                                "startswith" => edge_value.starts_with(value),
                                "endswith" => edge_value.ends_with(value),
                                _ => return None,
                            };

                            if matches { Some(edge_index) } else { None }
                        })
                );
                
                result
            } else {
                Vec::new()
            }
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

    /// Remove edge and clean up attribute storage (legacy interface)  
    pub fn remove_edge_legacy(&self, edge_index: usize) {
        // Find all attributes this edge has by scanning all attribute maps
        let mut edge_attr_uids = HashSet::new();
        
        for attr_entry in self.edge_attributes.iter() {
            let attr_uid = attr_entry.key();
            let attr_map = attr_entry.value();
            
            if attr_map.contains_key(&edge_index) {
                edge_attr_uids.insert(attr_uid.clone());
            }
        }
        
        // Use the new method with the found attr_uids
        self.remove_edge(edge_index, &edge_attr_uids);
    }
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
            node_attributes: self.node_attributes.clone(),
            edge_attributes: self.edge_attributes.clone(),
            node_value_bitmaps: self.node_value_bitmaps.clone(),
            edge_value_bitmaps: self.edge_value_bitmaps.clone(),
            bitmaps_dirty: std::sync::atomic::AtomicBool::new(
                self.bitmaps_dirty.load(std::sync::atomic::Ordering::Relaxed),
            ),
            max_node_index: std::sync::atomic::AtomicUsize::new(
                self.max_node_index.load(std::sync::atomic::Ordering::Relaxed),
            ),
            max_edge_index: std::sync::atomic::AtomicUsize::new(
                self.max_edge_index.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

/// Complex query structure for multi-stage filtering
#[derive(Debug, Clone)]
pub struct ComplexQuery {
    /// Exact value filters (fast bitmap operations)
    pub exact_filters: HashMap<String, JsonValue>,
    /// Numeric comparison filters: (attribute, operator, value)
    pub numeric_filters: Vec<(String, String, f64)>,
    /// String comparison filters: (attribute, operator, value)
    pub string_filters: Vec<(String, String, String)>,
}

impl ComplexQuery {
    pub fn new() -> Self {
        Self {
            exact_filters: HashMap::new(),
            numeric_filters: Vec::new(),
            string_filters: Vec::new(),
        }
    }

    /// Add exact value filter
    pub fn with_exact(mut self, attr_name: String, value: JsonValue) -> Self {
        self.exact_filters.insert(attr_name, value);
        self
    }

    /// Add numeric comparison filter
    pub fn with_numeric(mut self, attr_name: String, operator: String, value: f64) -> Self {
        self.numeric_filters.push((attr_name, operator, value));
        self
    }

    /// Add string comparison filter
    pub fn with_string(mut self, attr_name: String, operator: String, value: String) -> Self {
        self.string_filters.push((attr_name, operator, value));
        self
    }
}

impl ColumnarStore {
    /// Batch set node attributes for multiple nodes efficiently
    /// Uses attr_uid mapping and value deduplication
    pub fn batch_set_node_attributes(&self, 
        batch_data: Vec<(usize, Vec<(String, JsonValue)>)>
    ) -> HashMap<String, AttrUID> {
        let mut attr_name_to_uid = HashMap::new();
        
        // Pre-allocate or get attr_uids for all attribute names in batch
        let mut all_attr_names = HashSet::new();
        for (_, attrs) in &batch_data {
            for (attr_name, _) in attrs {
                all_attr_names.insert(attr_name.clone());
            }
        }
        
        // Get/create all attr_uids at once
        for attr_name in all_attr_names {
            let attr_uid = self.get_or_create_attr_uid(&attr_name);
            attr_name_to_uid.insert(attr_name, attr_uid);
        }
        
        // Track max node index for this batch
        let mut max_node_in_batch = 0;
        
        // Group by attribute for efficient columnar updates
        let mut attr_updates: HashMap<AttrUID, Vec<(usize, JsonValue)>> = HashMap::new();
        
        for (node_index, attrs) in batch_data {
            max_node_in_batch = max_node_in_batch.max(node_index);
            
            for (attr_name, value) in attrs {
                if let Some(attr_uid) = attr_name_to_uid.get(&attr_name) {
                    attr_updates.entry(attr_uid.clone())
                        .or_default()
                        .push((node_index, value));
                }
            }
        }
        
        // Update max node index atomically
        let current_max = self.max_node_index.load(std::sync::atomic::Ordering::Relaxed);
        if max_node_in_batch > current_max {
            self.max_node_index.store(max_node_in_batch, std::sync::atomic::Ordering::Relaxed);
        }
        
        // Apply all updates per attribute in batch
        for (attr_uid, updates) in attr_updates {
            let mut attr_map = self.node_attributes.entry(attr_uid).or_default();
            
            for (node_index, value) in updates {
                attr_map.insert(node_index, value);
            }
        }
        
        // Mark bitmaps as dirty - they'll be rebuilt on demand
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        
        attr_name_to_uid
    }

    /// Batch set edge attributes for multiple edges efficiently  
    pub fn batch_set_edge_attributes(&self, 
        batch_data: Vec<(usize, Vec<(String, JsonValue)>)>
    ) -> HashMap<String, AttrUID> {
        let mut attr_name_to_uid = HashMap::new();
        
        // Pre-allocate or get attr_uids for all attribute names in batch
        let mut all_attr_names = HashSet::new();
        for (_, attrs) in &batch_data {
            for (attr_name, _) in attrs {
                all_attr_names.insert(attr_name.clone());
            }
        }
        
        // Get/create all attr_uids at once
        for attr_name in all_attr_names {
            let attr_uid = self.get_or_create_attr_uid(&attr_name);
            attr_name_to_uid.insert(attr_name, attr_uid);
        }
        
        // Track max edge index for this batch
        let mut max_edge_in_batch = 0;
        
        // Group by attribute for efficient columnar updates
        let mut attr_updates: HashMap<AttrUID, Vec<(usize, JsonValue)>> = HashMap::new();
        
        for (edge_index, attrs) in batch_data {
            max_edge_in_batch = max_edge_in_batch.max(edge_index);
            
            for (attr_name, value) in attrs {
                if let Some(attr_uid) = attr_name_to_uid.get(&attr_name) {
                    attr_updates.entry(attr_uid.clone())
                        .or_default()
                        .push((edge_index, value));
                }
            }
        }
        
        // Update max edge index atomically
        let current_max = self.max_edge_index.load(std::sync::atomic::Ordering::Relaxed);
        if max_edge_in_batch > current_max {
            self.max_edge_index.store(max_edge_in_batch, std::sync::atomic::Ordering::Relaxed);
        }
        
        // Apply all updates per attribute in batch
        for (attr_uid, updates) in attr_updates {
            let mut attr_map = self.edge_attributes.entry(attr_uid).or_default();
            
            for (edge_index, value) in updates {
                attr_map.insert(edge_index, value);
            }
        }
        
        // Mark bitmaps as dirty - they'll be rebuilt on demand
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        
        attr_name_to_uid
    }

    /// High-performance multi-attribute filtering using sparse intersection
    /// Instead of building full bitmaps, directly intersect sparse attribute maps
    pub fn filter_nodes_sparse(&self, filters: &HashMap<String, JsonValue>) -> Vec<usize> {
        if filters.is_empty() {
            return Vec::new();
        }

        // Convert attribute names to UIDs
        let mut attr_uid_filters = Vec::new();
        for (attr_name, expected_value) in filters {
            if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
                attr_uid_filters.push((attr_uid.clone(), expected_value.clone()));
            } else {
                // Attribute doesn't exist - return empty immediately
                return Vec::new();
            }
        }

        // Start with candidates from the first (hopefully most selective) attribute
        let mut candidates: HashSet<usize> = HashSet::new();
        let mut first_iteration = true;

        for (attr_uid, expected_value) in attr_uid_filters {
            if let Some(attr_map) = self.node_attributes.get(&attr_uid) {
                let matching_nodes: HashSet<usize> = attr_map
                    .iter()
                    .filter_map(|(&node_index, node_value)| {
                        if node_value == &expected_value {
                            Some(node_index)
                        } else {
                            None
                        }
                    })
                    .collect();

                if first_iteration {
                    candidates = matching_nodes;
                    first_iteration = false;
                } else {
                    // Intersect with previous candidates
                    candidates.retain(|node_index| matching_nodes.contains(node_index));
                }

                // Early termination if no candidates remain
                if candidates.is_empty() {
                    return Vec::new();
                }
            } else {
                // No nodes have this attribute - return empty
                return Vec::new();
            }
        }

        let mut result: Vec<usize> = candidates.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Get unique values for an attribute (for analytics and optimization)
    pub fn get_unique_values_for_attribute(&self, attr_name: &str, is_node: bool) -> Vec<JsonValue> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let attr_map = if is_node {
                self.node_attributes.get(&attr_uid)
            } else {
                self.edge_attributes.get(&attr_uid)
            };

            if let Some(attr_map) = attr_map {
                let mut unique_values: HashSet<JsonValue> = HashSet::new();
                for (_, value) in attr_map.iter() {
                    unique_values.insert(value.clone());
                }
                unique_values.into_iter().collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Get value distribution for an attribute (useful for query optimization)
    pub fn get_value_distribution(&self, attr_name: &str, is_node: bool) -> HashMap<JsonValue, usize> {
        if let Some(attr_uid) = self.attr_name_to_uid.get(attr_name) {
            let attr_map = if is_node {
                self.node_attributes.get(&attr_uid)
            } else {
                self.edge_attributes.get(&attr_uid)
            };

            if let Some(attr_map) = attr_map {
                let mut distribution: HashMap<JsonValue, usize> = HashMap::new();
                for (_, value) in attr_map.iter() {
                    *distribution.entry(value.clone()).or_default() += 1;
                }
                distribution
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        }
    }
    
    /// Bulk set node attributes for efficient batch operations
    pub fn bulk_set_node_attributes(
        &self,
        attr_name: &str,
        node_value_pairs: Vec<(usize, JsonValue)>
    ) {
        if node_value_pairs.is_empty() {
            return;
        }

        // Get or create attribute UID
        let attr_uid = self.get_or_create_attr_uid(attr_name);
        
        // Update max node index if needed
        if let Some(max_index) = node_value_pairs.iter().map(|(idx, _)| *idx).max() {
            self.max_node_index.fetch_max(max_index, std::sync::atomic::Ordering::Relaxed);
        }

        // Bulk insert into columnar storage
        {
            let mut attr_map = self.node_attributes.entry(attr_uid).or_insert_with(HashMap::new);
            for (node_index, value) in node_value_pairs {
                attr_map.insert(node_index, value);
            }
        }

        // Mark bitmaps as dirty
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bulk set edge attributes for efficient batch operations
    pub fn bulk_set_edge_attributes(
        &self,
        attr_name: &str,
        edge_value_pairs: Vec<(usize, JsonValue)>
    ) {
        if edge_value_pairs.is_empty() {
            return;
        }

        // Get or create attribute UID
        let attr_uid = self.get_or_create_attr_uid(attr_name);
        
        // Update max edge index if needed
        if let Some(max_index) = edge_value_pairs.iter().map(|(idx, _)| *idx).max() {
            self.max_edge_index.fetch_max(max_index, std::sync::atomic::Ordering::Relaxed);
        }

        // Bulk insert into columnar storage
        {
            let mut attr_map = self.edge_attributes.entry(attr_uid).or_insert_with(HashMap::new);
            for (edge_index, value) in edge_value_pairs {
                attr_map.insert(edge_index, value);
            }
        }

        // Mark bitmaps as dirty
        self.bitmaps_dirty.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bulk set multiple node attributes efficiently - optimized for graph creation
    pub fn bulk_set_multiple_node_attributes(
        &self,
        nodes_attrs: Vec<(usize, HashMap<String, JsonValue>)>
    ) {
        if nodes_attrs.is_empty() {
            return;
        }

        // Group by attribute name for better cache locality
        let mut attr_groups: HashMap<String, Vec<(usize, JsonValue)>> = HashMap::new();
        
        for (node_index, attributes) in nodes_attrs {
            // Update max node index
            self.max_node_index.fetch_max(node_index, std::sync::atomic::Ordering::Relaxed);
            
            for (attr_name, attr_value) in attributes {
                attr_groups
                    .entry(attr_name)
                    .or_insert_with(Vec::new)
                    .push((node_index, attr_value));
            }
        }

        // Bulk insert each attribute group
        for (attr_name, pairs) in attr_groups {
            self.bulk_set_node_attributes(&attr_name, pairs);
        }
    }

    /// Bulk set multiple edge attributes efficiently - optimized for graph creation  
    pub fn bulk_set_multiple_edge_attributes(
        &self,
        edges_attrs: Vec<(usize, HashMap<String, JsonValue>)>
    ) {
        if edges_attrs.is_empty() {
            return;
        }

        // Group by attribute name for better cache locality
        let mut attr_groups: HashMap<String, Vec<(usize, JsonValue)>> = HashMap::new();
        
        for (edge_index, attributes) in edges_attrs {
            // Update max edge index
            self.max_edge_index.fetch_max(edge_index, std::sync::atomic::Ordering::Relaxed);
            
            for (attr_name, attr_value) in attributes {
                attr_groups
                    .entry(attr_name)
                    .or_insert_with(Vec::new)
                    .push((edge_index, attr_value));
            }
        }

        // Bulk insert each attribute group
        for (attr_name, pairs) in attr_groups {
            self.bulk_set_edge_attributes(&attr_name, pairs);
        }
    }

    /// Optimized multi-criteria filtering that handles exact matches, numeric comparisons, and string comparisons
    /// All filtering and intersection logic is done in Rust to avoid Python overhead
    pub fn filter_nodes_multi_criteria(
        &self,
        exact_filters: &HashMap<String, JsonValue>,
        numeric_filters: &[(String, String, f64)], // (attr_name, operator, value)
        string_filters: &[(String, String, String)], // (attr_name, operator, value)
    ) -> Vec<usize> {
        
        // Start with exact matches if any (usually most selective)
        let mut candidates: Option<HashSet<usize>> = None;
        
        if !exact_filters.is_empty() {
            candidates = Some(self.filter_nodes_sparse(exact_filters).into_iter().collect());
            
            // Early termination if no exact matches
            if let Some(ref cands) = candidates {
                if cands.is_empty() {
                    return Vec::new();
                }
            }
        }
        
        // Apply numeric filters
        for (attr_name, operator, value) in numeric_filters {
            let matching_indices = self.filter_nodes_by_numeric_comparison(attr_name, operator, *value);
            
            if let Some(ref mut cands) = candidates {
                // Intersect with existing candidates - efficient approach
                let matching_set: HashSet<usize> = matching_indices.into_iter().collect();
                cands.retain(|&idx| matching_set.contains(&idx));
                
                // Early termination
                if cands.is_empty() {
                    return Vec::new();
                }
            } else {
                // First filter - initialize candidates
                candidates = Some(matching_indices.into_iter().collect());
            }
        }
        
        // Apply string filters
        for (attr_name, operator, value) in string_filters {
            let matching_indices = self.filter_nodes_by_string_comparison(attr_name, operator, value);
            
            if let Some(ref mut cands) = candidates {
                // Intersect with existing candidates - efficient approach
                let matching_set: HashSet<usize> = matching_indices.into_iter().collect();
                cands.retain(|&idx| matching_set.contains(&idx));
                
                // Early termination
                if cands.is_empty() {
                    return Vec::new();
                }
            } else {
                // First filter - initialize candidates
                candidates = Some(matching_indices.into_iter().collect());
            }
        }
        
        // Convert to sorted vector
        let mut result: Vec<usize> = candidates.unwrap_or_default().into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Similar multi-criteria filtering for edges
    pub fn filter_edges_multi_criteria(
        &self,
        exact_filters: &HashMap<String, JsonValue>,
        numeric_filters: &[(String, String, f64)],
        string_filters: &[(String, String, String)],
    ) -> Vec<usize> {
        
        // Start with exact matches if any (usually most selective)
        let mut candidates: Option<HashSet<usize>> = None;
        
        if !exact_filters.is_empty() {
            candidates = Some(self.filter_edges_by_attributes(exact_filters).into_iter().collect());
            
            // Early termination if no exact matches
            if let Some(ref cands) = candidates {
                if cands.is_empty() {
                    return Vec::new();
                }
            }
        }
        
        // Apply numeric filters
        for (attr_name, operator, value) in numeric_filters {
            let matching_indices = self.filter_edges_by_numeric_comparison(attr_name, operator, *value);
            
            if let Some(ref mut cands) = candidates {
                // Intersect with existing candidates
                let matching_set: HashSet<usize> = matching_indices.into_iter().collect();
                cands.retain(|&idx| matching_set.contains(&idx));
                
                // Early termination
                if cands.is_empty() {
                    return Vec::new();
                }
            } else {
                // First filter - initialize candidates
                candidates = Some(matching_indices.into_iter().collect());
            }
        }
        
        // Apply string filters
        for (attr_name, operator, value) in string_filters {
            let matching_indices = self.filter_edges_by_string_comparison(attr_name, operator, value);
            
            if let Some(ref mut cands) = candidates {
                // Intersect with existing candidates
                let matching_set: HashSet<usize> = matching_indices.into_iter().collect();
                cands.retain(|&idx| matching_set.contains(&idx));
                
                // Early termination
                if cands.is_empty() {
                    return Vec::new();
                }
            } else {
                // First filter - initialize candidates
                candidates = Some(matching_indices.into_iter().collect());
            }
        }
        
        // Convert to sorted vector
        let mut result: Vec<usize> = candidates.unwrap_or_default().into_iter().collect();
        result.sort_unstable();
        result
    }
}
