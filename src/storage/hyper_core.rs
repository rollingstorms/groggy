// src/storage/hyper_core.rs  
//! Ultra-high-performance graph storage combining:
//! - Arena allocation for cache locality
//! - SIMD vectorized operations  
//! - Packed memory layouts
//! - Minimal fragmentation

use crate::storage::arena_core::{ArenaGraphCore, AttributeArena, PackedAttributeValue, ComparisonOp};
use rustc_hash::FxHashMap;
use std::sync::RwLock;

// SIMD support for vectorized operations
#[cfg(feature = "simd")]
use wide::{i64x4, f64x4, CmpEq, CmpGt, CmpLt, CmpGe, CmpLe};

/// Ultra-fast graph core with optimized memory layout + SIMD
#[derive(Debug)]
pub struct HyperGraphCore {
    arena: ArenaGraphCore,
    
    // Compact attribute registry (no string duplication)
    attr_names: RwLock<Vec<String>>,                    // attr_id -> name
    attr_name_to_id: RwLock<FxHashMap<String, u32>>,   // name -> attr_id
    
    // Node and edge ID mappings (compact)
    node_ids: RwLock<Vec<String>>,                      // node_index -> id  
    node_id_to_index: RwLock<FxHashMap<String, u32>>,  // id -> index
    edge_ids: RwLock<Vec<(u32, u32)>>,                 // edge_index -> (src, tgt)
    edge_id_to_index: RwLock<FxHashMap<(u32, u32), u32>>, // (src, tgt) -> index
}

impl HyperGraphCore {
    pub fn new() -> Self {
        Self {
            arena: ArenaGraphCore::new(),
            attr_names: RwLock::new(Vec::new()),
            attr_name_to_id: RwLock::new(FxHashMap::default()),
            node_ids: RwLock::new(Vec::new()),
            node_id_to_index: RwLock::new(FxHashMap::default()),
            edge_ids: RwLock::new(Vec::new()),
            edge_id_to_index: RwLock::new(FxHashMap::default()),
        }
    }
    
    pub fn with_capacity(nodes: usize, edges: usize, attributes: usize) -> Self {
        Self {
            arena: ArenaGraphCore::with_capacity(nodes, edges, 5), // avg 5 attrs per entity
            attr_names: RwLock::new(Vec::with_capacity(attributes)),
            attr_name_to_id: RwLock::new(FxHashMap::default()),
            node_ids: RwLock::new(Vec::with_capacity(nodes)),
            node_id_to_index: RwLock::new(FxHashMap::default()),
            edge_ids: RwLock::new(Vec::with_capacity(edges)),
            edge_id_to_index: RwLock::new(FxHashMap::default()),
        }
    }
    
    /// Add node with automatic ID assignment
    pub fn add_node(&self, node_id: String) -> u32 {
        let mut node_ids = self.node_ids.write().unwrap();
        let mut node_id_to_index = self.node_id_to_index.write().unwrap();
        
        if let Some(&index) = node_id_to_index.get(&node_id) {
            return index;
        }
        
        let index = node_ids.len() as u32;
        node_ids.push(node_id.clone());
        node_id_to_index.insert(node_id, index);
        index
    }
    
    /// Add edge with automatic ID assignment  
    pub fn add_edge(&self, src_id: String, tgt_id: String) -> u32 {
        let src_idx = self.add_node(src_id);
        let tgt_idx = self.add_node(tgt_id);
        
        let mut edge_ids = self.edge_ids.write().unwrap();
        let mut edge_id_to_index = self.edge_id_to_index.write().unwrap();
        
        let edge_key = (src_idx, tgt_idx);
        if let Some(&index) = edge_id_to_index.get(&edge_key) {
            return index;
        }
        
        let index = edge_ids.len() as u32;
        edge_ids.push(edge_key);
        edge_id_to_index.insert(edge_key, index);
        index
    }
    
    /// Get or create attribute ID
    pub fn get_attr_id(&self, attr_name: &str) -> u32 {
        {
            let attr_name_to_id = self.attr_name_to_id.read().unwrap();
            if let Some(&id) = attr_name_to_id.get(attr_name) {
                return id;
            }
        }
        
        let mut attr_names = self.attr_names.write().unwrap();
        let mut attr_name_to_id = self.attr_name_to_id.write().unwrap();
        
        // Double-check after acquiring write lock
        if let Some(&id) = attr_name_to_id.get(attr_name) {
            return id;
        }
        
        let id = attr_names.len() as u32;
        attr_names.push(attr_name.to_string());
        attr_name_to_id.insert(attr_name.to_string(), id);
        id
    }
    
    /// Set node attribute (i64) with arena allocation
    pub fn set_node_i64(&self, node_id: &str, attr_name: &str, value: i64) {
        let node_idx = self.add_node(node_id.to_string());
        let attr_id = self.get_attr_id(attr_name);
        self.arena.add_node_i64(node_idx, attr_id, value);
    }
    
    /// Set node attribute (f64) with arena allocation
    pub fn set_node_f64(&self, node_id: &str, attr_name: &str, value: f64) {
        let node_idx = self.add_node(node_id.to_string());
        let attr_id = self.get_attr_id(attr_name);
        let mut arena = self.arena.node_arena.write().unwrap();
        arena.add_f64(node_idx, attr_id, value);
    }
    
    /// HYPER-FAST filtering with arena + SIMD
    #[cfg(feature = "simd")]
    pub fn hyper_filter_nodes_i64(&self, attr_name: &str, target: i64, op: ComparisonOp) -> Vec<String> {
        let attr_id = if let Some(&id) = self.attr_name_to_id.read().unwrap().get(attr_name) {
            id
        } else {
            return Vec::new(); // Attribute doesn't exist
        };
        
        let node_arena = self.arena.node_arena.read().unwrap();
        let node_ids = self.node_ids.read().unwrap();
        
        // Get raw memory slice for SIMD processing
        let matching_indices = self.simd_filter_arena_i64(&node_arena, attr_id, target, op);
        
        // Convert indices to node IDs
        matching_indices.into_iter()
            .filter_map(|idx| node_ids.get(idx as usize).cloned())
            .collect()
    }
    
    /// Ultra-fast SIMD filtering on arena memory
    #[cfg(feature = "simd")]
    fn simd_filter_arena_i64(&self, arena: &AttributeArena, attr_id: u32, target: i64, op: ComparisonOp) -> Vec<u32> {
        let mut results = Vec::new();
        let target_simd = i64x4::splat(target);
        
        // Get attribute values (already filtered by attr_id)
        let attr_values: Vec<&PackedAttributeValue> = arena.get_attribute_values(attr_id).collect();
        
        // Process in SIMD chunks of 4
        let chunks = attr_values.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let mut values = [0i64; 4];
            let mut entity_ids = [0u32; 4];
            let mut valid_mask = [false; 4];
            
            // Extract values for SIMD processing
            for (i, &packed_val) in chunk.iter().enumerate() {
                if packed_val.value_type == 1 { // i64 type
                    values[i] = packed_val.value_data as i64;
                    entity_ids[i] = packed_val.entity_id;
                    valid_mask[i] = true;
                }
            }
            
            // SIMD comparison
            let input_simd = i64x4::from(values);
            let mask_result = match op {
                ComparisonOp::Equal => input_simd.cmp_eq(target_simd),
                ComparisonOp::Greater => input_simd.cmp_gt(target_simd),
                ComparisonOp::Less => input_simd.cmp_lt(target_simd),
                ComparisonOp::GreaterEqual => input_simd.cmp_gt(target_simd) | input_simd.cmp_eq(target_simd),
                ComparisonOp::LessEqual => input_simd.cmp_lt(target_simd) | input_simd.cmp_eq(target_simd),
                ComparisonOp::NotEqual => !input_simd.cmp_eq(target_simd),
            };
            
            let mask_array: [i64; 4] = mask_result.into();
            
            // Extract matching entity IDs
            for (i, (&is_valid, &mask_val)) in valid_mask.iter().zip(mask_array.iter()).enumerate() {
                if is_valid && mask_val != 0 {
                    results.push(entity_ids[i]);
                }
            }
        }
        
        // Handle remainder (scalar)
        for &packed_val in remainder {
            if packed_val.value_type == 1 { // i64 type
                let actual = packed_val.value_data as i64;
                let matches = match op {
                    ComparisonOp::Equal => actual == target,
                    ComparisonOp::Greater => actual > target,
                    ComparisonOp::Less => actual < target,
                    ComparisonOp::GreaterEqual => actual >= target,
                    ComparisonOp::LessEqual => actual <= target,
                    ComparisonOp::NotEqual => actual != target,
                };
                
                if matches {
                    results.push(packed_val.entity_id);
                }
            }
        }
        
        results
    }
    
    /// Fallback filtering without SIMD
    #[cfg(not(feature = "simd"))]
    pub fn hyper_filter_nodes_i64(&self, attr_name: &str, target: i64, op: ComparisonOp) -> Vec<String> {
        let attr_id = if let Some(&id) = self.attr_name_to_id.read().unwrap().get(attr_name) {
            id
        } else {
            return Vec::new();
        };
        
        let matching_indices = self.arena.filter_nodes_i64(attr_id, target, op);
        let node_ids = self.node_ids.read().unwrap();
        
        matching_indices.into_iter()
            .filter_map(|idx| node_ids.get(idx as usize).cloned())
            .collect()
    }
    
    /// Bulk add nodes with attributes (optimal for large datasets)
    pub fn bulk_add_nodes_i64(&self, nodes: Vec<(&str, &str, i64)>) {
        // Pre-allocate to avoid fragmentation
        let node_count = nodes.iter().map(|(id, _, _)| *id).collect::<std::collections::HashSet<_>>().len();
        
        {
            let mut node_ids = self.node_ids.write().unwrap();
            let mut node_id_to_index = self.node_id_to_index.write().unwrap();
            node_ids.reserve(node_count);
            node_id_to_index.reserve(node_count);
        }
        
        // Batch insert for minimal lock contention
        for (node_id, attr_name, value) in nodes {
            self.set_node_i64(node_id, attr_name, value);
        }
        
        // Optimize layout after bulk insert
        self.arena.optimize_layout();
    }
    
    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            arena_bytes: self.arena.memory_usage_bytes(),
            index_bytes: self.estimate_index_memory(),
            total_bytes: self.arena.memory_usage_bytes() + self.estimate_index_memory(),
            fragmentation_ratio: self.estimate_fragmentation(),
        }
    }
    
    fn estimate_index_memory(&self) -> usize {
        let node_ids = self.node_ids.read().unwrap();
        let attr_names = self.attr_names.read().unwrap();
        
        node_ids.capacity() * std::mem::size_of::<String>() +
        attr_names.capacity() * std::mem::size_of::<String>() +
        1024 // Rough estimate for hash map overhead
    }
    
    fn estimate_fragmentation(&self) -> f64 {
        // Low fragmentation due to arena allocation
        0.05 // 5% fragmentation estimate
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub arena_bytes: usize,
    pub index_bytes: usize, 
    pub total_bytes: usize,
    pub fragmentation_ratio: f64,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Memory Stats:\n  Arena: {:.2} MB\n  Indices: {:.2} MB\n  Total: {:.2} MB\n  Fragmentation: {:.1}%",
            self.arena_bytes as f64 / 1_000_000.0,
            self.index_bytes as f64 / 1_000_000.0, 
            self.total_bytes as f64 / 1_000_000.0,
            self.fragmentation_ratio * 100.0)
    }
}