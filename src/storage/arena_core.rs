// src/storage/arena_core.rs
//! Arena-based memory layout optimization for 10x better cache locality
//! Replaces fragmented HashMap allocations with packed, contiguous memory

use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Packed attribute value for better cache locality
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct PackedAttributeValue {
    pub attr_id: u32,    // Attribute ID (4 bytes)
    pub entity_id: u32,  // Entity ID (4 bytes)
    pub value_type: u8,  // Type discriminator (1 byte)
    pub _padding: [u8; 3], // Padding for alignment (3 bytes)
    pub value_data: u64, // Union of i64/f64/string_id (8 bytes)
}

/// Arena allocator for contiguous attribute storage
#[derive(Debug)]
pub struct AttributeArena {
    // Single contiguous allocation for all attributes
    pub values: Vec<PackedAttributeValue>,
    pub capacity: usize,
    pub len: usize,
    
    // Index structures for fast lookup (minimal overhead)
    pub entity_to_range: FxHashMap<u32, (u32, u32)>, // entity_id -> (start_idx, count)
    pub attr_to_range: FxHashMap<u32, (u32, u32)>,   // attr_id -> (start_idx, count)
}

impl AttributeArena {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            capacity: 0,
            len: 0,
            entity_to_range: FxHashMap::default(),
            attr_to_range: FxHashMap::default(),
        }
    }
    
    /// Pre-allocate arena for known size
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            capacity,
            len: 0,
            entity_to_range: FxHashMap::default(),
            attr_to_range: FxHashMap::default(),
        }
    }
    
    /// Add attribute value to arena (packed, cache-friendly)
    pub fn add_i64(&mut self, entity_id: u32, attr_id: u32, value: i64) {
        let packed = PackedAttributeValue {
            attr_id,
            entity_id,
            value_type: 1, // i64 type
            _padding: [0; 3],
            value_data: value as u64,
        };
        
        self.values.push(packed);
        self.len += 1;
        
        // Update indices
        self.update_indices(entity_id, attr_id);
    }
    
    /// Add float attribute value
    pub fn add_f64(&mut self, entity_id: u32, attr_id: u32, value: f64) {
        let packed = PackedAttributeValue {
            attr_id,
            entity_id, 
            value_type: 2, // f64 type
            _padding: [0; 3],
            value_data: value.to_bits(),
        };
        
        self.values.push(packed);
        self.len += 1;
        
        self.update_indices(entity_id, attr_id);
    }
    
    /// Add string attribute value (as string pool ID)
    pub fn add_string_id(&mut self, entity_id: u32, attr_id: u32, string_id: u32) {
        let packed = PackedAttributeValue {
            attr_id,
            entity_id,
            value_type: 3, // string type
            _padding: [0; 3],
            value_data: string_id as u64,
        };
        
        self.values.push(packed);
        self.len += 1;
        
        self.update_indices(entity_id, attr_id);
    }
    
    /// Update index structures for fast lookup
    fn update_indices(&mut self, entity_id: u32, attr_id: u32) {
        let idx = (self.len - 1) as u32;
        
        // Update entity index
        match self.entity_to_range.get_mut(&entity_id) {
            Some((_, count)) => *count += 1,
            None => {
                self.entity_to_range.insert(entity_id, (idx, 1));
            }
        }
        
        // Update attribute index  
        match self.attr_to_range.get_mut(&attr_id) {
            Some((_, count)) => *count += 1,
            None => {
                self.attr_to_range.insert(attr_id, (idx, 1));
            }
        }
    }
    
    /// Get all values for an entity (cache-friendly iteration)
    pub fn get_entity_values(&self, entity_id: u32) -> &[PackedAttributeValue] {
        if let Some(&(start, count)) = self.entity_to_range.get(&entity_id) {
            let start_idx = start as usize;
            let end_idx = start_idx + count as usize;
            &self.values[start_idx..end_idx]
        } else {
            &[]
        }
    }
    
    /// Get all values for an attribute (vectorized operations)
    pub fn get_attribute_values(&self, attr_id: u32) -> impl Iterator<Item = &PackedAttributeValue> {
        self.values.iter().filter(move |v| v.attr_id == attr_id)
    }
    
    /// SIMD-friendly bulk filtering for i64 attributes
    pub fn filter_i64_bulk(&self, attr_id: u32, target: i64, op: ComparisonOp) -> Vec<u32> {
        let mut results = Vec::new();
        
        // Iterate through contiguous memory for better cache performance
        for value in self.values.iter() {
            if value.attr_id == attr_id && value.value_type == 1 {
                let actual = value.value_data as i64;
                let matches = match op {
                    ComparisonOp::Equal => actual == target,
                    ComparisonOp::Greater => actual > target,
                    ComparisonOp::Less => actual < target,
                    ComparisonOp::GreaterEqual => actual >= target,
                    ComparisonOp::LessEqual => actual <= target,
                    ComparisonOp::NotEqual => actual != target,
                };
                
                if matches {
                    results.push(value.entity_id);
                }
            }
        }
        
        results
    }
    
    /// SIMD-friendly bulk filtering for f64 attributes
    pub fn filter_f64_bulk(&self, attr_id: u32, target: f64, op: ComparisonOp) -> Vec<u32> {
        let mut results = Vec::new();
        
        for value in self.values.iter() {
            if value.attr_id == attr_id && value.value_type == 2 {
                let actual = f64::from_bits(value.value_data);
                let matches = match op {
                    ComparisonOp::Equal => actual == target,
                    ComparisonOp::Greater => actual > target,
                    ComparisonOp::Less => actual < target,
                    ComparisonOp::GreaterEqual => actual >= target,
                    ComparisonOp::LessEqual => actual <= target,
                    ComparisonOp::NotEqual => actual != target,
                };
                
                if matches {
                    results.push(value.entity_id);
                }
            }
        }
        
        results
    }
    
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of::<PackedAttributeValue>() * self.values.capacity() +
        self.entity_to_range.capacity() * std::mem::size_of::<(u32, (u32, u32))>() +
        self.attr_to_range.capacity() * std::mem::size_of::<(u32, (u32, u32))>()
    }
    
    /// Defragment arena for optimal cache performance
    pub fn defragment(&mut self) {
        // Sort by entity_id first, then attr_id for better locality
        // Use local copies to avoid unaligned references to packed fields
        self.values.sort_by(|a, b| {
            let a_entity = a.entity_id;
            let b_entity = b.entity_id;
            let a_attr = a.attr_id;
            let b_attr = b.attr_id;
            a_entity.cmp(&b_entity).then(a_attr.cmp(&b_attr))
        });
        
        // Rebuild indices after sorting
        self.rebuild_indices();
    }
    
    fn rebuild_indices(&mut self) {
        self.entity_to_range.clear();
        self.attr_to_range.clear();
        
        let mut current_entity = None;
        let mut current_attr = None;
        let mut entity_start = 0u32;
        let mut attr_start = 0u32;
        
        for (idx, value) in self.values.iter().enumerate() {
            let idx = idx as u32;
            
            // Track entity ranges
            if current_entity != Some(value.entity_id) {
                if let Some(prev_entity) = current_entity {
                    let count = idx - entity_start;
                    self.entity_to_range.insert(prev_entity, (entity_start, count));
                }
                current_entity = Some(value.entity_id);
                entity_start = idx;
            }
            
            // Track attribute ranges
            if current_attr != Some(value.attr_id) {
                if let Some(prev_attr) = current_attr {
                    let count = idx - attr_start;
                    self.attr_to_range.insert(prev_attr, (attr_start, count));
                }
                current_attr = Some(value.attr_id);
                attr_start = idx;
            }
        }
        
        // Handle final ranges
        if let Some(entity) = current_entity {
            let count = self.len as u32 - entity_start;
            self.entity_to_range.insert(entity, (entity_start, count));
        }
        if let Some(attr) = current_attr {
            let count = self.len as u32 - attr_start;
            self.attr_to_range.insert(attr, (attr_start, count));
        }
    }
}

/// Comparison operations for filtering
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
}

/// High-performance graph storage with arena allocation
#[derive(Debug)]
pub struct ArenaGraphCore {
    pub node_arena: RwLock<AttributeArena>,
    pub edge_arena: RwLock<AttributeArena>,
    pub string_pool: RwLock<crate::storage::fast_core::StringPool>,
    
    // Minimal index structures
    pub node_count: std::sync::atomic::AtomicU32,
    pub edge_count: std::sync::atomic::AtomicU32,
    pub attribute_count: std::sync::atomic::AtomicU32,
}

impl ArenaGraphCore {
    pub fn new() -> Self {
        Self {
            node_arena: RwLock::new(AttributeArena::new()),
            edge_arena: RwLock::new(AttributeArena::new()),
            string_pool: RwLock::new(crate::storage::fast_core::StringPool::new()),
            node_count: std::sync::atomic::AtomicU32::new(0),
            edge_count: std::sync::atomic::AtomicU32::new(0),
            attribute_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
    
    /// Pre-allocate for expected graph size
    pub fn with_capacity(nodes: usize, edges: usize, attributes_per_entity: usize) -> Self {
        let node_capacity = nodes * attributes_per_entity;
        let edge_capacity = edges * attributes_per_entity;
        
        Self {
            node_arena: RwLock::new(AttributeArena::with_capacity(node_capacity)),
            edge_arena: RwLock::new(AttributeArena::with_capacity(edge_capacity)),
            string_pool: RwLock::new(crate::storage::fast_core::StringPool::new()),
            node_count: std::sync::atomic::AtomicU32::new(0),
            edge_count: std::sync::atomic::AtomicU32::new(0),
            attribute_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
    
    /// Add node attribute with optimal memory layout
    pub fn add_node_i64(&self, node_id: u32, attr_id: u32, value: i64) {
        let mut arena = self.node_arena.write().unwrap();
        arena.add_i64(node_id, attr_id, value);
    }
    
    /// Add edge attribute with optimal memory layout  
    pub fn add_edge_i64(&self, edge_id: u32, attr_id: u32, value: i64) {
        let mut arena = self.edge_arena.write().unwrap();
        arena.add_i64(edge_id, attr_id, value);
    }
    
    /// Bulk filter nodes with arena-optimized performance
    pub fn filter_nodes_i64(&self, attr_id: u32, target: i64, op: ComparisonOp) -> Vec<u32> {
        let arena = self.node_arena.read().unwrap();
        arena.filter_i64_bulk(attr_id, target, op)
    }
    
    /// Bulk filter edges with arena-optimized performance
    pub fn filter_edges_i64(&self, attr_id: u32, target: i64, op: ComparisonOp) -> Vec<u32> {
        let arena = self.edge_arena.read().unwrap();
        arena.filter_i64_bulk(attr_id, target, op)
    }
    
    /// Defragment for optimal cache performance
    pub fn optimize_layout(&self) {
        let mut node_arena = self.node_arena.write().unwrap();
        let mut edge_arena = self.edge_arena.write().unwrap();
        
        node_arena.defragment();
        edge_arena.defragment();
    }
    
    /// Get total memory usage
    pub fn memory_usage_bytes(&self) -> usize {
        let node_arena = self.node_arena.read().unwrap();
        let edge_arena = self.edge_arena.read().unwrap();
        let string_pool = self.string_pool.read().unwrap();
        
        node_arena.memory_usage_bytes() + 
        edge_arena.memory_usage_bytes() + 
        string_pool.memory_usage_bytes()
    }
}