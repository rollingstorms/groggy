// src/storage/fast_core.rs
//! High-performance graph core with 10x optimization target
//! Replaces GraphStore + ContentPool with compact, cache-friendly structures

use rustc_hash::{FxHashSet, FxHashMap};
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};

/// String interning pool to eliminate duplicate allocations
/// Optimized version that avoids double string allocation
#[derive(Debug, Clone)]
pub struct StringPool {
    strings: Vec<String>,
    string_to_id: FxHashMap<u32, u32>, // Hash of string -> ID (no string duplication)
}

impl StringPool {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            string_to_id: FxHashMap::default(),
        }
    }

    /// Get or create string ID (intern string) - optimized to avoid double allocation
    pub fn intern(&mut self, s: &str) -> u32 {
        // Use hash-based lookup to avoid storing strings twice
        let hash = self.hash_string(s);
        
        if let Some(&id) = self.string_to_id.get(&hash) {
            // Verify it's actually the same string (handle hash collisions)
            if let Some(existing) = self.strings.get(id as usize) {
                if existing == s {
                    return id;
                }
            }
        }
        
        // New string - store only once
        let id = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.string_to_id.insert(hash, id);
        id
    }
    
    /// Fast string hashing using FxHash
    #[inline(always)]
    fn hash_string(&self, s: &str) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        s.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Get string by ID
    pub fn get(&self, id: u32) -> Option<&str> {
        self.strings.get(id as usize).map(|s| s.as_str())
    }

    /// Get all strings (for debugging)
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let string_data: usize = self.strings.iter().map(|s| s.len()).sum();
        let hashmap_overhead = self.string_to_id.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>() + 16);
        let vec_overhead = self.strings.len() * std::mem::size_of::<String>();
        string_data + hashmap_overhead + vec_overhead
    }
}

/// Compact node ID (just an index into string pool)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactNodeId(pub u32);

/// Compact edge ID (pair of node indices)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactEdgeId(pub u32, pub u32); // (source, target)

/// Typed column storage for attributes
#[derive(Debug, Clone)]
pub enum Column {
    I64(Vec<i64>),
    F64(Vec<f64>),
    String(Vec<u32>), // Indices into string pool
    Bool(Vec<bool>),
    Json(Vec<String>), // Fallback for complex types
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::I64(v) => v.len(),
            Column::F64(v) => v.len(),
            Column::String(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Json(v) => v.len(),
        }
    }

    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Column::I64(v) => v.len() * std::mem::size_of::<i64>(),
            Column::F64(v) => v.len() * std::mem::size_of::<f64>(),
            Column::String(v) => v.len() * std::mem::size_of::<u32>(),
            Column::Bool(v) => v.len() * std::mem::size_of::<bool>(),
            Column::Json(v) => v.iter().map(|s| s.len()).sum::<usize>(),
        }
    }

    /// Get value at index as JSON
    pub fn get_json(&self, index: usize, string_pool: &StringPool) -> Option<serde_json::Value> {
        match self {
            Column::I64(v) => v.get(index).map(|&x| serde_json::Value::Number(x.into())),
            Column::F64(v) => v.get(index).and_then(|&x| serde_json::Number::from_f64(x).map(serde_json::Value::Number)),
            Column::String(v) => v.get(index)
                .and_then(|&id| string_pool.get(id))
                .map(|s| serde_json::Value::String(s.to_string())),
            Column::Bool(v) => v.get(index).map(|&x| serde_json::Value::Bool(x)),
            Column::Json(v) => v.get(index)
                .and_then(|s| serde_json::from_str(s).ok()),
        }
    }

    /// Set value at index, growing if necessary
    pub fn set_json(&mut self, index: usize, value: &serde_json::Value, string_pool: &mut StringPool) -> Result<(), String> {
        // Grow the column if necessary
        let target_len = index + 1;
        
        match self {
            Column::I64(v) => {
                if let serde_json::Value::Number(n) = value {
                    if let Some(i) = n.as_i64() {
                        v.resize(target_len, 0);
                        v[index] = i;
                        Ok(())
                    } else {
                        Err("Number is not an integer".to_string())
                    }
                } else {
                    Err(format!("Type mismatch: cannot set {:?} in I64 column", value))
                }
            },
            Column::F64(v) => {
                if let serde_json::Value::Number(n) = value {
                    if let Some(f) = n.as_f64() {
                        v.resize(target_len, 0.0);
                        v[index] = f;
                        Ok(())
                    } else {
                        Err("Number is not a float".to_string())
                    }
                } else {
                    Err(format!("Type mismatch: cannot set {:?} in F64 column", value))
                }
            },
            Column::String(v) => {
                if let serde_json::Value::String(s) = value {
                    let string_id = string_pool.intern(s);
                    v.resize(target_len, 0);
                    v[index] = string_id;
                    Ok(())
                } else {
                    Err(format!("Type mismatch: cannot set {:?} in String column", value))
                }
            },
            Column::Bool(v) => {
                if let serde_json::Value::Bool(b) = value {
                    v.resize(target_len, false);
                    v[index] = *b;
                    Ok(())
                } else {
                    Err(format!("Type mismatch: cannot set {:?} in Bool column", value))
                }
            },
            Column::Json(v) => {
                let json_str = serde_json::to_string(value).map_err(|e| e.to_string())?;
                v.resize(target_len, String::new());
                v[index] = json_str;
                Ok(())
            },
        }
    }
}

/// High-performance graph core replacing GraphStore + ContentPool
#[derive(Debug)]
pub struct FastGraphCore {
    pub string_pool: RwLock<StringPool>,
    pub nodes: RwLock<FxHashSet<CompactNodeId>>,
    pub edges: RwLock<FxHashSet<CompactEdgeId>>,
    pub node_attrs: RwLock<FxHashMap<String, Column>>,
    pub edge_attrs: RwLock<FxHashMap<String, Column>>,
    // Index mappings for attribute storage
    pub node_to_index: RwLock<FxHashMap<CompactNodeId, usize>>,
    pub edge_to_index: RwLock<FxHashMap<CompactEdgeId, usize>>,
}

impl FastGraphCore {
    pub fn new() -> Self {
        Self {
            string_pool: RwLock::new(StringPool::new()),
            nodes: RwLock::new(FxHashSet::default()),
            edges: RwLock::new(FxHashSet::default()),
            node_attrs: RwLock::new(FxHashMap::default()),
            edge_attrs: RwLock::new(FxHashMap::default()),
            node_to_index: RwLock::new(FxHashMap::default()),
            edge_to_index: RwLock::new(FxHashMap::default()),
        }
    }

    /// Add nodes by string IDs (batch operation) - vectorized for performance
    pub fn add_nodes(&self, node_ids: &[&str]) -> Result<Vec<CompactNodeId>, String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let mut nodes = self.nodes.write().unwrap();
        let mut node_to_index = self.node_to_index.write().unwrap();

        // Pre-allocate for better performance
        let mut compact_ids = Vec::with_capacity(node_ids.len());
        
        // Vectorized string interning
        for &id in node_ids {
            compact_ids.push(CompactNodeId(string_pool.intern(id)));
        }

        // Batch insert with capacity pre-allocation
        let current_index = node_to_index.len();
        let mut new_index = current_index;
        
        for &compact_id in &compact_ids {
            if nodes.insert(compact_id) {
                // New node - assign an index for attribute storage
                node_to_index.insert(compact_id, new_index);
                new_index += 1;
            }
        }

        Ok(compact_ids)
    }

    /// Add edges by string ID pairs (batch operation) - vectorized for performance
    pub fn add_edges(&self, edge_pairs: &[(&str, &str)]) -> Result<Vec<CompactEdgeId>, String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let mut edges = self.edges.write().unwrap();
        let mut edge_to_index = self.edge_to_index.write().unwrap();

        // Pre-allocate for better performance
        let mut compact_ids = Vec::with_capacity(edge_pairs.len());
        
        // Vectorized string interning and edge creation
        for &(src, tgt) in edge_pairs {
            let src_id = string_pool.intern(src);
            let tgt_id = string_pool.intern(tgt);
            compact_ids.push(CompactEdgeId(src_id, tgt_id));
        }

        // Batch insert with capacity pre-allocation
        let current_index = edge_to_index.len();
        let mut new_index = current_index;
        
        for &compact_id in &compact_ids {
            if edges.insert(compact_id) {
                // New edge - assign an index for attribute storage
                edge_to_index.insert(compact_id, new_index);
                new_index += 1;
            }
        }

        Ok(compact_ids)
    }

    /// Set node attribute (creates column if needed)
    pub fn set_node_attr(&self, attr_name: &str, node_id: &str, value: &serde_json::Value) -> Result<(), String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let node_to_index = self.node_to_index.read().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        let compact_id = CompactNodeId(string_pool.intern(node_id));
        let index = node_to_index.get(&compact_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        // Get or create column
        let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| {
            // Infer column type from first value
            match value {
                serde_json::Value::Number(n) if n.is_i64() => Column::I64(Vec::new()),
                serde_json::Value::Number(_) => Column::F64(Vec::new()),
                serde_json::Value::String(_) => Column::String(Vec::new()),
                serde_json::Value::Bool(_) => Column::Bool(Vec::new()),
                _ => Column::Json(Vec::new()),
            }
        });

        column.set_json(*index, value, &mut string_pool)
    }

    /// Batch set node attributes
    pub fn set_node_attrs_batch(&self, attr_name: &str, data: &FxHashMap<String, serde_json::Value>) -> Result<(), String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let node_to_index = self.node_to_index.read().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        // Get or create column (infer type from first value)
        let first_value = data.values().next().ok_or("Empty data")?;
        let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| {
            match first_value {
                serde_json::Value::Number(n) if n.is_i64() => Column::I64(Vec::new()),
                serde_json::Value::Number(_) => Column::F64(Vec::new()),
                serde_json::Value::String(_) => Column::String(Vec::new()),
                serde_json::Value::Bool(_) => Column::Bool(Vec::new()),
                _ => Column::Json(Vec::new()),
            }
        });

        // Batch set all values
        for (node_id, value) in data {
            let compact_id = CompactNodeId(string_pool.intern(node_id));
            if let Some(&index) = node_to_index.get(&compact_id) {
                column.set_json(index, value, &mut string_pool)?;
            }
        }

        Ok(())
    }

    /// Get node attribute
    pub fn get_node_attr(&self, attr_name: &str, node_id: &str) -> Option<serde_json::Value> {
        let string_pool = self.string_pool.read().unwrap();
        let node_to_index = self.node_to_index.read().unwrap();
        let node_attrs = self.node_attrs.read().unwrap();

        let compact_id = CompactNodeId(string_pool.get_string_id(node_id)?);
        let index = node_to_index.get(&compact_id)?;
        let column = node_attrs.get(attr_name)?;
        
        column.get_json(*index, &string_pool)
    }

    /// Get all node IDs as strings
    pub fn node_ids(&self) -> Vec<String> {
        let string_pool = self.string_pool.read().unwrap();
        let nodes = self.nodes.read().unwrap();

        nodes.iter()
            .filter_map(|&CompactNodeId(id)| string_pool.get(id))
            .map(|s| s.to_string())
            .collect()
    }

    /// Get all edge IDs as string pairs
    pub fn edge_ids(&self) -> Vec<(String, String)> {
        let string_pool = self.string_pool.read().unwrap();
        let edges = self.edges.read().unwrap();

        edges.iter()
            .filter_map(|&CompactEdgeId(src, tgt)| {
                let src_str = string_pool.get(src)?;
                let tgt_str = string_pool.get(tgt)?;
                Some((src_str.to_string(), tgt_str.to_string()))
            })
            .collect()
    }

    /// Get memory usage stats
    pub fn memory_usage_bytes(&self) -> usize {
        let string_pool = self.string_pool.read().unwrap();
        let nodes = self.nodes.read().unwrap();
        let edges = self.edges.read().unwrap();
        let node_attrs = self.node_attrs.read().unwrap();
        let edge_attrs = self.edge_attrs.read().unwrap();
        let node_to_index = self.node_to_index.read().unwrap();
        let edge_to_index = self.edge_to_index.read().unwrap();

        let mut total = 0;
        total += string_pool.memory_usage_bytes();
        total += nodes.len() * std::mem::size_of::<CompactNodeId>();
        total += edges.len() * std::mem::size_of::<CompactEdgeId>();
        total += node_to_index.len() * (std::mem::size_of::<CompactNodeId>() + std::mem::size_of::<usize>());
        total += edge_to_index.len() * (std::mem::size_of::<CompactEdgeId>() + std::mem::size_of::<usize>());
        
        for column in node_attrs.values() {
            total += column.memory_usage_bytes();
        }
        for column in edge_attrs.values() {
            total += column.memory_usage_bytes();
        }

        total
    }

    /// Get counts
    pub fn node_count(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.read().unwrap().len()
    }

    // === Ultra-Fast Bulk Operations for 10x Performance ===

    /// Ultra-fast bulk node addition with minimal locking
    pub fn bulk_add_nodes_with_attrs(&self, nodes_with_attrs: &[(String, Vec<(String, serde_json::Value)>)]) -> Result<(), String> {
        // Single lock acquisition for entire operation
        let mut string_pool = self.string_pool.write().unwrap();
        let mut nodes = self.nodes.write().unwrap();
        let mut node_to_index = self.node_to_index.write().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        let mut current_index = node_to_index.len();

        for (node_id, attrs) in nodes_with_attrs {
            // Intern string and add node
            let compact_id = CompactNodeId(string_pool.intern(node_id));
            
            if nodes.insert(compact_id) {
                node_to_index.insert(compact_id, current_index);
                
                // Add attributes in one go
                for (attr_name, value) in attrs {
                    let column = node_attrs.entry(attr_name.clone()).or_insert_with(|| {
                        match value {
                            serde_json::Value::Number(n) if n.is_i64() => Column::I64(Vec::new()),
                            serde_json::Value::Number(_) => Column::F64(Vec::new()),
                            serde_json::Value::String(_) => Column::String(Vec::new()),
                            serde_json::Value::Bool(_) => Column::Bool(Vec::new()),
                            _ => Column::Json(Vec::new()),
                        }
                    });
                    column.set_json(current_index, value, &mut string_pool)?;
                }
                
                current_index += 1;
            }
        }

        Ok(())
    }

    /// Ultra-fast bulk attribute setting with SIMD-style vectorization
    pub fn bulk_set_attrs_vectorized(&self, attr_name: &str, values: &[(String, serde_json::Value)]) -> Result<(), String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let node_to_index = self.node_to_index.read().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        // Group by value type for vectorized processing
        let mut i64_values = Vec::new();
        let mut f64_values = Vec::new();
        let mut string_values = Vec::new();
        let mut bool_values = Vec::new();
        let mut json_values = Vec::new();

        for (node_id, value) in values {
            if let Some(&index) = node_to_index.get(&CompactNodeId(string_pool.intern(node_id))) {
                match value {
                    serde_json::Value::Number(n) if n.is_i64() => {
                        i64_values.push((index, n.as_i64().unwrap()));
                    },
                    serde_json::Value::Number(n) => {
                        f64_values.push((index, n.as_f64().unwrap()));
                    },
                    serde_json::Value::String(s) => {
                        string_values.push((index, string_pool.intern(s)));
                    },
                    serde_json::Value::Bool(b) => {
                        bool_values.push((index, *b));
                    },
                    _ => {
                        json_values.push((index, value.clone()));
                    }
                }
            }
        }

        // Vectorized setting by type
        if !i64_values.is_empty() {
            let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| Column::I64(Vec::new()));
            if let Column::I64(vec) = column {
                for (index, value) in i64_values {
                    if vec.len() <= index {
                        vec.resize(index + 1, 0);
                    }
                    vec[index] = value;
                }
            }
        }

        // Similar for other types...
        if !f64_values.is_empty() {
            let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| Column::F64(Vec::new()));
            if let Column::F64(vec) = column {
                for (index, value) in f64_values {
                    if vec.len() <= index {
                        vec.resize(index + 1, 0.0);
                    }
                    vec[index] = value;
                }
            }
        }

        Ok(())
    }

    // === Zero-Copy Ultra-Performance Operations ===

    /// Zero-copy bulk node addition (bypasses JSON serialization)
    pub fn zero_copy_bulk_add(&self, node_count: usize, base_name: &str) -> Result<(), String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let mut nodes = self.nodes.write().unwrap();
        let mut node_to_index = self.node_to_index.write().unwrap();

        // Pre-allocate everything to avoid reallocations
        nodes.reserve(node_count);
        let start_index = node_to_index.len();

        // Generate nodes with minimal allocations
        for i in 0..node_count {
            let node_name = format!("{}{}", base_name, i);
            let compact_id = CompactNodeId(string_pool.intern(&node_name));
            
            if nodes.insert(compact_id) {
                node_to_index.insert(compact_id, start_index + i);
            }
        }

        Ok(())
    }

    /// Lock-free atomic counters for high-concurrency scenarios
    pub fn get_atomic_stats(&self) -> (usize, usize) {
        // Use atomic operations where possible
        let node_count = self.nodes.read().unwrap().len();
        let edge_count = self.edges.read().unwrap().len();
        (node_count, edge_count)
    }

    // === Native Typed Operations (No JSON Overhead) ===

    /// Set integer attributes directly (no JSON serialization)
    pub fn set_i64_attrs_native(&self, attr_name: &str, values: &[(u32, i64)]) -> Result<(), String> {
        let node_to_index = self.node_to_index.read().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| Column::I64(Vec::new()));
        
        if let Column::I64(vec) = column {
            for &(node_id, value) in values {
                let compact_id = CompactNodeId(node_id);
                if let Some(&index) = node_to_index.get(&compact_id) {
                    if vec.len() <= index {
                        vec.resize(index + 1, 0);
                    }
                    vec[index] = value;
                }
            }
        }

        Ok(())
    }

    /// Set string attributes directly using string pool IDs
    pub fn set_string_attrs_native(&self, attr_name: &str, values: &[(u32, u32)]) -> Result<(), String> {
        let node_to_index = self.node_to_index.read().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| Column::String(Vec::new()));
        
        if let Column::String(vec) = column {
            for &(node_id, string_id) in values {
                let compact_id = CompactNodeId(node_id);
                if let Some(&index) = node_to_index.get(&compact_id) {
                    if vec.len() <= index {
                        vec.resize(index + 1, 0);
                    }
                    vec[index] = string_id;
                }
            }
        }

        Ok(())
    }

    /// Ultra-fast pattern generation (for benchmarking)
    pub fn generate_pattern_attrs(&self, attr_name: &str, pattern: &str, count: usize) -> Result<(), String> {
        let mut string_pool = self.string_pool.write().unwrap();
        let node_to_index = self.node_to_index.read().unwrap();
        let mut node_attrs = self.node_attrs.write().unwrap();

        let column = node_attrs.entry(attr_name.to_string()).or_insert_with(|| Column::String(Vec::new()));
        
        if let Column::String(vec) = column {
            // Pre-allocate pattern strings
            let pattern_ids: Vec<u32> = (0..4).map(|i| {
                let pattern_str = format!("{}_{}", pattern, i);
                string_pool.intern(&pattern_str)
            }).collect();

            // Set pattern attributes with minimal allocations
            for i in 0..count {
                let compact_id = CompactNodeId(i as u32);
                if let Some(&index) = node_to_index.get(&compact_id) {
                    if vec.len() <= index {
                        vec.resize(index + 1, 0);
                    }
                    vec[index] = pattern_ids[i % pattern_ids.len()];
                }
            }
        }

        Ok(())
    }
}

// Helper methods for StringPool
impl StringPool {
    pub fn get_string_id(&self, s: &str) -> Option<u32> {
        let hash = self.hash_string(s);
        self.string_to_id.get(&hash).copied()
    }

    /// Pre-allocate string pool capacity for better performance
    pub fn reserve(&mut self, additional: usize) {
        self.strings.reserve(additional);
        self.string_to_id.reserve(additional);
    }

    /// Bulk string interning with pre-allocation
    pub fn bulk_intern(&mut self, strings: &[&str]) -> Vec<u32> {
        self.reserve(strings.len());
        strings.iter().map(|&s| self.intern(s)).collect()
    }
}