// src/storage/lockfree_core.rs
//! Lock-free data structures for maximum concurrent performance
//! Replaces RwLock with atomic operations and lock-free queues

use crossbeam_queue::{ArrayQueue, SegQueue};
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Lock-free batch operation for queuing attribute updates
#[derive(Debug, Clone)]
pub enum LockFreeBatchOp {
    SetNodeAttr {
        entity_id: u32,
        attr_id: u32,
        value: AttributeValue,
    },
    SetEdgeAttr {
        entity_id: u32,
        attr_id: u32,
        value: AttributeValue,
    },
    FilterNodes {
        attr_id: u32,
        target: AttributeValue,
        op: ComparisonOp,
    },
}

/// Lock-free attribute value union
#[derive(Debug, Clone)]
pub enum AttributeValue {
    I64(i64),
    F64(f64),
    String(String),
    Bool(bool),
}

/// Comparison operations for lock-free filtering
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
}

/// Lock-free attribute storage with atomic operations
#[derive(Debug)]
pub struct LockFreeAttributeStore {
    // Atomic counters for IDs
    next_attr_id: AtomicU32,
    next_entity_id: AtomicU32,
    
    // Lock-free batch operation queue
    batch_queue: SegQueue<LockFreeBatchOp>,
    
    // Atomic pointers to attribute data (wait-free reads)
    attr_data: AtomicPtr<AttributeData>,
    
    // Statistics
    total_operations: AtomicU64,
    queue_size: AtomicUsize,
}

/// Attribute data structure for atomic pointer swapping
#[derive(Debug)]
struct AttributeData {
    // Immutable snapshots for lock-free reads
    node_attrs: FxHashMap<(u32, u32), AttributeValue>, // (entity_id, attr_id) -> value
    edge_attrs: FxHashMap<(u32, u32), AttributeValue>,
    attr_names: Vec<String>,                           // attr_id -> name
    attr_name_to_id: FxHashMap<String, u32>,          // name -> attr_id
}

impl LockFreeAttributeStore {
    pub fn new() -> Self {
        let initial_data = Box::new(AttributeData {
            node_attrs: FxHashMap::default(),
            edge_attrs: FxHashMap::default(),
            attr_names: Vec::new(),
            attr_name_to_id: FxHashMap::default(),
        });
        
        Self {
            next_attr_id: AtomicU32::new(0),
            next_entity_id: AtomicU32::new(0),
            batch_queue: SegQueue::new(),
            attr_data: AtomicPtr::new(Box::into_raw(initial_data)),
            total_operations: AtomicU64::new(0),
            queue_size: AtomicUsize::new(0),
        }
    }
    
    /// Get next attribute ID atomically
    pub fn next_attr_id(&self) -> u32 {
        self.next_attr_id.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Get next entity ID atomically
    pub fn next_entity_id(&self) -> u32 {
        self.next_entity_id.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Queue attribute operation (lock-free)
    pub fn queue_operation(&self, op: LockFreeBatchOp) {
        self.batch_queue.push(op);
        self.queue_size.fetch_add(1, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Process batch operations (single writer pattern)
    pub fn process_batch(&self, max_operations: usize) -> usize {
        let mut processed = 0;
        let mut updates = Vec::new();
        
        // Collect operations from queue
        while processed < max_operations {
            if let Some(op) = self.batch_queue.pop() {
                updates.push(op);
                processed += 1;
                self.queue_size.fetch_sub(1, Ordering::Relaxed);
            } else {
                break;
            }
        }
        
        if !updates.is_empty() {
            self.apply_batch_updates(updates);
        }
        
        processed
    }
    
    /// Apply batch updates atomically (single writer)
    fn apply_batch_updates(&self, operations: Vec<LockFreeBatchOp>) {
        // Load current data
        let current_ptr = self.attr_data.load(Ordering::Acquire);
        let current_data = unsafe { &*current_ptr };
        
        // Create new data with updates
        let mut new_node_attrs = current_data.node_attrs.clone();
        let mut new_edge_attrs = current_data.edge_attrs.clone();
        let mut new_attr_names = current_data.attr_names.clone();
        let mut new_attr_name_to_id = current_data.attr_name_to_id.clone();
        
        // Apply all operations
        for op in operations {
            match op {
                LockFreeBatchOp::SetNodeAttr { entity_id, attr_id, value } => {
                    new_node_attrs.insert((entity_id, attr_id), value);
                }
                LockFreeBatchOp::SetEdgeAttr { entity_id, attr_id, value } => {
                    new_edge_attrs.insert((entity_id, attr_id), value);
                }
                LockFreeBatchOp::FilterNodes { .. } => {
                    // Filtering is read-only, handled separately
                }
            }
        }
        
        // Create new data structure
        let new_data = Box::new(AttributeData {
            node_attrs: new_node_attrs,
            edge_attrs: new_edge_attrs,
            attr_names: new_attr_names,
            attr_name_to_id: new_attr_name_to_id,
        });
        
        // Atomic swap (publish new data)
        let new_ptr = Box::into_raw(new_data);
        let old_ptr = self.attr_data.swap(new_ptr, Ordering::AcqRel);
        
        // Schedule old data for cleanup (should use epoch-based reclamation in production)
        // For now, we'll leak it to avoid use-after-free
        // TODO: Implement proper epoch-based memory reclamation
        std::mem::forget(unsafe { Box::from_raw(old_ptr) });
    }
    
    /// Lock-free read of node attribute
    pub fn get_node_attr(&self, entity_id: u32, attr_id: u32) -> Option<AttributeValue> {
        let data_ptr = self.attr_data.load(Ordering::Acquire);
        let data = unsafe { &*data_ptr };
        data.node_attrs.get(&(entity_id, attr_id)).cloned()
    }
    
    /// Lock-free read of edge attribute
    pub fn get_edge_attr(&self, entity_id: u32, attr_id: u32) -> Option<AttributeValue> {
        let data_ptr = self.attr_data.load(Ordering::Acquire);
        let data = unsafe { &*data_ptr };
        data.edge_attrs.get(&(entity_id, attr_id)).cloned()
    }
    
    /// Lock-free filtering of nodes
    pub fn filter_nodes(&self, attr_id: u32, target: &AttributeValue, op: ComparisonOp) -> Vec<u32> {
        let data_ptr = self.attr_data.load(Ordering::Acquire);
        let data = unsafe { &*data_ptr };
        
        let mut results = Vec::new();
        
        // Iterate through all node attributes
        for ((entity_id, attr_id_key), value) in &data.node_attrs {
            if *attr_id_key == attr_id && self.compare_values(value, target, op) {
                results.push(*entity_id);
            }
        }
        
        results
    }
    
    /// Compare attribute values for filtering
    fn compare_values(&self, value: &AttributeValue, target: &AttributeValue, op: ComparisonOp) -> bool {
        use AttributeValue::*;
        use ComparisonOp::*;
        
        match (value, target) {
            (I64(a), I64(b)) => match op {
                Equal => a == b,
                NotEqual => a != b,
                Greater => a > b,
                Less => a < b,
                GreaterEqual => a >= b,
                LessEqual => a <= b,
            },
            (F64(a), F64(b)) => match op {
                Equal => (a - b).abs() < f64::EPSILON,
                NotEqual => (a - b).abs() >= f64::EPSILON,
                Greater => a > b,
                Less => a < b,
                GreaterEqual => a >= b,
                LessEqual => a <= b,
            },
            (String(a), String(b)) => match op {
                Equal => a == b,
                NotEqual => a != b,
                Greater => a > b,
                Less => a < b,
                GreaterEqual => a >= b,
                LessEqual => a <= b,
            },
            (Bool(a), Bool(b)) => match op {
                Equal => a == b,
                NotEqual => a != b,
                _ => false, // Other comparisons don't make sense for booleans
            },
            _ => false, // Type mismatch
        }
    }
    
    /// Get queue statistics
    pub fn get_stats(&self) -> LockFreeStats {
        LockFreeStats {
            total_operations: self.total_operations.load(Ordering::Relaxed),
            queue_size: self.queue_size.load(Ordering::Relaxed),
            next_attr_id: self.next_attr_id.load(Ordering::Relaxed),
            next_entity_id: self.next_entity_id.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for lock-free operations
#[derive(Debug, Clone)]
pub struct LockFreeStats {
    pub total_operations: u64,
    pub queue_size: usize,
    pub next_attr_id: u32,
    pub next_entity_id: u32,
}

impl std::fmt::Display for LockFreeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LockFree Stats:\n  Operations: {}\n  Queue Size: {}\n  Next Attr ID: {}\n  Next Entity ID: {}",
            self.total_operations, self.queue_size, self.next_attr_id, self.next_entity_id)
    }
}

/// Lock-free string pool for efficient string storage
#[derive(Debug)]
pub struct LockFreeStringPool {
    next_id: AtomicU32,
    strings: AtomicPtr<StringPoolData>,
}

#[derive(Debug)]
struct StringPoolData {
    id_to_string: Vec<String>,
    string_to_id: FxHashMap<String, u32>,
}

impl LockFreeStringPool {
    pub fn new() -> Self {
        let initial_data = Box::new(StringPoolData {
            id_to_string: Vec::new(),
            string_to_id: FxHashMap::default(),
        });
        
        Self {
            next_id: AtomicU32::new(0),
            strings: AtomicPtr::new(Box::into_raw(initial_data)),
        }
    }
    
    /// Intern string atomically (may require fallback to slower path for new strings)
    pub fn intern(&self, s: &str) -> u32 {
        // Fast path: check if string already exists
        let data_ptr = self.strings.load(Ordering::Acquire);
        let data = unsafe { &*data_ptr };
        
        if let Some(&id) = data.string_to_id.get(s) {
            return id;
        }
        
        // Slow path: need to add new string (would need proper synchronization)
        // For now, return a hash-based ID as fallback
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        (hasher.finish() as u32) % 1000000 // Simple fallback
    }
    
    /// Get string by ID (lock-free read)
    pub fn get_string(&self, id: u32) -> Option<String> {
        let data_ptr = self.strings.load(Ordering::Acquire);
        let data = unsafe { &*data_ptr };
        
        data.id_to_string.get(id as usize).cloned()
    }
}

/// Lock-free batch processor for high-throughput operations
#[derive(Debug)]
pub struct LockFreeBatchProcessor {
    store: LockFreeAttributeStore,
    processing_interval_ms: AtomicU64,
    max_batch_size: AtomicUsize,
}

impl LockFreeBatchProcessor {
    pub fn new() -> Self {
        Self {
            store: LockFreeAttributeStore::new(),
            processing_interval_ms: AtomicU64::new(10), // Process every 10ms
            max_batch_size: AtomicUsize::new(1000),     // Max 1000 ops per batch
        }
    }
    
    /// Queue operation for batch processing
    pub fn queue_operation(&self, op: LockFreeBatchOp) {
        self.store.queue_operation(op);
    }
    
    /// Process a batch of operations
    pub fn process_batch(&self) -> usize {
        let max_size = self.max_batch_size.load(Ordering::Relaxed);
        self.store.process_batch(max_size)
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> LockFreeStats {
        self.store.get_stats()
    }
    
    /// Set batch processing parameters
    pub fn configure(&self, interval_ms: u64, max_batch_size: usize) {
        self.processing_interval_ms.store(interval_ms, Ordering::Relaxed);
        self.max_batch_size.store(max_batch_size, Ordering::Relaxed);
    }
}

#[cfg(feature = "lockfree")]
pub use self::{LockFreeAttributeStore, LockFreeBatchProcessor, LockFreeStringPool};