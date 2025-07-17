// src_new/storage/content_pool.rs
#![allow(non_local_definitions)]
// High-performance content-addressed storage for graph components
/// Content pool for high-performance batch operations
use rustc_hash::FxHashMap;
use std::sync::RwLock;
use std::sync::Arc;
use serde_json::Value;
use crate::graph::types::{NodeId, EdgeId};

// Using JSON Value as generic storage for now
pub type NodeType = Value;
pub type EdgeType = Value;
pub type ContentHash = u64;

pub struct ContentPool {
    // Optimized: Use FxHashMap + RwLock instead of DashMap for single-threaded performance
    nodes: RwLock<FxHashMap<ContentHash, Arc<NodeType>>>,
    edges: RwLock<FxHashMap<ContentHash, Arc<EdgeType>>>,
    node_refs: RwLock<FxHashMap<ContentHash, usize>>,
    edge_refs: RwLock<FxHashMap<ContentHash, usize>>,
    
    // Fast lookups for the git-like architecture
    node_id_to_hash: RwLock<FxHashMap<NodeId, ContentHash>>,
    edge_id_to_hash: RwLock<FxHashMap<EdgeId, ContentHash>>,
}


impl ContentPool {
    /// Optimized memory usage calculation
    pub fn memory_usage_bytes(&self) -> usize {
        use std::mem::size_of;
        let mut total = 0usize;
        let hashmap_overhead = 16; // FxHashMap is more efficient
        
        let nodes = self.nodes.read().unwrap();
        let edges = self.edges.read().unwrap();
        let node_refs = self.node_refs.read().unwrap();
        let edge_refs = self.edge_refs.read().unwrap();

        // More accurate accounting
        total += nodes.len() * (size_of::<ContentHash>() + size_of::<Arc<NodeType>>() + hashmap_overhead);
        total += edges.len() * (size_of::<ContentHash>() + size_of::<Arc<EdgeType>>() + hashmap_overhead);
        total += node_refs.len() * (size_of::<ContentHash>() + size_of::<usize>() + hashmap_overhead);
        total += edge_refs.len() * (size_of::<ContentHash>() + size_of::<usize>() + hashmap_overhead);

        // Fast approximation for JSON data (avoid expensive sampling)
        let avg_node_size = 50; // Reasonable estimate for node JSON
        let avg_edge_size = 80; // Reasonable estimate for edge JSON
        total += nodes.len() * avg_node_size;
        total += edges.len() * avg_edge_size;

        total
    }
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(FxHashMap::default()),
            edges: RwLock::new(FxHashMap::default()),
            node_refs: RwLock::new(FxHashMap::default()),
            edge_refs: RwLock::new(FxHashMap::default()),
            node_id_to_hash: RwLock::new(FxHashMap::default()),
            edge_id_to_hash: RwLock::new(FxHashMap::default()),
        }
    }

    /// Optimized batch node interning with minimal locking
    pub fn batch_intern_nodes(&self, nodes: Vec<NodeType>) -> Vec<ContentHash> {
        let mut nodes_map = self.nodes.write().unwrap();
        let mut node_refs_map = self.node_refs.write().unwrap();
        
        // Pre-allocate to avoid reallocations
        let mut hashes = Vec::with_capacity(nodes.len());
        
        for node in nodes {
            let hash = crate::utils::hash::hash_node(&node);
            
            // Use entry API for efficiency
            nodes_map.entry(hash).or_insert_with(|| Arc::new(node));
            *node_refs_map.entry(hash).or_insert(0) += 1;
            hashes.push(hash);
        }
        
        hashes
    }
    
    /// Optimized batch edge interning with minimal locking
    pub fn batch_intern_edges(&self, edges: Vec<EdgeType>) -> Vec<ContentHash> {
        let mut edges_map = self.edges.write().unwrap();
        let mut edge_refs_map = self.edge_refs.write().unwrap();
        
        // Pre-allocate to avoid reallocations
        let mut hashes = Vec::with_capacity(edges.len());
        
        for edge in edges {
            let hash = crate::utils::hash::hash_edge(&edge);
            
            // Use entry API for efficiency
            edges_map.entry(hash).or_insert_with(|| Arc::new(edge));
            *edge_refs_map.entry(hash).or_insert(0) += 1;
            hashes.push(hash);
        }
        
        hashes
    }
    
    /// Store node in pool and return its content hash
    pub fn intern_node(&self, node: NodeType) -> ContentHash {
        let hash = crate::utils::hash::hash_node(&node);
        let arc_node = Arc::new(node);
        
        // Single lock acquisition
        {
            let mut nodes = self.nodes.write().unwrap();
            nodes.entry(hash).or_insert(arc_node);
        }
        {
            let mut node_refs = self.node_refs.write().unwrap();
            *node_refs.entry(hash).or_insert(0) += 1;
        }
        
        hash
    }
    
    /// Store edge in pool and return its content hash
    pub fn intern_edge(&self, edge: EdgeType) -> ContentHash {
        let hash = crate::utils::hash::hash_edge(&edge);
        let arc_edge = Arc::new(edge);
        
        // Single lock acquisition  
        {
            let mut edges = self.edges.write().unwrap();
            edges.entry(hash).or_insert(arc_edge);
        }
        {
            let mut edge_refs = self.edge_refs.write().unwrap();
            *edge_refs.entry(hash).or_insert(0) += 1;
        }
        
        hash
    }
    /// Synchronizes the content pool with persistent storage.
    pub fn sync(&self) {
        // Stub: Implement backend flush/consistency logic here
    }
    /// Returns all node hashes in the pool.
    pub fn all_node_hashes(&self) -> Vec<ContentHash> {
        self.nodes.read().unwrap().keys().copied().collect()
    }
    /// Returns all edge hashes in the pool.
    pub fn all_edge_hashes(&self) -> Vec<ContentHash> {
        self.edges.read().unwrap().keys().copied().collect()
    }
    /// Returns the number of nodes in the pool.
    pub fn node_count(&self) -> usize {
        self.nodes.read().unwrap().len()
    }
    /// Returns the number of edges in the pool.
    pub fn edge_count(&self) -> usize {
        self.edges.read().unwrap().len()
    }
    /// Checks if a node exists by hash.
    pub fn has_node_hash(&self, hash: &ContentHash) -> bool {
        self.nodes.read().unwrap().contains_key(hash)
    }
    /// Checks if an edge exists by hash.
    pub fn has_edge_hash(&self, hash: &ContentHash) -> bool {
        self.edges.read().unwrap().contains_key(hash)
    }
    /// Adds nodes by hash (increments refcount or inserts if not present).
    pub fn add_node_hashes(&self, hashes: &[ContentHash]) {
        let mut nodes = self.nodes.write().unwrap();
        let mut node_refs = self.node_refs.write().unwrap();
        
        for hash in hashes {
            *node_refs.entry(*hash).or_insert(0) += 1;
            nodes.entry(*hash).or_insert_with(|| Arc::new(Value::Null));
        }
    }
    /// Adds edges by hash (increments refcount or inserts if not present).
    pub fn add_edge_hashes(&self, hashes: &[ContentHash]) {
        let mut edges = self.edges.write().unwrap();
        let mut edge_refs = self.edge_refs.write().unwrap();
        
        for hash in hashes {
            *edge_refs.entry(*hash).or_insert(0) += 1;
            edges.entry(*hash).or_insert_with(|| Arc::new(Value::Null));
        }
    }
    /// Removes nodes by hash (decrements refcount and removes if zero).
    pub fn remove_node_hashes(&self, hashes: &[ContentHash]) {
        let mut nodes = self.nodes.write().unwrap();
        let mut node_refs = self.node_refs.write().unwrap();
        
        for hash in hashes {
            if let Some(refcount) = node_refs.get_mut(hash) {
                if *refcount > 1 {
                    *refcount -= 1;
                } else {
                    node_refs.remove(hash);
                    nodes.remove(hash);
                }
            }
        }
    }
    /// Removes edges by hash (decrements refcount and removes if zero).
    pub fn remove_edge_hashes(&self, hashes: &[ContentHash]) {
        let mut edges = self.edges.write().unwrap();
        let mut edge_refs = self.edge_refs.write().unwrap();
        
        for hash in hashes {
            if let Some(refcount) = edge_refs.get_mut(hash) {
                if *refcount > 1 {
                    *refcount -= 1;
                } else {
                    edge_refs.remove(hash);
                    edges.remove(hash);
                }
            }
        }
    }

    // --- Ergonomic NodeId/EdgeId wrappers ---

    /// Returns all NodeIds in the pool (best-effort, expects stored Value to be a string or object with "id").
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        let nodes = self.nodes.read().unwrap();
        nodes.values().filter_map(|arc_value| {
            let value = arc_value.as_ref();
            if let Some(s) = value.as_str() {
                Some(NodeId(s.to_string()))
            } else if let Some(obj) = value.as_object() {
                obj.get("id").and_then(|v| v.as_str()).map(|s| NodeId(s.to_string()))
            } else {
                None
            }
        }).collect()
    }
    /// Returns all EdgeIds in the pool (best-effort, expects stored Value to be an object with "source" and "target").
    pub fn all_edge_ids(&self) -> Vec<EdgeId> {
        let edges = self.edges.read().unwrap();
        edges.values().filter_map(|arc_value| {
            let value = arc_value.as_ref();
            if let Some(obj) = value.as_object() {
                let source = obj.get("source").and_then(|v| v.as_str());
                let target = obj.get("target").and_then(|v| v.as_str());
                match (source, target) {
                    (Some(s), Some(t)) => Some(EdgeId(NodeId(s.to_string()), NodeId(t.to_string()))),
                    _ => None
                }
            } else {
                None
            }
        }).collect()
    }
    /// Checks if a node exists by NodeId.
    pub fn has_node(&self, node_id: &NodeId) -> bool {
        let hash = crate::utils::hash::hash_node(&node_id.0);
        self.has_node_hash(&hash)
    }
    /// Checks if an edge exists by EdgeId.
    pub fn has_edge(&self, edge_id: &EdgeId) -> bool {
        let hash = crate::utils::hash::hash_edge(edge_id);
        self.has_edge_hash(&hash)
    }
    /// Adds nodes by NodeId.
    pub fn add_nodes(&self, node_ids: &[NodeId]) {
        let mut nodes = self.nodes.write().unwrap();
        let mut node_refs = self.node_refs.write().unwrap();
        
        for node_id in node_ids {
            let hash = crate::utils::hash::hash_node(&node_id.0);
            *node_refs.entry(hash).or_insert(0) += 1;
            // Store the actual node ID as a JSON string value
            nodes.entry(hash).or_insert_with(|| Arc::new(Value::String(node_id.0.clone())));
        }
    }
    /// Adds edges by EdgeId.
    pub fn add_edges(&self, edge_ids: &[EdgeId]) {
        let mut edges = self.edges.write().unwrap();
        let mut edge_refs = self.edge_refs.write().unwrap();
        
        for edge_id in edge_ids {
            let hash = crate::utils::hash::hash_edge(edge_id);
            *edge_refs.entry(hash).or_insert(0) += 1;
            // Store the actual edge as JSON with source and target
            let edge_value = serde_json::json!({
                "source": edge_id.source().0,
                "target": edge_id.target().0
            });
            edges.entry(hash).or_insert_with(|| Arc::new(edge_value));
        }
    }
    /// Removes nodes by NodeId.
    pub fn remove_nodes(&self, node_ids: &[NodeId]) {
        let hashes: Vec<ContentHash> = node_ids.iter().map(|id| crate::utils::hash::hash_node(&id.0)).collect();
        self.remove_node_hashes(&hashes);
    }
    /// Removes edges by EdgeId.
    pub fn remove_edges(&self, edge_ids: &[EdgeId]) {
        let hashes: Vec<ContentHash> = edge_ids.iter().map(|id| crate::utils::hash::hash_edge(id)).collect();
        self.remove_edge_hashes(&hashes);
    }

}

