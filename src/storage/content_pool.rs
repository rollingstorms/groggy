// src_new/storage/content_pool.rs
#![allow(non_local_definitions)]
// High-performance content-addressed storage for graph components
/// Content pool for high-performance batch operations
use dashmap::DashMap;
use std::sync::Arc;
use serde_json::Value;
use crate::graph::types::{NodeId, EdgeId};

// Using JSON Value as generic storage for now
pub type NodeType = Value;
pub type EdgeType = Value;
pub type ContentHash = u64;

pub struct ContentPool {
    nodes: DashMap<ContentHash, Arc<NodeType>>,
    edges: DashMap<ContentHash, Arc<EdgeType>>,
    node_refs: DashMap<ContentHash, usize>,
    edge_refs: DashMap<ContentHash, usize>,
}


impl ContentPool {
    /// Estimate the total heap memory usage in bytes for this ContentPool.
    pub fn memory_usage_bytes(&self) -> usize {
        use std::mem::size_of;
        let mut total = 0usize;
        // Estimate DashMap overhead per entry (very rough)
        let dashmap_entry_overhead = 32;

        // nodes: DashMap<ContentHash, Arc<NodeType>>
        total += self.nodes.len() * (size_of::<ContentHash>() + size_of::<Arc<NodeType>>() + dashmap_entry_overhead);
        // edges: DashMap<ContentHash, Arc<EdgeType>>
        total += self.edges.len() * (size_of::<ContentHash>() + size_of::<Arc<EdgeType>>() + dashmap_entry_overhead);
        // node_refs: DashMap<ContentHash, usize>
        total += self.node_refs.len() * (size_of::<ContentHash>() + size_of::<usize>() + dashmap_entry_overhead);
        // edge_refs: DashMap<ContentHash, usize>
        total += self.edge_refs.len() * (size_of::<ContentHash>() + size_of::<usize>() + dashmap_entry_overhead);

        // Approximate heap usage for the actual node/edge data (JSON Value)
        // This is expensive, so only sum the first N and extrapolate
        let sample = 10.max(self.nodes.len().min(100));
        if self.nodes.len() > 0 {
            let mut sample_size = 0usize;
            let mut count = 0usize;
            for entry in self.nodes.iter().take(sample) {
                if let Ok(json) = serde_json::to_vec(entry.value().as_ref()) {
                    sample_size += json.len();
                    count += 1;
                }
            }
            if count > 0 {
                total += sample_size * self.nodes.len() / count;
            }
        }
        if self.edges.len() > 0 {
            let mut sample_size = 0usize;
            let mut count = 0usize;
            for entry in self.edges.iter().take(sample) {
                if let Ok(json) = serde_json::to_vec(entry.value().as_ref()) {
                    sample_size += json.len();
                    count += 1;
                }
            }
            if count > 0 {
                total += sample_size * self.edges.len() / count;
            }
        }
        total
    }
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            edges: DashMap::new(),
            node_refs: DashMap::new(),
            edge_refs: DashMap::new(),
        }
    }

    /// Interns a batch of nodes in a single operation.
    pub fn batch_intern_nodes(&self, nodes: Vec<NodeType>) -> Vec<ContentHash> {
        nodes.into_iter().map(|node| self.intern_node(node)).collect()
    }
    /// Interns a batch of edges in a single operation.
    pub fn batch_intern_edges(&self, edges: Vec<EdgeType>) -> Vec<ContentHash> {
        edges.into_iter().map(|edge| self.intern_edge(edge)).collect()
    }
    /// Store node in pool and return its content hash
    pub fn intern_node(&self, node: NodeType) -> ContentHash {
        let hash = crate::utils::hash::hash_node(&node);
        let arc_node = Arc::new(node);
        self.nodes.entry(hash).or_insert(arc_node);
        *self.node_refs.entry(hash).or_insert(0) += 1;
        hash
    }
    /// Store edge in pool and return its content hash
    pub fn intern_edge(&self, edge: EdgeType) -> ContentHash {
        let hash = crate::utils::hash::hash_edge(&edge);
        let arc_edge = Arc::new(edge);
        self.edges.entry(hash).or_insert(arc_edge);
        *self.edge_refs.entry(hash).or_insert(0) += 1;
        hash
    }
    /// Synchronizes the content pool with persistent storage.
    pub fn sync(&self) {
        // Stub: Implement backend flush/consistency logic here
    }
    /// Returns all node hashes in the pool.
    pub fn all_node_hashes(&self) -> Vec<ContentHash> {
        self.nodes.iter().map(|entry| *entry.key()).collect()
    }
    /// Returns all edge hashes in the pool.
    pub fn all_edge_hashes(&self) -> Vec<ContentHash> {
        self.edges.iter().map(|entry| *entry.key()).collect()
    }
    /// Returns the number of nodes in the pool.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Returns the number of edges in the pool.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    /// Checks if a node exists by hash.
    pub fn has_node_hash(&self, hash: &ContentHash) -> bool {
        self.nodes.contains_key(hash)
    }
    /// Checks if an edge exists by hash.
    pub fn has_edge_hash(&self, hash: &ContentHash) -> bool {
        self.edges.contains_key(hash)
    }
    /// Adds nodes by hash (increments refcount or inserts if not present).
    pub fn add_node_hashes(&self, hashes: &[ContentHash]) {
        for hash in hashes {
            self.node_refs.entry(*hash).and_modify(|c| *c += 1).or_insert(1);
            self.nodes.entry(*hash).or_insert_with(|| Arc::new(Value::Null));
        }
    }
    /// Adds edges by hash (increments refcount or inserts if not present).
    pub fn add_edge_hashes(&self, hashes: &[ContentHash]) {
        for hash in hashes {
            self.edge_refs.entry(*hash).and_modify(|c| *c += 1).or_insert(1);
            self.edges.entry(*hash).or_insert_with(|| Arc::new(Value::Null));
        }
    }
    /// Removes nodes by hash (decrements refcount and removes if zero).
    pub fn remove_node_hashes(&self, hashes: &[ContentHash]) {
        for hash in hashes {
            if let Some(mut refcount) = self.node_refs.get_mut(hash) {
                if *refcount > 1 {
                    *refcount -= 1;
                } else {
                    self.node_refs.remove(hash);
                    self.nodes.remove(hash);
                }
            }
        }
    }
    /// Removes edges by hash (decrements refcount and removes if zero).
    pub fn remove_edge_hashes(&self, hashes: &[ContentHash]) {
        for hash in hashes {
            if let Some(mut refcount) = self.edge_refs.get_mut(hash) {
                if *refcount > 1 {
                    *refcount -= 1;
                } else {
                    self.edge_refs.remove(hash);
                    self.edges.remove(hash);
                }
            }
        }
    }

    // --- Ergonomic NodeId/EdgeId wrappers ---

    /// Returns all NodeIds in the pool (best-effort, expects stored Value to be a string or object with "id").
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().filter_map(|entry| {
            let value = entry.value();
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
        self.edges.iter().filter_map(|entry| {
            let value = entry.value();
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
        let hashes: Vec<ContentHash> = node_ids.iter().map(|id| crate::utils::hash::hash_node(&id.0)).collect();
        self.add_node_hashes(&hashes);
    }
    /// Adds edges by EdgeId.
    pub fn add_edges(&self, edge_ids: &[EdgeId]) {
        let hashes: Vec<ContentHash> = edge_ids.iter().map(|id| crate::utils::hash::hash_edge(id)).collect();
        self.add_edge_hashes(&hashes);
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

