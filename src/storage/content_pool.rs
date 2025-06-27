use pyo3::prelude::*;
use dashmap::DashMap;
use std::sync::Arc;
use crate::graph::types::{NodeData, EdgeData};
use crate::utils::hash::fast_hash;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ContentHash(pub u64);

/// High-performance content-addressed storage for graph components
#[pyclass]
pub struct ContentPool {
    nodes: DashMap<ContentHash, Arc<NodeData>>,
    edges: DashMap<ContentHash, Arc<EdgeData>>,
    node_refs: DashMap<ContentHash, usize>,
    edge_refs: DashMap<ContentHash, usize>,
}

#[pymethods]
impl ContentPool {
    #[new]
    pub fn new() -> Self {
        Self {
            nodes: DashMap::new(),
            edges: DashMap::new(),
            node_refs: DashMap::new(),
            edge_refs: DashMap::new(),
        }
    }
    
    /// Get storage statistics
    pub fn get_stats(&self) -> std::collections::HashMap<String, usize> {
        let mut stats = std::collections::HashMap::new();
        stats.insert("pooled_nodes".to_string(), self.nodes.len());
        stats.insert("pooled_edges".to_string(), self.edges.len());
        stats.insert("node_refs_tracked".to_string(), self.node_refs.len());
        stats.insert("edge_refs_tracked".to_string(), self.edge_refs.len());
        stats
    }
}

impl ContentPool {
    /// Hash a node for content addressing
    pub fn hash_node(node: &NodeData) -> ContentHash {
        let serialized = serde_json::to_string(node).unwrap_or_default();
        ContentHash(fast_hash(serialized.as_bytes()))
    }
    
    /// Hash an edge for content addressing
    pub fn hash_edge(edge: &EdgeData) -> ContentHash {
        let serialized = serde_json::to_string(edge).unwrap_or_default();
        ContentHash(fast_hash(serialized.as_bytes()))
    }
    
    /// Store node in pool and return its content hash
    pub fn intern_node(&self, node: NodeData) -> ContentHash {
        let hash = Self::hash_node(&node);
        let arc_node = Arc::new(node);
        
        // Insert or get existing
        self.nodes.entry(hash.clone()).or_insert(arc_node);
        
        // Increment reference count
        *self.node_refs.entry(hash.clone()).or_insert(0) += 1;
        
        hash
    }
    
    /// Store edge in pool and return its content hash
    pub fn intern_edge(&self, edge: EdgeData) -> ContentHash {
        let hash = Self::hash_edge(&edge);
        let arc_edge = Arc::new(edge);
        
        // Insert or get existing
        self.edges.entry(hash.clone()).or_insert(arc_edge);
        
        // Increment reference count
        *self.edge_refs.entry(hash.clone()).or_insert(0) += 1;
        
        hash
    }
    
    /// Get node by content hash
    pub fn get_node(&self, hash: &ContentHash) -> Option<Arc<NodeData>> {
        self.nodes.get(hash).map(|entry| entry.clone())
    }
    
    /// Get edge by content hash
    pub fn get_edge(&self, hash: &ContentHash) -> Option<Arc<EdgeData>> {
        self.edges.get(hash).map(|entry| entry.clone())
    }
    
    /// Release a node reference (for garbage collection)
    pub fn release_node(&self, hash: &ContentHash) {
        if let Some(mut entry) = self.node_refs.get_mut(hash) {
            *entry -= 1;
            let count = *entry;
            drop(entry);
            
            if count == 0 {
                self.nodes.remove(hash);
                self.node_refs.remove(hash);
            }
        }
    }
    
    /// Release an edge reference (for garbage collection)
    pub fn release_edge(&self, hash: &ContentHash) {
        if let Some(mut entry) = self.edge_refs.get_mut(hash) {
            *entry -= 1;
            let count = *entry;
            drop(entry);
            
            if count == 0 {
                self.edges.remove(hash);
                self.edge_refs.remove(hash);
            }
        }
    }
    
    /// Compact the pool by removing unreferenced items
    pub fn compact(&self) -> usize {
        let mut removed = 0;
        
        // Remove nodes with zero references
        let node_hashes_to_remove: Vec<_> = self.node_refs.iter()
            .filter(|entry| *entry.value() == 0)
            .map(|entry| entry.key().clone())
            .collect();
        
        for hash in node_hashes_to_remove {
            self.nodes.remove(&hash);
            self.node_refs.remove(&hash);
            removed += 1;
        }
        
        // Remove edges with zero references
        let edge_hashes_to_remove: Vec<_> = self.edge_refs.iter()
            .filter(|entry| *entry.value() == 0)
            .map(|entry| entry.key().clone())
            .collect();
        
        for hash in edge_hashes_to_remove {
            self.edges.remove(&hash);
            self.edge_refs.remove(&hash);
            removed += 1;
        }
        
        removed
    }
}
