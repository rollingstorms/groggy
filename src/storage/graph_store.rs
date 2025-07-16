// src_new/storage/graph_store.rs
#![allow(non_local_definitions)]
#![allow(dead_code)]
#![allow(clippy::uninlined_format_args)]
// High-performance persistent graph state and branch management
/// Persistent graph storage backend
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct GraphState {
    pub hash: String,
    pub parent_hash: Option<String>,
    pub operation: String,
    pub timestamp: u64,
    pub node_hashes: HashMap<String, u64>,
    pub edge_hashes: HashMap<String, u64>,
}

pub struct GraphStore {
    pub node_id_to_index: DashMap<crate::graph::types::NodeId, usize>,
    pub index_to_node_id: DashMap<usize, crate::graph::types::NodeId>,
    pub edge_id_to_index: DashMap<crate::graph::types::EdgeId, usize>,
    pub index_to_edge_id: DashMap<usize, crate::graph::types::EdgeId>,
    states: DashMap<String, GraphState>,
    content_pool: Arc<crate::storage::content_pool::ContentPool>,
    current_hash: RwLock<Option<String>>,
    branches: DashMap<String, String>, // branch_name -> current_hash
}


impl GraphStore {
    /// Estimate the total heap memory usage in bytes for this GraphStore (including ContentPool).
    pub fn memory_usage_bytes(&self) -> usize {
        use std::mem::size_of;
        let dashmap_entry_overhead = 32;
        let mut total = 0usize;
        total += self.node_id_to_index.len() * (size_of::<crate::graph::types::NodeId>() + size_of::<usize>() + dashmap_entry_overhead);
        total += self.index_to_node_id.len() * (size_of::<usize>() + size_of::<crate::graph::types::NodeId>() + dashmap_entry_overhead);
        total += self.edge_id_to_index.len() * (size_of::<crate::graph::types::EdgeId>() + size_of::<usize>() + dashmap_entry_overhead);
        total += self.index_to_edge_id.len() * (size_of::<usize>() + size_of::<crate::graph::types::EdgeId>() + dashmap_entry_overhead);
        total += self.states.len() * (size_of::<String>() + size_of::<GraphState>() + dashmap_entry_overhead);
        total += self.branches.len() * (size_of::<String>() * 2 + dashmap_entry_overhead);
        // RwLock and Arc overheads are ignored for simplicity
        // Add content_pool
        total += self.content_pool.memory_usage_bytes();
        total
    }

    /// Returns the estimated heap memory usage in bytes for the content pool only.
    pub fn content_pool_memory_usage_bytes(&self) -> usize {
        self.content_pool.memory_usage_bytes()
    }
    pub fn node_index(&self, id: &crate::graph::types::NodeId) -> Option<usize> {
        self.node_id_to_index.get(id).map(|v| *v)
    }
    pub fn node_id(&self, idx: usize) -> Option<crate::graph::types::NodeId> {
        self.index_to_node_id.get(&idx).map(|v| v.clone())
    }
    pub fn edge_index(&self, id: &crate::graph::types::EdgeId) -> Option<usize> {
        self.edge_id_to_index.get(id).map(|v| *v)
    }
    pub fn edge_id(&self, idx: usize) -> Option<crate::graph::types::EdgeId> {
        self.index_to_edge_id.get(&idx).map(|v| v.clone())
    }
    pub fn new() -> Self {
        let store = Self {
            node_id_to_index: DashMap::new(),
            index_to_node_id: DashMap::new(),
            edge_id_to_index: DashMap::new(),
            index_to_edge_id: DashMap::new(),
            states: DashMap::new(),
            content_pool: Arc::new(crate::storage::content_pool::ContentPool::new()),
            current_hash: RwLock::new(None),
            branches: DashMap::new(),
        };
        store.create_initial_state();
        store
    }

    /// Saves the current graph state to persistent storage.
    pub fn save_state(&self, node_hashes: HashMap<String, u64>, edge_hashes: HashMap<String, u64>, operation: String) -> String {
        use crate::utils::hash::hash_node;
        let state_data = format!("{:?}{:?}", node_hashes, edge_hashes);
        let state_hash = format!("{:x}", crate::utils::hash::hash_node(&state_data));
        let state = GraphState {
            hash: state_hash.clone(),
            parent_hash: self.current_hash.read().clone(),
            operation,
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            node_hashes,
            edge_hashes,
        };
        self.states.insert(state_hash.clone(), state);
        *self.current_hash.write() = Some(state_hash.clone());
        state_hash
    }
    /// Loads a graph state by its ID from persistent storage.
    pub fn load_state(&self, state_id: &str) -> Option<GraphState> {
        self.states.get(state_id).map(|entry| entry.value().clone())
    }
    /// Creates a new branch from the current graph state.
    pub fn branch(&self, branch_name: String, from_hash: Option<String>) {
        let base_hash = from_hash.or_else(|| self.current_hash.read().clone());
        if let Some(hash) = base_hash {
            self.branches.insert(branch_name, hash);
        }
    }
    /// Lists all available graph states and branches in storage.
    pub fn list_branches(&self) -> Vec<(String, String)> {
        self.branches.iter().map(|entry| (entry.key().clone(), entry.value().clone())).collect()
    }
    /// Switches to a different graph state or branch by ID.
    pub fn switch_branch(&self, state_id: &str) {
        if self.states.contains_key(state_id) {
            *self.current_hash.write() = Some(state_id.to_string());
        }
    }
    /// Deletes a branch by its ID.
    pub fn delete_branch(&self, branch_name: &str) {
        self.branches.remove(branch_name);
    }
    fn create_initial_state(&self) {
        let initial_state = GraphState {
            hash: "initial".to_string(),
            parent_hash: None,
            operation: "initialize".to_string(),
            timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            node_hashes: HashMap::new(),
            edge_hashes: HashMap::new(),
        };
        self.states.insert("initial".to_string(), initial_state);
        *self.current_hash.write() = Some("initial".to_string());
        self.branches.insert("main".to_string(), "initial".to_string());
    }

    // --- Node/Edge management ergonomic pass-throughs ---
    pub fn all_node_ids(&self) -> Vec<crate::graph::types::NodeId> {
        self.content_pool.all_node_ids()
    }
    pub fn add_nodes(&self, node_ids: &[crate::graph::types::NodeId]) {
        self.content_pool.add_nodes(node_ids);
        // Also populate the index maps for attribute storage
        for node_id in node_ids {
            if !self.node_id_to_index.contains_key(node_id) {
                let next_index = self.node_id_to_index.len();
                self.node_id_to_index.insert(node_id.clone(), next_index);
                self.index_to_node_id.insert(next_index, node_id.clone());
            }
        }
    }
    pub fn remove_nodes(&self, node_ids: &[crate::graph::types::NodeId]) {
        self.content_pool.remove_nodes(node_ids)
    }
    pub fn node_count(&self) -> usize {
        self.content_pool.node_count()
    }
    pub fn has_node(&self, node_id: &crate::graph::types::NodeId) -> bool {
        self.content_pool.has_node(node_id)
    }
    pub fn all_edge_ids(&self) -> Vec<crate::graph::types::EdgeId> {
        self.content_pool.all_edge_ids()
    }
    pub fn add_edges(&self, edge_ids: &[crate::graph::types::EdgeId]) {
        self.content_pool.add_edges(edge_ids);
        // Also populate the index maps for attribute storage  
        for edge_id in edge_ids {
            if !self.edge_id_to_index.contains_key(edge_id) {
                let next_index = self.edge_id_to_index.len();
                self.edge_id_to_index.insert(edge_id.clone(), next_index);
                self.index_to_edge_id.insert(next_index, edge_id.clone());
            }
        }
    }
    pub fn remove_edges(&self, edge_ids: &[crate::graph::types::EdgeId]) {
        self.content_pool.remove_edges(edge_ids)
    }
    pub fn edge_count(&self) -> usize {
        self.content_pool.edge_count()
    }
    pub fn has_edge(&self, edge_id: &crate::graph::types::EdgeId) -> bool {
        self.content_pool.has_edge(edge_id)
    }
}

