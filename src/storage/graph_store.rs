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

pub struct GraphState {
    pub hash: String,
    pub parent_hash: Option<String>,
    pub operation: String,
    pub timestamp: u64,
    pub node_hashes: HashMap<String, u64>,
    pub edge_hashes: HashMap<String, u64>,
}

pub struct GraphStore {
    states: DashMap<String, GraphState>,
    content_pool: Arc<crate::storage::content_pool::ContentPool>,
    current_hash: RwLock<Option<String>>,
    branches: DashMap<String, String>, // branch_name -> current_hash
}


impl GraphStore {
    pub fn new() -> Self {
        let store = Self {
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
        self.states.get(state_id).map(|entry| entry.clone())
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
        self.content_pool.add_nodes(node_ids)
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
        self.content_pool.add_edges(edge_ids)
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

