use pyo3::prelude::*;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::collections::HashMap;
use crate::graph::FastGraph;
use crate::storage::{ContentPool, ContentHash};
use crate::utils::hash::fast_hash;

#[derive(Debug, Clone)]
pub struct GraphState {
    pub hash: String,
    pub parent_hash: Option<String>,
    pub operation: String,
    pub timestamp: u64,
    pub node_hashes: HashMap<String, ContentHash>,
    pub edge_hashes: HashMap<String, ContentHash>,
}

/// High-performance graph state management
#[pyclass]
pub struct GraphStore {
    states: DashMap<String, GraphState>,
    content_pool: Arc<ContentPool>,
    current_hash: RwLock<Option<String>>,
    branches: DashMap<String, String>, // branch_name -> current_hash
}

#[pymethods]
impl GraphStore {
    #[new]  
    fn new() -> Self {
        let store = Self {
            states: DashMap::new(),
            content_pool: Arc::new(ContentPool::new()),
            current_hash: RwLock::new(None),
            branches: DashMap::new(),
        };
        
        // Initialize with empty state
        store.create_initial_state();
        store
    }
    
    /// Get current graph hash
    fn get_current_hash(&self) -> Option<String> {
        self.current_hash.read().clone()
    }
    
    /// Get storage statistics
    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = self.content_pool.get_stats();
        stats.insert("total_states".to_string(), self.states.len());
        stats.insert("branches".to_string(), self.branches.len());
        stats
    }
    
    /// Create a new branch
    fn create_branch(&self, branch_name: String, from_hash: Option<String>) -> PyResult<()> {
        let base_hash = from_hash.or_else(|| self.get_current_hash())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("No base state available"))?;
        
        if !self.states.contains_key(&base_hash) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("State '{}' not found", base_hash)
            ));
        }
        
        self.branches.insert(branch_name, base_hash);
        Ok(())
    }
    
    /// List all branches
    fn list_branches(&self) -> Vec<(String, String)> {
        self.branches.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
}

impl GraphStore {
    /// Create initial empty state
    fn create_initial_state(&self) {
        let initial_state = GraphState {
            hash: "initial".to_string(),
            parent_hash: None,
            operation: "initialize".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            node_hashes: HashMap::new(),
            edge_hashes: HashMap::new(),
        };
        
        self.states.insert("initial".to_string(), initial_state);
        *self.current_hash.write() = Some("initial".to_string());
        
        // Create main branch
        self.branches.insert("main".to_string(), "initial".to_string());
    }
    
    /// Store a graph and return its state hash
    pub fn store_graph(&self, graph: &FastGraph, operation: String) -> String {
        let mut node_hashes = HashMap::new();
        let edge_hashes = HashMap::new();
        
        // Store all nodes in content pool
        for node_id in graph.get_node_ids() {
            let node_data = crate::graph::core::NodeData {
                id: node_id.clone(),
                attributes: HashMap::new(), // TODO: Convert PyObject to HashMap<String, JsonValue>
            };
            let hash = self.content_pool.intern_node(node_data);
            node_hashes.insert(node_id, hash);
        }
        
        // TODO: Store edges similarly
        
        // Create state hash
        let state_data = format!("{:?}{:?}", node_hashes, edge_hashes);
        let state_hash = format!("{:x}", fast_hash(state_data.as_bytes()));
        
        let state = GraphState {
            hash: state_hash.clone(),
            parent_hash: self.get_current_hash(),
            operation,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            node_hashes,
            edge_hashes,
        };
        
        self.states.insert(state_hash.clone(), state);
        *self.current_hash.write() = Some(state_hash.clone());
        
        state_hash
    }
    
    /// Reconstruct graph from state hash
    pub fn reconstruct_graph(&self, state_hash: &str) -> Option<FastGraph> {
        let state = self.states.get(state_hash)?;
        let mut graph = FastGraph::new();
        
        // Reconstruct nodes
        for (node_id, content_hash) in &state.node_hashes {
            if let Some(_node_data) = self.content_pool.get_node(content_hash) {
                // Convert attributes back to Python-compatible format
                graph.add_node(node_id.clone(), None).ok()?;
            }
        }
        
        // TODO: Reconstruct edges
        
        Some(graph)
    }
}
