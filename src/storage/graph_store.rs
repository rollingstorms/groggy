#![allow(non_local_definitions)]
#![allow(dead_code)]

use crate::graph::types::{LegacyEdgeData, LegacyNodeData};
use crate::graph::FastGraph;
use crate::storage::{ContentHash, ContentPool};
use crate::utils::hash::fast_hash;
use dashmap::DashMap;
use parking_lot::RwLock;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

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
        let base_hash = from_hash
            .or_else(|| self.get_current_hash())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("No base state available")
            })?;

        if !self.states.contains_key(&base_hash) {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "State '{}' not found",
                base_hash
            )));
        }

        self.branches.insert(branch_name, base_hash);
        Ok(())
    }

    /// List all branches
    fn list_branches(&self) -> Vec<(String, String)> {
        self.branches
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Store a graph and return its state hash (exposed to Python)
    fn store_current_graph(&self, graph: &FastGraph, operation: String) -> String {
        self.store_graph(graph, operation)
    }

    /// Reconstruct graph from state hash (exposed to Python)
    fn get_graph_from_state(&self, state_hash: String) -> Option<FastGraph> {
        self.reconstruct_graph(&state_hash)
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
        self.branches
            .insert("main".to_string(), "initial".to_string());
    }

    /// Store a graph and return its state hash
    pub fn store_graph(&self, graph: &FastGraph, operation: String) -> String {
        let mut node_hashes = HashMap::new();
        let mut edge_hashes = HashMap::new();

        // Store all nodes in content pool
        for node_id in graph.get_node_ids() {
            // Get the actual node data from the graph
            if let Some(node_idx) = graph.node_id_to_index.get(&node_id) {
                if let Some(node_data) = graph.get_node_weight(*node_idx) {
                    // Convert new data to legacy format for content pool
                    let node_attributes =
                        graph.columnar_store.get_node_attributes(node_idx.index());
                    let legacy_node_data = LegacyNodeData {
                        id: node_data.id.clone(),
                        attributes: node_attributes,
                    };
                    let hash = self.content_pool.intern_node(legacy_node_data);
                    node_hashes.insert(node_id, hash);
                }
            }
        }

        // Store all edges in content pool
        for edge_idx in graph.get_edge_indices() {
            if let Some(edge_data) = graph.get_edge_weight(edge_idx) {
                let edge_id = format!("{}->{}", edge_data.source, edge_data.target);
                // Convert new data to legacy format for content pool
                let edge_attributes = graph.columnar_store.get_edge_attributes(edge_idx.index());
                let legacy_edge_data = LegacyEdgeData {
                    source: edge_data.source.clone(),
                    target: edge_data.target.clone(),
                    attributes: edge_attributes,
                };
                let hash = self.content_pool.intern_edge(legacy_edge_data);
                edge_hashes.insert(edge_id, hash);
            }
        }

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
        let mut graph = FastGraph::new(true); // Default to directed for now

        // Reconstruct nodes
        for (node_id, content_hash) in &state.node_hashes {
            if let Some(legacy_node_data) = self.content_pool.get_node(content_hash) {
                // Convert legacy data to new format
                let node_data = crate::graph::types::NodeData::from((*legacy_node_data).clone());
                let new_node_idx = graph.add_node_to_graph_public(node_data);
                graph.node_id_to_index.insert(node_id.clone(), new_node_idx);
                graph.node_index_to_id.insert(new_node_idx, node_id.clone());

                // Store attributes in columnar format
                for (attr_name, attr_value) in &legacy_node_data.attributes {
                    graph.columnar_store.set_node_attribute(
                        new_node_idx.index(),
                        attr_name,
                        attr_value.clone(),
                    );
                }
            }
        }

        // Reconstruct edges
        for (_edge_id, content_hash) in &state.edge_hashes {
            if let Some(legacy_edge_data) = self.content_pool.get_edge(content_hash) {
                let source_idx = *graph.node_id_to_index.get(&legacy_edge_data.source)?;
                let target_idx = *graph.node_id_to_index.get(&legacy_edge_data.target)?;
                // Convert legacy data to new format
                let edge_data = crate::graph::types::EdgeData::from((*legacy_edge_data).clone());
                let edge_idx = graph.add_edge_to_graph_public(source_idx, target_idx, edge_data);

                // Store attributes in columnar format
                for (attr_name, attr_value) in &legacy_edge_data.attributes {
                    graph.columnar_store.set_edge_attribute(
                        edge_idx.index(),
                        attr_name,
                        attr_value.clone(),
                    );
                }
            }
        }

        Some(graph)
    }
}
