// src/graph/native_graph.rs
//! Native graph implementation that eliminates JSON serialization

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use crate::graph::native_attributes::NativeAttributeManager;
use crate::graph::native_proxy::{NativeNodeProxy, NativeEdgeProxy};
use crate::graph::attribute_value::AttributeValue;

/// Native graph that works with Python objects directly
#[pyclass]
pub struct NativeGraph {
    /// Node storage: just IDs for now
    nodes: HashSet<String>,
    /// Edge storage: source -> target relationships
    edges: HashMap<String, HashSet<String>>,
    /// Reverse edges for efficient lookups
    reverse_edges: HashMap<String, HashSet<String>>,
    /// Unified attribute manager
    attr_manager: Arc<Mutex<NativeAttributeManager>>,
    /// Graph metadata
    directed: bool,
}

#[pymethods]
impl NativeGraph {
    #[new]
    pub fn new(directed: Option<bool>) -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            attr_manager: Arc::new(Mutex::new(NativeAttributeManager::new())),
            directed: directed.unwrap_or(true),
        }
    }
    
    /// Add a single node
    pub fn add_node(&mut self, node_id: &str) -> PyResult<()> {
        self.nodes.insert(node_id.to_string());
        Ok(())
    }
    
    /// Add multiple nodes
    pub fn add_nodes(&mut self, node_ids: Vec<&str>) -> PyResult<()> {
        for node_id in node_ids {
            self.nodes.insert(node_id.to_string());
        }
        Ok(())
    }
    
    /// Add an edge
    pub fn add_edge(&mut self, source: &str, target: &str) -> PyResult<()> {
        // Ensure nodes exist
        self.nodes.insert(source.to_string());
        self.nodes.insert(target.to_string());
        
        // Add edge
        self.edges.entry(source.to_string())
            .or_insert_with(HashSet::new)
            .insert(target.to_string());
        
        // Add reverse edge for efficient lookups
        self.reverse_edges.entry(target.to_string())
            .or_insert_with(HashSet::new)
            .insert(source.to_string());
        
        Ok(())
    }
    
    /// Add multiple edges
    pub fn add_edges(&mut self, edge_list: Vec<(String, String)>) -> PyResult<()> {
        for (source, target) in edge_list {
            self.add_edge(&source, &target)?;
        }
        Ok(())
    }
    
    /// Get node proxy
    pub fn get_node(&self, node_id: &str) -> PyResult<Option<NativeNodeProxy>> {
        if self.nodes.contains(node_id) {
            let mut proxy = NativeNodeProxy::new(node_id.to_string());
            proxy.attr_manager = self.attr_manager.clone();
            Ok(Some(proxy))
        } else {
            Ok(None)
        }
    }
    
    /// Get edge proxy
    pub fn get_edge(&self, source: &str, target: &str) -> PyResult<Option<NativeEdgeProxy>> {
        if let Some(targets) = self.edges.get(source) {
            if targets.contains(target) {
                let edge_id = format!("{}->{}", source, target);
                let mut proxy = NativeEdgeProxy::new(edge_id);
                proxy.attr_manager = self.attr_manager.clone();
                return Ok(Some(proxy));
            }
        }
        Ok(None)
    }
    
    /// Set node attributes (batch)
    pub fn set_node_attributes(&mut self, py: Python, attrs: &PyDict) -> PyResult<()> {
        let mut manager = self.attr_manager.lock().unwrap();
        manager.set_node_attributes(py, attrs)
    }
    
    /// Set edge attributes (batch)
    pub fn set_edge_attributes(&mut self, py: Python, attrs: &PyDict) -> PyResult<()> {
        let mut manager = self.attr_manager.lock().unwrap();
        manager.set_edge_attributes(py, attrs)
    }
    
    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.nodes.iter().cloned().collect()
    }
    
    /// Get all edges as (source, target) tuples
    pub fn get_edges(&self) -> Vec<(String, String)> {
        let mut edges = Vec::new();
        for (source, targets) in &self.edges {
            for target in targets {
                edges.push((source.clone(), target.clone()));
            }
        }
        edges
    }
    
    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        if let Some(neighbors) = self.edges.get(node_id) {
            Ok(neighbors.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get predecessors of a node
    pub fn get_predecessors(&self, node_id: &str) -> PyResult<Vec<String>> {
        if let Some(predecessors) = self.reverse_edges.get(node_id) {
            Ok(predecessors.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|targets| targets.len()).sum()
    }
    
    /// Check if node exists
    pub fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains(node_id)
    }
    
    /// Check if edge exists
    pub fn has_edge(&self, source: &str, target: &str) -> bool {
        if let Some(targets) = self.edges.get(source) {
            targets.contains(target)
        } else {
            false
        }
    }
    
    /// Get graph statistics
    pub fn info(&self) -> PyResult<HashMap<String, PyObject>> {
        let mut info = HashMap::new();
        Python::with_gil(|py| {
            info.insert("directed".to_string(), self.directed.to_object(py));
            info.insert("node_count".to_string(), self.node_count().to_object(py));
            info.insert("edge_count".to_string(), self.edge_count().to_object(py));
            
            // Get attribute manager stats
            let manager = self.attr_manager.lock().unwrap();
            let attr_stats = manager.get_stats();
            info.insert("attribute_memory_bytes".to_string(), 
                       attr_stats.get("memory_bytes").unwrap_or(&0).to_object(py));
            info.insert("total_node_attributes".to_string(),
                       attr_stats.get("total_node_attrs").unwrap_or(&0).to_object(py));
            info.insert("total_edge_attributes".to_string(),
                       attr_stats.get("total_edge_attrs").unwrap_or(&0).to_object(py));
        });
        Ok(info)
    }
    
    /// Clear all data
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.reverse_edges.clear();
        let mut manager = self.attr_manager.lock().unwrap();
        manager.clear();
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("NativeGraph(nodes={}, edges={}, directed={})", 
                self.node_count(), self.edge_count(), self.directed)
    }
}