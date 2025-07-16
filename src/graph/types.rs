// src_new/graph/types.rs
//! Core type definitions for Groggy graphs, including node/edge IDs and graph metadata.

use pyo3::prelude::*;

/// Strongly-typed node identifier (string-based for generality)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[pyclass]
pub struct NodeId(#[pyo3(get, set, name = "value")] pub String);

#[pymethods]
impl NodeId {
    #[new]
    pub fn new(id: String) -> Self {
        Self(id)
    }
    
    pub fn __str__(&self) -> String {
        self.0.clone()
    }
    
    pub fn __repr__(&self) -> String {
        format!("NodeId('{}')", self.0)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strongly-typed edge identifier (tuple of source and target NodeId)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[pyclass]
pub struct EdgeId(#[pyo3(get, name = "source")] pub NodeId, #[pyo3(get, name = "target")] pub NodeId);

#[pymethods]
impl EdgeId {
    #[new]
    pub fn new(source: NodeId, target: NodeId) -> Self {
        Self(source, target)
    }
    
    pub fn source(&self) -> NodeId {
        self.0.clone()
    }
    
    pub fn target(&self) -> NodeId {
        self.1.clone()
    }
    
    pub fn __str__(&self) -> String {
        format!("{}->{}", self.0, self.1)
    }
    
    pub fn __repr__(&self) -> String {
        format!("EdgeId({}, {})", self.0, self.1)
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}->{}", self.0, self.1)
    }
}

/// Metadata and info about the graph
#[pyclass]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphInfo {
    pub name: Option<String>,
    pub directed: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
}

#[pymethods]
impl GraphInfo {
    #[new]
    #[pyo3(signature = (name=None, directed=false, node_count=0, edge_count=0, attributes=std::collections::HashMap::new()))]
    pub fn new(name: Option<String>, directed: bool, node_count: usize, edge_count: usize, attributes: std::collections::HashMap<String, serde_json::Value>) -> Self {
        Self { name, directed, node_count, edge_count, attributes }
    }
    pub fn name(&self) -> Option<String> { self.name.clone() }
    pub fn directed(&self) -> bool { self.directed }
    pub fn node_count(&self) -> usize { self.node_count }
    pub fn edge_count(&self) -> usize { self.edge_count }
    pub fn attributes(&self) -> std::collections::HashMap<String, serde_json::Value> { self.attributes.clone() }
}

impl Default for GraphInfo {
    fn default() -> Self {
        Self {
            name: None,
            directed: true,
            node_count: 0,
            edge_count: 0,
            attributes: std::collections::HashMap::new(),
        }
    }
}
