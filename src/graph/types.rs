// src_new/graph/types.rs
//! Core type definitions for Groggy graphs, including node/edge IDs and graph metadata.

/// Strongly-typed node identifier (string-based for generality)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct NodeId(pub String);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Strongly-typed edge identifier (tuple of source and target NodeId)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct EdgeId(pub NodeId, pub NodeId);

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}->{}", self.0, self.1)
    }
}

/// Metadata and info about the graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphInfo {
    pub name: Option<String>,
    pub directed: bool,
    pub node_count: usize,
    pub edge_count: usize,
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
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
