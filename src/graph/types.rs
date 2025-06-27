use pyo3::prelude::*;
use petgraph::{Graph as PetGraph, Directed, Undirected};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    pub id: String,
    pub attributes: HashMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub source: String,
    pub target: String,
    pub attributes: HashMap<String, JsonValue>,
}

/// Graph type enum to support both directed and undirected graphs
#[derive(Debug, Clone)]
pub enum GraphType {
    Directed(PetGraph<NodeData, EdgeData, Directed>),
    Undirected(PetGraph<NodeData, EdgeData, Undirected>),
}

impl GraphType {
    pub fn new_directed() -> Self {
        GraphType::Directed(PetGraph::new())
    }

    pub fn new_undirected() -> Self {
        GraphType::Undirected(PetGraph::new_undirected())
    }

    pub fn node_count(&self) -> usize {
        match self {
            GraphType::Directed(g) => g.node_count(),
            GraphType::Undirected(g) => g.node_count(),
        }
    }

    pub fn edge_count(&self) -> usize {
        match self {
            GraphType::Directed(g) => g.edge_count(),
            GraphType::Undirected(g) => g.edge_count(),
        }
    }

    pub fn add_node(&mut self, node_data: NodeData) -> NodeIndex {
        match self {
            GraphType::Directed(g) => g.add_node(node_data),
            GraphType::Undirected(g) => g.add_node(node_data),
        }
    }

    pub fn add_edge(&mut self, source_idx: NodeIndex, target_idx: NodeIndex, edge_data: EdgeData) -> petgraph::graph::EdgeIndex {
        match self {
            GraphType::Directed(g) => g.add_edge(source_idx, target_idx, edge_data),
            GraphType::Undirected(g) => g.add_edge(source_idx, target_idx, edge_data),
        }
    }

    pub fn remove_node(&mut self, node_idx: NodeIndex) -> Option<NodeData> {
        match self {
            GraphType::Directed(g) => g.remove_node(node_idx),
            GraphType::Undirected(g) => g.remove_node(node_idx),
        }
    }

    pub fn remove_edge(&mut self, edge_idx: petgraph::graph::EdgeIndex) -> Option<EdgeData> {
        match self {
            GraphType::Directed(g) => g.remove_edge(edge_idx),
            GraphType::Undirected(g) => g.remove_edge(edge_idx),
        }
    }

    pub fn find_edge(&self, source_idx: NodeIndex, target_idx: NodeIndex) -> Option<petgraph::graph::EdgeIndex> {
        match self {
            GraphType::Directed(g) => g.find_edge(source_idx, target_idx),
            GraphType::Undirected(g) => g.find_edge(source_idx, target_idx),
        }
    }

    pub fn neighbors(&self, node_idx: NodeIndex) -> Vec<NodeIndex> {
        match self {
            GraphType::Directed(g) => g.neighbors(node_idx).collect(),
            GraphType::Undirected(g) => g.neighbors(node_idx).collect(),
        }
    }

    pub fn node_weight(&self, node_idx: NodeIndex) -> Option<&NodeData> {
        match self {
            GraphType::Directed(g) => g.node_weight(node_idx),
            GraphType::Undirected(g) => g.node_weight(node_idx),
        }
    }

    pub fn node_weight_mut(&mut self, node_idx: NodeIndex) -> Option<&mut NodeData> {
        match self {
            GraphType::Directed(g) => g.node_weight_mut(node_idx),
            GraphType::Undirected(g) => g.node_weight_mut(node_idx),
        }
    }

    pub fn edge_weight(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&EdgeData> {
        match self {
            GraphType::Directed(g) => g.edge_weight(edge_idx),
            GraphType::Undirected(g) => g.edge_weight(edge_idx),
        }
    }

    pub fn edge_weight_mut(&mut self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&mut EdgeData> {
        match self {
            GraphType::Directed(g) => g.edge_weight_mut(edge_idx),
            GraphType::Undirected(g) => g.edge_weight_mut(edge_idx),
        }
    }

    pub fn edge_indices(&self) -> Vec<petgraph::graph::EdgeIndex> {
        match self {
            GraphType::Directed(g) => g.edge_indices().collect(),
            GraphType::Undirected(g) => g.edge_indices().collect(),
        }
    }

    pub fn edges_directed(&self, node_idx: NodeIndex, direction: petgraph::Direction) -> Vec<petgraph::graph::EdgeReference<EdgeData>> {
        match self {
            GraphType::Directed(g) => g.edges_directed(node_idx, direction).collect(),
            GraphType::Undirected(g) => g.edges_directed(node_idx, direction).collect(),
        }
    }

    pub fn edge_endpoints(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        match self {
            GraphType::Directed(g) => g.edge_endpoints(edge_idx),
            GraphType::Undirected(g) => g.edge_endpoints(edge_idx),
        }
    }
}
