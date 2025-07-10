use petgraph::{Graph as PetGraph, Directed, Undirected};
use petgraph::graph::NodeIndex;
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

/// Lightweight node data that only stores ID and attribute UIDs for hybrid/columnar storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridNodeData {
    pub id: String,
    /// Only store the attribute UIDs, not the values
    pub attr_uids: HashSet<u64>,
}

/// Lightweight edge data that only stores source/target and attribute UIDs for hybrid/columnar storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEdgeData {
    pub source: String,
    pub target: String,
    /// Only store the attribute UIDs, not the values
    pub attr_uids: HashSet<u64>,
}

/// Graph type enum to support both directed and undirected graphs with hybrid/columnar storage
#[derive(Debug, Clone)]
pub enum HybridGraphType {
    Directed(PetGraph<HybridNodeData, HybridEdgeData, Directed>),
    Undirected(PetGraph<HybridNodeData, HybridEdgeData, Undirected>),
}

impl HybridGraphType {
    pub fn new_directed() -> Self {
        HybridGraphType::Directed(PetGraph::new())
    }

    pub fn new_undirected() -> Self {
        HybridGraphType::Undirected(PetGraph::new_undirected())
    }

    pub fn node_count(&self) -> usize {
        match self {
            HybridGraphType::Directed(g) => g.node_count(),
            HybridGraphType::Undirected(g) => g.node_count(),
        }
    }

    pub fn edge_count(&self) -> usize {
        match self {
            HybridGraphType::Directed(g) => g.edge_count(),
            HybridGraphType::Undirected(g) => g.edge_count(),
        }
    }

    pub fn add_node(&mut self, node_data: HybridNodeData) -> NodeIndex {
        match self {
            HybridGraphType::Directed(g) => g.add_node(node_data),
            HybridGraphType::Undirected(g) => g.add_node(node_data),
        }
    }

    pub fn add_edge(&mut self, source_idx: NodeIndex, target_idx: NodeIndex, edge_data: HybridEdgeData) -> petgraph::graph::EdgeIndex {
        match self {
            HybridGraphType::Directed(g) => g.add_edge(source_idx, target_idx, edge_data),
            HybridGraphType::Undirected(g) => g.add_edge(source_idx, target_idx, edge_data),
        }
    }

    pub fn remove_node(&mut self, node_idx: NodeIndex) -> Option<HybridNodeData> {
        match self {
            HybridGraphType::Directed(g) => g.remove_node(node_idx),
            HybridGraphType::Undirected(g) => g.remove_node(node_idx),
        }
    }

    pub fn remove_edge(&mut self, edge_idx: petgraph::graph::EdgeIndex) -> Option<HybridEdgeData> {
        match self {
            HybridGraphType::Directed(g) => g.remove_edge(edge_idx),
            HybridGraphType::Undirected(g) => g.remove_edge(edge_idx),
        }
    }

    pub fn find_edge(&self, source_idx: NodeIndex, target_idx: NodeIndex) -> Option<petgraph::graph::EdgeIndex> {
        match self {
            HybridGraphType::Directed(g) => g.find_edge(source_idx, target_idx),
            HybridGraphType::Undirected(g) => g.find_edge(source_idx, target_idx),
        }
    }

    pub fn neighbors(&self, node_idx: NodeIndex) -> Vec<NodeIndex> {
        match self {
            HybridGraphType::Directed(g) => g.neighbors(node_idx).collect(),
            HybridGraphType::Undirected(g) => g.neighbors(node_idx).collect(),
        }
    }

    pub fn node_weight(&self, node_idx: NodeIndex) -> Option<&HybridNodeData> {
        match self {
            HybridGraphType::Directed(g) => g.node_weight(node_idx),
            HybridGraphType::Undirected(g) => g.node_weight(node_idx),
        }
    }

    pub fn node_weight_mut(&mut self, node_idx: NodeIndex) -> Option<&mut HybridNodeData> {
        match self {
            HybridGraphType::Directed(g) => g.node_weight_mut(node_idx),
            HybridGraphType::Undirected(g) => g.node_weight_mut(node_idx),
        }
    }

    pub fn edge_weight(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&HybridEdgeData> {
        match self {
            HybridGraphType::Directed(g) => g.edge_weight(edge_idx),
            HybridGraphType::Undirected(g) => g.edge_weight(edge_idx),
        }
    }

    pub fn edge_weight_mut(&mut self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&mut HybridEdgeData> {
        match self {
            HybridGraphType::Directed(g) => g.edge_weight_mut(edge_idx),
            HybridGraphType::Undirected(g) => g.edge_weight_mut(edge_idx),
        }
    }

    pub fn edge_indices(&self) -> Vec<petgraph::graph::EdgeIndex> {
        match self {
            HybridGraphType::Directed(g) => g.edge_indices().collect(),
            HybridGraphType::Undirected(g) => g.edge_indices().collect(),
        }
    }

    pub fn edges_directed(&self, node_idx: NodeIndex, direction: petgraph::Direction) -> Vec<petgraph::graph::EdgeReference<HybridEdgeData>> {
        match self {
            HybridGraphType::Directed(g) => g.edges_directed(node_idx, direction).collect(),
            HybridGraphType::Undirected(g) => g.edges_directed(node_idx, direction).collect(),
        }
    }

    pub fn neighbors_directed(&self, node_idx: NodeIndex, direction: petgraph::Direction) -> Vec<NodeIndex> {
        match self {
            HybridGraphType::Directed(g) => g.neighbors_directed(node_idx, direction).collect(),
            HybridGraphType::Undirected(g) => g.neighbors_directed(node_idx, direction).collect(),
        }
    }

    pub fn edge_endpoints(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<(NodeIndex, NodeIndex)> {
        match self {
            HybridGraphType::Directed(g) => g.edge_endpoints(edge_idx),
            HybridGraphType::Undirected(g) => g.edge_endpoints(edge_idx),
        }
    }

    pub fn clear(&mut self) {
        match self {
            HybridGraphType::Directed(g) => g.clear(),
            HybridGraphType::Undirected(g) => g.clear(),
        }
    }
}
