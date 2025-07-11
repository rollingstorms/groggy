use crate::storage::columnar::AttrUID;
use petgraph::graph::NodeIndex;
use petgraph::{Directed, Graph as PetGraph, Undirected};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};

/// Node data that only stores ID and attribute UIDs for columnar storage (PRIMARY)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    pub id: String,
    /// Only store the attribute UIDs, not the values
    pub attr_uids: HashSet<AttrUID>,
}

/// Edge data that only stores source/target and attribute UIDs for columnar storage (PRIMARY)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub source: String,
    pub target: String,
    /// Only store the attribute UIDs, not the values
    pub attr_uids: HashSet<AttrUID>,
}

/// Legacy types for storage compatibility only - DO NOT USE IN NEW CODE
/// These exist only to support the ContentPool storage format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyNodeData {
    pub id: String,
    pub attributes: HashMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyEdgeData {
    pub source: String,
    pub target: String,
    pub attributes: HashMap<String, JsonValue>,
}

/// Conversion utilities between legacy and new formats (for storage only)
impl From<LegacyNodeData> for NodeData {
    fn from(node_data: LegacyNodeData) -> Self {
        NodeData {
            id: node_data.id,
            attr_uids: HashSet::new(), // Will be populated by the columnar store
        }
    }
}

impl From<LegacyEdgeData> for EdgeData {
    fn from(edge_data: LegacyEdgeData) -> Self {
        EdgeData {
            source: edge_data.source,
            target: edge_data.target,
            attr_uids: HashSet::new(), // Will be populated by the columnar store
        }
    }
}

/// Graph type enum to support both directed and undirected graphs with columnar storage
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

    pub fn add_edge(
        &mut self,
        source_idx: NodeIndex,
        target_idx: NodeIndex,
        edge_data: EdgeData,
    ) -> petgraph::graph::EdgeIndex {
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

    pub fn find_edge(
        &self,
        source_idx: NodeIndex,
        target_idx: NodeIndex,
    ) -> Option<petgraph::graph::EdgeIndex> {
        match self {
            GraphType::Directed(g) => g.find_edge(source_idx, target_idx),
            GraphType::Undirected(g) => g.find_edge(source_idx, target_idx),
        }
    }

    pub fn neighbors(&self, node_idx: NodeIndex) -> Box<dyn Iterator<Item = NodeIndex> + '_> {
        match self {
            GraphType::Directed(g) => Box::new(g.neighbors(node_idx)),
            GraphType::Undirected(g) => Box::new(g.neighbors(node_idx)),
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

    pub fn edge_weight_mut(
        &mut self,
        edge_idx: petgraph::graph::EdgeIndex,
    ) -> Option<&mut EdgeData> {
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

    pub fn edges_directed(
        &self,
        node_idx: NodeIndex,
        direction: petgraph::Direction,
    ) -> Vec<petgraph::graph::EdgeReference<EdgeData>> {
        match self {
            GraphType::Directed(g) => g.edges_directed(node_idx, direction).collect(),
            GraphType::Undirected(g) => g.edges_directed(node_idx, direction).collect(),
        }
    }

    pub fn neighbors_directed(
        &self,
        node_idx: NodeIndex,
        direction: petgraph::Direction,
    ) -> Box<dyn Iterator<Item = NodeIndex> + '_> {
        match self {
            GraphType::Directed(g) => Box::new(g.neighbors_directed(node_idx, direction)),
            GraphType::Undirected(g) => Box::new(g.neighbors_directed(node_idx, direction)),
        }
    }

    pub fn edge_endpoints(
        &self,
        edge_idx: petgraph::graph::EdgeIndex,
    ) -> Option<(NodeIndex, NodeIndex)> {
        match self {
            GraphType::Directed(g) => g.edge_endpoints(edge_idx),
            GraphType::Undirected(g) => g.edge_endpoints(edge_idx),
        }
    }

    pub fn clear(&mut self) {
        match self {
            GraphType::Directed(g) => g.clear(),
            GraphType::Undirected(g) => g.clear(),
        }
    }
}
