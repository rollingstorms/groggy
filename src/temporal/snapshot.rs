use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;

use crate::api::graph::Graph;
use crate::errors::GraphError;
use crate::state::history::{Commit, HistoryForest};
use crate::state::state::GraphSnapshot;
use crate::subgraphs::Subgraph;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};

/// Convenience index for checking whether nodes/edges existed at a snapshot.
#[derive(Debug, Clone)]
pub struct ExistenceIndex {
    nodes: HashSet<NodeId>,
    edges: HashSet<EdgeId>,
}

impl ExistenceIndex {
    pub fn new(nodes: HashSet<NodeId>, edges: HashSet<EdgeId>) -> Self {
        Self { nodes, edges }
    }

    pub fn contains_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains(&node_id)
    }

    pub fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains(&edge_id)
    }

    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    pub fn edges(&self) -> &HashSet<EdgeId> {
        &self.edges
    }
}

/// Metadata describing the lineage of a commit.
#[derive(Debug, Clone)]
pub struct LineageMetadata {
    pub commit_id: StateId,
    pub parent_commits: Vec<StateId>,
    pub message: String,
    pub author: String,
    pub timestamp: u64,
}

/// Immutable snapshot of graph state at a specific commit or timestamp.
#[derive(Debug, Clone)]
pub struct TemporalSnapshot {
    lineage: LineageMetadata,
    existence: ExistenceIndex,
    node_attributes: HashMap<NodeId, HashMap<AttrName, AttrValue>>,
    edge_attributes: HashMap<EdgeId, HashMap<AttrName, AttrValue>>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
    snapshot: GraphSnapshot,
}

impl TemporalSnapshot {
    /// Build a snapshot for a specific commit.
    pub fn at_commit(history: &HistoryForest, commit_id: StateId) -> Result<Self, GraphError> {
        let commit = history.get_commit(commit_id)?;
        let graph_snapshot = history.reconstruct_state_at(commit_id)?;
        Ok(Self::from_components(commit, graph_snapshot))
    }

    /// Build a snapshot for the latest commit at or before the timestamp.
    pub fn at_timestamp(history: &HistoryForest, timestamp: u64) -> Result<Self, GraphError> {
        if let Some(commit_id) = history.commit_at_or_before(timestamp) {
            Self::at_commit(history, commit_id)
        } else {
            Err(GraphError::InvalidInput(format!(
                "No commit found at or before timestamp {}",
                timestamp
            )))
        }
    }

    fn from_components(commit: Arc<Commit>, snapshot: GraphSnapshot) -> Self {
        let nodes: HashSet<NodeId> = snapshot.active_nodes.iter().copied().collect();
        let edges: HashSet<EdgeId> = snapshot.edges.keys().copied().collect();

        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &node in &snapshot.active_nodes {
            adjacency.entry(node).or_default();
        }
        for (&edge_id, &(source, target)) in &snapshot.edges {
            adjacency.entry(source).or_default().push(target);
            adjacency.entry(target).or_default();
            // For undirected graphs, we may need to add reverse neighbor.
            // Snapshot does not indicate graph type, so preserve outgoing edge only.
            let _ = edge_id; // silence unused warning when not compiled for undirected
        }

        let lineage = LineageMetadata {
            commit_id: commit.id,
            parent_commits: commit.parents.clone(),
            message: commit.message.clone(),
            author: commit.author.clone(),
            timestamp: commit.timestamp,
        };

        Self {
            lineage,
            existence: ExistenceIndex::new(nodes, edges),
            node_attributes: snapshot.node_attributes.clone(),
            edge_attributes: snapshot.edge_attributes.clone(),
            adjacency,
            snapshot,
        }
    }

    pub fn lineage(&self) -> &LineageMetadata {
        &self.lineage
    }

    pub fn existence(&self) -> &ExistenceIndex {
        &self.existence
    }

    pub fn node_exists(&self, node_id: NodeId) -> bool {
        self.existence.contains_node(node_id)
    }

    pub fn edge_exists(&self, edge_id: EdgeId) -> bool {
        self.existence.contains_edge(edge_id)
    }

    pub fn node_attr(&self, node_id: NodeId, attr: &AttrName) -> Option<AttrValue> {
        self.node_attributes
            .get(&node_id)
            .and_then(|attrs| attrs.get(attr).cloned())
    }

    pub fn edge_attr(&self, edge_id: EdgeId, attr: &AttrName) -> Option<AttrValue> {
        self.edge_attributes
            .get(&edge_id)
            .and_then(|attrs| attrs.get(attr).cloned())
    }

    pub fn neighbors(&self, node_id: NodeId) -> Option<&Vec<NodeId>> {
        self.adjacency.get(&node_id)
    }

    pub fn neighbors_bulk(&self, nodes: &[NodeId]) -> HashMap<NodeId, Vec<NodeId>> {
        nodes
            .iter()
            .filter_map(|node| self.neighbors(*node).map(|n| (*node, n.clone())))
            .collect()
    }

    pub fn as_subgraph(&self) -> Result<Subgraph, GraphError> {
        let graph = Graph::from_snapshot(self.snapshot.clone())?;
        let rc = Rc::new(std::cell::RefCell::new(graph));
        let nodes = self.existence.nodes().clone();
        let edges = self.existence.edges().clone();
        Ok(Subgraph::new(rc, nodes, edges, "TemporalSnapshot".into()))
    }
}
