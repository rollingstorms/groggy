use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;

use crate::api::graph::Graph;
use crate::errors::GraphError;
use crate::state::history::{Commit, HistoryForest};
use crate::state::state::GraphSnapshot;
use crate::subgraphs::Subgraph;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId, StateId};

pub use crate::state::read_state::ExistenceIndex;
use crate::state::PinnedReadState;

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
    history_identity: Arc<()>,
    lineage: LineageMetadata,
    state: PinnedReadState,
    materialization: Option<Arc<crate::api::graph::snapshot::SnapshotMaterialization>>,
}

impl TemporalSnapshot {
    /// Build a snapshot for a specific commit.
    pub fn at_commit(history: &HistoryForest, commit_id: StateId) -> Result<Self, GraphError> {
        let commit = history.get_commit(commit_id)?;
        let graph_snapshot = history.reconstruct_state_at(commit_id)?;
        Self::from_components(commit, graph_snapshot, Arc::clone(history.identity()))
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

    fn from_components(
        commit: Arc<Commit>,
        snapshot: GraphSnapshot,
        history_identity: Arc<()>,
    ) -> Result<Self, GraphError> {
        let lineage = LineageMetadata {
            commit_id: commit.id,
            parent_commits: commit.parents.clone(),
            message: commit.message.clone(),
            author: commit.author.clone(),
            timestamp: commit.timestamp,
        };

        Ok(Self {
            history_identity,
            lineage,
            state: PinnedReadState::from_snapshot(snapshot)?,
            materialization: None,
        })
    }

    pub(crate) fn with_materialization(
        mut self,
        context: Arc<crate::api::graph::snapshot::SnapshotMaterialization>,
    ) -> Self {
        self.materialization = Some(context);
        self
    }

    /// Whether this capture explicitly owns full backing storage and retained history.
    pub fn is_owned(&self) -> bool {
        self.materialization
            .as_ref()
            .is_some_and(|context| context.is_owned())
    }

    /// Configuration captured from the graph owner, if constructed through Graph.
    pub fn configuration(&self) -> Option<&crate::GraphConfig> {
        self.materialization
            .as_ref()
            .map(|context| context.configuration())
    }

    /// Restore a full explicitly owned capture into an independent writable graph.
    pub fn to_graph(&self) -> Result<Graph, GraphError> {
        self.materialization
            .as_ref()
            .ok_or_else(|| {
                GraphError::InvalidInput(
                    "Snapshot has no owned backing context; capture an owned snapshot first".into(),
                )
            })?
            .restore_owned(&self.state)
    }

    pub(crate) fn with_groups(
        mut self,
        groups: Arc<
            std::collections::HashMap<
                crate::types::SubgraphId,
                crate::storage::retained_group::RetainedGroup,
            >,
        >,
    ) -> Self {
        self.state = self.state.with_groups(groups);
        self
    }
    pub(crate) fn same_history(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.history_identity, &other.history_identity)
    }

    pub(crate) fn belongs_to_history(&self, history: &HistoryForest) -> bool {
        Arc::ptr_eq(&self.history_identity, history.identity())
    }

    pub fn lineage(&self) -> &LineageMetadata {
        &self.lineage
    }

    pub fn existence(&self) -> &ExistenceIndex {
        self.state.existence()
    }

    pub fn node_exists(&self, node_id: NodeId) -> bool {
        self.state.existence().contains_node(node_id)
    }

    pub fn edge_exists(&self, edge_id: EdgeId) -> bool {
        self.state.existence().contains_edge(edge_id)
    }

    /// Find a node by its external identity in this immutable state (a snapshot scan).
    pub fn node_by_external_id(&self, external_id: &str, group_id: Option<&str>) -> Option<NodeId> {
        self.state
            .snapshot()
            .node_attributes
            .iter()
            .find_map(|(&id, attrs)| {
                (self.node_exists(id) && Self::identity_matches(attrs, external_id, group_id))
                    .then_some(id)
            })
    }

    /// Find an edge by its external identity in this immutable state (a snapshot scan).
    pub fn edge_by_external_id(&self, external_id: &str, group_id: Option<&str>) -> Option<EdgeId> {
        self.state
            .snapshot()
            .edge_attributes
            .iter()
            .find_map(|(&id, attrs)| {
                (self.edge_exists(id) && Self::identity_matches(attrs, external_id, group_id))
                    .then_some(id)
            })
    }

    fn identity_matches(
        attrs: &HashMap<AttrName, AttrValue>,
        name: &str,
        group: Option<&str>,
    ) -> bool {
        attrs
            .get(crate::EXTERNAL_ID)
            .and_then(AttrValue::get_text)
            .as_deref()
            == Some(name)
            && group.map_or(true, |group| {
                attrs
                    .get("group_id")
                    .and_then(AttrValue::get_text)
                    .as_deref()
                    == Some(group)
            })
    }

    pub fn node_attr(&self, node_id: NodeId, attr: &AttrName) -> Option<AttrValue> {
        self.state.node_attr(node_id, attr)
    }
    pub fn edge_attr(&self, edge_id: EdgeId, attr: &AttrName) -> Option<AttrValue> {
        self.state.edge_attr(edge_id, attr)
    }
    pub fn neighbors(&self, node_id: NodeId) -> Option<&Vec<NodeId>> {
        self.state.neighbors(node_id)
    }
    /// Shared immutable state underlying the temporal facade.
    pub fn read_state(&self) -> &PinnedReadState {
        &self.state
    }

    pub fn neighbors_bulk(&self, nodes: &[NodeId]) -> HashMap<NodeId, Vec<NodeId>> {
        nodes
            .iter()
            .filter_map(|node| self.neighbors(*node).map(|n| (*node, n.clone())))
            .collect()
    }

    pub fn as_subgraph(&self) -> Result<Subgraph, GraphError> {
        let graph = if let Some(context) = &self.materialization {
            if context.is_owned() {
                context.restore_owned(&self.state)?
            } else {
                context.materialize_selection(&self.state)?
            }
        } else {
            Graph::from_snapshot(self.state.snapshot().clone())?
        };
        let rc = Rc::new(std::cell::RefCell::new(graph));
        let nodes = self.state.existence().nodes().clone();
        let edges = self.state.existence().edges().clone();
        Ok(Subgraph::new(rc, nodes, edges, "TemporalSnapshot".into()))
    }
}
