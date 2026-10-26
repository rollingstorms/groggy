//! Shared immutable read-state ownership for current captures and historical snapshots.
use super::state::GraphSnapshot;
use crate::{
    types::GraphType, AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId, StateId,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// Active membership at a captured state; independent of later working-graph edits.
#[derive(Debug, Clone)]
pub struct ExistenceIndex {
    nodes: HashSet<NodeId>,
    edges: HashSet<EdgeId>,
}
impl ExistenceIndex {
    /// Build membership from explicit sets.
    pub fn new(nodes: HashSet<NodeId>, edges: HashSet<EdgeId>) -> Self {
        Self { nodes, edges }
    }
    /// Whether the node belongs to the captured state.
    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes.contains(&id)
    }
    /// Whether the edge belongs to the captured state.
    pub fn contains_edge(&self, id: EdgeId) -> bool {
        self.edges.contains(&id)
    }
    /// Captured node membership.
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.nodes
    }
    /// Captured edge membership.
    pub fn edges(&self) -> &HashSet<EdgeId> {
        &self.edges
    }
}
#[derive(Debug)]
struct Adjacency {
    neighbors: Vec<NodeId>,
    edges: Vec<EdgeId>,
}
#[derive(Debug)]
struct ReadData {
    snapshot: GraphSnapshot,
    existence: ExistenceIndex,
    adjacency: HashMap<NodeId, Adjacency>,
    incident: HashMap<NodeId, Vec<EdgeId>>,
}
/// In-process checked-write provenance, shared by derived immutable selections.
#[derive(Debug)]
pub(crate) struct ReadOrigin {
    pub history: Arc<()>,
    pub branch: String,
    pub state: StateId,
    pub clean: bool,
}
/// Cheap clones share frozen topology and values; no writable graph owner is exposed.
/// Capturing current state does not create a checkpoint or record a mutation.
#[derive(Debug, Clone)]
pub struct PinnedReadState {
    data: Arc<ReadData>,
    origin: Option<Arc<ReadOrigin>>,
    read_time: Option<Arc<super::effective_read::ReadTime>>,
}
impl PinnedReadState {
    pub(crate) fn from_snapshot(snapshot: GraphSnapshot) -> GraphResult<Self> {
        let nodes: HashSet<_> = snapshot.active_nodes.iter().copied().collect();
        if nodes.len() != snapshot.active_nodes.len() {
            return Err(GraphError::InvalidInput(
                "Duplicate node in read snapshot".into(),
            ));
        }
        let edges = snapshot.edges.keys().copied().collect();
        let mut adjacency: HashMap<_, Vec<_>> =
            nodes.iter().map(|&node| (node, Vec::new())).collect();
        let mut incident: HashMap<_, Vec<_>> =
            nodes.iter().map(|&node| (node, Vec::new())).collect();
        for (&edge, &(source, target)) in &snapshot.edges {
            if !nodes.contains(&source) || !nodes.contains(&target) {
                return Err(GraphError::InvalidInput(
                    "Read snapshot edge has an inactive endpoint".into(),
                ));
            }
            incident.entry(source).or_default().push(edge);
            if source != target {
                incident.entry(target).or_default().push(edge);
            }
            adjacency.entry(source).or_default().push((target, edge));
            if snapshot.graph_type == GraphType::Undirected && source != target {
                adjacency.entry(target).or_default().push((source, edge));
            }
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
        }
        for edges in incident.values_mut() {
            edges.sort_unstable();
        }
        let adjacency = adjacency
            .into_iter()
            .map(|(node, pairs)| {
                let (neighbors, edges) = pairs.into_iter().unzip();
                (node, Adjacency { neighbors, edges })
            })
            .collect();
        Ok(Self {
            origin: None,
            read_time: None,
            data: Arc::new(ReadData {
                snapshot,
                existence: ExistenceIndex::new(nodes, edges),
                adjacency,
                incident,
            }),
        })
    }
    pub(crate) fn with_read_time(mut self, time: super::effective_read::ReadTime) -> Self {
        self.read_time = Some(Arc::new(time));
        self
    }
    /// Effective clocks/head for temporal captures, retained by derived views/tables.
    pub fn read_time(&self) -> Option<&super::effective_read::ReadTime> {
        self.read_time.as_deref()
    }
    pub(crate) fn with_origin(
        mut self,
        history: Arc<()>,
        branch: String,
        state: StateId,
        clean: bool,
    ) -> Self {
        self.origin = Some(Arc::new(ReadOrigin {
            history,
            branch,
            state,
            clean,
        }));
        self
    }
    pub(crate) fn origin(&self) -> Option<&ReadOrigin> {
        self.origin.as_deref()
    }
    /// Accepted source state for checked writeback; dirty and standalone captures are read-only.
    pub fn write_state_id(&self) -> Option<StateId> {
        self.origin()
            .filter(|origin| origin.clean)
            .map(|origin| origin.state)
    }
    pub(crate) fn shares_capture(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
    /// Borrow frozen values/topology. state_id is checkpoint ancestry for a working capture,
    /// not an assertion that uncheckpointed edits have been recorded in history.
    pub fn snapshot(&self) -> &GraphSnapshot {
        &self.data.snapshot
    }
    /// Captured active membership.
    pub fn existence(&self) -> &ExistenceIndex {
        &self.data.existence
    }
    /// Current value at capture, or none for missing/inactive cells.
    pub fn node_attr(&self, id: NodeId, name: &AttrName) -> Option<AttrValue> {
        if !self.existence().contains_node(id) {
            return None;
        }
        self.data
            .snapshot
            .node_attributes
            .get(&id)
            .and_then(|attrs| attrs.get(name))
            .cloned()
    }
    /// Current edge value at capture, or none for missing/inactive cells.
    pub fn edge_attr(&self, id: EdgeId, name: &AttrName) -> Option<AttrValue> {
        if !self.existence().contains_edge(id) {
            return None;
        }
        self.data
            .snapshot
            .edge_attributes
            .get(&id)
            .and_then(|attrs| attrs.get(name))
            .cloned()
    }
    /// Sorted neighbor occurrences; parallel edges repeat neighbors, self-loops occur once.
    pub fn neighbors(&self, id: NodeId) -> Option<&Vec<NodeId>> {
        self.data.adjacency.get(&id).map(|entry| &entry.neighbors)
    }
    /// Edge IDs aligned with neighbor occurrences, enabling edge-scoped views.
    pub fn neighbor_edges(&self, id: NodeId) -> Option<&Vec<EdgeId>> {
        self.data.adjacency.get(&id).map(|entry| &entry.edges)
    }
    /// Incident edge instances in stable ID order; loops occur once.
    pub fn incident_edges(&self, id: NodeId) -> Option<&Vec<EdgeId>> {
        self.data.incident.get(&id)
    }
}
