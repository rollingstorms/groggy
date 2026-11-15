//! Shared immutable read-state ownership for current captures and historical snapshots.
use super::state::GraphSnapshot;
use crate::{
    types::GraphType, AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId, StateId,
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
};

/// Active membership at a captured state; independent of later working-graph edits.
#[derive(Debug, Clone)]
pub struct ExistenceIndex {
    nodes: Arc<HashSet<NodeId>>,
    edges: Arc<HashSet<EdgeId>>,
}
impl ExistenceIndex {
    /// Build membership from explicit sets.
    pub fn new(nodes: HashSet<NodeId>, edges: HashSet<EdgeId>) -> Self {
        Self {
            nodes: Arc::new(nodes),
            edges: Arc::new(edges),
        }
    }
    pub(crate) fn shared_membership(&self) -> (Arc<HashSet<NodeId>>, Arc<HashSet<EdgeId>>) {
        (self.nodes.clone(), self.edges.clone())
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
// Active isolates and directed sinks share immutable empty rows. Membership,
// rather than an allocated empty map entry, distinguishes them from absent IDs.
static EMPTY_ADJACENCY: Adjacency = Adjacency {
    neighbors: Vec::new(),
    edges: Vec::new(),
};
#[derive(Debug)]
struct ReadData {
    snapshot: GraphSnapshot,
    existence: ExistenceIndex,
    adjacency: HashMap<NodeId, Adjacency>,
    incident: HashMap<NodeId, Vec<EdgeId>>,
}
/// A non-owning current-state cache; readers alone determine payload lifetime.
#[derive(Debug)]
pub(crate) struct WeakReadData(Weak<ReadData>);
impl WeakReadData {
    pub(crate) fn upgrade(&self) -> Option<PinnedReadState> {
        self.0.upgrade().map(|data| PinnedReadState {
            data,
            capture: Arc::new(()),
            origin: None,
            read_time: None,
            groups: Default::default(),
        })
    }
}
/// Non-owning cache of one logical capture, including its identity. Metadata does
/// not retain graph/history payloads; both values and group catalogs stay weak.
#[derive(Debug)]
pub(crate) struct WeakReadCapture {
    data: Weak<ReadData>,
    capture: Weak<()>,
    groups: Weak<HashMap<crate::types::SubgraphId, crate::storage::retained_group::RetainedGroup>>,
    origin: Option<Arc<ReadOrigin>>,
    read_time: Option<Arc<super::effective_read::ReadTime>>,
}
impl WeakReadCapture {
    pub(crate) fn upgrade(&self) -> Option<PinnedReadState> {
        Some(PinnedReadState {
            data: self.data.upgrade()?,
            capture: self.capture.upgrade()?,
            groups: self.groups.upgrade()?,
            origin: self.origin.clone(),
            read_time: self.read_time.clone(),
        })
    }
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
    // Writeback distinguishes independent captures even when their payload is shared.
    capture: Arc<()>,
    origin: Option<Arc<ReadOrigin>>,
    read_time: Option<Arc<super::effective_read::ReadTime>>,
    groups: Arc<HashMap<crate::types::SubgraphId, crate::storage::retained_group::RetainedGroup>>,
}
impl PinnedReadState {
    pub(crate) fn downgrade_capture(&self) -> WeakReadCapture {
        WeakReadCapture {
            data: Arc::downgrade(&self.data),
            capture: Arc::downgrade(&self.capture),
            groups: Arc::downgrade(&self.groups),
            origin: self.origin.clone(),
            read_time: self.read_time.clone(),
        }
    }
    pub(crate) fn downgrade_data(&self) -> WeakReadData {
        WeakReadData(Arc::downgrade(&self.data))
    }

    fn snapshot_nodes(snapshot: &GraphSnapshot) -> GraphResult<HashSet<NodeId>> {
        let nodes: HashSet<_> = snapshot.active_nodes.iter().copied().collect();
        if nodes.len() != snapshot.active_nodes.len() {
            return Err(GraphError::InvalidInput(
                "Duplicate node in read snapshot".into(),
            ));
        }
        Ok(nodes)
    }

    fn validate_endpoints(
        nodes: &HashSet<NodeId>,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<()> {
        if !nodes.contains(&source) || !nodes.contains(&target) {
            return Err(GraphError::InvalidInput(
                "Read snapshot edge has an inactive endpoint".into(),
            ));
        }
        Ok(())
    }

    /// Validate historical topology without constructing disposable read indexes.
    pub(crate) fn validate_snapshot(snapshot: &GraphSnapshot) -> GraphResult<()> {
        let nodes = Self::snapshot_nodes(snapshot)?;
        for &(source, target) in snapshot.edges.values() {
            Self::validate_endpoints(&nodes, source, target)?;
        }
        Ok(())
    }

    pub(crate) fn from_snapshot(snapshot: GraphSnapshot) -> GraphResult<Self> {
        let nodes = Self::snapshot_nodes(&snapshot)?;
        let edges = snapshot.edges.keys().copied().collect();
        // Bound occupied rows by incidences, avoiding dense empty entries and
        // repeated table growth on connected captures. Saturation is intentional
        // for the capacity bound; validation below still checks every endpoint.
        let incident_bound = snapshot.edges.len().saturating_mul(2).min(nodes.len());
        let adjacency_bound = if snapshot.graph_type == GraphType::Directed {
            snapshot.edges.len().min(nodes.len())
        } else {
            incident_bound
        };
        let mut adjacency: HashMap<_, Vec<_>> = HashMap::with_capacity(adjacency_bound);
        let mut incident: HashMap<_, Vec<_>> = HashMap::with_capacity(incident_bound);
        for (&edge, &(source, target)) in &snapshot.edges {
            Self::validate_endpoints(&nodes, source, target)?;
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
            capture: Arc::new(()),
            origin: None,
            read_time: None,
            groups: Default::default(),
            data: Arc::new(ReadData {
                snapshot,
                existence: ExistenceIndex::new(nodes, edges),
                adjacency,
                incident,
            }),
        })
    }
    pub(crate) fn with_groups(
        mut self,
        groups: Arc<
            HashMap<crate::types::SubgraphId, crate::storage::retained_group::RetainedGroup>,
        >,
    ) -> Self {
        self.groups = groups;
        self
    }
    pub(crate) fn groups(
        &self,
    ) -> &Arc<HashMap<crate::types::SubgraphId, crate::storage::retained_group::RetainedGroup>>
    {
        &self.groups
    }
    /// Retrieve an immutable retained source revision. Legacy membership-only IDs
    /// need an explicit source pin; current working values are never substituted.
    pub fn retained_group(&self, id: crate::types::SubgraphId) -> GraphResult<crate::GraphView> {
        self.groups
            .get(&id)
            .map(|group| group.view.clone())
            .ok_or_else(|| {
                GraphError::InvalidInput(format!(
                    "Group {id} has no frozen source revision; pin an explicit source view"
                ))
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
    /// Process-local logical capture key. Weak authority records keep the token
    /// allocation alive, so a live record cannot be confused by pointer reuse.
    pub(crate) fn capture_key(&self) -> usize {
        Arc::as_ptr(&self.capture) as usize
    }
    pub(crate) fn shares_capture(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.capture, &other.capture)
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
        self.adjacency_row(id).map(|entry| &entry.neighbors)
    }
    /// Edge IDs aligned with neighbor occurrences, enabling edge-scoped views.
    pub fn neighbor_edges(&self, id: NodeId) -> Option<&Vec<EdgeId>> {
        self.adjacency_row(id).map(|entry| &entry.edges)
    }
    /// Incident edge instances in stable ID order; loops occur once.
    pub fn incident_edges(&self, id: NodeId) -> Option<&Vec<EdgeId>> {
        self.data.incident.get(&id).or_else(|| {
            self.existence()
                .contains_node(id)
                .then_some(&EMPTY_ADJACENCY.edges)
        })
    }

    fn adjacency_row(&self, id: NodeId) -> Option<&Adjacency> {
        self.data.adjacency.get(&id).or_else(|| {
            self.existence()
                .contains_node(id)
                .then_some(&EMPTY_ADJACENCY)
        })
    }
}

#[cfg(test)]
mod validation_only_tests {
    use super::*;

    #[test]
    fn validation_only_preserves_capture_topology_rejections() {
        let mut snapshot = GraphSnapshot::empty_with_type(7, GraphType::Directed);
        snapshot.active_nodes = vec![10, 20];
        snapshot
            .edges
            .extend([(4, (10, 20)), (8, (10, 20)), (9, (20, 20))]);
        assert!(PinnedReadState::validate_snapshot(&snapshot).is_ok());
        assert!(PinnedReadState::from_snapshot(snapshot.clone()).is_ok());
        for invalid in [vec![10, 10], vec![10]] {
            snapshot.active_nodes = invalid;
            let validation = PinnedReadState::validate_snapshot(&snapshot).unwrap_err();
            let capture = PinnedReadState::from_snapshot(snapshot.clone()).unwrap_err();
            assert_eq!(validation.to_string(), capture.to_string());
        }
        assert!(PinnedReadState::validate_snapshot(&GraphSnapshot::empty(0)).is_ok());
    }
}
