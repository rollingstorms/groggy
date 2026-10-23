//! Canonical immutable membership over shared frozen state, independent of write owners.
use crate::{
    state::{state::GraphSnapshot, PinnedReadState},
    types::GraphType,
    AttrName, AttrValue, EdgeId, GraphError, GraphResult, NodeId,
};
use std::{collections::HashSet, sync::Arc};
#[derive(Debug, Clone)]
enum Membership {
    Full,
    Selected(Arc<Selection>),
}
#[derive(Debug)]
struct Selection {
    nodes: HashSet<NodeId>,
    edges: HashSet<EdgeId>,
}
/// Immutable graph selection. Clones and nested selections share frozen state.
/// Working-graph mutation targets belong to adapters, never to this read representation.
#[derive(Debug, Clone)]
pub struct GraphView {
    state: PinnedReadState,
    membership: Membership,
}
impl GraphView {
    /// Select the complete captured state without copying its membership sets.
    pub fn all(state: PinnedReadState) -> Self {
        Self {
            state,
            membership: Membership::Full,
        }
    }
    /// Select explicit nodes/edges. Every selected edge must have both endpoints selected.
    pub fn from_membership(
        state: PinnedReadState,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
    ) -> GraphResult<Self> {
        if nodes.iter().any(|&id| !state.existence().contains_node(id)) {
            return Err(Self::invalid(
                "View contains a node outside its captured state",
            ));
        }
        for &edge in &edges {
            let &(source, target) =
                state.snapshot().edges.get(&edge).ok_or_else(|| {
                    Self::invalid("View contains an edge outside its captured state")
                })?;
            if !nodes.contains(&source) || !nodes.contains(&target) {
                return Err(Self::invalid("View edge endpoints must belong to the view"));
            }
        }
        Ok(Self {
            state,
            membership: Membership::Selected(Arc::new(Selection { nodes, edges })),
        })
    }
    fn invalid(message: &str) -> GraphError {
        GraphError::InvalidInput(message.into())
    }
    /// Shared immutable state underlying this selection.
    pub fn read_state(&self) -> &PinnedReadState {
        &self.state
    }
    /// Selected node membership, including isolated nodes.
    pub fn nodes(&self) -> &HashSet<NodeId> {
        match &self.membership {
            Membership::Full => self.state.existence().nodes(),
            Membership::Selected(selection) => &selection.nodes,
        }
    }
    /// Selected edge instances; a selection need not be induced.
    pub fn edges(&self) -> &HashSet<EdgeId> {
        match &self.membership {
            Membership::Full => self.state.existence().edges(),
            Membership::Selected(selection) => &selection.edges,
        }
    }
    /// Stable ascending node IDs.
    pub fn node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<_> = self.nodes().iter().copied().collect();
        ids.sort_unstable();
        ids
    }
    /// Stable ascending edge IDs.
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        let mut ids: Vec<_> = self.edges().iter().copied().collect();
        ids.sort_unstable();
        ids
    }
    fn require_node(&self, id: NodeId) -> GraphResult<()> {
        if self.nodes().contains(&id) {
            Ok(())
        } else {
            Err(Self::invalid("Node does not belong to this view"))
        }
    }
    /// Frozen node attribute. Out-of-scope IDs fail; absent attributes return none.
    pub fn node_attr(&self, id: NodeId, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.require_node(id)?;
        Ok(self.state.node_attr(id, name))
    }
    /// Frozen edge attribute. Out-of-scope IDs fail.
    pub fn edge_attr(&self, id: EdgeId, name: &AttrName) -> GraphResult<Option<AttrValue>> {
        self.edge_endpoints(id)?;
        Ok(self.state.edge_attr(id, name))
    }
    /// Attribute presence distinguishes an explicit null from an absent property.
    pub fn has_node_attribute(&self, id: NodeId, name: &AttrName) -> GraphResult<bool> {
        self.require_node(id)?;
        Ok(self
            .state
            .snapshot()
            .node_attributes
            .get(&id)
            .is_some_and(|attrs| attrs.contains_key(name)))
    }
    /// Edge attribute presence, requiring selected membership.
    pub fn has_edge_attribute(&self, id: EdgeId, name: &AttrName) -> GraphResult<bool> {
        self.edge_endpoints(id)?;
        Ok(self
            .state
            .snapshot()
            .edge_attributes
            .get(&id)
            .is_some_and(|attrs| attrs.contains_key(name)))
    }
    /// Stable attribute names for a selected node.
    pub fn node_attribute_keys(&self, id: NodeId) -> GraphResult<Vec<AttrName>> {
        self.require_node(id)?;
        let mut keys: Vec<_> = self
            .state
            .snapshot()
            .node_attributes
            .get(&id)
            .into_iter()
            .flat_map(|attrs| attrs.keys().cloned())
            .collect();
        keys.sort();
        Ok(keys)
    }
    /// Stable attribute names for a selected edge.
    pub fn edge_attribute_keys(&self, id: EdgeId) -> GraphResult<Vec<AttrName>> {
        self.edge_endpoints(id)?;
        let mut keys: Vec<_> = self
            .state
            .snapshot()
            .edge_attributes
            .get(&id)
            .into_iter()
            .flat_map(|attrs| attrs.keys().cloned())
            .collect();
        keys.sort();
        Ok(keys)
    }
    /// Frozen endpoints, requiring selected edge membership.
    pub fn edge_endpoints(&self, id: EdgeId) -> GraphResult<(NodeId, NodeId)> {
        if !self.edges().contains(&id) {
            return Err(Self::invalid("Edge does not belong to this view"));
        }
        self.state
            .snapshot()
            .edges
            .get(&id)
            .copied()
            .ok_or_else(|| Self::invalid("Captured edge has no endpoints"))
    }
    fn outgoing_occurrences(
        &self,
        id: NodeId,
    ) -> GraphResult<impl Iterator<Item = (NodeId, EdgeId)> + '_> {
        self.require_node(id)?;
        let neighbors = self
            .state
            .neighbors(id)
            .ok_or_else(|| Self::invalid("Captured node has no adjacency entry"))?;
        let edges = self
            .state
            .neighbor_edges(id)
            .ok_or_else(|| Self::invalid("Captured node has no edge adjacency entry"))?;
        let selected = self.edges();
        Ok(neighbors
            .iter()
            .copied()
            .zip(edges.iter().copied())
            .filter(move |(_, edge)| selected.contains(edge)))
    }
    pub(crate) fn selected_incident(
        &self,
        id: NodeId,
    ) -> GraphResult<impl Iterator<Item = EdgeId> + '_> {
        self.require_node(id)?;
        let edges = self
            .state
            .incident_edges(id)
            .ok_or_else(|| Self::invalid("Captured node has no incident entry"))?;
        let selected = self.edges();
        Ok(edges
            .iter()
            .copied()
            .filter(move |edge| selected.contains(edge)))
    }
    /// Outgoing neighbor occurrences (both directions when undirected).
    /// Parallel selected edges repeat neighbors; self-loops occur once.
    pub fn neighbors(&self, id: NodeId) -> GraphResult<Vec<NodeId>> {
        Ok(self
            .outgoing_occurrences(id)?
            .map(|(node, _)| node)
            .collect())
    }
    /// Selected incident instances in stable edge order, including incoming directed edges.
    pub fn incident_edges(&self, id: NodeId) -> GraphResult<Vec<EdgeId>> {
        Ok(self.selected_incident(id)?.collect())
    }
    /// Incident-instance count; every loop and parallel edge instance counts once.
    pub fn degree(&self, id: NodeId) -> GraphResult<usize> {
        Ok(self.selected_incident(id)?.count())
    }
    /// Outgoing-instance count; undirected edges count at both endpoints, loops once.
    pub fn out_degree(&self, id: NodeId) -> GraphResult<usize> {
        Ok(self.outgoing_occurrences(id)?.count())
    }
    /// Incoming-instance count for directed graphs; incident degree when undirected.
    pub fn in_degree(&self, id: NodeId) -> GraphResult<usize> {
        if self.state.snapshot().graph_type == GraphType::Undirected {
            return self.degree(id);
        }
        let mut count = 0;
        for edge in self.selected_incident(id)? {
            if self.edge_endpoints(edge)?.1 == id {
                count += 1;
            }
        }
        Ok(count)
    }
    /// Number of selected nodes.
    pub fn node_count(&self) -> usize {
        self.nodes().len()
    }
    /// Number of selected edge instances.
    pub fn edge_count(&self) -> usize {
        self.edges().len()
    }
    /// Whether a node belongs to this selection.
    pub fn contains_node(&self, id: NodeId) -> bool {
        self.nodes().contains(&id)
    }
    /// Whether an edge belongs to this selection.
    pub fn contains_edge(&self, id: EdgeId) -> bool {
        self.edges().contains(&id)
    }
    /// Whether the captured graph is directed.
    pub fn is_directed(&self) -> bool {
        self.state.snapshot().graph_type == GraphType::Directed
    }

    /// Induce a node selection within this view's existing edge scope, sharing its pin.
    pub fn select_nodes(&self, ids: &[NodeId]) -> GraphResult<Self> {
        let mut nodes = HashSet::with_capacity(ids.len());
        for &id in ids {
            self.require_node(id)?;
            nodes.insert(id);
        }
        let mut edges = HashSet::new();
        for &node in &nodes {
            for edge in self.incident_edges(node)? {
                let (source, target) = self.edge_endpoints(edge)?;
                if nodes.contains(&source) && nodes.contains(&target) {
                    edges.insert(edge);
                }
            }
        }
        Self::from_membership(self.state.clone(), nodes, edges)
    }
    /// Select edge instances and their endpoints, within the parent scope.
    pub fn select_edges(&self, ids: &[EdgeId]) -> GraphResult<Self> {
        let mut nodes = HashSet::new();
        let mut edges = HashSet::with_capacity(ids.len());
        for &id in ids {
            let (source, target) = self.edge_endpoints(id)?;
            nodes.insert(source);
            nodes.insert(target);
            edges.insert(id);
        }
        Self::from_membership(self.state.clone(), nodes, edges)
    }
    /// Filter selected captured nodes; the child cannot expand its parent's edge scope.
    pub fn filter_nodes(&self, filter: &crate::query::NodeFilter) -> GraphResult<Self> {
        let ids = crate::query::filtering::nodes(
            &crate::query::filtering::Frozen(self),
            &self.node_ids(),
            filter,
        )?;
        self.select_nodes(&ids)
    }
    /// Filter selected captured edge instances and retain their endpoints.
    pub fn filter_edges(&self, filter: &crate::query::EdgeFilter) -> GraphResult<Self> {
        let ids = crate::query::filtering::edges(
            &crate::query::filtering::Frozen(self),
            &self.edge_ids(),
            filter,
        )?;
        self.select_edges(&ids)
    }
    /// Prepare reusable immutable CSR/row identity for native algorithms.
    pub fn algorithm_input(&self) -> GraphResult<crate::algorithms::AlgorithmRead> {
        crate::algorithms::AlgorithmRead::new(self.clone())
    }
    /// Captured component views, ordered by smallest node ID, without writes.
    pub fn connected_components(
        &self,
        mode: crate::algorithms::community::ComponentMode,
    ) -> GraphResult<Vec<Self>> {
        self.algorithm_input()?.connected_components(mode, None)
    }
    /// Reachable hop distances as owned (node ID, distance) rows in ascending ID order.
    pub fn bfs_distances(&self, start: NodeId) -> GraphResult<Vec<(NodeId, i64)>> {
        self.algorithm_input()?.bfs_distances(start, None)
    }
    /// Ascending-neighbor depth-first preorder as owned (node ID, order) rows.
    pub fn dfs_order(&self, start: NodeId) -> GraphResult<Vec<(NodeId, i64)>> {
        self.algorithm_input()?.dfs_order(start, None)
    }
    /// Captured nonnegative weighted distances as owned f64 rows in node-ID order.
    pub fn dijkstra_distances(
        &self,
        start: NodeId,
        weight_attr: Option<&str>,
    ) -> GraphResult<Vec<(NodeId, f64)>> {
        self.algorithm_input()?
            .dijkstra_distances(start, weight_attr, None)
    }
    /// Owned shortest path with actual selected edge IDs and captured total weight.
    pub fn astar_path(
        &self,
        start: NodeId,
        goal: NodeId,
        weight_attr: Option<&str>,
        heuristic_attr: Option<&str>,
    ) -> GraphResult<crate::query::Path> {
        self.algorithm_input()?
            .astar_path(start, goal, weight_attr, heuristic_attr, None)
    }
    /// Owned adjacency from captured edge instances and captured weights.
    /// Missing weights are one; explicit values follow the shared graph conversion policy.
    pub fn adjacency_matrix<
        T: crate::storage::matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    >(
        &self,
        sparse: bool,
        weight_attr: Option<&str>,
    ) -> GraphResult<crate::storage::matrix::GraphMatrix<T>> {
        let nodes = self.node_ids();
        let edges = self.edge_ids();
        let attr = weight_attr.map(str::to_owned);
        crate::storage::matrix::graph_projection::adjacency(
            &nodes,
            &edges,
            sparse,
            attr.as_ref(),
            self.is_directed(),
            |edge| self.edge_endpoints(edge),
            |edge, name| {
                Ok(self
                    .state
                    .snapshot()
                    .edge_attributes
                    .get(&edge)
                    .and_then(|attrs| attrs.get(name)))
            },
        )
    }
    /// Owned scalar feature matrix in stable node/attribute order.
    /// Missing/null/nonnumeric cells retain the graph API's zero-imputation policy.
    pub fn to_matrix<
        T: crate::storage::matrix::NumericType + crate::storage::matrix::FromAttrValue<T>,
    >(
        &self,
    ) -> GraphResult<crate::storage::matrix::GraphMatrix<T>> {
        let nodes = self.node_ids();
        let mut names = std::collections::BTreeSet::new();
        for &node in &nodes {
            if let Some(attrs) = self.state.snapshot().node_attributes.get(&node) {
                names.extend(attrs.keys());
            }
        }
        crate::storage::matrix::graph_projection::features(
            &nodes,
            names.into_iter().cloned().collect(),
            |node, name| {
                Ok(self
                    .state
                    .snapshot()
                    .node_attributes
                    .get(&node)
                    .and_then(|attrs| attrs.get(name)))
            },
        )
    }

    /// Owned node columns from this captured selection, transferable between threads.
    pub fn node_table_data(
        &self,
    ) -> GraphResult<crate::storage::table::projection::OwnedTableData> {
        let nodes = self.node_ids();
        let attrs = &self.state.snapshot().node_attributes;
        let names: std::collections::BTreeSet<_> = nodes
            .iter()
            .filter_map(|id| attrs.get(id))
            .flat_map(|attrs| attrs.keys())
            .collect();
        crate::storage::table::projection::nodes(
            &nodes,
            names.into_iter().cloned().collect(),
            |id, name| Ok(attrs.get(&id).and_then(|attrs| attrs.get(name))),
        )
    }
    /// Owned edge columns from captured endpoints and selected edge instances.
    pub fn edge_table_data(
        &self,
    ) -> GraphResult<crate::storage::table::projection::OwnedTableData> {
        let edges = self.edge_ids();
        let attrs = &self.state.snapshot().edge_attributes;
        let names: std::collections::BTreeSet<_> = edges
            .iter()
            .filter_map(|id| attrs.get(id))
            .flat_map(|attrs| attrs.keys())
            .collect();
        crate::storage::table::projection::edges(
            &edges,
            names.into_iter().cloned().collect(),
            |id| self.edge_endpoints(id),
            |id, name| Ok(attrs.get(&id).and_then(|attrs| attrs.get(name))),
        )
    }
    /// Independent node table; missing properties occupy aligned null cells.
    pub fn nodes_table(&self) -> GraphResult<crate::storage::table::NodesTable> {
        self.node_table_data()?.into_nodes_table()
    }
    /// Independent edge table retaining captured endpoint identity.
    pub fn edges_table(&self) -> GraphResult<crate::storage::table::EdgesTable> {
        self.edge_table_data()?.into_edges_table()
    }
    /// Independent node/edge tables from one immutable state and selection.
    pub fn table(&self) -> GraphResult<crate::storage::table::GraphTable> {
        Ok(crate::storage::table::GraphTable::new(
            self.nodes_table()?,
            self.edges_table()?,
        ))
    }

    /// Explicit owned projection for consumers that require materialized state.
    /// Does not recover retained meta-group payloads or create history.
    pub fn to_snapshot(&self) -> GraphSnapshot {
        let mut snapshot = GraphSnapshot::empty_with_type(
            self.state.snapshot().state_id,
            self.state.snapshot().graph_type,
        );
        snapshot.active_nodes = self.node_ids();
        for &node in self.nodes() {
            if let Some(attrs) = self.state.snapshot().node_attributes.get(&node) {
                snapshot.node_attributes.insert(node, attrs.clone());
            }
        }
        for &edge in self.edges() {
            if let Some(&endpoints) = self.state.snapshot().edges.get(&edge) {
                snapshot.edges.insert(edge, endpoints);
            }
            if let Some(attrs) = self.state.snapshot().edge_attributes.get(&edge) {
                snapshot.edge_attributes.insert(edge, attrs.clone());
            }
        }
        snapshot
    }
}
