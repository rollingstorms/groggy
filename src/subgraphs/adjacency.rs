//! Detached selected topology for adjacency export, shared with existing CSR readers.
use super::{GraphView, Subgraph};
use crate::{state::topology::Csr, GraphResult, NodeId};
use std::{collections::BTreeMap, sync::Arc};

/// Borrow one endpoint authority for a bulk algorithm read. Live iteration holds
/// one owner guard; captured iteration reads its immutable source directly.
pub(crate) enum EndpointRead<'a> {
    Frozen(&'a GraphView),
    Live(crate::ownership::OwnerReadGuard<'a, crate::Graph>),
}
impl EndpointRead<'_> {
    pub(crate) fn get(&self, edge: crate::EdgeId) -> GraphResult<(NodeId, NodeId)> {
        match self {
            Self::Frozen(view) => view.edge_endpoints(edge),
            Self::Live(graph) => graph.edge_endpoints(edge),
        }
    }
}

/// Immutable selected rows and CSR. This retains no writable owner or property payload.
#[derive(Debug, Clone)]
pub struct AdjacencyRead {
    nodes: Arc<[NodeId]>,
    csr: Arc<Csr>,
}
impl AdjacencyRead {
    pub(crate) fn new(nodes: Arc<[NodeId]>, csr: Arc<Csr>) -> Self {
        Self { nodes, csr }
    }
    /// Ascending node keys, including isolates; neighbors use ascending node IDs.
    /// Parallel occurrences remain distinct, and self-loops occur once.
    pub fn adjacency_list(&self) -> BTreeMap<NodeId, Vec<NodeId>> {
        self.nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(row, node)| {
                (node, {
                    let mut neighbors: Vec<_> = self
                        .csr
                        .neighbors(row)
                        .iter()
                        .map(|&index| self.nodes[index])
                        .collect();
                    neighbors.sort_unstable();
                    neighbors
                })
            })
            .collect()
    }
}
impl Subgraph {
    /// Capture only current selected topology for owner-free worker/export calls.
    /// Invalid live membership/endpoints reject before cached CSR can be reused.
    pub fn adjacency_read(&self) -> GraphResult<AdjacencyRead> {
        if let Some(view) = self.frozen_view() {
            return view.adjacency_read();
        }

        let reverse = self.graph().borrow().is_undirected();
        Ok(AdjacencyRead::new(
            self.ordered_nodes(),
            self.get_or_build_csr_internal(reverse)?,
        ))
    }
    /// Export selected outgoing adjacency, using both directions when undirected.
    pub fn adjacency_list(&self) -> GraphResult<BTreeMap<NodeId, Vec<NodeId>>> {
        Ok(self.adjacency_read()?.adjacency_list())
    }
}
impl GraphView {
    /// Prepare exactly selected frozen topology, independent of source owner/liveness.
    pub fn adjacency_read(&self) -> GraphResult<AdjacencyRead> {
        Ok(crate::algorithms::AlgorithmRead::new(self.clone())?.into_adjacency_read())
    }
    /// Export frozen selected outgoing adjacency, retaining isolates and edge occurrences.
    pub fn adjacency_list(&self) -> GraphResult<BTreeMap<NodeId, Vec<NodeId>>> {
        Ok(self.adjacency_read()?.adjacency_list())
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn bulk_endpoint_sources_preserve_frozen_reads_and_reject_live_inactive_ids() {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(2).unwrap();
        graph.add_edge(0, 1).unwrap();
        let owner = graph.into_shared();
        let frozen = Subgraph::from_frozen_view(
            owner.clone(),
            owner.borrow().read_view().unwrap(),
            "frozen endpoints".into(),
        );
        let live = Subgraph::new(
            owner.clone(),
            [0, 1].into(),
            [0].into(),
            "live endpoints".into(),
        );
        owner.borrow_mut().remove_node(1).unwrap();
        {
            let _writer = owner.borrow_mut();
            let endpoints = frozen.endpoint_read();
            assert_eq!(endpoints.get(0).unwrap(), (0, 1));
            assert!(endpoints.get(1).is_err());
            assert!(!frozen.read_is_directed());
        }
        assert!(live.endpoint_read().get(0).is_err());
    }
}
