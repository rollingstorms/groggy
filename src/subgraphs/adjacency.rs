//! Detached selected topology for adjacency export, shared with existing CSR readers.
use super::{GraphView, Subgraph};
use crate::{state::topology::Csr, GraphResult, NodeId};
use std::{collections::BTreeMap, sync::Arc};

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
