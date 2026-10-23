//! Immutable graph input for algorithm kernels, independent of working writers.
use crate::{
    state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions},
    subgraphs::GraphView,
    GraphError, GraphResult, NodeId,
};
use rustc_hash::FxHashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Prepared immutable algorithm input. Clones share state, row identity and CSR.
/// Construction does not publish a mutation or create a checkpoint.
#[derive(Debug, Clone)]
pub struct AlgorithmRead {
    view: GraphView,
    nodes: Arc<[NodeId]>,
    index: Arc<FxHashMap<NodeId, usize>>,
    csr: Arc<Csr>,
}
impl AlgorithmRead {
    /// Prepare exactly the captured view's selected nodes and edge instances.
    pub fn new(view: GraphView) -> GraphResult<Self> {
        let nodes: Arc<[NodeId]> = view.node_ids().into();
        let index: FxHashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row))
            .collect();
        let edges = view.edge_ids();
        let mut failure = None;
        let mut csr = Csr::default();
        build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |id| index.get(&id).copied(),
            |edge| match view.edge_endpoints(edge) {
                Ok((s, t)) if index.contains_key(&s) && index.contains_key(&t) => Some((s, t)),
                Ok(_) => {
                    failure = Some(GraphError::InvalidInput(
                        "Algorithm edge endpoints must belong to the captured selection".into(),
                    ));
                    None
                }
                Err(error) => {
                    failure = Some(error);
                    None
                }
            },
            CsrOptions {
                add_reverse_edges: !view.is_directed(),
                sort_neighbors: true,
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(Self {
            view,
            nodes,
            index: Arc::new(index),
            csr: Arc::new(csr),
        })
    }
    /// Captured selection and properties used by this prepared input.
    pub fn view(&self) -> &GraphView {
        &self.view
    }
    /// Ascending selected node IDs corresponding to CSR row indices.
    pub fn node_ids(&self) -> &[NodeId] {
        &self.nodes
    }
    /// Shared directed/undirected adjacency, preserving parallel occurrences and loops.
    pub fn csr(&self) -> &Csr {
        &self.csr
    }
    fn source(&self, start: NodeId) -> GraphResult<usize> {
        self.index.get(&start).copied().ok_or_else(|| {
            GraphError::InvalidInput(format!(
                "Traversal start {start} does not belong to the captured selection"
            ))
        })
    }
    /// Reachable hop distances, ordered by original node ID; no attributes are written.
    /// A supplied cancellation flag rejects the operation without partial output.
    pub fn bfs_distances(
        &self,
        start: NodeId,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<(NodeId, i64)>> {
        let source = self.source(start)?;
        super::pathfinding::bfs_dfs::bfs_distances(
            || cancel.is_some_and(|token| token.load(Ordering::Relaxed)),
            &self.csr,
            &self.nodes,
            source,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))
    }
    /// Reachable ascending-neighbor DFS preorder, returned in discovery order.
    pub fn dfs_order(
        &self,
        start: NodeId,
        cancel: Option<&AtomicBool>,
    ) -> GraphResult<Vec<(NodeId, i64)>> {
        let source = self.source(start)?;
        super::pathfinding::bfs_dfs::dfs_order(
            || cancel.is_some_and(|token| token.load(Ordering::Relaxed)),
            &self.csr,
            &self.nodes,
            source,
        )
        .map_err(|error| GraphError::InvalidInput(error.to_string()))
    }
}
