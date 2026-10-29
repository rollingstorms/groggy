//! Captured unweighted summaries delegate to the shared topology-only input.
use super::GraphView;
use crate::{GraphResult, NodeId};
impl GraphView {
    /// Prepare reusable immutable undirected topology; no property payload is retained by the result.
    pub fn structural_read(&self) -> GraphResult<crate::algorithms::UndirectedStructure> {
        self.algorithm_input()?.structural_read()
    }
    /// Simple undirected per-node clustering, ordered by node ID.
    pub fn clustering_coefficients(&self) -> GraphResult<Vec<(NodeId, f64)>> {
        self.structural_read()?.clustering_coefficients()
    }
    /// Simple undirected local clustering, or mean including isolates.
    pub fn clustering_coefficient(&self, node: Option<NodeId>) -> GraphResult<f64> {
        self.structural_read()?.clustering_coefficient(node)
    }
    /// Simple undirected transitivity, with loops/parallel multiplicity removed.
    pub fn transitivity(&self) -> GraphResult<f64> {
        self.structural_read()?.transitivity()
    }
    /// Edge-instance density with the captured direction; loops/parallel edges can exceed one.
    pub fn density(&self) -> f64 {
        crate::algorithms::structural::density(
            self.node_count(),
            self.edge_count(),
            self.is_directed(),
        )
    }
}
