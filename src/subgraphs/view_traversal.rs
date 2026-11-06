//! Captured traversal selections share the generic scoped tree/path kernel.
use super::GraphView;
use crate::query::scoped_traversal as traversal;
use crate::{GraphResult, NodeId};
impl GraphView {
    /// Neighborhood of the lowest selected node ID, with an explicit hop/direction
    /// policy. Empty selections retain their source coordinate and graph type.
    pub fn neighborhood_from_first(
        &self,
        hops: usize,
        direction: crate::query::selection::TraversalDirection,
    ) -> GraphResult<Self> {
        match self.nodes().iter().copied().min() {
            Some(center) => self.within_hops(&[center], hops, direction),
            None => Ok(self.clone()),
        }
    }

    /// Ascending-neighbor BFS discovery tree, following selected outgoing instances.
    pub fn bfs(&self, start: NodeId, max_depth: Option<usize>) -> GraphResult<Self> {
        let (nodes, edges) = traversal::bfs(self, start, max_depth)?;
        Self::from_membership(self.read_state().clone(), nodes, edges)
    }
    /// Iterative ascending-neighbor DFS tree, bounded by first-discovery depth.
    pub fn dfs(&self, start: NodeId, max_depth: Option<usize>) -> GraphResult<Self> {
        let (nodes, edges) = traversal::dfs(self, start, max_depth)?;
        Self::from_membership(self.read_state().clone(), nodes, edges)
    }
    /// Hop reachability; out-of-scope IDs return false.
    pub fn has_path(&self, source: NodeId, target: NodeId) -> GraphResult<bool> {
        traversal::has_path(self, source, target)
    }
    /// One exact shortest hop path; unreachable/out-of-scope endpoints return None.
    pub fn shortest_path_subgraph(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Self>> {
        traversal::shortest_path(self, source, target)?
            .map(|(nodes, edges)| Self::from_membership(self.read_state().clone(), nodes, edges))
            .transpose()
    }
    /// Nonempty connectivity with an explicit weak/strong component definition.
    pub fn is_connected(
        &self,
        mode: crate::algorithms::community::ComponentMode,
    ) -> GraphResult<bool> {
        if self.node_count() == 0 {
            return Ok(false);
        }
        Ok(self.connected_components(mode)?.len() == 1)
    }
}
