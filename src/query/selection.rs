//! Bounded captured traversal; membership and source clocks remain owned by GraphView.
use crate::{GraphError, GraphResult, GraphView, NodeId};
use std::{
    collections::{HashSet, VecDeque},
    str::FromStr,
};

/// Edge orientation used during bounded traversal. Undirected graphs use both
/// endpoints for every mode; directed "both" follows either incident orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}
impl FromStr for TraversalDirection {
    type Err = GraphError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "outgoing" => Ok(Self::Outgoing),
            "incoming" => Ok(Self::Incoming),
            "both" => Ok(Self::Both),
            _ => Err(GraphError::InvalidInput(
                "direction must be outgoing, incoming or both".into(),
            )),
        }
    }
}
impl GraphView {
    /// Select all seeds and nodes reachable within at most max_hops, then retain
    /// only selected internal edges. Duplicate seeds coalesce as in select_nodes.
    /// Every seed must belong to this capture, including for zero-hop requests.
    pub fn within_hops(
        &self,
        seeds: &[NodeId],
        max_hops: usize,
        direction: TraversalDirection,
    ) -> GraphResult<Self> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        for &seed in seeds {
            if !self.contains_node(seed) {
                return Err(GraphError::InvalidInput(format!(
                    "Traversal seed {seed} does not belong to this view"
                )));
            }
            if visited.insert(seed) {
                queue.push_back((seed, 0usize));
            }
        }
        while let Some((node, depth)) = queue.pop_front() {
            if depth == max_hops {
                continue;
            }
            for edge in self.selected_incident(node)? {
                let (source, target) = self.edge_endpoints(edge)?;
                let neighbor = if !self.is_directed() || direction == TraversalDirection::Both {
                    Some(if source == node { target } else { source })
                } else {
                    match direction {
                        TraversalDirection::Outgoing if source == node => Some(target),
                        TraversalDirection::Incoming if target == node => Some(source),
                        _ => None,
                    }
                };
                if let Some(neighbor) = neighbor {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        self.select_nodes(&visited.into_iter().collect::<Vec<_>>())
    }
}
