//! Shared captured endpoint/incident postings for entity filters and pattern joins.
use super::TraversalDirection;
use crate::{EdgeId, GraphResult, GraphView, NodeId};
use std::collections::HashMap;
#[derive(Debug, Default)]
pub(crate) struct EndpointIndex {
    directed: bool,
    pairs: HashMap<(NodeId, NodeId), Vec<EdgeId>>,
    incident: HashMap<NodeId, Vec<EdgeId>>,
}
impl EndpointIndex {
    pub fn new(directed: bool) -> Self {
        Self {
            directed,
            ..Default::default()
        }
    }
    pub fn scan(view: &GraphView) -> GraphResult<Self> {
        let mut index = Self::new(view.is_directed());
        for edge in view.edge_ids() {
            let (s, t) = view.edge_endpoints(edge)?;
            index.insert(edge, s, t);
        }
        Ok(index)
    }
    // Callers insert ascending edge IDs, so all postings remain deterministic.
    pub fn insert(&mut self, edge: EdgeId, s: NodeId, t: NodeId) {
        let pair = if !self.directed && s > t {
            (t, s)
        } else {
            (s, t)
        };
        self.pairs.entry(pair).or_default().push(edge);
        self.incident.entry(s).or_default().push(edge);
        if s != t {
            self.incident.entry(t).or_default().push(edge);
        }
    }
    pub fn incident(&self, node: NodeId) -> &[EdgeId] {
        self.incident.get(&node).map_or(&[], Vec::as_slice)
    }
    pub fn between(&self, s: NodeId, t: NodeId, direction: TraversalDirection) -> Vec<EdgeId> {
        if !self.directed {
            return self
                .pairs
                .get(&if s > t { (t, s) } else { (s, t) })
                .cloned()
                .unwrap_or_default();
        }
        let forward = || self.pairs.get(&(s, t)).cloned().unwrap_or_default();
        let reverse = || self.pairs.get(&(t, s)).cloned().unwrap_or_default();
        match direction {
            TraversalDirection::Outgoing => forward(),
            TraversalDirection::Incoming => reverse(),
            TraversalDirection::Both => {
                let mut rows = forward();
                if s != t {
                    rows.extend(reverse());
                    rows.sort_unstable();
                }
                rows
            }
        }
    }
}
