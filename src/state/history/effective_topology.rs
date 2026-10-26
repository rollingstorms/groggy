//! Incremental effective lifetimes. No property payloads or graph snapshots.
use super::*;
use crate::{ChangeKey, PublicationMetadata};
type Clock = (i64, StateId, u8);
type EdgeState = Option<(NodeId, NodeId)>;
#[derive(Debug)]
pub(crate) struct EffectiveTopology {
    head: StateId,
    precise: bool,
    nodes: HashMap<NodeId, Vec<(Clock, bool)>>,
    edges: HashMap<EdgeId, Vec<(Clock, EdgeState)>>,
    incident: HashMap<NodeId, HashSet<EdgeId>>,
}
impl Default for EffectiveTopology {
    fn default() -> Self {
        Self {
            head: 0,
            precise: true,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            incident: HashMap::new(),
        }
    }
}
#[derive(Default)]
pub(crate) struct TopologyPatch {
    pub nodes: HashMap<NodeId, Vec<(Clock, bool)>>,
    pub edges: HashMap<EdgeId, Vec<(Clock, EdgeState)>>,
}
impl TopologyPatch {
    pub(crate) fn from_metadata(
        metadata: &PublicationMetadata,
        state: StateId,
        additions: &HashMap<EdgeId, (NodeId, NodeId)>,
    ) -> GraphResult<Self> {
        let mut patch = Self::default();
        for (key, &at) in &metadata.change_times {
            match key {
                ChangeKey::NodeAdded(id) => patch
                    .nodes
                    .entry(*id)
                    .or_default()
                    .push(((at, state, 0), true)),
                ChangeKey::NodeRemoved(id) => patch
                    .nodes
                    .entry(*id)
                    .or_default()
                    .push(((at, state, 2), false)),
                ChangeKey::EdgeRemoved(id) => patch
                    .edges
                    .entry(*id)
                    .or_default()
                    .push(((at, state, 1), None)),
                ChangeKey::EdgeAdded(id) => {
                    let endpoints = additions.get(id).copied().ok_or_else(|| {
                        GraphError::InvalidInput("Missing effective edge addition endpoints".into())
                    })?;
                    patch
                        .edges
                        .entry(*id)
                        .or_default()
                        .push(((at, state, 3), Some(endpoints)));
                }
                _ => {}
            }
        }
        Ok(patch)
    }
}
/// Collapse equal-time events using acceptance order and lifecycle phase.
fn segments<T: Copy + PartialEq>(
    old: Option<&Vec<(Clock, T)>>,
    new: Option<&Vec<(Clock, T)>>,
) -> Vec<(i64, Option<i64>, T)> {
    let mut events: Vec<_> = old
        .into_iter()
        .flatten()
        .copied()
        .chain(new.into_iter().flatten().copied())
        .collect();
    events.sort_unstable_by_key(|(clock, _)| *clock);
    let mut times = Vec::new();
    for ((at, _, _), value) in events {
        if let Some((last_at, last_value)) = times.last_mut() {
            if *last_at == at {
                *last_value = value;
                continue;
            }
        }
        times.push((at, value));
    }
    times.dedup_by(|right, left| right.1 == left.1);
    times
        .iter()
        .enumerate()
        .map(|(i, &(start, value))| (start, times.get(i + 1).map(|&(at, _)| at), value))
        .collect()
}
impl EffectiveTopology {
    pub(crate) fn append_if_current(&mut self, commit: &Commit) {
        if commit.parents.len() > 1 || commit.parents.first().copied().unwrap_or(0) != self.head {
            return;
        }
        self.head = commit.id;
        let Some(metadata) = &commit.publication else {
            self.precise = false;
            return;
        };
        let additions = commit
            .delta
            .edges_added
            .iter()
            .map(|&(id, from, to)| (id, (from, to)))
            .collect();
        let Ok(patch) = TopologyPatch::from_metadata(metadata, commit.id, &additions) else {
            self.precise = false;
            return;
        };
        for (id, events) in patch.nodes {
            self.nodes.entry(id).or_default().extend(events);
        }
        for (id, events) in patch.edges {
            for (_, endpoints) in &events {
                if let Some((from, to)) = endpoints {
                    self.incident.entry(*from).or_default().insert(id);
                    self.incident.entry(*to).or_default().insert(id);
                }
            }
            self.edges.entry(id).or_default().extend(events);
        }
    }
    fn validate(&self, patch: &TopologyPatch) -> GraphResult<()> {
        let mut affected: HashSet<_> = patch.edges.keys().copied().collect();
        for node in patch.nodes.keys() {
            affected.extend(self.incident.get(node).into_iter().flatten().copied());
        }
        let mut lifetimes = HashMap::new();
        for edge in affected {
            for (start, end, endpoints) in segments(self.edges.get(&edge), patch.edges.get(&edge)) {
                let Some((from, to)) = endpoints else {
                    continue;
                };
                for node in [from, to] {
                    let intervals = lifetimes
                        .entry(node)
                        .or_insert_with(|| segments(self.nodes.get(&node), patch.nodes.get(&node)));
                    let position = intervals.partition_point(|(at, _, _)| *at <= start);
                    let covered = position
                        .checked_sub(1)
                        .and_then(|i| intervals.get(i))
                        .is_some_and(|(_, until, alive)| {
                            *alive
                                && match (until, end) {
                                    (None, _) => true,
                                    (Some(_), None) => false,
                                    (Some(until), Some(end)) => *until >= end,
                                }
                        });
                    if !covered {
                        return Err(GraphError::InvalidInput(format!("Invalid effective endpoint timeline: edge {edge} at {start} extends outside node {node}'s active lifetime")));
                    }
                }
            }
        }
        Ok(())
    }
}
impl HistoryForest {
    pub(crate) fn validate_effective_topology(
        &self,
        head: StateId,
        patch: &TopologyPatch,
        explicit: bool,
    ) -> GraphResult<()> {
        if patch.nodes.is_empty() && patch.edges.is_empty() {
            return Ok(());
        }
        let mut cache = self.effective_topology.lock();
        if cache.head != head {
            let mut lineage = Vec::new();
            let mut seen = HashSet::new();
            let mut current = head;
            while current != 0 {
                if !seen.insert(current) {
                    return Err(GraphError::InvalidInput(
                        "Cycle in effective topology ancestry".into(),
                    ));
                }
                let commit = self.get_commit(current)?;
                if commit.parents.len() > 1 {
                    return Err(GraphError::InvalidInput(
                        "Effective topology validation requires single-parent ancestry".into(),
                    ));
                }
                current = commit.parents.first().copied().unwrap_or(0);
                lineage.push(commit);
            }
            *cache = EffectiveTopology::default();
            for commit in lineage.into_iter().rev() {
                cache.append_if_current(&commit);
            }
        }
        if !cache.precise {
            if explicit {
                return Err(GraphError::InvalidInput("Explicit effective topology changes require precisely timed ancestry; legacy manual history has no effective lifetimes".into()));
            }
            return Ok(()); // Preserve untimed legacy compatibility; no temporal guarantee.
        }
        cache.validate(patch)
    }
}
