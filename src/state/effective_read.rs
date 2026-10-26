//! Effective-time replay of a pinned, precisely timed publication lineage.
use super::{Commit, GraphSnapshot, PinnedReadState};
use crate::{
    types::GraphType, AttrName, AttrValue, ChangeKey, GraphError, GraphResult, GraphView, StateId,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// The branch/head and clocks captured for an immutable effective-time read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadTime {
    /// Branch selected when the read was prepared.
    pub branch: String,
    /// Published head whose ancestry bounds this read.
    pub published_head: StateId,
    /// Requested effective time, signed Unix microseconds.
    pub effective_at: i64,
    /// Optional engine-recorded acceptance cutoff, signed Unix microseconds.
    pub recorded_by: Option<i64>,
}

/// Owned immutable input: replay can run without a mutable graph or Python GIL.
#[derive(Debug)]
pub struct EffectiveReadPlan {
    pub(crate) commits: Vec<Arc<Commit>>,
    pub(crate) graph_type: GraphType,
    pub(crate) time: ReadTime,
}
enum Change<'a> {
    NodeAdded(usize),
    EdgeRemoved(usize),
    NodeRemoved(usize),
    EdgeAdded(usize, usize, usize),
    NodeAttr(usize, &'a AttrName, &'a AttrValue),
    EdgeAttr(usize, &'a AttrName, &'a AttrValue),
}
impl EffectiveReadPlan {
    /// Reconstruct an immutable state. Equal effective times use accepted lineage
    /// order; properties outside an entity's active lifetime do not carry into a
    /// later activation. Invalid endpoint intervals through T reject the read.
    pub fn read_view(self) -> GraphResult<GraphView> {
        let invalid = |message: &str| GraphError::InvalidInput(message.into());
        let mut events = Vec::new();
        for (accepted, commit) in self.commits.iter().enumerate() {
            let metadata = commit.publication.as_ref().ok_or_else(|| invalid("Effective-time reads require precise publication metadata; legacy manual versions are unsupported"))?;
            if self
                .time
                .recorded_by
                .is_some_and(|cutoff| metadata.recorded_at > cutoff)
            {
                continue;
            }
            let mut keys = HashSet::new();
            let mut push = |key: ChangeKey, phase, change| -> GraphResult<()> {
                if !keys.insert(key.clone()) {
                    return Err(invalid("Duplicate logical change in publication"));
                }
                let at = *metadata
                    .change_times
                    .get(&key)
                    .ok_or_else(|| invalid("Missing effective time for published change"))?;
                if at <= self.time.effective_at {
                    events.push((at, accepted, phase, change));
                }
                Ok(())
            };
            let delta = &commit.delta;
            for &id in &delta.nodes_added {
                push(ChangeKey::NodeAdded(id), 0, Change::NodeAdded(id))?;
            }
            for &id in &delta.edges_removed {
                push(ChangeKey::EdgeRemoved(id), 1, Change::EdgeRemoved(id))?;
            }
            for &id in &delta.nodes_removed {
                push(ChangeKey::NodeRemoved(id), 2, Change::NodeRemoved(id))?;
            }
            for &(id, from, to) in &delta.edges_added {
                push(ChangeKey::EdgeAdded(id), 3, Change::EdgeAdded(id, from, to))?;
            }
            for (id, name, _, value) in &delta.node_attr_changes {
                push(
                    ChangeKey::NodeAttribute(*id, name.clone()),
                    4,
                    Change::NodeAttr(*id, name, value),
                )?;
            }
            for (id, name, _, value) in &delta.edge_attr_changes {
                push(
                    ChangeKey::EdgeAttribute(*id, name.clone()),
                    5,
                    Change::EdgeAttr(*id, name, value),
                )?;
            }
        }
        events.sort_by_key(|(at, accepted, phase, _)| (*at, *accepted, *phase));
        let mut snapshot =
            GraphSnapshot::empty_with_type(self.time.published_head, self.graph_type);
        let mut nodes = HashSet::new();
        let mut incident: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut touched = HashSet::new();
        for (index, (at, _, _, change)) in events.iter().enumerate() {
            match change {
                Change::NodeAdded(id) => {
                    nodes.insert(*id);
                    snapshot.node_attributes.remove(id);
                    touched.extend(incident.get(id).into_iter().flatten().copied());
                }
                Change::NodeRemoved(id) => {
                    nodes.remove(id);
                    snapshot.node_attributes.remove(id);
                    touched.extend(incident.get(id).into_iter().flatten().copied());
                }
                Change::EdgeRemoved(id) => {
                    if let Some((from, to)) = snapshot.edges.remove(id) {
                        if let Some(edges) = incident.get_mut(&from) {
                            edges.remove(id);
                        }
                        if let Some(edges) = incident.get_mut(&to) {
                            edges.remove(id);
                        }
                    }
                    snapshot.edge_attributes.remove(id);
                    touched.insert(*id);
                }
                Change::EdgeAdded(id, from, to) => {
                    if let Some((old_from, old_to)) = snapshot.edges.insert(*id, (*from, *to)) {
                        if let Some(edges) = incident.get_mut(&old_from) {
                            edges.remove(id);
                        }
                        if let Some(edges) = incident.get_mut(&old_to) {
                            edges.remove(id);
                        }
                    }
                    incident.entry(*from).or_default().insert(*id);
                    incident.entry(*to).or_default().insert(*id);
                    snapshot.edge_attributes.remove(id);
                    touched.insert(*id);
                }
                Change::NodeAttr(id, name, value) if nodes.contains(id) => {
                    if matches!(value, AttrValue::SubgraphRef(_)) {
                        return Err(invalid("Effective-time pooled subgraph references require source-pool reconstruction"));
                    }
                    snapshot
                        .node_attributes
                        .entry(*id)
                        .or_default()
                        .insert((*name).clone(), (*value).clone());
                }
                Change::EdgeAttr(id, name, value) if snapshot.edges.contains_key(id) => {
                    if matches!(value, AttrValue::SubgraphRef(_)) {
                        return Err(invalid("Effective-time pooled subgraph references require source-pool reconstruction"));
                    }
                    snapshot
                        .edge_attributes
                        .entry(*id)
                        .or_default()
                        .insert((*name).clone(), (*value).clone());
                }
                _ => {}
            }
            if events.get(index + 1).is_none_or(|next| next.0 != *at) {
                for edge in touched.drain() {
                    if let Some((from, to)) = snapshot.edges.get(&edge) {
                        if !nodes.contains(from) || !nodes.contains(to) {
                            return Err(GraphError::InvalidInput(format!("Invalid effective endpoint timeline at {at}: edge {edge} has an inactive endpoint")));
                        }
                    }
                }
            }
        }
        snapshot.active_nodes = nodes.into_iter().collect();
        snapshot.active_nodes.sort_unstable();
        Ok(GraphView::all(
            PinnedReadState::from_snapshot(snapshot)?.with_read_time(self.time),
        ))
    }
}
