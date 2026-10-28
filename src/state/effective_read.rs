//! Effective-time replay of a pinned, precisely timed publication lineage.
use super::{Commit, GraphSnapshot, PinnedReadState};
use crate::{
    types::GraphType, AttrName, AttrValue, ChangeKey, GraphError, GraphResult, GraphView, StateId,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{Arc, OnceLock},
};

mod property_intervals;
pub use property_intervals::PropertyInterval;
type PropertyPostings = HashMap<(bool, usize), HashMap<AttrName, Vec<usize>>>;

/// The branch/head and clocks captured for an immutable effective-time read.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReadTime {
    /// Branch selected when the read was prepared.
    pub branch: String,
    /// Published head whose ancestry bounds this read.
    pub published_head: StateId,
    /// Requested effective time, signed Unix microseconds.
    pub effective_at: i64,
    /// Optional engine-recorded acceptance cutoff, signed Unix microseconds.
    pub recorded_by: Option<i64>,
    /// Caller-assigned clocks for legacy versions; never inferred or written back.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub legacy_clocks: BTreeMap<StateId, (i64, i64)>,
    /// Explicit legacy source bindings, from membership ID to frozen revision ID.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub group_sources: BTreeMap<crate::types::SubgraphId, crate::types::SubgraphId>,
}

/// Explicit reconstruction inputs for information absent from legacy storage.
#[derive(Debug, Clone, Default)]
pub struct EffectiveReadOptions {
    /// Restrict acceptance time only when supplied.
    pub recorded_by: Option<i64>,
    /// Caller-assigned publication clocks for legacy versions.
    pub legacy_publications: BTreeMap<StateId, crate::PublicationMetadata>,
    /// Explicit source views for legacy membership-only references.
    pub group_sources: BTreeMap<crate::types::SubgraphId, GraphView>,
}
/// Cached immutable ancestry and one lazily built effective event index.
#[derive(Debug)]
pub(crate) struct ReadContext {
    commits: Arc<[Arc<Commit>]>,
    index: OnceLock<Arc<EffectiveReadIndex>>,
}
impl ReadContext {
    pub(crate) fn new(commits: Vec<Arc<Commit>>) -> Self {
        Self {
            commits: commits.into(),
            index: OnceLock::new(),
        }
    }
    pub(crate) fn recorded_commits(&self) -> Arc<[Arc<Commit>]> {
        Arc::clone(&self.commits)
    }
    pub(crate) fn commit_at_or_before(&self, timestamp: u64) -> Option<StateId> {
        self.commits
            .iter()
            .filter(|commit| commit.timestamp <= timestamp)
            .max_by_key(|commit| (commit.timestamp, commit.id))
            .map(|commit| commit.id)
    }

    pub(crate) fn validate_legacy(
        &self,
        legacy: &BTreeMap<StateId, crate::PublicationMetadata>,
    ) -> GraphResult<()> {
        if legacy.is_empty() {
            return Ok(());
        }
        let versions: HashMap<_, _> = self
            .commits
            .iter()
            .map(|commit| (commit.id, commit.publication.is_some()))
            .collect();
        for id in legacy.keys() {
            match versions.get(id) {
                None => {
                    return Err(GraphError::InvalidInput(
                        "Legacy clock override is outside the selected first-parent lineage".into(),
                    ))
                }
                Some(true) => {
                    return Err(GraphError::InvalidInput(
                        "Cannot override precise publication clocks".into(),
                    ))
                }
                _ => {}
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
struct EffectiveReadIndex {
    events: Vec<(i64, usize, u8, Position)>,
}
#[derive(Debug, Clone, Copy)]
enum Position {
    NodeAdded(usize),
    NodeRemoved(usize),
    EdgeAdded(usize, usize, usize),
    EdgeEndpoints(usize, usize, usize),
    EdgeRemoved(usize),
    NodeAttr(usize),
    EdgeAttr(usize),
}
/// Owned immutable input: no mutable graph or GIL is needed for indexed replay.
#[derive(Debug, Clone)]
pub struct EffectiveReadPlan {
    pub(crate) context: Arc<ReadContext>,
    pub(crate) graph_type: GraphType,
    pub(crate) time: ReadTime,
    pub(crate) groups:
        Arc<HashMap<crate::types::SubgraphId, crate::storage::retained_group::RetainedGroup>>,
    pub(crate) legacy_publications: BTreeMap<StateId, crate::PublicationMetadata>,
}
enum Change<'a> {
    NodeAdded(usize),
    EdgeRemoved(usize),
    NodeRemoved(usize),
    EdgeAdded(usize, usize, usize),
    EdgeEndpoints(usize, usize, usize),
    NodeAttr(usize, &'a AttrName, &'a Option<AttrValue>),
    EdgeAttr(usize, &'a AttrName, &'a Option<AttrValue>),
}
impl EffectiveReadIndex {
    fn build(
        commits: &[Arc<Commit>],
        legacy: &BTreeMap<StateId, crate::PublicationMetadata>,
    ) -> GraphResult<Self> {
        let invalid = |message: &str| GraphError::InvalidInput(message.into());
        let mut events = Vec::new();
        for (accepted, commit) in commits.iter().enumerate() {
            let metadata = commit.publication.as_ref().or_else(||legacy.get(&commit.id)).ok_or_else(|| invalid("Effective-time reads require precise publication metadata; provide explicit legacy clocks for manual versions"))?;
            let mut keys = HashSet::new();
            let mut push = |key: ChangeKey, phase, change| -> GraphResult<()> {
                if !keys.insert(key.clone()) {
                    return Err(invalid("Duplicate logical change in publication"));
                }
                let at = metadata
                    .change_times
                    .get(&key)
                    .copied()
                    .or_else(|| {
                        legacy
                            .contains_key(&commit.id)
                            .then_some(metadata.effective_at)
                    })
                    .ok_or_else(|| invalid("Missing effective time for published change"))?;
                events.push((at, accepted, phase, change));
                Ok(())
            };
            let delta = &commit.delta;
            for &id in &delta.nodes_added {
                push(ChangeKey::NodeAdded(id), 0, Position::NodeAdded(id))?;
            }
            for &id in &delta.edges_removed {
                push(ChangeKey::EdgeRemoved(id), 1, Position::EdgeRemoved(id))?;
            }
            for &id in &delta.nodes_removed {
                push(ChangeKey::NodeRemoved(id), 2, Position::NodeRemoved(id))?;
            }
            for &(id, from, to) in &delta.edges_added {
                push(
                    ChangeKey::EdgeAdded(id),
                    3,
                    Position::EdgeAdded(id, from, to),
                )?;
            }
            for change in &delta.edge_endpoint_changes {
                push(
                    ChangeKey::EdgeEndpoints(change.edge_id),
                    4,
                    Position::EdgeEndpoints(
                        change.edge_id,
                        change.new_endpoints.0,
                        change.new_endpoints.1,
                    ),
                )?;
            }
            for (row, (id, name, _, _)) in delta.node_attr_changes.iter().enumerate() {
                push(
                    ChangeKey::NodeAttribute(*id, name.clone()),
                    4,
                    Position::NodeAttr(row),
                )?;
            }
            for (row, (id, name, _, _)) in delta.edge_attr_changes.iter().enumerate() {
                push(
                    ChangeKey::EdgeAttribute(*id, name.clone()),
                    5,
                    Position::EdgeAttr(row),
                )?;
            }
            if legacy
                .get(&commit.id)
                .is_some_and(|metadata| metadata.change_times.keys().any(|key| !keys.contains(key)))
            {
                return Err(invalid(
                    "Legacy clock override names a change absent from its version",
                ));
            }
        }
        events.sort_by_key(|(at, accepted, phase, _)| (*at, *accepted, *phase));
        Ok(Self { events })
    }
}
impl EffectiveReadPlan {
    /// Reconstruct an immutable state. Equal effective times use accepted lineage
    /// order; properties outside an entity's active lifetime do not carry into a
    /// later activation. Invalid endpoint intervals through T reject the read.
    pub fn read_view(self) -> GraphResult<GraphView> {
        let index = self.event_index()?;
        self.read_indexed(&index)
    }
    fn event_index(&self) -> GraphResult<Arc<EffectiveReadIndex>> {
        Ok(if self.legacy_publications.is_empty() {
            if let Some(index) = self.context.index.get() {
                index.clone()
            } else {
                let built = Arc::new(EffectiveReadIndex::build(
                    &self.context.commits,
                    &self.legacy_publications,
                )?);
                let _ = self.context.index.set(built.clone());
                self.context.index.get().cloned().unwrap_or(built)
            }
        } else {
            Arc::new(EffectiveReadIndex::build(
                &self.context.commits,
                &self.legacy_publications,
            )?)
        })
    }
    fn read_indexed(self, index: &EffectiveReadIndex) -> GraphResult<GraphView> {
        let invalid = |message: &str| GraphError::InvalidInput(message.into());
        let end = index
            .events
            .partition_point(|(at, _, _, _)| *at <= self.time.effective_at);
        let events: Vec<_> = index.events[..end]
            .iter()
            .filter_map(|&(at, accepted, phase, position)| {
                let commit = &self.context.commits[accepted];
                let metadata = commit
                    .publication
                    .as_ref()
                    .or_else(|| self.legacy_publications.get(&commit.id))?;
                if self
                    .time
                    .recorded_by
                    .is_some_and(|cutoff| metadata.recorded_at > cutoff)
                {
                    return None;
                }
                let change = match position {
                    Position::NodeAdded(id) => Change::NodeAdded(id),
                    Position::NodeRemoved(id) => Change::NodeRemoved(id),
                    Position::EdgeAdded(id, from, to) => Change::EdgeAdded(id, from, to),
                    Position::EdgeEndpoints(id, from, to) => Change::EdgeEndpoints(id, from, to),
                    Position::EdgeRemoved(id) => Change::EdgeRemoved(id),
                    Position::NodeAttr(row) => {
                        let (id, name, _, value) = &commit.delta.node_attr_changes[row];
                        Change::NodeAttr(*id, name, value)
                    }
                    Position::EdgeAttr(row) => {
                        let (id, name, _, value) = &commit.delta.edge_attr_changes[row];
                        Change::EdgeAttr(*id, name, value)
                    }
                };
                Some((at, accepted, phase, change))
            })
            .collect();
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
                Change::EdgeAdded(id, from, to) | Change::EdgeEndpoints(id, from, to) => {
                    if matches!(change, Change::EdgeEndpoints(..))
                        && !snapshot.edges.contains_key(id)
                    {
                        return Err(GraphError::InvalidInput(format!(
                            "Endpoint revision of inactive edge {id} at {at}"
                        )));
                    }
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
                    if matches!(change, Change::EdgeAdded(..)) {
                        snapshot.edge_attributes.remove(id);
                    }
                    touched.insert(*id);
                }
                Change::NodeAttr(id, name, value) if nodes.contains(id) => {
                    let attrs = snapshot.node_attributes.entry(*id).or_default();
                    if let Some(value) = value {
                        attrs.insert((*name).clone(), value.clone());
                    } else {
                        attrs.remove(*name);
                    }
                }
                Change::EdgeAttr(id, name, value) if snapshot.edges.contains_key(id) => {
                    let attrs = snapshot.edge_attributes.entry(*id).or_default();
                    if let Some(value) = value {
                        attrs.insert((*name).clone(), value.clone());
                    } else {
                        attrs.remove(*name);
                    }
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
        for id in crate::storage::retained_group::references(&snapshot) {
            if !self.groups.contains_key(&id) {
                return Err(invalid(
                    "Group reference has no frozen source revision; pin an explicit source view",
                ));
            }
        }
        snapshot.active_nodes = nodes.into_iter().collect();
        snapshot.active_nodes.sort_unstable();
        Ok(GraphView::all(
            PinnedReadState::from_snapshot(snapshot)?
                .with_groups(self.groups)
                .with_read_time(self.time),
        ))
    }
}

/// An accepted logical change, ordered by effective time and canonical replay order.
/// Values remain in retained deltas; this record identifies their entity/property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveTimeEvent {
    pub state_id: StateId,
    pub effective_at: i64,
    pub recorded_at: i64,
    pub change: ChangeKey,
}
/// Structural activation interval [start, end); None is an unbounded end.
/// Attribute-based validity and endpoint predicates are separate from activation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationInterval {
    pub start: i64,
    pub end: Option<i64>,
}
/// Explicit immutable event index over one retained first-parent lineage.
/// Captures survive owner edits/closure/pruning; replacement is always explicit.
#[derive(Debug, Clone)]
pub struct EffectiveTimeIndex {
    plan: EffectiveReadPlan,
    events: Arc<EffectiveReadIndex>,
    by_entity: Arc<HashMap<(bool, usize), Vec<usize>>>,
    by_activation: Arc<HashMap<(bool, usize), Vec<usize>>>,
    by_property: Arc<PropertyPostings>,
}
impl Position {
    fn entity(self, delta: &super::Delta) -> (bool, usize) {
        match self {
            Self::NodeAdded(id) | Self::NodeRemoved(id) => (true, id),
            Self::EdgeAdded(id, ..) | Self::EdgeRemoved(id) | Self::EdgeEndpoints(id, ..) => {
                (false, id)
            }
            Self::NodeAttr(row) => (true, delta.node_attr_changes[row].0),
            Self::EdgeAttr(row) => (false, delta.edge_attr_changes[row].0),
        }
    }
    fn key(self, delta: &super::Delta) -> ChangeKey {
        match self {
            Self::NodeAdded(id) => ChangeKey::NodeAdded(id),
            Self::NodeRemoved(id) => ChangeKey::NodeRemoved(id),
            Self::EdgeAdded(id, ..) => ChangeKey::EdgeAdded(id),
            Self::EdgeRemoved(id) => ChangeKey::EdgeRemoved(id),
            Self::EdgeEndpoints(id, ..) => ChangeKey::EdgeEndpoints(id),
            Self::NodeAttr(row) => {
                let (id, name, ..) = &delta.node_attr_changes[row];
                ChangeKey::NodeAttribute(*id, name.clone())
            }
            Self::EdgeAttr(row) => {
                let (id, name, ..) = &delta.edge_attr_changes[row];
                ChangeKey::EdgeAttribute(*id, name.clone())
            }
        }
    }
}
impl EffectiveReadPlan {
    /// Explicitly index the complete prepared lineage, independently of effective_at.
    /// Apply recorded cutoffs per query, rather than silently binding one at build.
    pub fn build_index(self) -> GraphResult<EffectiveTimeIndex> {
        if self.time.recorded_by.is_some() {
            return Err(GraphError::InvalidInput(
                "Build a temporal index without recorded_by; supply the cutoff when querying"
                    .into(),
            ));
        }
        let events = self.event_index()?;
        let mut by_entity: HashMap<_, Vec<_>> = HashMap::new();
        let mut by_activation: HashMap<_, Vec<_>> = HashMap::new();
        let mut by_property = PropertyPostings::new();
        for (ordinal, &(_, accepted, _, position)) in events.events.iter().enumerate() {
            let delta = &self.context.commits[accepted].delta;
            let property = match position {
                Position::NodeAttr(row) => Some(&delta.node_attr_changes[row].1),
                Position::EdgeAttr(row) => Some(&delta.edge_attr_changes[row].1),
                _ => None,
            };
            if let Some(name) = property {
                by_property
                    .entry(position.entity(delta))
                    .or_default()
                    .entry(name.clone())
                    .or_default()
                    .push(ordinal);
            }
            by_entity
                .entry(position.entity(&self.context.commits[accepted].delta))
                .or_default()
                .push(ordinal);
            if matches!(
                position,
                Position::NodeAdded(_)
                    | Position::NodeRemoved(_)
                    | Position::EdgeAdded(..)
                    | Position::EdgeRemoved(_)
            ) {
                by_activation
                    .entry(position.entity(&self.context.commits[accepted].delta))
                    .or_default()
                    .push(ordinal);
            }
        }
        Ok(EffectiveTimeIndex {
            plan: self,
            events,
            by_entity: Arc::new(by_entity),
            by_activation: Arc::new(by_activation),
            by_property: Arc::new(by_property),
        })
    }
}
impl EffectiveTimeIndex {
    /// Captured branch name; later branch switching cannot change this index.
    pub fn branch(&self) -> &str {
        &self.plan.time.branch
    }
    /// Last accepted version bounding this captured lineage.
    pub fn captured_head(&self) -> StateId {
        self.plan.time.published_head
    }
    /// Number of logical events, including inactive entities and unset properties.
    pub fn event_count(&self) -> usize {
        self.events.events.len()
    }
    /// Reconstruct through the canonical engine using the already built event index.
    pub fn read_view_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<GraphView> {
        let mut plan = self.plan.clone();
        plan.time.effective_at = effective_at;
        plan.time.recorded_by = recorded_by;
        plan.read_indexed(&self.events)
    }
    fn event(&self, ordinal: usize, cutoff: Option<i64>) -> Option<EffectiveTimeEvent> {
        let (at, accepted, _, position) = self.events.events[ordinal];
        let commit = &self.plan.context.commits[accepted];
        let metadata = commit
            .publication
            .as_ref()
            .or_else(|| self.plan.legacy_publications.get(&commit.id))?;
        if cutoff.is_some_and(|u| metadata.recorded_at > u) {
            return None;
        }
        Some(EffectiveTimeEvent {
            state_id: commit.id,
            effective_at: at,
            recorded_at: metadata.recorded_at,
            change: position.key(&commit.delta),
        })
    }
    fn check_range(start: i64, end: i64) -> GraphResult<()> {
        if start > end {
            return Err(GraphError::InvalidInput(
                "Temporal range start exceeds end".into(),
            ));
        }
        Ok(())
    }
    /// Accepted change events in the inclusive effective window [start, end].
    /// This is an event stream, not a filter for current liveness or fact validity.
    pub fn events_between(
        &self,
        start: i64,
        end: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<Vec<EffectiveTimeEvent>> {
        Self::check_range(start, end)?;
        let lower = self.events.events.partition_point(|(at, ..)| *at < start);
        let upper = self.events.events.partition_point(|(at, ..)| *at <= end);
        Ok((lower..upper)
            .filter_map(|n| self.event(n, recorded_by))
            .collect())
    }
    /// Entity-local change events, including its retained inactive periods.
    fn entity_events(
        &self,
        node: bool,
        id: usize,
        start: i64,
        end: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<Vec<EffectiveTimeEvent>> {
        Self::check_range(start, end)?;
        let Some(rows) = self.by_entity.get(&(node, id)) else {
            return Ok(Vec::new());
        };
        let lower = rows.partition_point(|&n| self.events.events[n].0 < start);
        let upper = rows.partition_point(|&n| self.events.events[n].0 <= end);
        Ok(rows[lower..upper]
            .iter()
            .filter_map(|&n| self.event(n, recorded_by))
            .collect())
    }
    /// Node-local events in an inclusive effective-time window.
    pub fn node_events(
        &self,
        id: usize,
        start: i64,
        end: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<Vec<EffectiveTimeEvent>> {
        self.entity_events(true, id, start, end, recorded_by)
    }
    /// Edge-local events, including endpoint revisions and property unsets.
    pub fn edge_events(
        &self,
        id: usize,
        start: i64,
        end: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<Vec<EffectiveTimeEvent>> {
        self.entity_events(false, id, start, end, recorded_by)
    }
    /// Node structural activation intervals; these do not infer attribute validity.
    pub fn node_activation_intervals(
        &self,
        id: usize,
        recorded_by: Option<i64>,
    ) -> Vec<ActivationInterval> {
        self.activation_intervals(true, id, recorded_by)
    }
    /// Edge structural activation intervals; endpoint validity is checked by reads.
    pub fn edge_activation_intervals(
        &self,
        id: usize,
        recorded_by: Option<i64>,
    ) -> Vec<ActivationInterval> {
        self.activation_intervals(false, id, recorded_by)
    }
    /// Structural activation from accepted add/remove events. Equal-time changes
    /// resolve in replay order; zero-duration gaps and intervals are coalesced.
    /// This does not validate graph endpoints or attribute-based fact validity.
    fn activation_intervals(
        &self,
        node: bool,
        id: usize,
        recorded_by: Option<i64>,
    ) -> Vec<ActivationInterval> {
        let mut toggles: Vec<(i64, bool)> = Vec::new();
        for &ordinal in self.by_activation.get(&(node, id)).into_iter().flatten() {
            let Some(event) = self.event(ordinal, recorded_by) else {
                continue;
            };
            let active = match event.change {
                ChangeKey::NodeAdded(_) | ChangeKey::EdgeAdded(_) => true,
                ChangeKey::NodeRemoved(_) | ChangeKey::EdgeRemoved(_) => false,
                _ => continue,
            };
            if let Some(last) = toggles
                .last_mut()
                .filter(|last| last.0 == event.effective_at)
            {
                last.1 = active;
            } else {
                toggles.push((event.effective_at, active));
            }
        }
        let mut intervals = Vec::new();
        let mut open = None;
        for (at, active) in toggles {
            match (open, active) {
                (None, true) => open = Some(at),
                (Some(start), false) => {
                    intervals.push(ActivationInterval {
                        start,
                        end: Some(at),
                    });
                    open = None;
                }
                _ => {}
            }
        }
        if let Some(start) = open {
            intervals.push(ActivationInterval { start, end: None });
        }
        intervals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_history_cannot_hide_an_invalid_earlier_endpoint_interval() {
        let mut history = crate::state::HistoryForest::new();
        let mut changes = crate::state::ChangeSet::new();
        changes.nodes_added = vec![0, 1];
        changes.edges_added = vec![(0, 0, 1)];
        let first = history
            .create_commit_with_publication(
                changes,
                "raw".into(),
                "test".into(),
                None,
                Some(crate::PublicationMetadata {
                    recorded_at: 100,
                    effective_at: 10,
                    change_times: Default::default(),
                }),
            )
            .unwrap();
        let mut changes = crate::state::ChangeSet::new();
        changes.nodes_removed = vec![0];
        changes.edges_removed = vec![0];
        let second = history
            .create_commit_with_publication(
                changes,
                "invalid raw".into(),
                "test".into(),
                Some(first),
                Some(crate::PublicationMetadata {
                    recorded_at: 101,
                    effective_at: 20,
                    change_times: std::collections::BTreeMap::from([(
                        crate::ChangeKey::EdgeRemoved(0),
                        30,
                    )]),
                }),
            )
            .unwrap();
        let plan = EffectiveReadPlan {
            context: Arc::new(ReadContext::new(vec![
                history.get_commit(first).unwrap(),
                history.get_commit(second).unwrap(),
            ])),
            groups: Default::default(),
            legacy_publications: Default::default(),
            graph_type: GraphType::Undirected,
            time: ReadTime {
                branch: "raw".into(),
                published_head: second,
                effective_at: 40,
                recorded_by: None,
                legacy_clocks: Default::default(),
                group_sources: Default::default(),
            },
        };
        assert!(plan
            .read_view()
            .unwrap_err()
            .to_string()
            .contains("endpoint timeline at 20"));
    }
}
