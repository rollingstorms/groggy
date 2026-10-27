//! Pure three-way snapshot planning with explicit caller conflict answers.
//! Publication and identity/schema checks belong to the owning graph coordinator;
//! no owner is mutated here.
use super::state::{validate_snapshot, GraphSnapshot};
use crate::{types::*, GraphError, GraphResult};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Whole typed property map carried by a logical merge entity.
pub type MergeProperties = HashMap<AttrName, AttrValue>;

/// A complete entity value at one merge input (absence means deleted/not created).
#[derive(Debug, Clone)]
pub struct MergeEntity {
    /// None for nodes; ordered endpoints for edges.
    pub endpoints: Option<(NodeId, NodeId)>,
    pub properties: MergeProperties,
}
/// Explicit conflict inputs. Values retain whole types and exact floating bits.
#[derive(Debug, Clone)]
pub enum SnapshotMergeConflict {
    /// Delete/modify or colliding creations of a node ID.
    Node {
        id: NodeId,
        base: Option<MergeEntity>,
        left: Option<MergeEntity>,
        right: Option<MergeEntity>,
    },
    /// Delete/modify or colliding creations of an edge ID.
    Edge {
        id: EdgeId,
        base: Option<MergeEntity>,
        left: Option<MergeEntity>,
        right: Option<MergeEntity>,
    },
    /// Both sides changed the same property differently. None is absence, not Null.
    NodeProperty {
        id: NodeId,
        name: AttrName,
        base: Option<AttrValue>,
        left: Option<AttrValue>,
        right: Option<AttrValue>,
    },
    /// Both sides changed the same edge property differently.
    EdgeProperty {
        id: EdgeId,
        name: AttrName,
        base: Option<AttrValue>,
        left: Option<AttrValue>,
        right: Option<AttrValue>,
    },
    /// Both sides changed the ordered pair differently.
    EdgeEndpoints {
        id: EdgeId,
        base: (NodeId, NodeId),
        left: (NodeId, NodeId),
        right: (NodeId, NodeId),
    },
    /// Otherwise independent changes would leave a dangling edge. No implicit cascade.
    Topology {
        id: EdgeId,
        endpoints: (NodeId, NodeId),
    },
}
/// Select an exact retained input value without round-tripping through a language
/// binding's default numeric type conversion.
#[derive(Debug, Clone, Copy)]
pub enum SnapshotMergeSide {
    Base,
    Left,
    Right,
}
impl SnapshotMergeConflict {
    /// Expand an explicit input-side choice into a whole typed answer. Topology
    /// repair has no implicit winning side and requires a concrete repair answer.
    pub fn resolution_for(&self, side: SnapshotMergeSide) -> GraphResult<SnapshotMergeResolution> {
        let select = |base, left, right| match side {
            SnapshotMergeSide::Base => base,
            SnapshotMergeSide::Left => left,
            SnapshotMergeSide::Right => right,
        };
        match self {
            Self::Node {
                base, left, right, ..
            }
            | Self::Edge {
                base, left, right, ..
            } => Ok(SnapshotMergeResolution::Entity(
                select(base, left, right).clone(),
            )),
            Self::NodeProperty {
                base, left, right, ..
            }
            | Self::EdgeProperty {
                base, left, right, ..
            } => Ok(SnapshotMergeResolution::Property(
                match side {
                    SnapshotMergeSide::Base => base,
                    SnapshotMergeSide::Left => left,
                    SnapshotMergeSide::Right => right,
                }
                .clone(),
            )),
            Self::EdgeEndpoints {
                base, left, right, ..
            } => Ok(SnapshotMergeResolution::Endpoints(match side {
                SnapshotMergeSide::Base => *base,
                SnapshotMergeSide::Left => *left,
                SnapshotMergeSide::Right => *right,
            })),
            Self::Topology { .. } => Err(GraphError::InvalidInput(
                "Topology conflicts require explicit edge removal or endpoint restoration".into(),
            )),
        }
    }
}
/// An explicit caller answer for one typed conflict. No default winner or
/// implicit deletion is selected by the engine.
#[derive(Debug, Clone)]
pub enum SnapshotMergeResolution {
    /// Complete node/edge record; None explicitly removes that entity.
    Entity(Option<MergeEntity>),
    /// Whole typed property; None explicitly means absence.
    Property(Option<AttrValue>),
    /// Ordered edge endpoints.
    Endpoints((NodeId, NodeId)),
    /// Explicitly discard a resulting dangling edge and its properties.
    DropEdge,
    /// Explicitly retain missing endpoint nodes with caller-supplied properties.
    /// Exactly the still-missing endpoints must be supplied, without duplicates.
    EndpointNodes(Vec<(NodeId, MergeProperties)>),
}
/// Deterministic direct conflicts and a private candidate. Conflicted candidates
/// cannot escape as usable snapshots without explicit caller resolutions.
#[derive(Debug)]
pub struct SnapshotMergePlan {
    candidate: GraphSnapshot,
    conflicts: Vec<SnapshotMergeConflict>,
}
impl SnapshotMergePlan {
    /// Deterministic node, edge, property and resulting-topology conflicts.
    pub fn conflicts(&self) -> &[SnapshotMergeConflict] {
        &self.conflicts
    }
    /// Resolve direct conflicts with whole typed caller answers, then recompute
    /// resulting topology and request explicit answers for dangling edges. No
    /// input/owner mutation is performed, even if the callback fails partway.
    /// Pool IDs/references, external identity and schema still require owner checks.
    pub fn resolve_with(
        mut self,
        mut resolve: impl FnMut(&SnapshotMergeConflict) -> GraphResult<SnapshotMergeResolution>,
    ) -> GraphResult<GraphSnapshot> {
        let mut nodes: HashSet<_> = self.candidate.active_nodes.iter().copied().collect();
        for conflict in &self.conflicts {
            if matches!(conflict, SnapshotMergeConflict::Topology { .. }) {
                continue;
            }
            let answer = resolve(conflict)?;
            match (conflict, answer) {
                (
                    SnapshotMergeConflict::Node { id, .. },
                    SnapshotMergeResolution::Entity(value),
                ) => match value {
                    Some(record) if record.endpoints.is_none() => {
                        nodes.insert(*id);
                        self.candidate
                            .node_attributes
                            .insert(*id, record.properties);
                    }
                    None => {
                        nodes.remove(id);
                        self.candidate.node_attributes.remove(id);
                    }
                    _ => return Err(invalid_resolution("node records cannot have endpoints")),
                },
                (
                    SnapshotMergeConflict::Edge { id, .. },
                    SnapshotMergeResolution::Entity(value),
                ) => match value {
                    Some(record) => {
                        let pair = record
                            .endpoints
                            .ok_or_else(|| invalid_resolution("edge records require endpoints"))?;
                        self.candidate.edges.insert(*id, pair);
                        self.candidate
                            .edge_attributes
                            .insert(*id, record.properties);
                    }
                    None => {
                        self.candidate.edges.remove(id);
                        self.candidate.edge_attributes.remove(id);
                    }
                },
                (
                    SnapshotMergeConflict::NodeProperty { id, name, .. },
                    SnapshotMergeResolution::Property(value),
                ) => {
                    set_property(
                        &mut self.candidate.node_attributes,
                        *id,
                        name.clone(),
                        value,
                    );
                }
                (
                    SnapshotMergeConflict::EdgeProperty { id, name, .. },
                    SnapshotMergeResolution::Property(value),
                ) => {
                    set_property(
                        &mut self.candidate.edge_attributes,
                        *id,
                        name.clone(),
                        value,
                    );
                }
                (
                    SnapshotMergeConflict::EdgeEndpoints { id, .. },
                    SnapshotMergeResolution::Endpoints(pair),
                ) => {
                    self.candidate.edges.insert(*id, pair);
                }
                _ => return Err(invalid_resolution("answer does not match conflict type")),
            }
        }
        // Resolutions may create or eliminate dependencies. Never use provisional
        // topology conflicts as authoritative after direct decisions.
        let mut edges: Vec<_> = self
            .candidate
            .edges
            .iter()
            .filter(|(_, pair)| !nodes.contains(&pair.0) || !nodes.contains(&pair.1))
            .map(|(&id, _)| id)
            .collect();
        edges.sort_unstable();
        for id in edges {
            let pair = self.candidate.edges[&id];
            let missing: HashSet<_> = [pair.0, pair.1]
                .into_iter()
                .filter(|n| !nodes.contains(n))
                .collect();
            if missing.is_empty() {
                continue;
            }
            let conflict = SnapshotMergeConflict::Topology {
                id,
                endpoints: pair,
            };
            match resolve(&conflict)? {
                SnapshotMergeResolution::DropEdge => {
                    self.candidate.edges.remove(&id);
                    self.candidate.edge_attributes.remove(&id);
                }
                SnapshotMergeResolution::EndpointNodes(records) => {
                    let supplied: HashSet<_> = records.iter().map(|r| r.0).collect();
                    if supplied != missing || supplied.len() != records.len() {
                        return Err(invalid_resolution(
                            "supply exactly the missing endpoint nodes",
                        ));
                    }
                    for (id, properties) in records {
                        nodes.insert(id);
                        self.candidate.node_attributes.insert(id, properties);
                    }
                }
                _ => {
                    return Err(invalid_resolution(
                        "dangling edges require explicit removal or endpoint restoration",
                    ))
                }
            }
        }
        self.candidate.active_nodes = nodes.into_iter().collect();
        self.candidate.active_nodes.sort_unstable();
        validate_snapshot(&self.candidate)?;
        Ok(self.candidate)
    }
    /// Return the candidate only when every change merges without a conflict.
    pub fn into_snapshot(self) -> GraphResult<GraphSnapshot> {
        if !self.conflicts.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "Snapshot merge has {} unresolved conflicts",
                self.conflicts.len()
            )));
        }
        validate_snapshot(&self.candidate)?;
        Ok(self.candidate)
    }
}
fn invalid_resolution(message: &str) -> GraphError {
    GraphError::InvalidInput(format!("Invalid merge resolution: {message}"))
}
fn set_property(
    rows: &mut HashMap<usize, MergeProperties>,
    id: usize,
    name: AttrName,
    value: Option<AttrValue>,
) {
    if let Some(value) = value {
        rows.entry(id).or_default().insert(name, value);
    } else if let Some(row) = rows.get_mut(&id) {
        row.remove(&name);
    }
}
fn equal_value(a: Option<&AttrValue>, b: Option<&AttrValue>) -> GraphResult<bool> {
    match (a, b) {
        (None, None) => Ok(true),
        (Some(a), Some(b)) => Ok(crate::storage::retained_group::encode(a)?
            == crate::storage::retained_group::encode(b)?),
        _ => Ok(false),
    }
}
fn equal_properties(a: &MergeProperties, b: &MergeProperties) -> GraphResult<bool> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (name, value) in a {
        if !equal_value(Some(value), b.get(name))? {
            return Ok(false);
        }
    }
    Ok(true)
}
#[derive(Clone, Copy)]
struct EntityRef<'a> {
    endpoints: Option<(NodeId, NodeId)>,
    properties: &'a MergeProperties,
}
impl EntityRef<'_> {
    fn owned(self) -> MergeEntity {
        MergeEntity {
            endpoints: self.endpoints,
            properties: self.properties.clone(),
        }
    }
}
fn equal_entity(a: Option<EntityRef<'_>>, b: Option<EntityRef<'_>>) -> GraphResult<bool> {
    match (a, b) {
        (None, None) => Ok(true),
        (Some(a), Some(b)) => {
            Ok(a.endpoints == b.endpoints && equal_properties(a.properties, b.properties)?)
        }
        _ => Ok(false),
    }
}

fn properties(
    id: usize,
    node: bool,
    base: &MergeProperties,
    left: &MergeProperties,
    right: &MergeProperties,
    conflicts: &mut Vec<SnapshotMergeConflict>,
) -> GraphResult<MergeProperties> {
    let names: BTreeSet<_> = base.keys().chain(left.keys()).chain(right.keys()).collect();
    let mut merged = HashMap::new();
    for name in names {
        let (b, l, r) = (base.get(name), left.get(name), right.get(name));
        let selected = if equal_value(l, r)? || equal_value(b, r)? {
            l
        } else if equal_value(b, l)? {
            r
        } else {
            conflicts.push(if node {
                SnapshotMergeConflict::NodeProperty {
                    id,
                    name: name.clone(),
                    base: b.cloned(),
                    left: l.cloned(),
                    right: r.cloned(),
                }
            } else {
                SnapshotMergeConflict::EdgeProperty {
                    id,
                    name: name.clone(),
                    base: b.cloned(),
                    left: l.cloned(),
                    right: r.cloned(),
                }
            });
            l // private provisional value; into_snapshot rejects all conflicts
        };
        if let Some(value) = selected {
            merged.insert(name.clone(), value.clone());
        }
    }
    Ok(merged)
}
fn node<'a>(
    snapshot: &'a GraphSnapshot,
    nodes: &HashSet<NodeId>,
    id: NodeId,
    empty: &'a MergeProperties,
) -> Option<EntityRef<'a>> {
    nodes.contains(&id).then(|| EntityRef {
        endpoints: None,
        properties: snapshot.node_attributes.get(&id).unwrap_or(empty),
    })
}
fn edge<'a>(
    snapshot: &'a GraphSnapshot,
    id: EdgeId,
    empty: &'a MergeProperties,
) -> Option<EntityRef<'a>> {
    snapshot.edges.get(&id).map(|&pair| EntityRef {
        endpoints: Some(pair),
        properties: snapshot.edge_attributes.get(&id).unwrap_or(empty),
    })
}

fn merge_entity(
    id: usize,
    is_node: bool,
    base: Option<EntityRef<'_>>,
    left: Option<EntityRef<'_>>,
    right: Option<EntityRef<'_>>,
    conflicts: &mut Vec<SnapshotMergeConflict>,
) -> GraphResult<Option<MergeEntity>> {
    if let (Some(b), Some(l), Some(r)) = (base, left, right) {
        let endpoints = if l.endpoints == r.endpoints || b.endpoints == r.endpoints {
            l.endpoints
        } else if b.endpoints == l.endpoints {
            r.endpoints
        } else {
            // All three edge records have a pair; nodes always have None.
            if let (Some(base), Some(left), Some(right)) = (b.endpoints, l.endpoints, r.endpoints) {
                conflicts.push(SnapshotMergeConflict::EdgeEndpoints {
                    id,
                    base,
                    left,
                    right,
                });
            }
            l.endpoints
        };
        return Ok(Some(MergeEntity {
            endpoints,
            properties: properties(
                id,
                is_node,
                b.properties,
                l.properties,
                r.properties,
                conflicts,
            )?,
        }));
    }
    if equal_entity(left, right)? || equal_entity(base, right)? {
        return Ok(left.map(EntityRef::owned));
    }
    if equal_entity(base, left)? {
        return Ok(right.map(EntityRef::owned));
    }
    let selected = left.map(EntityRef::owned);
    let (base, left, right) = (
        base.map(EntityRef::owned),
        left.map(EntityRef::owned),
        right.map(EntityRef::owned),
    );
    conflicts.push(if is_node {
        SnapshotMergeConflict::Node {
            id,
            base,
            left,
            right,
        }
    } else {
        SnapshotMergeConflict::Edge {
            id,
            base,
            left,
            right,
        }
    });
    Ok(selected)
}
/// Plan independent changes and inventory all direct conflicts. Inputs must share
/// directionality and the same identity domain/base chosen by the caller. Separate
/// edge IDs always remain separate parallel facts, including equal endpoint pairs.
/// Pool-reference, external-name and schema validation still requires the owner.
/// Entity/property unions are sorted for deterministic planning; output and
/// conflict values are owned, while unchanged input records are borrowed during
/// planning. No mutable graph borrow or publication authority is retained.
///
/// ```
/// use groggy::{state::{GraphSnapshot, plan_snapshot_merge, SnapshotMergeConflict,
///     SnapshotMergeResolution}, AttrValue};
/// let mut base = GraphSnapshot::empty(1);
/// base.active_nodes.push(0);
/// let mut left = base.clone();
/// let mut right = base.clone();
/// left.node_attributes.insert(0, [("value".into(), AttrValue::Int(1))].into());
/// right.node_attributes.insert(0, [("value".into(), AttrValue::Int(2))].into());
/// let plan = plan_snapshot_merge(&base, &left, &right, 4)?;
/// assert_eq!(plan.conflicts().len(), 1);
/// let merged = plan.resolve_with(|conflict| match conflict {
///     SnapshotMergeConflict::NodeProperty { right, .. } =>
///         Ok(SnapshotMergeResolution::Property(right.clone())),
///     _ => Err(groggy::GraphError::InvalidInput("explicit policy required".into())),
/// })?;
/// assert_eq!(merged.node_attributes[&0]["value"], AttrValue::Int(2));
/// # Ok::<(), groggy::GraphError>(())
/// ```
pub fn plan_snapshot_merge(
    base: &GraphSnapshot,
    left: &GraphSnapshot,
    right: &GraphSnapshot,
    target: StateId,
) -> GraphResult<SnapshotMergePlan> {
    for snapshot in [base, left, right] {
        validate_snapshot(snapshot)?;
    }
    if base.graph_type != left.graph_type || base.graph_type != right.graph_type {
        return Err(GraphError::InvalidInput(
            "Snapshot merge directionality differs".into(),
        ));
    }
    let mut candidate = GraphSnapshot::empty_with_type(target, base.graph_type);
    let mut conflicts = Vec::new();
    let empty = MergeProperties::new();
    let (bn, ln, rn): (HashSet<_>, HashSet<_>, HashSet<_>) = (
        base.active_nodes.iter().copied().collect(),
        left.active_nodes.iter().copied().collect(),
        right.active_nodes.iter().copied().collect(),
    );
    let nodes: BTreeSet<_> = bn
        .iter()
        .chain(ln.iter())
        .chain(rn.iter())
        .copied()
        .collect();
    for id in nodes {
        if let Some(record) = merge_entity(
            id,
            true,
            node(base, &bn, id, &empty),
            node(left, &ln, id, &empty),
            node(right, &rn, id, &empty),
            &mut conflicts,
        )? {
            candidate.active_nodes.push(id);
            if !record.properties.is_empty() {
                candidate.node_attributes.insert(id, record.properties);
            }
        }
    }
    let edges: BTreeSet<_> = base
        .edges
        .keys()
        .chain(left.edges.keys())
        .chain(right.edges.keys())
        .copied()
        .collect();
    for id in edges {
        if let Some(record) = merge_entity(
            id,
            false,
            edge(base, id, &empty),
            edge(left, id, &empty),
            edge(right, id, &empty),
            &mut conflicts,
        )? {
            if let Some(pair) = record.endpoints {
                candidate.edges.insert(id, pair);
            }
            if !record.properties.is_empty() {
                candidate.edge_attributes.insert(id, record.properties);
            }
        }
    }
    let nodes: HashSet<_> = candidate.active_nodes.iter().copied().collect();
    let mut dangling: Vec<_> = candidate
        .edges
        .iter()
        .filter(|(_, pair)| !nodes.contains(&pair.0) || !nodes.contains(&pair.1))
        .map(|(&id, &pair)| (id, pair))
        .collect();
    dangling.sort_unstable();
    conflicts.extend(
        dangling
            .into_iter()
            .map(|(id, endpoints)| SnapshotMergeConflict::Topology { id, endpoints }),
    );
    Ok(SnapshotMergePlan {
        candidate,
        conflicts,
    })
}
