//! Owner-bound merge capture, exact first-parent delta preparation and checked
//! two-parent publication. Logical outcomes remain unpublished until acceptance.
use super::*;
use crate::state::{
    GraphSnapshot, SnapshotMergeConflict, SnapshotMergePlan, SnapshotMergeResolution,
};

#[derive(Debug)]
struct MergeCoordinate {
    owner: Arc<()>,
    branch: String,
    left: StateId,
    right: StateId,
    base: StateId,
}
/// Owned retained merge inputs. Reconstruct and plan without a graph borrow.
#[derive(Debug)]
pub struct GraphMergeReadPlan {
    coordinate: MergeCoordinate,
    base: crate::RecordedReadPlan,
    left: crate::RecordedReadPlan,
    right: crate::RecordedReadPlan,
}
/// Native logical conflict plan tied to the captured owner/head/branch/base.
#[derive(Debug)]
pub struct GraphMergePlan {
    coordinate: MergeCoordinate,
    plan: SnapshotMergePlan,
    before: GraphSnapshot,
}
/// Explicitly resolved logical result. It has no mutation authority and has not
/// reserved a version. The owning graph must revalidate it before publication.
#[derive(Debug)]
pub struct GraphMergeOutcome {
    coordinate: MergeCoordinate,
    snapshot: GraphSnapshot,
    changes: crate::state::ChangeSet,
}
impl GraphMergeReadPlan {
    /// Reconstruct three first-parent selections and plan outside the owner/GIL.
    pub fn execute(self) -> GraphResult<GraphMergePlan> {
        let base = self.base.execute()?;
        let left = self.left.execute()?;
        let right = self.right.execute()?;
        let plan = crate::state::plan_snapshot_merge(&base, &left, &right, 0)?;
        Ok(GraphMergePlan {
            coordinate: self.coordinate,
            plan,
            before: left,
        })
    }
}
impl GraphMergePlan {
    /// Captured explicit base (zero is the explicitly chosen universal empty base).
    pub fn base_state(&self) -> StateId {
        self.coordinate.base
    }
    /// Ordered two-parent selections; the first is the expected working state.
    pub fn parents(&self) -> [StateId; 2] {
        [self.coordinate.left, self.coordinate.right]
    }
    /// Deterministic logical conflicts, including provisional topology dependencies.
    pub fn conflicts(&self) -> &[SnapshotMergeConflict] {
        self.plan.conflicts()
    }
    /// Finish conflict-free planning. No owner mutation or version allocation.
    pub fn into_outcome(self) -> GraphResult<GraphMergeOutcome> {
        merge_outcome(self.coordinate, &self.before, self.plan.into_snapshot()?)
    }
    /// Explicit typed resolution, with topology recomputed after direct answers.
    pub fn resolve_with(
        self,
        resolve: impl FnMut(&SnapshotMergeConflict) -> GraphResult<SnapshotMergeResolution>,
    ) -> GraphResult<GraphMergeOutcome> {
        merge_outcome(
            self.coordinate,
            &self.before,
            self.plan.resolve_with(resolve)?,
        )
    }
}
/// Effective-time options for one resolved merge. Recorded time is engine-owned.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphMergeOptions {
    /// Batch default effective time; None uses the engine acceptance clock.
    pub effective_at: Option<crate::TimestampMicros>,
    /// Overrides for actual first-parent resolution changes only.
    pub change_times: std::collections::BTreeMap<crate::ChangeKey, crate::TimestampMicros>,
}
pub(crate) struct PreparedMerge {
    pub outcome: GraphMergeOutcome,
    pub metadata: crate::PublicationMetadata,
    bindings: Vec<(String, ExternalEntityRef)>,
}
impl GraphMergeOutcome {
    /// Exact typed resolution delta relative to the first parent, not the base.
    pub fn changes(&self) -> &crate::state::ChangeSet {
        &self.changes
    }
    /// Captured branch receiving the resolved merge.
    pub fn branch(&self) -> &str {
        &self.coordinate.branch
    }

    /// Captured explicit base, without a newly allocated merge version.
    pub fn base_state(&self) -> StateId {
        self.coordinate.base
    }
    /// Ordered retained parent selections.
    pub fn parents(&self) -> [StateId; 2] {
        [self.coordinate.left, self.coordinate.right]
    }
    /// Logical immutable result; state zero marks that it is unpublished.
    pub fn snapshot(&self) -> &GraphSnapshot {
        &self.snapshot
    }
}
impl Graph {
    /// Capture a merge with an explicit right-hand retained version and expected
    /// working state. None selects only a unique best base. Unrelated roots require
    /// explicit Some(0); multiple best bases require an explicit common ancestor.
    pub fn prepare_merge_read(
        &self,
        right: StateId,
        base: Option<StateId>,
        expected_state: StateId,
    ) -> GraphResult<GraphMergeReadPlan> {
        self.validate_write_head(expected_state)?;
        self.history.get_commit(expected_state)?;
        self.history.get_commit(right)?;
        if expected_state == right {
            return Err(GraphError::InvalidInput(
                "Merge requires distinct parent states".into(),
            ));
        }
        let base = match base {
            Some(base) => base,
            None => self
                .history
                .find_common_ancestor(expected_state, right)?
                .ok_or_else(|| {
                    GraphError::InvalidInput(
                        "Unrelated merge roots require an explicit empty base (0)".into(),
                    )
                })?,
        };
        self.validate_merge_base(base, expected_state, right)?;
        Ok(GraphMergeReadPlan {
            coordinate: MergeCoordinate {
                owner: Arc::clone(self.history.identity()),
                branch: self.current_branch.clone(),
                left: expected_state,
                right,
                base,
            },
            base: self.history.prepare_recorded_read(base)?,
            left: self.history.prepare_recorded_read(expected_state)?,
            right: self.history.prepare_recorded_read(right)?,
        })
    }
    fn validate_merge_base(&self, base: StateId, left: StateId, right: StateId) -> GraphResult<()> {
        if base != 0
            && (!self.history.is_ancestor(base, left)? || !self.history.is_ancestor(base, right)?)
        {
            return Err(GraphError::InvalidInput(
                "Explicit merge base is not a common ancestor".into(),
            ));
        }
        Ok(())
    }
    /// Read-only owner preflight for an explicitly resolved logical outcome.
    /// Validation is not publication and does not authorize later unchecked use.
    /// Clocks, exact first-parent deltas and two-parent acceptance remain separate.
    pub fn validate_merge_outcome(
        &self,
        outcome: &GraphMergeOutcome,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<()> {
        let coordinate = &outcome.coordinate;
        self.validate_write_head(coordinate.left)?;
        if !Arc::ptr_eq(self.history.identity(), &coordinate.owner)
            || self.current_branch != coordinate.branch
        {
            return Err(GraphError::InvalidInput(
                "Stale or foreign merge outcome".into(),
            ));
        }
        self.history.get_commit(coordinate.left)?;
        self.history.get_commit(coordinate.right)?;
        self.validate_merge_base(coordinate.base, coordinate.left, coordinate.right)?;
        let snapshot = &outcome.snapshot;
        crate::state::validate_snapshot(snapshot)?;
        let pool = self.pool.borrow();
        if snapshot.state_id != 0
            || snapshot.graph_type != self.graph_type()
            || snapshot
                .active_nodes
                .iter()
                .any(|&id| !pool.is_node_allocated(id))
            || snapshot.edges.keys().any(|&id| !pool.is_edge_allocated(id))
        {
            return Err(GraphError::InvalidInput(
                "Merge outcome contains unallocated entities or incompatible state".into(),
            ));
        }
        for id in crate::storage::retained_group::references(snapshot) {
            pool.validate_stored_subgraph(id)?;
        }
        drop(pool);
        self.snapshot_identity_bindings(snapshot)?;
        if let Some(schema) = schema {
            schema::reject_violations(schema.validate_snapshot(snapshot)?)?;
        }
        Ok(())
    }
}

fn merge_outcome(
    coordinate: MergeCoordinate,
    before: &GraphSnapshot,
    snapshot: GraphSnapshot,
) -> GraphResult<GraphMergeOutcome> {
    let changes = exact_merge_changes(before, &snapshot)?;
    Ok(GraphMergeOutcome {
        coordinate,
        snapshot,
        changes,
    })
}
fn exact_merge_changes(
    before: &GraphSnapshot,
    after: &GraphSnapshot,
) -> GraphResult<crate::state::ChangeSet> {
    use std::collections::BTreeSet;
    let mut changes = crate::state::ChangeSet::new();
    let old_nodes: HashSet<_> = before.active_nodes.iter().copied().collect();
    let new_nodes: HashSet<_> = after.active_nodes.iter().copied().collect();
    changes.nodes_added = new_nodes.difference(&old_nodes).copied().collect();
    changes.nodes_removed = old_nodes.difference(&new_nodes).copied().collect();
    changes.nodes_added.sort_unstable();
    changes.nodes_removed.sort_unstable();
    for (&id, &(u, v)) in &after.edges {
        match before.edges.get(&id) {
            None => changes.edges_added.push((id, u, v)),
            Some(&old) if old != (u, v) => {
                changes
                    .edge_endpoint_changes
                    .push(crate::EdgeEndpointChange {
                        edge_id: id,
                        old_endpoints: old,
                        new_endpoints: (u, v),
                    })
            }
            _ => {}
        }
    }
    changes.edges_added.sort_unstable();
    changes.edges_removed = before
        .edges
        .keys()
        .filter(|id| !after.edges.contains_key(id))
        .copied()
        .collect();
    changes.edges_removed.sort_unstable();
    changes
        .edge_endpoint_changes
        .sort_unstable_by_key(|c| c.edge_id);
    for (nodes, old_rows, new_rows, output) in [
        (
            true,
            &before.node_attributes,
            &after.node_attributes,
            &mut changes.node_attr_changes,
        ),
        (
            false,
            &before.edge_attributes,
            &after.edge_attributes,
            &mut changes.edge_attr_changes,
        ),
    ] {
        let ids: BTreeSet<_> = if nodes {
            new_nodes.iter().copied().collect()
        } else {
            after.edges.keys().copied().collect()
        };
        for id in ids {
            let was_active = if nodes {
                old_nodes.contains(&id)
            } else {
                before.edges.contains_key(&id)
            };
            let old = was_active.then(|| old_rows.get(&id)).flatten();
            let new = new_rows.get(&id);
            let names: BTreeSet<_> = old
                .into_iter()
                .flat_map(|row| row.keys())
                .chain(new.into_iter().flat_map(|row| row.keys()))
                .collect();
            for name in names {
                let a = old.and_then(|r| r.get(name));
                let b = new.and_then(|r| r.get(name));
                let equal = match (a, b) {
                    (None, None) => true,
                    (Some(a), Some(b)) => archive::same_value(a, b)?,
                    _ => false,
                };
                if !equal {
                    output.push((id, name.clone(), a.cloned(), b.cloned()));
                }
            }
        }
    }
    Ok(changes)
}
impl Graph {
    /// Publish one explicitly resolved two-parent merge, including data-identical
    /// merges which still join ancestry. No node/edge IDs are allocated.
    pub fn finish_merge(
        &mut self,
        outcome: GraphMergeOutcome,
        options: GraphMergeOptions,
        message: String,
        author: String,
    ) -> GraphResult<StateId> {
        self.finish_merge_with_schema(outcome, options, message, author, None)
    }
    /// Publish under an optional complete-target schema; ordinary publication
    /// policy is preserved, and pending manual writes reject.
    pub fn finish_merge_with_schema(
        &mut self,
        outcome: GraphMergeOutcome,
        options: GraphMergeOptions,
        message: String,
        author: String,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        let prepared = self.prepare_merge_publication(outcome, &options, schema, None, None)?;
        self.execute_merge_publication(prepared, message, author)
    }
    pub(crate) fn prepare_merge_publication(
        &self,
        outcome: GraphMergeOutcome,
        options: &GraphMergeOptions,
        schema: Option<&GraphSchema>,
        additional_schema: Option<&GraphSchema>,
        recorded_at: Option<i64>,
    ) -> GraphResult<PreparedMerge> {
        self.validate_merge_outcome(&outcome, schema)?;
        if let Some(policy) = additional_schema {
            schema::reject_violations(policy.validate_snapshot(&outcome.snapshot)?)?;
        }
        let bindings = self.snapshot_identity_bindings(&outcome.snapshot)?;
        let changes = &outcome.changes;
        if !changes.edge_endpoint_changes.is_empty()
            && !self.change_tracker.supports_endpoint_changes()
        {
            return Err(GraphError::InvalidInput(
                "Temporal strategy does not support endpoint changes".into(),
            ));
        }
        let pool = self.pool.borrow();
        let rewired: HashSet<_> = outcome
            .snapshot
            .edges
            .iter()
            .filter_map(|(&id, &pair)| (pool.get_edge_endpoints(id) != Some(pair)).then_some(id))
            .collect();
        if pool.legacy_groups_own_edges(&rewired) {
            return Err(GraphError::InvalidInput("Merge endpoint changes require frozen group sources; edge belongs to a legacy membership-only group".into()));
        }
        drop(pool);
        let recorded_at = match recorded_at {
            Some(time) => time,
            None => crate::state::publication::now_micros()?,
        };
        let effective_at = options.effective_at.unwrap_or(recorded_at);
        let mut metadata = crate::PublicationMetadata {
            recorded_at,
            effective_at,
            change_times: Default::default(),
        };
        use crate::ChangeKey as K;
        let keys = changes
            .nodes_added
            .iter()
            .map(|&id| K::NodeAdded(id))
            .chain(changes.nodes_removed.iter().map(|&id| K::NodeRemoved(id)))
            .chain(
                changes
                    .edges_added
                    .iter()
                    .map(|&(id, _, _)| K::EdgeAdded(id)),
            )
            .chain(changes.edges_removed.iter().map(|&id| K::EdgeRemoved(id)))
            .chain(
                changes
                    .edge_endpoint_changes
                    .iter()
                    .map(|c| K::EdgeEndpoints(c.edge_id)),
            )
            .chain(
                changes
                    .node_attr_changes
                    .iter()
                    .map(|(id, name, _, _)| K::NodeAttribute(*id, name.clone())),
            )
            .chain(
                changes
                    .edge_attr_changes
                    .iter()
                    .map(|(id, name, _, _)| K::EdgeAttribute(*id, name.clone())),
            );
        for key in keys {
            let fallback = match &key {
                K::NodeAttribute(id, _) => metadata
                    .change_times
                    .get(&K::NodeAdded(*id))
                    .copied()
                    .unwrap_or(effective_at),
                K::EdgeAttribute(id, _) => metadata
                    .change_times
                    .get(&K::EdgeAdded(*id))
                    .copied()
                    .unwrap_or(effective_at),
                K::EdgeRemoved(id) => {
                    let (u, v) = self.edge_endpoints(*id)?;
                    [u, v]
                        .iter()
                        .filter_map(|id| metadata.change_times.get(&K::NodeRemoved(*id)))
                        .min()
                        .copied()
                        .unwrap_or(effective_at)
                }
                _ => effective_at,
            };
            metadata.change_times.insert(
                key.clone(),
                options.change_times.get(&key).copied().unwrap_or(fallback),
            );
        }
        if options
            .change_times
            .keys()
            .any(|key| !metadata.change_times.contains_key(key))
        {
            return Err(GraphError::InvalidInput(
                "Merge timestamp override does not target an actual resolution change".into(),
            ));
        }
        let additions = changes
            .edges_added
            .iter()
            .map(|&(id, u, v)| (id, (u, v)))
            .chain(
                changes
                    .edge_endpoint_changes
                    .iter()
                    .map(|c| (c.edge_id, c.new_endpoints)),
            )
            .collect();
        let patch = crate::state::history::TopologyPatch::from_metadata(
            &metadata,
            self.history.next_state_id(),
            &additions,
        )?;
        self.history.validate_effective_topology(
            outcome.coordinate.left,
            &patch,
            options.effective_at.is_some()
                || options
                    .change_times
                    .keys()
                    .any(|key| !matches!(key, K::NodeAttribute(..) | K::EdgeAttribute(..))),
        )?;
        Ok(PreparedMerge {
            outcome,
            metadata,
            bindings,
        })
    }
    pub(crate) fn execute_merge_publication(
        &mut self,
        prepared: PreparedMerge,
        message: String,
        author: String,
    ) -> GraphResult<StateId> {
        // Every fallible caller/input check precedes this exclusive installation.
        // Retained allocated IDs and logical values were preflighted; payload
        // installation cannot encounter a normal input failure after publication.
        let PreparedMerge {
            outcome,
            metadata,
            bindings,
        } = prepared;
        let GraphMergeOutcome {
            coordinate,
            mut snapshot,
            changes,
        } = outcome;
        let id = self.history.create_merge_commit_with_publication(
            changes,
            message,
            author,
            vec![coordinate.left, coordinate.right],
            Some(metadata),
        )?;
        snapshot.state_id = id;
        self.install_snapshot_payload(snapshot, bindings)?;
        self.history.update_branch_head(&coordinate.branch, id)?;
        self.current_commit = id;
        Ok(id)
    }
}

impl Graph {
    pub(crate) fn resolved_merge_for_replay(
        &self,
        left: StateId,
        right: StateId,
        base: StateId,
        changes: crate::state::ChangeSet,
    ) -> GraphResult<GraphMergeOutcome> {
        let plan = self.prepare_merge_read(right, Some(base), left)?;
        let before = plan.left.execute()?;
        let declared = crate::state::Delta::from_changes(changes);
        let snapshot = before.apply_delta(&declared, 0);
        let outcome = merge_outcome(plan.coordinate, &before, snapshot)?;
        let actual = crate::state::Delta::from_changes(outcome.changes.clone());
        if !crate::state::history::exact_delta(&declared, &actual)? {
            return Err(GraphError::InvalidInput(
                "Merge record delta differs from exact first-parent resolution".into(),
            ));
        }
        Ok(outcome)
    }
}
