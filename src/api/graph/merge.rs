//! Owner-bound immutable merge capture and resolved-target preflight. Publication
//! remains a separate coordinator step; a validated logical result is not a version.
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
}
/// Explicitly resolved logical result. It has no mutation authority and has not
/// reserved a version. The owning graph must revalidate it before publication.
#[derive(Debug)]
pub struct GraphMergeOutcome {
    coordinate: MergeCoordinate,
    snapshot: GraphSnapshot,
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
        Ok(GraphMergeOutcome {
            coordinate: self.coordinate,
            snapshot: self.plan.into_snapshot()?,
        })
    }
    /// Explicit typed resolution, with topology recomputed after direct answers.
    pub fn resolve_with(
        self,
        resolve: impl FnMut(&SnapshotMergeConflict) -> GraphResult<SnapshotMergeResolution>,
    ) -> GraphResult<GraphMergeOutcome> {
        Ok(GraphMergeOutcome {
            coordinate: self.coordinate,
            snapshot: self.plan.resolve_with(resolve)?,
        })
    }
}
impl GraphMergeOutcome {
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
