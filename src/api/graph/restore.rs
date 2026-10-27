//! Explicit retained-state selection; history collection remains a separate action.
use super::*;
use crate::state::GraphSnapshot;

/// Explicit restoration history behavior. Neither mode implicitly collects history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StateRestoreMode {
    /// Select the target state and move the current branch reference.
    Switch,
    /// Publish the target contents as one new delta on the current branch.
    Tracked,
}
impl std::str::FromStr for StateRestoreMode {
    type Err = GraphError;
    fn from_str(value: &str) -> GraphResult<Self> {
        match value {
            "switch" => Ok(Self::Switch),
            "tracked" => Ok(Self::Tracked),
            _ => Err(GraphError::InvalidInput(
                "Restore mode must be 'switch' or 'tracked'".into(),
            )),
        }
    }
}

/// Detached logical target capture. It carries no mutable graph access.
#[derive(Debug)]
pub struct StateRestorePlan {
    read: crate::RecordedReadPlan,
    before: Option<crate::RecordedReadPlan>,
    mode: StateRestoreMode,
    owner: Arc<()>,
    branch: String,
    from: StateId,
    target: StateId,
}
/// Materialized target, installable only into the still-matching owner/selection.
#[derive(Debug)]
pub struct StateRestoreOutcome {
    snapshot: GraphSnapshot,
    changes: Option<crate::state::ChangeSet>,
    mode: StateRestoreMode,
    owner: Arc<()>,
    branch: String,
    from: StateId,
    target: StateId,
}
impl StateRestorePlan {
    /// Reconstruct the captured target off-owner, without altering retained roots.
    pub fn execute(self) -> GraphResult<StateRestoreOutcome> {
        let snapshot = self.read.execute()?;
        let changes = self
            .before
            .map(|before| crate::state::change::snapshot_changes(&before.execute()?, &snapshot))
            .transpose()?;
        Ok(StateRestoreOutcome {
            snapshot,
            changes,
            mode: self.mode,
            owner: self.owner,
            branch: self.branch,
            from: self.from,
            target: self.target,
        })
    }
}
impl StateRestoreOutcome {
    /// Captured restoration semantics; cannot be changed after detached execution.
    pub fn mode(&self) -> StateRestoreMode {
        self.mode
    }
    /// Exact current-parent delta, present only for tracked restoration.
    pub fn changes(&self) -> Option<&crate::state::ChangeSet> {
        self.changes.as_ref()
    }

    /// Explicitly selected retained published state.
    pub fn target_state(&self) -> StateId {
        self.target
    }
    pub(crate) fn expected_state(&self) -> StateId {
        self.from
    }
    pub(crate) fn check_schema(&self, policy: &GraphSchema) -> GraphResult<()> {
        schema::reject_violations(policy.validate_snapshot(&self.snapshot)?)
    }
    pub(crate) fn branch(&self) -> &str {
        &self.branch
    }
}
impl Graph {
    /// Restore an explicit retained state using the required history mode.
    /// Tracked mode publishes one new delta; switch mode retargets the branch.
    /// Both preserve retained history and reject pending manual changes.
    pub fn restore_state(
        &mut self,
        target: StateId,
        expected_state: StateId,
        mode: StateRestoreMode,
    ) -> GraphResult<StateId> {
        let outcome = self
            .prepare_state_restore(target, expected_state, mode)?
            .execute()?;
        self.finish_state_restore(outcome)
    }
    /// Capture immutable target ancestry; zero is the empty root, not a retained version.
    pub fn prepare_state_restore(
        &self,
        target: StateId,
        expected_state: StateId,
        mode: StateRestoreMode,
    ) -> GraphResult<StateRestorePlan> {
        self.validate_write_head(expected_state)?;
        self.history.get_commit(target)?;
        Ok(StateRestorePlan {
            read: self.history.prepare_recorded_read(target)?,
            before: if mode == StateRestoreMode::Tracked {
                Some(self.history.prepare_recorded_read(expected_state)?)
            } else {
                None
            },
            mode,
            owner: Arc::clone(self.history.identity()),
            branch: self.current_branch.clone(),
            from: expected_state,
            target,
        })
    }
    /// Validate ownership, selected branch/head, retained target, references and
    /// optional current schema before any pool or branch mutation.
    pub(crate) fn validate_state_restore(
        &self,
        outcome: &StateRestoreOutcome,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<Vec<(String, ExternalEntityRef)>> {
        self.validate_write_head(outcome.from)?;
        if !Arc::ptr_eq(self.history.identity(), &outcome.owner)
            || self.current_branch != outcome.branch
        {
            return Err(GraphError::InvalidInput(
                "Stale or foreign state restoration".into(),
            ));
        }
        self.history.get_commit(outcome.target)?;
        let snapshot = &outcome.snapshot;
        crate::state::validate_snapshot(snapshot)?;
        let pool = self.pool.borrow();
        if snapshot.state_id != outcome.target
            || snapshot.graph_type != self.graph_type()
            || snapshot
                .active_nodes
                .iter()
                .any(|&id| !pool.is_node_allocated(id))
            || snapshot.edges.keys().any(|&id| !pool.is_edge_allocated(id))
        {
            return Err(GraphError::InvalidInput(
                "Invalid retained restoration topology".into(),
            ));
        }
        for id in crate::storage::retained_group::references(snapshot) {
            pool.validate_stored_subgraph(id)?;
        }
        drop(pool);
        let bindings = self.snapshot_identity_bindings(snapshot)?;
        if let Some(schema) = schema {
            schema::reject_violations(schema.validate_snapshot(snapshot)?)?;
        }
        Ok(bindings)
    }
    /// Install a validated target. The original history identity, all versions,
    /// allocation counters, reservations, other branches and publication clocks survive.
    pub fn finish_state_restore(&mut self, outcome: StateRestoreOutcome) -> GraphResult<StateId> {
        self.finish_state_restore_with_schema(outcome, None)
    }
    pub fn finish_state_restore_with_schema(
        &mut self,
        outcome: StateRestoreOutcome,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        self.finish_state_restore_with_options(
            outcome,
            GraphPublicationOptions::default(),
            "restore".into(),
            "writer".into(),
            schema,
        )
    }
    /// Accept the captured explicit mode, timing and current complete-target policy.
    pub fn finish_state_restore_with_options(
        &mut self,
        outcome: StateRestoreOutcome,
        options: GraphPublicationOptions,
        message: String,
        author: String,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        match outcome.mode {
            StateRestoreMode::Switch => {
                Self::validate_switch_options(&options)?;
                let bindings = self.validate_state_restore(&outcome, schema)?;
                self.install_validated_state_restore(outcome, bindings)
            }
            StateRestoreMode::Tracked => {
                let prepared =
                    self.prepare_tracked_restore(outcome, &options, schema, None, None)?;
                self.execute_snapshot_publication(prepared, message, author)
            }
        }
    }
    pub(crate) fn validate_switch_options(options: &GraphPublicationOptions) -> GraphResult<()> {
        if options.effective_at.is_some() || !options.change_times.is_empty() {
            return Err(GraphError::InvalidInput(
                "Switch restoration records no changes and accepts no change timestamps".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn prepare_tracked_restore(
        &self,
        outcome: StateRestoreOutcome,
        options: &GraphPublicationOptions,
        schema: Option<&GraphSchema>,
        additional_schema: Option<&GraphSchema>,
        recorded_at: Option<i64>,
    ) -> GraphResult<super::snapshot_publication::PreparedSnapshotPublication> {
        self.validate_state_restore(&outcome, schema)?;
        if let Some(schema) = additional_schema {
            outcome.check_schema(schema)?;
        }
        if outcome.mode != StateRestoreMode::Tracked {
            return Err(GraphError::InvalidInput(
                "Tracked restoration requires a tracked capture".into(),
            ));
        }
        let StateRestoreOutcome {
            mut snapshot,
            changes,
            from,
            branch,
            ..
        } = outcome;
        let changes = changes
            .ok_or_else(|| GraphError::InvalidInput("Missing tracked restoration delta".into()))?;
        snapshot.state_id = 0;
        self.prepare_snapshot_publication(
            snapshot,
            changes,
            from,
            branch,
            vec![from],
            options,
            recorded_at,
        )
    }
    pub(crate) fn install_validated_state_restore(
        &mut self,
        outcome: StateRestoreOutcome,
        bindings: Vec<(String, ExternalEntityRef)>,
    ) -> GraphResult<StateId> {
        if outcome.mode != StateRestoreMode::Switch {
            return Err(GraphError::InvalidInput(
                "Reference installation requires switch mode".into(),
            ));
        }
        if outcome.target == self.current_commit {
            return Ok(outcome.target);
        }
        self.install_snapshot_payload(outcome.snapshot, bindings)?;
        self.history
            .update_branch_head(&self.current_branch, outcome.target)?;
        self.current_commit = outcome.target;
        Ok(outcome.target)
    }
    /// Shared payload installer for recorded switching, owned copies and explicit restore.
    pub(super) fn install_snapshot_payload(
        &mut self,
        snapshot: GraphSnapshot,
        identity_bindings: Vec<(String, ExternalEntityRef)>,
    ) -> GraphResult<()> {
        // Reset selected membership and column mappings, not the retained pool.
        // Older branches, stored subgraphs and column indices still own data here.
        self.space = crate::state::space::GraphSpace::new(self.pool.clone(), snapshot.state_id);
        self.config.graph_type = snapshot.graph_type;

        // 2. Restore nodes
        for &node_id in &snapshot.active_nodes {
            // Ensure node ID exists in pool
            self.pool.borrow_mut().ensure_node_id_exists(node_id)?;
            // Activate in space
            self.space.activate_node(node_id);

            // Restore node attributes
            if let Some(attrs) = snapshot.node_attributes.get(&node_id) {
                for (attr_name, attr_value) in attrs {
                    let index =
                        self.pool
                            .borrow_mut()
                            .restore_attr(attr_name.clone(), attr_value, true);
                    self.space
                        .set_node_attr_index(node_id, attr_name.clone(), index);
                }
            }
        }

        // 3. Restore edges
        for (&edge_id, &(source, target)) in &snapshot.edges {
            // Store topology in pool with specific ID
            self.pool
                .borrow_mut()
                .restore_edge_revision(edge_id, source, target)?;
            // Activate in space
            self.space.activate_edge(edge_id, source, target);

            // Restore edge attributes
            if let Some(attrs) = snapshot.edge_attributes.get(&edge_id) {
                for (attr_name, attr_value) in attrs {
                    let index =
                        self.pool
                            .borrow_mut()
                            .restore_attr(attr_name.clone(), attr_value, false);
                    self.space
                        .set_edge_attr_index(edge_id, attr_name.clone(), index);
                }
            }
        }

        self.retain_snapshot_identity_bindings(identity_bindings);
        Ok(())
    }
}
