//! Explicit retained-state selection; history collection remains a separate action.
use super::*;
use crate::state::GraphSnapshot;

/// Detached logical target capture. It carries no mutable graph access.
#[derive(Debug)]
pub struct StateRestorePlan {
    read: crate::RecordedReadPlan,
    owner: Arc<()>,
    branch: String,
    from: StateId,
    target: StateId,
}
/// Materialized target, installable only into the still-matching owner/selection.
#[derive(Debug)]
pub struct StateRestoreOutcome {
    snapshot: GraphSnapshot,
    owner: Arc<()>,
    branch: String,
    from: StateId,
    target: StateId,
}
impl StateRestorePlan {
    /// Reconstruct the captured target off-owner, without altering retained roots.
    pub fn execute(self) -> GraphResult<StateRestoreOutcome> {
        Ok(StateRestoreOutcome {
            snapshot: self.read.execute()?,
            owner: self.owner,
            branch: self.branch,
            from: self.from,
            target: self.target,
        })
    }
}
impl StateRestoreOutcome {
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
    /// Switch the current branch to an explicit retained published state. No new
    /// mutation version or implicit history pruning; pending manual changes reject.
    pub fn restore_state(
        &mut self,
        target: StateId,
        expected_state: StateId,
    ) -> GraphResult<StateId> {
        let outcome = self
            .prepare_state_restore(target, expected_state)?
            .execute()?;
        self.finish_state_restore(outcome)
    }
    /// Capture immutable target ancestry; zero is the empty root, not a retained version.
    pub fn prepare_state_restore(
        &self,
        target: StateId,
        expected_state: StateId,
    ) -> GraphResult<StateRestorePlan> {
        self.validate_write_head(expected_state)?;
        self.history.get_commit(target)?;
        Ok(StateRestorePlan {
            read: self.history.prepare_recorded_read(target)?,
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
        let nodes: HashSet<_> = snapshot.active_nodes.iter().copied().collect();
        let pool = self.pool.borrow();
        if snapshot.state_id != outcome.target
            || snapshot.graph_type != self.graph_type()
            || nodes.len() != snapshot.active_nodes.len()
            || snapshot
                .node_attributes
                .iter()
                .any(|(id, attrs)| !attrs.is_empty() && !nodes.contains(id))
            || snapshot
                .edge_attributes
                .iter()
                .any(|(id, attrs)| !attrs.is_empty() && !snapshot.edges.contains_key(id))
            || nodes.iter().any(|&id| !pool.is_node_allocated(id))
            || snapshot.edges.iter().any(|(&id, &(u, v))| {
                !pool.is_edge_allocated(id) || !nodes.contains(&u) || !nodes.contains(&v)
            })
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
        let bindings = self.validate_state_restore(&outcome, schema)?;
        self.install_validated_state_restore(outcome, bindings)
    }
    pub(crate) fn install_validated_state_restore(
        &mut self,
        outcome: StateRestoreOutcome,
        bindings: Vec<(String, ExternalEntityRef)>,
    ) -> GraphResult<StateId> {
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
