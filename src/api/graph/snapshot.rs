//! Explicit owned storage capture, separate from ordinary immutable read pins.
use super::*;
use crate::state::PinnedReadState;

#[derive(Debug)]
pub(crate) struct SnapshotMaterialization {
    config: GraphConfig,
    allocation: (usize, usize),
    owned: Option<OwnedContents>,
}
#[derive(Debug)]
struct OwnedContents {
    pool: GraphPool,
    history: HistoryForest,
    identities: ExternalIdentityIndex,
    branch: String,
    automatic_publication: bool,
}
impl SnapshotMaterialization {
    pub(crate) fn is_owned(&self) -> bool {
        self.owned.is_some()
    }
    pub(crate) fn configuration(&self) -> &GraphConfig {
        &self.config
    }

    pub(crate) fn materialize_selection(&self, state: &PinnedReadState) -> GraphResult<Graph> {
        let mut graph = Graph::with_config(self.config.clone());
        {
            let mut pool = graph.pool.borrow_mut();
            pool.install_snapshot_groups(state.groups().clone());
            pool.preserve_allocation(self.allocation.0, self.allocation.1);
        }
        // This preserves the legacy manual selection container contract. Full
        // original history/raw storage and publishing require an owned capture.
        graph.reset_to_snapshot(state.snapshot().clone())?;
        Ok(graph)
    }

    pub(crate) fn restore_owned(&self, state: &PinnedReadState) -> GraphResult<Graph> {
        let owned = self.owned.as_ref().ok_or_else(|| {
            GraphError::InvalidInput(
                "to_graph requires an owned snapshot; capture with owned=True".into(),
            )
        })?;
        let mut graph = Graph::with_config(self.config.clone());
        graph.pool = Rc::new(RefCell::new(owned.pool.clone()));
        graph.history = owned.history.detached_copy();
        graph.identities = owned.identities.clone();
        graph.current_branch = owned.branch.clone();
        graph.current_commit = state.snapshot().state_id;
        // Selecting an older state retargets only the copy's selected branch.
        // Other branches, all retained versions and allocation reservations survive.
        graph
            .history
            .update_branch_head(&graph.current_branch, graph.current_commit)?;
        graph.reset_to_snapshot(state.snapshot().clone())?;
        graph.automatic_publication = owned.automatic_publication;
        Ok(graph)
    }
}
impl Graph {
    pub(super) fn snapshot_materialization(&self, owned: bool) -> Arc<SnapshotMaterialization> {
        let pool = self.pool.borrow();
        let stats = pool.statistics();
        Arc::new(SnapshotMaterialization {
            config: self.config.clone(),
            allocation: (stats.node_count, stats.edge_count),
            owned: owned.then(|| OwnedContents {
                pool: pool.clone(),
                history: self.history.detached_copy(),
                identities: self.identities.clone(),
                branch: self.current_branch.clone(),
                automatic_publication: self.automatic_publication,
            }),
        })
    }

    /// Capture a recorded state plus all currently retained backing storage,
    /// history, configuration and allocation/identity reservations. Explicitly
    /// copies the pool; ordinary read snapshots never pay this cost.
    pub fn owned_snapshot_at_commit(&self, state: StateId) -> GraphResult<TemporalSnapshot> {
        self.validate_write_head(self.current_state_id())?;
        Ok(self
            .snapshot_at_commit(state)?
            .with_materialization(self.snapshot_materialization(true)))
    }

    /// Full owned capture selected by recorded timestamp (Unix seconds).
    pub fn owned_snapshot_at_timestamp(&self, timestamp: u64) -> GraphResult<TemporalSnapshot> {
        self.validate_write_head(self.current_state_id())?;
        Ok(self
            .snapshot_at_timestamp(timestamp)?
            .with_materialization(self.snapshot_materialization(true)))
    }
}
