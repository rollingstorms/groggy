//! Explicit history/reference pruning. Raw pool/identity reclamation is separate.
use super::*;

pub(crate) struct PreparedHistoryCollection {
    pub(crate) branch: Option<String>,
    pub(crate) removed: Vec<StateId>,
    keep: HashSet<StateId>,
}
impl Graph {
    /// Delete a non-current, non-main branch and collect only unreachable history.
    /// Retained pool values and lifetime identity reservations are not reclaimed.
    pub fn prune_branch(&mut self, name: &str) -> GraphResult<usize> {
        let prepared = self.prepare_history_collection(Some(name.into()))?;
        self.execute_history_collection(prepared)
    }

    pub(crate) fn prepare_history_collection(
        &self,
        branch: Option<String>,
    ) -> GraphResult<PreparedHistoryCollection> {
        self.validate_write_head(self.current_state_id())?;
        if let Some(name) = &branch {
            if name == "main" || name == &self.current_branch {
                return Err(GraphError::InvalidInput(
                    "Cannot prune main or the currently selected branch".into(),
                ));
            }
            self.history.get_branch_head(name)?;
        }
        let (keep, removed) = self
            .history
            .collection_plan(branch.as_deref(), &HashSet::new());
        Ok(PreparedHistoryCollection {
            branch,
            removed,
            keep,
        })
    }

    pub(crate) fn execute_history_collection(
        &mut self,
        prepared: PreparedHistoryCollection,
    ) -> GraphResult<usize> {
        if let Some(name) = &prepared.branch {
            self.history.delete_branch(name)?;
        }
        Ok(self.history.retain_commits(&prepared.keep))
    }
}
