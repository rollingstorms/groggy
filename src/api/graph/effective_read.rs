//! Capture only accepted ancestry before detached effective-time replay.
use super::*;
use crate::state::effective_read::{EffectiveReadPlan, ReadTime};
impl Graph {
    /// Pin the selected published head and prepare a detached effective-time read.
    /// Pending manual edits reject; siblings and future publications are excluded.
    pub fn prepare_read_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<EffectiveReadPlan> {
        if self.has_uncommitted_changes() || (self.current_commit == 0 && !self.is_empty()) {
            return Err(GraphError::InvalidInput("Effective-time reads require a clean published graph; publish pending manual edits first".into()));
        }
        let mut commits = Vec::new();
        let mut seen = HashSet::new();
        let mut state = self.current_commit;
        while state != 0 {
            if !seen.insert(state) {
                return Err(GraphError::InvalidInput(
                    "Cycle in publication ancestry".into(),
                ));
            }
            let commit = self.history.get_commit(state)?;
            if commit.parents.len() > 1 {
                return Err(GraphError::InvalidInput(
                    "Effective-time replay of merge ancestry is not yet supported".into(),
                ));
            }
            state = commit.parents.first().copied().unwrap_or(0);
            commits.push(commit);
        }
        commits.reverse();
        Ok(EffectiveReadPlan {
            commits,
            graph_type: self.graph_type(),
            time: ReadTime {
                branch: self.current_branch.clone(),
                published_head: self.current_commit,
                effective_at,
                recorded_by,
            },
        })
    }
    /// Reconstruct a read-only graph at effective time T, optionally restricted to
    /// publications recorded by U. Neither a checkpoint nor a mutation is created.
    pub fn read_view_at(
        &self,
        effective_at: i64,
        recorded_by: Option<i64>,
    ) -> GraphResult<crate::GraphView> {
        self.prepare_read_at(effective_at, recorded_by)?.read_view()
    }
}
