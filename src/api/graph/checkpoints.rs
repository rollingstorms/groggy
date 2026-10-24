//! Explicit checkpoints do not manufacture a second mutation version.
use super::*;
use crate::state::publication::now_micros;
use crate::Checkpoint;

impl Graph {
    /// Mark the current clean published state, preserving its ID and mutation clocks.
    /// Unpublished compatibility writes must be published before this operation.
    pub fn checkpoint(&mut self, message: String, author: String) -> GraphResult<StateId> {
        self.checkpoint_at(self.current_state_id(), message, author)
    }

    /// Mark a clean state only if the optimistic publication head still matches.
    pub fn checkpoint_at(
        &mut self,
        expected_state: StateId,
        message: String,
        author: String,
    ) -> GraphResult<StateId> {
        self.checkpoint_with_schema(expected_state, message, author, None)
    }

    /// Mark a state under an additional schema policy, checked before metadata changes.
    pub fn checkpoint_with_schema(
        &mut self,
        expected_state: StateId,
        message: String,
        author: String,
        schema: Option<&GraphSchema>,
    ) -> GraphResult<StateId> {
        self.validate_write_head(expected_state)?;
        if let Some(schema) = schema {
            self.check_schema_policy(schema)?;
        }
        let record = self.prepare_checkpoint(expected_state, message, author)?;
        let state = record.state_id;
        self.record_checkpoint(record)?;
        Ok(state)
    }

    /// Explicit checkpoint records in acceptance order, including repeated labels
    /// for one state. They do not create extra permanent retention roots.
    pub fn checkpoints(&self) -> &[Checkpoint] {
        self.history.checkpoints()
    }

    pub(crate) fn prepare_checkpoint(
        &mut self,
        expected_state: StateId,
        message: String,
        author: String,
    ) -> GraphResult<Checkpoint> {
        self.validate_write_head(expected_state)?;
        let recorded_at = now_micros()?;
        self.history.reserve_checkpoint()?;
        Ok(Checkpoint {
            state_id: expected_state,
            branch: self.current_branch.clone(),
            message,
            author,
            recorded_at,
        })
    }

    pub(crate) fn record_checkpoint(&mut self, record: Checkpoint) -> GraphResult<()> {
        self.validate_write_head(record.state_id)?;
        if record.branch != self.current_branch {
            return Err(GraphError::InvalidInput(
                "Checkpoint branch does not match the selected graph branch".into(),
            ));
        }
        self.history.reserve_checkpoint()?;
        self.history.record_checkpoint(record);
        Ok(())
    }
}
