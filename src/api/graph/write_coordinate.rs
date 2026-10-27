//! Optimistic session coordinates retain identity, never graph data or mutation authority.
use super::*;
/// An owner/branch/head expectation for a checked write session.
/// This owns only an identity marker; it pins no retained state or graph payload.
#[derive(Debug, Clone)]
pub struct WriteCoordinate {
    binding: Arc<WriteBinding>,
    state: StateId,
}
#[derive(Debug)]
struct WriteBinding {
    owner: Arc<()>,
    branch: String,
}
impl WriteCoordinate {
    /// Expected published head; zero is the clean empty selection.
    pub fn state_id(&self) -> StateId {
        self.state
    }
    /// Expected selected branch.
    pub fn branch(&self) -> &str {
        &self.binding.branch
    }
    /// Retarget the expected state on this same owner/branch after acceptance.
    /// The graph still validates ownership, branch, head and cleanliness on use.
    pub fn at_state(&self, state: StateId) -> Self {
        Self {
            state,
            ..self.clone()
        }
    }
}
impl Graph {
    /// Capture the selected owner/branch/head without granting mutation authority.
    pub fn write_coordinate(&self) -> WriteCoordinate {
        WriteCoordinate {
            binding: Arc::new(WriteBinding {
                owner: Arc::clone(self.history.identity()),
                branch: self.current_branch.clone(),
            }),
            state: self.current_commit,
        }
    }
    /// Reject foreign owners and branch switches even when both heads share an ID.
    /// Validation is only valid for the current exclusive operation, never a lease.
    pub fn validate_write_coordinate(&self, coordinate: &WriteCoordinate) -> GraphResult<()> {
        if !Arc::ptr_eq(&coordinate.binding.owner, self.history.identity())
            || coordinate.binding.branch != self.current_branch
        {
            return Err(GraphError::InvalidInput(
                "Stale or foreign write coordinate: owner or branch changed; refresh the writer"
                    .into(),
            ));
        }
        self.validate_write_head(coordinate.state)
    }
}
