//! Explicit annotations of already-published states, independent of mutation deltas.
use crate::{BranchName, StateId, TimestampMicros};
use serde::{Deserialize, Serialize};

/// An explicit checkpoint of a published state. It allocates no state/version ID
/// and is not a permanent retention pin or a compressed recovery image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    /// Published state selected when this checkpoint was accepted; zero is empty.
    pub state_id: StateId,
    /// Branch selected at acceptance, retained as descriptive metadata.
    pub branch: BranchName,
    /// Caller-supplied checkpoint description.
    pub message: String,
    /// Caller-supplied author.
    pub author: String,
    /// Original engine acceptance time, in signed Unix microseconds.
    pub recorded_at: TimestampMicros,
}
