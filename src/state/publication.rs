//! Explicit effective time and engine-recorded publication time.
use crate::{AttrName, EdgeId, GraphError, GraphResult, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Signed Unix microseconds. Negative values represent times before the epoch.
pub type TimestampMicros = i64;

/// Identity of a logical change within a published delta.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ChangeKey {
    NodeAdded(NodeId),
    NodeRemoved(NodeId),
    EdgeAdded(EdgeId),
    EdgeRemoved(EdgeId),
    /// Endpoint revision of an existing edge; not a liveness toggle.
    EdgeEndpoints(EdgeId),
    NodeAttribute(NodeId, AttrName),
    EdgeAttribute(EdgeId, AttrName),
}

/// One version's time metadata. Effective times are keyed by logical change;
/// typed before/after data remain in the version's existing delta.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicationMetadata {
    /// Engine-assigned acceptance/publication clock; callers cannot override it.
    pub recorded_at: TimestampMicros,
    /// Explicit batch default, otherwise the same clock as recorded_at.
    pub effective_at: TimestampMicros,
    /// Effective time for every actual logical change, in deterministic order.
    pub change_times: BTreeMap<ChangeKey, TimestampMicros>,
}

pub(crate) fn now_micros() -> GraphResult<TimestampMicros> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let micros = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => i128::try_from(duration.as_micros()).map_err(|_| {
            GraphError::InvalidInput("Publication clock exceeds timestamp range".into())
        })?,
        Err(error) => -i128::try_from(error.duration().as_micros()).map_err(|_| {
            GraphError::InvalidInput("Publication clock exceeds timestamp range".into())
        })?,
    };
    i64::try_from(micros)
        .map_err(|_| GraphError::InvalidInput("Publication clock exceeds timestamp range".into()))
}
