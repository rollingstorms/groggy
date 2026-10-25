//! Temporal algorithm steps for time-series graph analysis.
//!
//! This module provides reusable step primitives for temporal graph analytics:
//! - Computing differences between snapshots
//! - Aggregating values over time windows
//! - Filtering based on temporal properties
//! - Tracking entity lifecycles

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::algorithms::temporal::ChangeType;
use crate::algorithms::{AlgorithmParamValue, Context, CostHint};
use crate::temporal::{TemporalIndex, TemporalSnapshot};
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId, StateId};

use super::{core::NodeColumn, Step, StepMetadata, StepScope};

const DEFAULT_TEMPORAL_INDEX_VAR: &str = "temporal_index";

/// Compute full-owner node membership and property-change masks between retained states.
///
/// Produces `<prefix>_nodes_added`, `_nodes_removed` and `_nodes_modified` node
/// maps with value 1. Modified means a property differs on a surviving node;
/// these are net snapshot differences, not all intervening historical events.
pub struct DiffNodesStep {
    /// Variable name containing the "before" snapshot (or use context ref)
    before_var: Option<String>,
    /// Variable name containing the "after" snapshot (or use context current)
    after_var: Option<String>,
    /// Prefix for output variables
    output_prefix: String,
}

impl DiffNodesStep {
    pub fn new(
        before_var: Option<String>,
        after_var: Option<String>,
        output_prefix: impl Into<String>,
    ) -> Self {
        Self {
            before_var,
            after_var,
            output_prefix: output_prefix.into(),
        }
    }
}

impl Step for DiffNodesStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "temporal.diff_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute node differences between two temporal snapshots".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("diff_nodes cancelled"));
        }

        let (before, after) = resolve_snapshot_pair(
            ctx,
            scope,
            self.before_var.as_ref(),
            self.after_var.as_ref(),
            "temporal.diff_nodes",
        )?;

        let delta = ctx.delta(&before, &after)?;

        let mut nodes_added = HashMap::new();
        for node in delta.nodes_added {
            check_cancelled(ctx)?;
            nodes_added.insert(node, AlgorithmParamValue::Int(1));
        }

        let mut nodes_removed = HashMap::new();
        for node in delta.nodes_removed {
            check_cancelled(ctx)?;
            nodes_removed.insert(node, AlgorithmParamValue::Int(1));
        }

        let mut nodes_modified = HashMap::new();
        for change in delta.node_attr_changes {
            check_cancelled(ctx)?;
            nodes_modified.insert(change.node_id, AlgorithmParamValue::Int(1));
        }
        check_cancelled(ctx)?;
        let prefix = &self.output_prefix;
        scope
            .variables_mut()
            .set_node_map(format!("{}_nodes_added", prefix), nodes_added);
        scope
            .variables_mut()
            .set_node_map(format!("{}_nodes_removed", prefix), nodes_removed);
        scope
            .variables_mut()
            .set_node_map(format!("{}_nodes_modified", prefix), nodes_modified);

        Ok(())
    }
}

/// Compute full-owner edge membership/property masks between retained states.
/// Outputs `<prefix>_edges_added`, `_edges_removed` and `_edges_modified`.
/// Common-edge endpoint changes reject until the delta format supports them.
pub struct DiffEdgesStep {
    before_var: Option<String>,
    after_var: Option<String>,
    output_prefix: String,
}

impl DiffEdgesStep {
    pub fn new(
        before_var: Option<String>,
        after_var: Option<String>,
        output_prefix: impl Into<String>,
    ) -> Self {
        Self {
            before_var,
            after_var,
            output_prefix: output_prefix.into(),
        }
    }
}

impl Step for DiffEdgesStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "temporal.diff_edges"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute edge differences between two temporal snapshots".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("diff_edges cancelled"));
        }

        let (before, after) = resolve_snapshot_pair(
            ctx,
            scope,
            self.before_var.as_ref(),
            self.after_var.as_ref(),
            "temporal.diff_edges",
        )?;

        let delta = ctx.delta(&before, &after)?;

        let mut edges_added = HashMap::new();
        for edge in delta.edges_added {
            check_cancelled(ctx)?;
            edges_added.insert(edge, AlgorithmParamValue::Int(1));
        }

        let mut edges_removed = HashMap::new();
        for edge in delta.edges_removed {
            check_cancelled(ctx)?;
            edges_removed.insert(edge, AlgorithmParamValue::Int(1));
        }

        let mut edges_modified = HashMap::new();
        for change in delta.edge_attr_changes {
            check_cancelled(ctx)?;
            edges_modified.insert(change.edge_id, AlgorithmParamValue::Int(1));
        }
        check_cancelled(ctx)?;
        let prefix = &self.output_prefix;
        scope
            .variables_mut()
            .set_edge_map(format!("{}_edges_added", prefix), edges_added);
        scope
            .variables_mut()
            .set_edge_map(format!("{}_edges_removed", prefix), edges_removed);
        scope
            .variables_mut()
            .set_edge_map(format!("{}_edges_modified", prefix), edges_modified);

        Ok(())
    }
}

/// Aggregate node attribute values over a temporal window.
///
/// This step queries the temporal index to get historical attribute values
/// for nodes within a specified time window and applies an aggregation function.
///
/// **Aggregation functions:**
/// - "count": Count number of changes
/// - "first": First value in window
/// - "last": Last value in window
/// - "sum": Sum numeric values
/// - "avg": Average numeric values
/// - "min": Minimum value
/// - "max": Maximum value
#[allow(dead_code)] // Fields used in placeholder implementation
pub struct WindowAggregateStep {
    /// Name of the node attribute to aggregate
    attr_name: String,
    /// Aggregation function to apply
    function: AggregateFunction,
    /// Output variable name
    output_var: String,
    /// Require temporal index in context (stored as scalar variable)
    index_var: String,
}

#[derive(Clone, Debug)]
pub enum AggregateFunction {
    Count,
    First,
    Last,
    Sum,
    Avg,
    Min,
    Max,
}

impl AggregateFunction {
    /// Parse aggregation function from string name.
    /// This is intentionally not implementing FromStr trait to avoid confusion
    /// with standard library trait and keep it as an explicit construction method.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "count" => Ok(Self::Count),
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            "sum" => Ok(Self::Sum),
            "avg" | "average" => Ok(Self::Avg),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            _ => Err(anyhow!("Unknown aggregation function: {}", s)),
        }
    }

    #[allow(dead_code)] // Used in tests, will be used when steps are fully implemented
    fn apply(&self, values: &[(StateId, AlgorithmParamValue)]) -> Result<AlgorithmParamValue> {
        if values.is_empty() {
            return Ok(AlgorithmParamValue::None);
        }

        match self {
            Self::Count => Ok(AlgorithmParamValue::Int(values.len() as i64)),
            Self::First => Ok(values.first().unwrap().1.clone()),
            Self::Last => Ok(values.last().unwrap().1.clone()),
            Self::Sum => {
                let mut sum = 0.0;
                for (_, val) in values {
                    sum += to_float(val)?;
                }
                Ok(AlgorithmParamValue::Float(sum))
            }
            Self::Avg => {
                let mut sum = 0.0;
                for (_, val) in values {
                    sum += to_float(val)?;
                }
                Ok(AlgorithmParamValue::Float(sum / values.len() as f64))
            }
            Self::Min => {
                let mut min = f64::INFINITY;
                for (_, val) in values {
                    let v = to_float(val)?;
                    if v < min {
                        min = v;
                    }
                }
                Ok(AlgorithmParamValue::Float(min))
            }
            Self::Max => {
                let mut max = f64::NEG_INFINITY;
                for (_, val) in values {
                    let v = to_float(val)?;
                    if v > max {
                        max = v;
                    }
                }
                Ok(AlgorithmParamValue::Float(max))
            }
        }
    }

    fn requires_numeric(&self) -> bool {
        matches!(
            self,
            AggregateFunction::Sum
                | AggregateFunction::Avg
                | AggregateFunction::Min
                | AggregateFunction::Max
        )
    }
}

/// Helper function to convert AlgorithmParamValue to f64
#[allow(dead_code)] // Used by apply method which is tested
fn to_float(val: &AlgorithmParamValue) -> Result<f64> {
    match val {
        AlgorithmParamValue::Int(i) => Ok(*i as f64),
        AlgorithmParamValue::Float(f) => Ok(*f),
        AlgorithmParamValue::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => Err(anyhow!("Cannot convert {:?} to float", val)),
    }
}

impl WindowAggregateStep {
    pub fn new(
        attr_name: impl Into<String>,
        function: AggregateFunction,
        output_var: impl Into<String>,
        index_var: impl Into<String>,
    ) -> Self {
        Self {
            attr_name: attr_name.into(),
            function,
            output_var: output_var.into(),
            index_var: index_var.into(),
        }
    }
}

impl Step for WindowAggregateStep {
    fn id(&self) -> &'static str {
        "temporal.window_aggregate"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Aggregate node attribute values over a temporal window".to_string(),
            cost_hint: CostHint::Quadratic, // Depends on window size
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("window_aggregate cancelled"));
        }

        let scope_info = ctx
            .temporal_scope()
            .ok_or_else(|| anyhow!("temporal.window_aggregate requires a temporal scope"))?;
        let (start, end) = scope_info
            .window
            .ok_or_else(|| anyhow!("temporal.window_aggregate requires a scope window"))?;
        if start > end {
            return Err(anyhow!(
                "temporal.window_aggregate received invalid window: start ({start}) > end ({end})"
            ));
        }

        let index = ensure_temporal_index(scope, &self.index_var)?;
        let attr_name = AttrName::from(self.attr_name.clone());
        let numeric_only = self.function.requires_numeric();
        let mut result = HashMap::new();

        for &node in scope.node_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("window_aggregate cancelled"));
            }

            let mut history = index.node_attr_history(node, &attr_name, start, end);
            if history.is_empty() || history.iter().all(|(_, value)| is_placeholder_value(value)) {
                history = snapshot_attr_history(scope, node, &attr_name, start, end)?;
            }

            let mut converted = Vec::with_capacity(history.len());
            for (commit, value) in history {
                let resolved = match resolve_history_value(scope, &attr_name, value)? {
                    Some(val) => val,
                    None => continue,
                };
                if let Some(param) = AlgorithmParamValue::from_attr_value(resolved) {
                    if numeric_only
                        && !matches!(
                            param,
                            AlgorithmParamValue::Int(_)
                                | AlgorithmParamValue::Float(_)
                                | AlgorithmParamValue::Bool(_)
                        )
                    {
                        continue;
                    }
                    converted.push((commit, param));
                }
            }

            let aggregated = self.function.apply(&converted)?;
            result.insert(node, aggregated);
        }

        scope
            .variables_mut()
            .set_node_map(self.output_var.clone(), result);

        Ok(())
    }
}

/// Filter nodes based on temporal properties.
///
/// This step filters the working set of nodes based on temporal criteria:
/// - Age (time since creation)
/// - Lifetime (creation to deletion span)
/// - Change frequency (number of modifications)
/// - Activity window (active during specific time range)
#[allow(dead_code)] // Predicate used in future full implementation
pub struct TemporalFilterStep {
    /// Predicate to apply
    predicate: TemporalPredicate,
    /// Output variable for filtered nodes (0 or 1)
    output_var: String,
}

#[derive(Clone, Debug)]
pub enum TemporalPredicate {
    /// Node was created after commit
    CreatedAfter(StateId),
    /// Node was created before commit
    CreatedBefore(StateId),
    /// Node existed at specific commit
    ExistedAt(StateId),
    /// Node was modified in commit range
    ModifiedInRange(StateId, StateId),
}

impl TemporalFilterStep {
    pub fn new(predicate: TemporalPredicate, output_var: impl Into<String>) -> Self {
        Self {
            predicate,
            output_var: output_var.into(),
        }
    }

    pub fn created_after(commit: StateId, output_var: impl Into<String>) -> Self {
        Self::new(TemporalPredicate::CreatedAfter(commit), output_var)
    }

    pub fn created_before(commit: StateId, output_var: impl Into<String>) -> Self {
        Self::new(TemporalPredicate::CreatedBefore(commit), output_var)
    }

    pub fn existed_at(commit: StateId, output_var: impl Into<String>) -> Self {
        Self::new(TemporalPredicate::ExistedAt(commit), output_var)
    }

    pub fn modified_in_range(start: StateId, end: StateId, output_var: impl Into<String>) -> Self {
        Self::new(TemporalPredicate::ModifiedInRange(start, end), output_var)
    }
}

impl Step for TemporalFilterStep {
    fn id(&self) -> &'static str {
        "temporal.filter"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Filter nodes based on temporal properties".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("temporal.filter cancelled"));
        }

        let index = ensure_temporal_index(scope, DEFAULT_TEMPORAL_INDEX_VAR)?;
        let mut result = HashMap::new();

        match self.predicate {
            TemporalPredicate::CreatedAfter(commit) => {
                for &node in scope.node_ids() {
                    let created = index.node_creation_commit(node);
                    let keep = created.map(|c| c > commit).unwrap_or(false);
                    result.insert(node, bool_to_param(keep));
                }
            }
            TemporalPredicate::CreatedBefore(commit) => {
                for &node in scope.node_ids() {
                    let created = index.node_creation_commit(node);
                    let keep = created.map(|c| c < commit).unwrap_or(false);
                    result.insert(node, bool_to_param(keep));
                }
            }
            TemporalPredicate::ExistedAt(commit) => {
                for &node in scope.node_ids() {
                    let keep = index.node_exists_at(node, commit);
                    result.insert(node, bool_to_param(keep));
                }
            }
            TemporalPredicate::ModifiedInRange(start, end) => {
                if start > end {
                    return Err(anyhow!(
                        "temporal.filter received invalid range: start ({start}) > end ({end})"
                    ));
                }
                let changed: HashSet<NodeId> = index
                    .nodes_changed_in_range(start, end)
                    .into_iter()
                    .collect();
                for &node in scope.node_ids() {
                    result.insert(node, bool_to_param(changed.contains(&node)));
                }
            }
        }

        scope
            .variables_mut()
            .set_node_map(self.output_var.clone(), result);

        Ok(())
    }
}

/// Mark nodes that changed within a time window.
///
/// This step uses the temporal index to identify nodes that were
/// created, deleted, or had attributes modified within a specified window.
#[allow(dead_code)] // change_type used in future full implementation
pub struct MarkChangedNodesStep {
    /// Output variable name
    output_var: String,
    /// Optional: specific change type to filter for
    change_type: Option<String>,
}

impl MarkChangedNodesStep {
    pub fn new(output_var: impl Into<String>, change_type: Option<String>) -> Self {
        Self {
            output_var: output_var.into(),
            change_type,
        }
    }

    pub fn all_changes(output_var: impl Into<String>) -> Self {
        Self::new(output_var, None)
    }

    pub fn created_only(output_var: impl Into<String>) -> Self {
        Self::new(output_var, Some("created".to_string()))
    }

    pub fn modified_only(output_var: impl Into<String>) -> Self {
        Self::new(output_var, Some("modified".to_string()))
    }
}

impl Step for MarkChangedNodesStep {
    fn id(&self) -> &'static str {
        "temporal.mark_changed_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Mark nodes that changed within a time window".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("temporal.mark_changed_nodes cancelled"));
        }

        let scope_info = ctx
            .temporal_scope()
            .ok_or_else(|| anyhow!("temporal.mark_changed_nodes requires a temporal scope"))?;
        if scope_info.window.is_none() {
            return Err(anyhow!(
                "temporal.mark_changed_nodes requires the temporal scope to define a window"
            ));
        }

        let index = ensure_temporal_index(scope, DEFAULT_TEMPORAL_INDEX_VAR)?;
        let changed_entities = ctx.changed_entities(index.as_ref())?;

        let desired = match self.change_type.as_deref() {
            None => None,
            Some("created") => Some(ChangeType::Created),
            Some("deleted") => Some(ChangeType::Deleted),
            Some("modified") => Some(ChangeType::AttributeModified),
            Some("any") => None,
            Some(other) => {
                return Err(anyhow!(
                    "unknown change_type '{other}' supplied to temporal.mark_changed_nodes"
                ))
            }
        };

        let mut result = HashMap::new();
        for &node in scope.node_ids() {
            let matches = changed_entities
                .node_change_types
                .get(&node)
                .map(|change| {
                    desired.map_or(true, |desired_change| {
                        change_matches(*change, desired_change)
                    })
                })
                .unwrap_or(false);
            result.insert(node, bool_to_param(matches));
        }

        scope
            .variables_mut()
            .set_node_map(self.output_var.clone(), result);

        Ok(())
    }
}

/// Capture an owned immutable retained state, excluding in-flight property writes.
/// Timestamp references select retained acceptance time in epoch seconds, not
/// per-property effective time. The snapshot covers the owner graph.
pub struct SnapshotAtStep {
    /// Commit ID or timestamp to snapshot at
    reference: SnapshotReference,
    /// Output variable name
    output: String,
}

#[derive(Clone, Debug)]
pub enum SnapshotReference {
    Commit(StateId),
    Timestamp(u64),
}

impl SnapshotAtStep {
    pub fn new_at_commit(commit_id: StateId, output: impl Into<String>) -> Self {
        Self {
            reference: SnapshotReference::Commit(commit_id),
            output: output.into(),
        }
    }

    pub fn new_at_timestamp(timestamp: u64, output: impl Into<String>) -> Self {
        Self {
            reference: SnapshotReference::Timestamp(timestamp),
            output: output.into(),
        }
    }
}

impl Step for SnapshotAtStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "temporal.snapshot_at"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Create temporal snapshot at commit or timestamp".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("snapshot_at cancelled"));
        }

        let snapshot = match &self.reference {
            SnapshotReference::Commit(id) => snapshot_at_commit(scope, *id)?,
            SnapshotReference::Timestamp(ts) => snapshot_at_timestamp(scope, *ts)?,
        };

        check_cancelled(ctx)?;
        scope
            .variables_mut()
            .set_snapshot(self.output.clone(), snapshot);

        Ok(())
    }
}

/// Filter to a temporal window between start and end timestamps.
pub struct TemporalWindowStep {
    start: u64,
    end: u64,
    output: String,
}

impl TemporalWindowStep {
    pub fn new(start: u64, end: u64, output: impl Into<String>) -> Self {
        Self {
            start,
            end,
            output: output.into(),
        }
    }
}

impl Step for TemporalWindowStep {
    fn id(&self) -> &'static str {
        "temporal.window"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Filter to temporal window [{}, {}]", self.start, self.end),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("temporal.window cancelled"));
        }

        if self.start > self.end {
            return Err(anyhow!(
                "invalid temporal window: start ({}) > end ({})",
                self.start,
                self.end
            ));
        }

        let index = ensure_temporal_index(scope, DEFAULT_TEMPORAL_INDEX_VAR)?;
        let mut result = HashMap::new();

        for &node in scope.node_ids() {
            let existed = node_existed_in_window(index.as_ref(), node, self.start, self.end);
            result.insert(node, bool_to_param(existed));
        }

        scope
            .variables_mut()
            .set_node_map(self.output.clone(), result);

        Ok(())
    }
}

/// Apply time-based decay to attribute values.
///
/// Uses exponential decay: value * exp(-λ * t) where λ = ln(2) / half_life
pub struct DecayStep {
    attr: String,
    half_life: f64,
    output: String,
    /// Time since reference (defaults to 0, can be overridden)
    time_delta: f64,
}

impl DecayStep {
    pub fn new(attr: impl Into<String>, half_life: f64, output: impl Into<String>) -> Self {
        Self {
            attr: attr.into(),
            half_life,
            output: output.into(),
            time_delta: 0.0,
        }
    }

    pub fn with_time_delta(mut self, time_delta: f64) -> Self {
        self.time_delta = time_delta;
        self
    }
}

impl Step for DecayStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "temporal.decay"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Apply exponential decay with half-life {}", self.half_life),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        check_cancelled(ctx)?;
        validate_decay_controls(self.half_life, self.time_delta)?;
        // Divide before multiplying: LN_2 / a subnormal half-life can overflow,
        // producing NaN for zero elapsed time. A nonnegative ratio may overflow
        // to infinity legitimately, in which case exp(-infinity) is zero.
        let factor = (-std::f64::consts::LN_2 * (self.time_delta / self.half_life)).exp();
        let variables = scope.variables();
        if let Ok(column) = variables.node_column(&self.attr) {
            let mut ids = Vec::new();
            let mut values = Vec::new();
            let mut seen = HashSet::new();
            for (node, value) in column.iter() {
                check_cancelled(ctx)?;
                validate_decay_entity(scope, node, false)?;
                if !seen.insert(node) {
                    return Err(anyhow!("temporal.decay duplicate node ID {node}"));
                }
                ids.push(node);
                values.push(decay_value(value, factor)?);
            }
            check_cancelled(ctx)?;
            scope
                .variables_mut()
                .set_node_column(self.output.clone(), NodeColumn::new(ids, values));
        } else {
            let (source, edges) = if let Ok(map) = variables.node_map(&self.attr) {
                (map, false)
            } else if let Ok(map) = variables.edge_map(&self.attr) {
                (map, true)
            } else {
                return Err(anyhow!(
                    "variable '{}' must be a node map, node column, or edge map",
                    self.attr
                ));
            };
            let mut result = HashMap::with_capacity(source.len());
            for (&entity, value) in source {
                check_cancelled(ctx)?;
                validate_decay_entity(scope, entity, edges)?;
                result.insert(entity, decay_value(value, factor)?);
            }
            check_cancelled(ctx)?;
            if edges {
                scope
                    .variables_mut()
                    .set_edge_map(self.output.clone(), result);
            } else {
                scope
                    .variables_mut()
                    .set_node_map(self.output.clone(), result);
            }
        }
        Ok(())
    }
}

pub(super) fn validate_decay_controls(half_life: f64, time_delta: f64) -> Result<()> {
    if !half_life.is_finite() || half_life <= 0.0 {
        return Err(anyhow!("half_life must be finite and positive"));
    }
    if !time_delta.is_finite() || time_delta < 0.0 {
        return Err(anyhow!("time_delta must be finite and non-negative"));
    }
    Ok(())
}

fn check_cancelled(ctx: &Context) -> Result<()> {
    if ctx.is_cancelled() {
        return Err(anyhow!("temporal step cancelled"));
    }
    Ok(())
}

fn decay_value(value: &AlgorithmParamValue, factor: f64) -> Result<AlgorithmParamValue> {
    let number = match value {
        AlgorithmParamValue::Int(value) => *value as f64,
        AlgorithmParamValue::Float(value) if value.is_finite() => *value,
        _ => return Err(anyhow!("temporal.decay requires finite numeric values")),
    };
    let result = number * factor;
    if !result.is_finite() {
        return Err(anyhow!("temporal.decay produced a non-finite value"));
    }
    Ok(AlgorithmParamValue::Float(result))
}

fn validate_decay_entity(scope: &StepScope<'_>, entity: usize, edges: bool) -> Result<()> {
    let selected = scope.subgraph();
    let owner = selected.graph();
    let graph = owner.borrow();
    if edges {
        if !selected.edge_set().contains(&entity) || !graph.has_edge(entity) {
            return Err(anyhow!(
                "temporal.decay edge {entity} is outside the active selection"
            ));
        }
        let (source, target) = graph.edge_endpoints(entity)?;
        if !selected.node_set().contains(&source)
            || !selected.node_set().contains(&target)
            || !graph.has_node(source)
            || !graph.has_node(target)
        {
            return Err(anyhow!(
                "temporal.decay edge {entity} endpoints are outside the active selection"
            ));
        }
    } else if !selected.node_set().contains(&entity) || !graph.has_node(entity) {
        return Err(anyhow!(
            "temporal.decay node {entity} is outside the active selection"
        ));
    }
    Ok(())
}

fn snapshot_attr_history(
    scope: &StepScope<'_>,
    node_id: NodeId,
    attr: &AttrName,
    start: StateId,
    end: StateId,
) -> Result<Vec<(StateId, AttrValue)>> {
    if start > end {
        return Ok(Vec::new());
    }

    let graph = scope.subgraph().graph();
    let mut entries = Vec::new();
    for commit in start..=end {
        let snapshot = graph
            .borrow()
            .snapshot_at_commit(commit)
            .map_err(|err| anyhow!("failed to build snapshot at commit {commit}: {err}"))?;
        if let Some(value) = snapshot.node_attr(node_id, attr) {
            entries.push((commit, value));
        }
    }
    Ok(entries)
}

fn is_placeholder_value(value: &AttrValue) -> bool {
    matches!(value, AttrValue::Text(text) if text.starts_with("index_"))
}

fn resolve_history_value(
    scope: &StepScope<'_>,
    attr: &AttrName,
    value: AttrValue,
) -> Result<Option<AttrValue>> {
    if let AttrValue::Text(text) = &value {
        if let Some(resolved) = resolve_placeholder_from_pool(scope, attr, text, true)? {
            return Ok(Some(resolved));
        }
    }
    Ok(Some(value))
}

fn resolve_placeholder_from_pool(
    scope: &StepScope<'_>,
    attr: &AttrName,
    placeholder: &str,
    is_node: bool,
) -> Result<Option<AttrValue>> {
    let Some(index_str) = placeholder.strip_prefix("index_") else {
        return Ok(None);
    };
    let Ok(index) = index_str.parse::<usize>() else {
        return Ok(None);
    };

    let graph_rc = scope.subgraph().graph();
    let graph_ref = graph_rc.borrow();
    let pool = graph_ref.pool();
    Ok(pool.get_attr_by_index(attr, index, is_node).cloned())
}

fn ensure_temporal_index(scope: &mut StepScope<'_>, name: &str) -> Result<Arc<TemporalIndex>> {
    if let Ok(index) = scope.variables().temporal_index(name) {
        return Ok(index);
    }

    let graph = scope.subgraph().graph();
    let built = graph
        .borrow()
        .build_temporal_index()
        .map_err(|err| anyhow!("failed to build temporal index: {err}"))?;
    let index = Arc::new(built);
    scope
        .variables_mut()
        .set_temporal_index(name.to_string(), Arc::clone(&index));
    Ok(index)
}

fn snapshot_at_commit(scope: &StepScope<'_>, commit_id: StateId) -> Result<TemporalSnapshot> {
    scope
        .subgraph()
        .graph()
        .borrow()
        .snapshot_at_commit(commit_id)
        .map_err(|err| anyhow!("failed to build snapshot at commit {commit_id}: {err}"))
}

fn snapshot_at_timestamp(scope: &StepScope<'_>, timestamp: u64) -> Result<TemporalSnapshot> {
    scope
        .subgraph()
        .graph()
        .borrow()
        .snapshot_at_timestamp(timestamp)
        .map_err(|err| anyhow!("failed to build snapshot at timestamp {timestamp}: {err}"))
}

fn resolve_snapshot_pair(
    ctx: &Context,
    scope: &StepScope<'_>,
    before_var: Option<&String>,
    after_var: Option<&String>,
    step_id: &str,
) -> Result<(TemporalSnapshot, TemporalSnapshot)> {
    let before_snapshot = if let Some(name) = before_var {
        scope.variables().snapshot(name)?.clone()
    } else {
        let temporal_scope = ctx.temporal_scope().ok_or_else(|| {
            anyhow!("{step_id} requires a temporal scope or explicit 'before' snapshot variable")
        })?;
        if let Some(snapshot) = &temporal_scope.reference_snapshot {
            snapshot.clone()
        } else if let Some((start, _)) = temporal_scope.window {
            snapshot_at_commit(scope, start)?
        } else {
            return Err(anyhow!(
                "{step_id} requires a reference snapshot via context or 'before' parameter"
            ));
        }
    };

    let after_snapshot = if let Some(name) = after_var {
        scope.variables().snapshot(name)?.clone()
    } else {
        let temporal_scope = ctx.temporal_scope().ok_or_else(|| {
            anyhow!("{step_id} requires a temporal scope or explicit 'after' snapshot variable")
        })?;
        snapshot_at_commit(scope, temporal_scope.current_commit)?
    };

    let owner = scope.subgraph().graph();
    let graph = owner.borrow();
    if !graph.owns_temporal_snapshot(&before_snapshot)
        || !graph.owns_temporal_snapshot(&after_snapshot)
    {
        return Err(anyhow!(
            "{step_id} snapshots must belong to the invocation owner history"
        ));
    }
    Ok((before_snapshot, after_snapshot))
}

fn bool_to_param(value: bool) -> AlgorithmParamValue {
    AlgorithmParamValue::Int(if value { 1 } else { 0 })
}

fn node_existed_in_window(
    index: &TemporalIndex,
    node_id: NodeId,
    start: StateId,
    end: StateId,
) -> bool {
    if let Some((created, deleted)) = index.node_lifetime_range(node_id) {
        let deletion_commit = deleted.unwrap_or(StateId::MAX);
        created <= end && start <= deletion_commit
    } else {
        false
    }
}

fn change_matches(change: ChangeType, desired: ChangeType) -> bool {
    change == desired || change == ChangeType::Multiple
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{algorithms::steps::core::StepVariables, api::graph::Graph, subgraphs::Subgraph};
    use std::{cell::RefCell, rc::Rc, sync::atomic::AtomicBool};

    fn fixture() -> Subgraph {
        let mut graph = Graph::new();
        graph.add_nodes(3);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        graph.commit("fixture".into(), "test".into()).unwrap();
        Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            [0, 1, 2].into_iter().collect(),
            "decay".into(),
        )
        .unwrap()
    }

    #[test]
    fn diff_step_rejects_foreign_owner_and_preserves_all_prior_outputs() {
        let selected = fixture();
        let foreign = fixture();
        let foreign_state = foreign.graph().borrow().current_state_id();
        let snapshot = foreign
            .graph()
            .borrow()
            .snapshot_at_commit(foreign_state)
            .unwrap();
        let mut vars = StepVariables::default();
        vars.set_snapshot("before", snapshot.clone());
        vars.set_snapshot("after", snapshot);
        for name in [
            "diff_nodes_added",
            "diff_nodes_removed",
            "diff_nodes_modified",
        ] {
            vars.set_node_map(
                name,
                [(55, AlgorithmParamValue::Int(91))].into_iter().collect(),
            );
        }
        let mut scope = StepScope::new(&selected, &mut vars);
        assert!(
            DiffNodesStep::new(Some("before".into()), Some("after".into()), "diff")
                .apply(&mut Context::new(), &mut scope)
                .is_err()
        );
        for name in [
            "diff_nodes_added",
            "diff_nodes_removed",
            "diff_nodes_modified",
        ] {
            assert_eq!(
                scope.variables().node_map(name).unwrap()[&55],
                AlgorithmParamValue::Int(91)
            );
        }
    }

    #[test]
    fn diff_steps_keep_historical_removed_ids_and_ignore_live_selection_clipping() {
        let selected = fixture();
        let state = selected.graph().borrow().current_state_id();
        let before = selected.graph().borrow().snapshot_at_commit(state).unwrap();
        selected.graph().borrow_mut().remove_node(2).unwrap();
        let next = selected
            .graph()
            .borrow_mut()
            .commit("removed".into(), "test".into())
            .unwrap();
        let after = selected.graph().borrow().snapshot_at_commit(next).unwrap();
        let mut vars = StepVariables::default();
        vars.set_snapshot("before", before);
        vars.set_snapshot("after", after);
        let mut scope = StepScope::new(&selected, &mut vars);
        DiffNodesStep::new(Some("before".into()), Some("after".into()), "diff")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        DiffEdgesStep::new(Some("before".into()), Some("after".into()), "diff")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        assert_eq!(
            scope.variables().node_map("diff_nodes_removed").unwrap()[&2],
            AlgorithmParamValue::Int(1)
        );
        assert_eq!(
            scope.variables().edge_map("diff_edges_removed").unwrap()[&1],
            AlgorithmParamValue::Int(1)
        );
        assert!(scope
            .variables()
            .node_map("diff_nodes_modified")
            .unwrap()
            .is_empty());
        assert!(scope
            .variables()
            .edge_map("diff_edges_modified")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn decay_preserves_column_order_and_supports_in_place_output() {
        let selected = fixture();
        let mut vars = StepVariables::default();
        vars.set_node_column(
            "values",
            NodeColumn::new(
                vec![2, 0],
                vec![
                    AlgorithmParamValue::Int(-8),
                    AlgorithmParamValue::Float(4.0),
                ],
            ),
        );
        let mut scope = StepScope::new(&selected, &mut vars);
        DecayStep::new("values", 2.0, "values")
            .with_time_delta(4.0)
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        let actual: Vec<_> = scope
            .variables()
            .node_column("values")
            .unwrap()
            .iter()
            .map(|(id, value)| (id, value.clone()))
            .collect();
        assert_eq!(
            actual,
            vec![
                (2, AlgorithmParamValue::Float(-2.0)),
                (0, AlgorithmParamValue::Float(1.0))
            ]
        );
    }

    #[test]
    fn decay_validates_controls_and_values_before_installing_output() {
        let selected = fixture();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "old",
            [(0, AlgorithmParamValue::Int(91))].into_iter().collect(),
        );
        for (half, elapsed) in [
            (0.0, 0.0),
            (-1.0, 0.0),
            (f64::INFINITY, 0.0),
            (f64::NAN, 0.0),
            (1.0, -1.0),
            (1.0, f64::INFINITY),
            (1.0, f64::NAN),
        ] {
            vars.set_node_map("values", HashMap::new());
            let mut scope = StepScope::new(&selected, &mut vars);
            assert!(DecayStep::new("values", half, "old")
                .with_time_delta(elapsed)
                .apply(&mut Context::new(), &mut scope)
                .is_err());
            assert_eq!(
                scope.variables().node_map("old").unwrap()[&0],
                AlgorithmParamValue::Int(91)
            );
        }
        for value in [
            AlgorithmParamValue::Bool(true),
            AlgorithmParamValue::None,
            AlgorithmParamValue::Text("x".into()),
            AlgorithmParamValue::Float(f64::NAN),
            AlgorithmParamValue::Float(f64::INFINITY),
        ] {
            vars.set_edge_map("values", [(0, value)].into_iter().collect());
            let mut scope = StepScope::new(&selected, &mut vars);
            assert!(DecayStep::new("values", 1.0, "old")
                .apply(&mut Context::new(), &mut scope)
                .is_err());
            assert_eq!(
                scope.variables().node_map("old").unwrap()[&0],
                AlgorithmParamValue::Int(91)
            );
        }
    }

    #[test]
    fn decay_extreme_finite_controls_have_finite_limits() {
        let selected = fixture();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "values",
            [(0, AlgorithmParamValue::Float(f64::MAX))]
                .into_iter()
                .collect(),
        );
        let mut scope = StepScope::new(&selected, &mut vars);
        for (elapsed, expected) in [(0.0, f64::MAX), (f64::MAX, 0.0)] {
            DecayStep::new("values", f64::from_bits(1), "result")
                .with_time_delta(elapsed)
                .apply(&mut Context::new(), &mut scope)
                .unwrap();
            assert_eq!(
                scope.variables().node_map("result").unwrap()[&0],
                AlgorithmParamValue::Float(expected)
            );
        }
    }

    #[test]
    fn decay_rejects_foreign_inactive_and_duplicate_ids() {
        let selected = fixture();
        let mut vars = StepVariables::default();
        for edges in [false, true] {
            let map = [(999, AlgorithmParamValue::Int(2))].into_iter().collect();
            if edges {
                vars.set_edge_map("values", map);
            } else {
                vars.set_node_map("values", map);
            }
            let mut scope = StepScope::new(&selected, &mut vars);
            assert!(DecayStep::new("values", 1.0, "result")
                .apply(&mut Context::new(), &mut scope)
                .is_err());
            assert!(!scope.variables().contains("result"));
        }
        vars.set_node_column(
            "values",
            NodeColumn::new(vec![0, 0], vec![AlgorithmParamValue::Int(1); 2]),
        );
        let mut scope = StepScope::new(&selected, &mut vars);
        assert!(DecayStep::new("values", 1.0, "result")
            .apply(&mut Context::new(), &mut scope)
            .is_err());
        selected.graph().borrow_mut().remove_node(2).unwrap();
        scope.variables_mut().set_node_map(
            "values",
            [(2, AlgorithmParamValue::Int(2))].into_iter().collect(),
        );
        assert!(DecayStep::new("values", 1.0, "result")
            .apply(&mut Context::new(), &mut scope)
            .is_err());
    }

    #[test]
    fn snapshot_step_retains_owner_state_without_working_changes() {
        let selected = fixture();
        let state = selected.graph().borrow().current_state_id();
        selected
            .graph()
            .borrow_mut()
            .set_node_attr(0, "score".into(), AttrValue::Int(3))
            .unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&selected, &mut vars);
        SnapshotAtStep::new_at_commit(state, "saved")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        let saved = scope.variables().snapshot("saved").unwrap().clone();
        assert!(saved.node_exists(2));
        assert_eq!(saved.node_attr(0, &"score".into()), None);
        let seconds = saved.lineage().timestamp;
        SnapshotAtStep::new_at_timestamp(seconds, "by_time")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        assert_eq!(
            scope
                .variables()
                .snapshot("by_time")
                .unwrap()
                .lineage()
                .commit_id,
            state
        );
        selected
            .graph()
            .borrow_mut()
            .commit("next".into(), "test".into())
            .unwrap();
        assert_eq!(saved.node_attr(0, &"score".into()), None);
        assert!(SnapshotAtStep::new_at_commit(u64::MAX, "saved")
            .apply(&mut Context::new(), &mut scope)
            .is_err());
        assert_eq!(
            scope
                .variables()
                .snapshot("saved")
                .unwrap()
                .lineage()
                .commit_id,
            state
        );
    }

    #[test]
    fn snapshot_and_decay_cancellation_preserve_existing_variables() {
        let selected = fixture();
        let mut vars = StepVariables::default();
        vars.set_scalar("result", AlgorithmParamValue::Int(71));
        vars.set_node_map(
            "values",
            [(0, AlgorithmParamValue::Int(2))].into_iter().collect(),
        );
        let mut scope = StepScope::new(&selected, &mut vars);
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(SnapshotAtStep::new_at_commit(1, "result")
            .apply(&mut ctx, &mut scope)
            .is_err());
        assert!(DecayStep::new("values", 1.0, "result")
            .apply(&mut ctx, &mut scope)
            .is_err());
        assert_eq!(
            scope.variables().scalar("result").unwrap(),
            &AlgorithmParamValue::Int(71)
        );
    }

    #[test]
    fn test_aggregate_function_from_str() {
        assert!(matches!(
            AggregateFunction::from_str("count").unwrap(),
            AggregateFunction::Count
        ));
        assert!(matches!(
            AggregateFunction::from_str("sum").unwrap(),
            AggregateFunction::Sum
        ));
        assert!(matches!(
            AggregateFunction::from_str("avg").unwrap(),
            AggregateFunction::Avg
        ));
    }

    #[test]
    fn test_aggregate_count() {
        let values = vec![
            (1, AlgorithmParamValue::Int(10)),
            (2, AlgorithmParamValue::Int(20)),
            (3, AlgorithmParamValue::Int(30)),
        ];

        let result = AggregateFunction::Count.apply(&values).unwrap();
        assert_eq!(result, AlgorithmParamValue::Int(3));
    }

    #[test]
    fn test_aggregate_sum() {
        let values = vec![
            (1, AlgorithmParamValue::Float(1.5)),
            (2, AlgorithmParamValue::Float(2.5)),
            (3, AlgorithmParamValue::Float(3.0)),
        ];

        let result = AggregateFunction::Sum.apply(&values).unwrap();
        if let AlgorithmParamValue::Float(sum) = result {
            assert!((sum - 7.0).abs() < 0.001);
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_aggregate_avg() {
        let values = vec![
            (1, AlgorithmParamValue::Float(10.0)),
            (2, AlgorithmParamValue::Float(20.0)),
            (3, AlgorithmParamValue::Float(30.0)),
        ];

        let result = AggregateFunction::Avg.apply(&values).unwrap();
        if let AlgorithmParamValue::Float(avg) = result {
            assert!((avg - 20.0).abs() < 0.001);
        } else {
            panic!("Expected Float result");
        }
    }

    #[test]
    fn test_aggregate_empty() {
        let values = vec![];
        let result = AggregateFunction::Count.apply(&values).unwrap();
        assert_eq!(result, AlgorithmParamValue::None);
    }
}
