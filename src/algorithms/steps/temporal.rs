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
use crate::types::{AttrName, AttrValue, NodeId, StateId};

use super::{Step, StepMetadata, StepScope};

const DEFAULT_TEMPORAL_INDEX_VAR: &str = "temporal_index";

/// Compute differences between two temporal snapshots and store results.
///
/// This step takes two snapshots (from variables or context) and computes
/// which nodes and edges were added, removed, or modified between them.
///
/// **Outputs:**
/// - `<prefix>_nodes_added`: NodeMap marking added nodes (value=1)
/// - `<prefix>_nodes_removed`: NodeMap marking removed nodes (value=1)
/// - `<prefix>_edges_added`: EdgeMap marking added edges (value=1)
/// - `<prefix>_edges_removed`: EdgeMap marking removed edges (value=1)
#[allow(dead_code)] // Fields are placeholders for future full implementation
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
            nodes_added.insert(node, AlgorithmParamValue::Int(1));
        }

        let mut nodes_removed = HashMap::new();
        for node in delta.nodes_removed {
            nodes_removed.insert(node, AlgorithmParamValue::Int(1));
        }

        let prefix = &self.output_prefix;
        scope
            .variables_mut()
            .set_node_map(format!("{}_nodes_added", prefix), nodes_added);
        scope
            .variables_mut()
            .set_node_map(format!("{}_nodes_removed", prefix), nodes_removed);

        Ok(())
    }
}

/// Compute edge differences between temporal snapshots.
#[allow(dead_code)] // Fields are placeholders for future full implementation
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
            edges_added.insert(edge, AlgorithmParamValue::Int(1));
        }

        let mut edges_removed = HashMap::new();
        for edge in delta.edges_removed {
            edges_removed.insert(edge, AlgorithmParamValue::Int(1));
        }

        let prefix = &self.output_prefix;
        scope
            .variables_mut()
            .set_edge_map(format!("{}_edges_added", prefix), edges_added);
        scope
            .variables_mut()
            .set_edge_map(format!("{}_edges_removed", prefix), edges_removed);

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

/// Create a temporal snapshot at a specific commit or timestamp.
///
/// This is a placeholder that stores metadata about the snapshot request.
/// Full implementation requires integration with TemporalSnapshot infrastructure.
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
        if ctx.is_cancelled() {
            return Err(anyhow!("decay cancelled"));
        }

        if self.half_life <= 0.0 {
            return Err(anyhow!(
                "half_life must be positive, got {}",
                self.half_life
            ));
        }

        // Compute decay factor: exp(-λ * t) where λ = ln(2) / half_life
        let lambda = std::f64::consts::LN_2 / self.half_life;
        let decay_factor = (-lambda * self.time_delta).exp();

        // Try node map first, then edge map
        if let Ok(map) = scope.variables().node_map(&self.attr) {
            let decayed = apply_decay(map, decay_factor)?;
            scope
                .variables_mut()
                .set_node_map(self.output.clone(), decayed);
            Ok(())
        } else if let Ok(map) = scope.variables().edge_map(&self.attr) {
            let decayed = apply_decay(map, decay_factor)?;
            scope
                .variables_mut()
                .set_edge_map(self.output.clone(), decayed);
            Ok(())
        } else {
            Err(anyhow!("variable '{}' not found or not a map", self.attr))
        }
    }
}

fn apply_decay<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    factor: f64,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + std::cmp::Eq + std::hash::Hash,
{
    let mut result = HashMap::with_capacity(map.len());

    for (&key, value) in map.iter() {
        let decayed_value = match value {
            AlgorithmParamValue::Float(v) => AlgorithmParamValue::Float(v * factor),
            AlgorithmParamValue::Int(v) => AlgorithmParamValue::Float(*v as f64 * factor),
            other => other.clone(), // Non-numeric values pass through unchanged
        };
        result.insert(key, decayed_value);
    }

    Ok(result)
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
