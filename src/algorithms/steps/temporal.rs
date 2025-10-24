//! Temporal algorithm steps for time-series graph analysis.
//!
//! This module provides reusable step primitives for temporal graph analytics:
//! - Computing differences between snapshots
//! - Aggregating values over time windows
//! - Filtering based on temporal properties
//! - Tracking entity lifecycles

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::algorithms::{AlgorithmParamValue, Context, CostHint};
use crate::types::StateId;

use super::{Step, StepMetadata, StepScope};

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
        // For now, we'll use a simplified implementation that works with
        // the temporal scope in the context
        let temporal_scope = ctx
            .temporal_scope()
            .ok_or_else(|| anyhow!("No temporal scope set in context"))?;

        let _reference = temporal_scope
            .reference_snapshot
            .as_ref()
            .ok_or_else(|| anyhow!("No reference snapshot in temporal scope"))?;

        // We need a "current" snapshot to compare against
        // This would typically come from the subgraph's current state
        // For now, we'll create placeholder outputs

        let nodes_added = HashMap::new();
        let nodes_removed = HashMap::new();

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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let edges_added = HashMap::new();
        let edges_removed = HashMap::new();

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
        let temporal_scope = ctx
            .temporal_scope()
            .ok_or_else(|| anyhow!("No temporal scope set in context"))?;

        let (_start, _end) = temporal_scope
            .window
            .ok_or_else(|| anyhow!("No time window in temporal scope"))?;

        // For now, create a placeholder result
        // In a real implementation, we'd query the temporal index from the variable
        let mut result = HashMap::new();
        for &node in scope.node_ids() {
            // Placeholder: just set to count=1 for demonstration
            result.insert(node, AlgorithmParamValue::Int(1));
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        // Placeholder implementation
        let mut result = HashMap::new();
        for &node in scope.node_ids() {
            // For now, include all nodes
            result.insert(node, AlgorithmParamValue::Int(1));
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        // Placeholder implementation
        let mut result = HashMap::new();
        for &node in scope.node_ids() {
            result.insert(node, AlgorithmParamValue::Int(0)); // Not changed
        }

        scope
            .variables_mut()
            .set_node_map(self.output_var.clone(), result);

        Ok(())
    }
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
