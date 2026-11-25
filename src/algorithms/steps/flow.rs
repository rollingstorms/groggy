//! Network flow primitives for max-flow/min-cut algorithms.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{AtomicStepEffect, Step, StepMetadata, StepScope};
use super::read_scope::{sealed, ReadStep, ReadStepScope};

/// Update flow values along a path or set of edges.
///
/// This step takes an existing flow map and updates it with new flow values,
/// typically representing flow along an augmenting path in max-flow algorithms.
///
/// # Parameters
/// - `flow`: Current flow map (edge id → flow value)
/// - `delta`: Flow change map (edge id → delta value) or scalar delta
/// - `target`: Output variable name for updated flow
///
/// # Flow Conservation
/// This step does NOT validate flow conservation - that's the responsibility
/// of the algorithm builder. It simply performs element-wise addition.
pub struct FlowUpdateStep {
    flow: String,
    delta: String,
    target: String,
}

impl FlowUpdateStep {
    pub fn new(
        flow: impl Into<String>,
        delta: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            flow: flow.into(),
            delta: delta.into(),
            target: target.into(),
        }
    }
}

impl Step for FlowUpdateStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.flow_update"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Update flow values along edges".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for FlowUpdateStep {}
impl ReadStep for FlowUpdateStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("flow_update cancelled"));
        }

        let flow_map = scope.variables().edge_map(&self.flow)?;

        // Try to determine if delta is an edge map or scalar
        // First try edge map, then scalar
        let is_edge_map = scope.variables().edge_map(&self.delta).is_ok();

        let mut result = HashMap::with_capacity(flow_map.len());

        if is_edge_map {
            let delta_map = scope.variables().edge_map(&self.delta)?;
            // Element-wise update: flow[e] += delta[e]
            for (&edge, flow_val) in flow_map.iter() {
                check_cancelled(ctx)?;
                validate_flow_value(flow_val, &self.flow)?;
                let delta_val = delta_map.get(&edge);
                let new_flow = if let Some(delta) = delta_val {
                    add_flow_values(flow_val, delta, &self.flow, &self.delta)?
                } else {
                    // No delta for this edge, keep original flow
                    flow_val.clone()
                };
                result.insert(edge, new_flow);
            }

            // Add any edges in delta that weren't in flow (treat original flow as 0)
            for (&edge, delta_val) in delta_map.iter() {
                check_cancelled(ctx)?;
                validate_flow_value(delta_val, &self.delta)?;
                if !flow_map.contains_key(&edge) {
                    result.insert(edge, delta_val.clone());
                }
            }
        } else {
            // Try as scalar
            let delta_scalar = scope.variables().scalar(&self.delta)?;
            validate_flow_value(delta_scalar, &self.delta)?;
            // Uniform update: flow[e] += delta for all edges
            for (&edge, flow_val) in flow_map.iter() {
                check_cancelled(ctx)?;
                validate_flow_value(flow_val, &self.flow)?;
                let new_flow = add_flow_values(flow_val, delta_scalar, &self.flow, &self.delta)?;
                result.insert(edge, new_flow);
            }
        }

        check_cancelled(ctx)?;
        scope
            .variables_mut()
            .set_edge_map(self.target.clone(), result);
        Ok(())
    }
}

/// Compute residual capacity for each edge given current flow and capacity.
///
/// For each edge (u, v) with capacity c[u,v] and flow f[u,v]:
/// - Forward residual: c[u,v] - f[u,v]  (remaining capacity)
/// - Backward residual: f[u,v]  (flow that can be pushed back)
///
/// # Parameters
/// - `capacity`: Edge capacity map (edge id → capacity)
/// - `flow`: Current flow map (edge id → flow value)
/// - `target`: Output variable for residual capacity map
///
/// # Output
/// Creates an edge map where each edge's value is the residual capacity.
/// Capacity keys produce positive forward residuals. Flow-only keys produce
/// positive backward residual amounts under their supplied IDs. This map-only
/// primitive does not create or traverse reverse edges.
pub struct ResidualCapacityStep {
    capacity: String,
    flow: String,
    target: String,
}

impl ResidualCapacityStep {
    pub fn new(
        capacity: impl Into<String>,
        flow: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            capacity: capacity.into(),
            flow: flow.into(),
            target: target.into(),
        }
    }
}

impl Step for ResidualCapacityStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.residual_capacity"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute residual capacity from capacity and flow".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for ResidualCapacityStep {}
impl ReadStep for ResidualCapacityStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("residual_capacity cancelled"));
        }

        let capacity_map = scope.variables().edge_map(&self.capacity)?;
        let flow_map = scope.variables().edge_map(&self.flow)?;

        let mut residual = HashMap::with_capacity(capacity_map.len());

        // Compute residual capacity: residual[e] = capacity[e] - flow[e]
        for (&edge, cap_val) in capacity_map.iter() {
            check_cancelled(ctx)?;
            let flow_val = flow_map.get(&edge).unwrap_or(&AlgorithmParamValue::Int(0));

            let residual_val = subtract_flow_values(cap_val, flow_val, &self.capacity, &self.flow)?;

            // Only include edges with positive residual capacity
            if is_positive(&residual_val)? {
                residual.insert(edge, residual_val);
            }
        }

        // For edges with flow but no explicit capacity entry,
        // backward residual equals the flow (can push back)
        for (&edge, flow_val) in flow_map.iter() {
            check_cancelled(ctx)?;
            validate_flow_value(flow_val, &self.flow)?;
            if !capacity_map.contains_key(&edge) && is_positive(flow_val)? {
                residual.insert(edge, flow_val.clone());
            }
        }

        check_cancelled(ctx)?;
        scope
            .variables_mut()
            .set_edge_map(self.target.clone(), residual);
        Ok(())
    }
}

// Reuse the expression evaluator's checked integer and finite f64 arithmetic.
fn check_cancelled(ctx: &Context) -> Result<()> {
    if ctx.is_cancelled() {
        return Err(anyhow!("flow computation cancelled"));
    }
    Ok(())
}

fn validate_flow_value(value: &AlgorithmParamValue, name: &str) -> Result<()> {
    if !matches!(
        value,
        AlgorithmParamValue::Int(_) | AlgorithmParamValue::Float(_)
    ) {
        return Err(anyhow!("flow value '{name}' must be numeric"));
    }
    super::expression::validate_value(value)
}

fn add_flow_values(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    validate_flow_value(lhs, lhs_name)?;
    validate_flow_value(rhs, rhs_name)?;
    super::expression::eval_binary_op(super::expression::BinaryOp::Add, lhs, rhs)
}

fn subtract_flow_values(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    validate_flow_value(lhs, lhs_name)?;
    validate_flow_value(rhs, rhs_name)?;
    super::expression::eval_binary_op(super::expression::BinaryOp::Sub, lhs, rhs)
}

fn is_positive(val: &AlgorithmParamValue) -> Result<bool> {
    validate_flow_value(val, "residual")?;
    match val {
        AlgorithmParamValue::Int(x) => Ok(*x > 0),
        AlgorithmParamValue::Float(x) => Ok(*x > 0.0),
        _ => Err(anyhow!("flow value must be numeric")),
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn test_add_flow_values() {
        let a = AlgorithmParamValue::Int(5);
        let b = AlgorithmParamValue::Int(3);
        let result = add_flow_values(&a, &b, "a", "b").unwrap();
        assert_eq!(result, AlgorithmParamValue::Int(8));

        let c = AlgorithmParamValue::Float(2.5);
        let d = AlgorithmParamValue::Int(3);
        let result = add_flow_values(&c, &d, "c", "d").unwrap();
        assert_eq!(result, AlgorithmParamValue::Float(5.5));
    }

    #[test]
    fn test_subtract_flow_values() {
        let a = AlgorithmParamValue::Int(10);
        let b = AlgorithmParamValue::Int(3);
        let result = subtract_flow_values(&a, &b, "a", "b").unwrap();
        assert_eq!(result, AlgorithmParamValue::Int(7));

        let c = AlgorithmParamValue::Float(5.5);
        let d = AlgorithmParamValue::Float(2.5);
        let result = subtract_flow_values(&c, &d, "c", "d").unwrap();
        assert_eq!(result, AlgorithmParamValue::Float(3.0));
    }

    #[test]
    fn test_is_positive() {
        assert!(is_positive(&AlgorithmParamValue::Int(5)).unwrap());
        assert!(!is_positive(&AlgorithmParamValue::Int(0)).unwrap());
        assert!(!is_positive(&AlgorithmParamValue::Int(-5)).unwrap());

        assert!(is_positive(&AlgorithmParamValue::Float(0.1)).unwrap());
        assert!(!is_positive(&AlgorithmParamValue::Float(0.0)).unwrap());
        assert!(!is_positive(&AlgorithmParamValue::Float(-1.5)).unwrap());
    }

    #[test]
    fn test_flow_arithmetic_type_promotion() {
        // Int + Float -> Float
        let int_val = AlgorithmParamValue::Int(10);
        let float_val = AlgorithmParamValue::Float(2.5);

        let result = add_flow_values(&int_val, &float_val, "i", "f").unwrap();
        match result {
            AlgorithmParamValue::Float(v) => assert!((v - 12.5).abs() < 1e-9),
            _ => panic!("Expected Float result"),
        }

        let result = subtract_flow_values(&float_val, &int_val, "f", "i").unwrap();
        match result {
            AlgorithmParamValue::Float(v) => assert!((v - (-7.5)).abs() < 1e-9),
            _ => panic!("Expected Float result"),
        }
    }

    #[test]
    fn test_is_positive_edge_cases() {
        // Zero is not positive
        assert!(!is_positive(&AlgorithmParamValue::Int(0)).unwrap());
        assert!(!is_positive(&AlgorithmParamValue::Float(0.0)).unwrap());

        // Very small positive value
        assert!(is_positive(&AlgorithmParamValue::Float(1e-10)).unwrap());

        // Negative values
        assert!(!is_positive(&AlgorithmParamValue::Int(-1)).unwrap());
        assert!(!is_positive(&AlgorithmParamValue::Float(-0.001)).unwrap());
    }
}

/// Copy/alias a variable to a new name.
///
/// This step is used within loops to create logical variable assignments
/// by copying the value of one variable to another. It's essential for
/// loop variable mapping where iteration N reads from variable X but
/// iteration N+1 needs to read from the updated value.
///
/// # Parameters
/// - `source`: Source variable name
/// - `target`: Target variable name (will be created or overwritten)
///
/// # Example
/// Inside a loop body:
/// ```text
/// ranks_new = compute_new_ranks(ranks)
/// alias(source=ranks_new, target=ranks)  # Copy ranks_new to ranks for next iteration
/// ```
pub struct AliasStep {
    source: String,
    target: String,
}

impl AliasStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

impl Step for AliasStep {
    fn atomic_effect(&self) -> super::core::AtomicStepEffect {
        super::core::AtomicStepEffect::ImmutableInputs
    }
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "alias"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Copy/alias variable to new name".to_string(),
            cost_hint: CostHint::Constant,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for AliasStep {}
impl ReadStep for AliasStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("alias cancelled"));
        }

        let value = scope
            .variables()
            .shared_value(&self.source)
            .ok_or_else(|| anyhow!("alias: source variable '{}' not found", self.source))?;
        if ctx.is_cancelled() {
            return Err(anyhow!("alias cancelled"));
        }
        scope.variables_mut().set_shared_value(&self.target, value);

        Ok(())
    }
}

#[cfg(test)]
mod checked_flow_tests {
    use super::super::core::StepVariables;
    use super::*;
    use crate::{ownership::OwnerCell, subgraphs::Subgraph, Graph};
    use std::{
        collections::HashSet,
        sync::{atomic::AtomicBool, Arc},
    };

    fn selection() -> Subgraph {
        Subgraph::new(
            Arc::new(OwnerCell::new(Graph::new_manual())),
            HashSet::new(),
            HashSet::new(),
            "flow values".into(),
        )
    }

    #[test]
    fn aliases_preserve_independent_map_and_column_mutation() {
        use super::super::core::NodeColumn;
        let selected = selection();
        let mut variables = StepVariables::default();
        variables.set_node_map("original", [(0, AlgorithmParamValue::Int(7))].into());
        variables.set_node_column(
            "column",
            NodeColumn::new(vec![0], vec![AlgorithmParamValue::Int(9)]),
        );
        let mut scope = StepScope::new(&selected, &mut variables);
        AliasStep::new("original", "copy")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        AliasStep::new("column", "column_copy")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        scope
            .variables_mut()
            .node_map_mut("copy")
            .unwrap()
            .insert(0, AlgorithmParamValue::Int(8));
        scope
            .variables_mut()
            .node_column_mut("column_copy")
            .unwrap()
            .values_mut()[0] = AlgorithmParamValue::Int(10);
        assert_eq!(
            scope.variables().node_map("original").unwrap()[&0],
            AlgorithmParamValue::Int(7)
        );
        assert_eq!(
            scope.variables().node_column("column").unwrap().get(0),
            Some(&AlgorithmParamValue::Int(9))
        );
        AliasStep::new("copy", "copy")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        assert_eq!(
            scope.variables().node_map("copy").unwrap()[&0],
            AlgorithmParamValue::Int(8)
        );
        assert!(AliasStep::new("missing", "copy")
            .apply(&mut Context::new(), &mut scope)
            .is_err());
        assert_eq!(
            scope.variables().node_map("copy").unwrap()[&0],
            AlgorithmParamValue::Int(8)
        );
    }

    #[test]
    fn flow_arithmetic_checks_overflow_and_nonfinite_values_without_rounding_integers() {
        use AlgorithmParamValue::{Float, Int};
        assert_eq!(
            add_flow_values(&Int((1i64 << 53) + 1), &Int(2), "f", "d").unwrap(),
            Int((1i64 << 53) + 3)
        );
        assert_eq!(
            subtract_flow_values(&Int((1i64 << 53) + 1), &Int(2), "c", "f").unwrap(),
            Int((1i64 << 53) - 1)
        );
        assert!(add_flow_values(&Int(i64::MAX), &Int(1), "f", "d").is_err());
        assert!(subtract_flow_values(&Int(i64::MIN), &Int(1), "c", "f").is_err());
        assert!(add_flow_values(&Float(f64::MAX), &Float(f64::MAX), "f", "d").is_err());
        assert!(subtract_flow_values(&Float(-f64::MAX), &Float(f64::MAX), "c", "f").is_err());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(add_flow_values(&Float(value), &Int(0), "f", "d").is_err());
            assert!(subtract_flow_values(&Int(0), &Float(value), "c", "f").is_err());
            assert!(is_positive(&Float(value)).is_err());
        }
        assert!(is_positive(&Float(f64::from_bits(1))).unwrap());
        assert!(!is_positive(&Float(-0.0)).unwrap());
    }

    #[test]
    fn invalid_passthrough_rows_and_empty_scalar_updates_preserve_prior_outputs() {
        use AlgorithmParamValue as V;
        let selected = selection();
        for invalid in [
            V::Text("bad".into()),
            V::Bool(true),
            V::None,
            V::Float(f64::NAN),
        ] {
            for branch in 0..4 {
                let mut vars = StepVariables::default();
                let old = HashMap::from([(8, V::Int(99))]);
                vars.set_edge_map("result", old.clone());
                vars.set_edge_map(
                    "flow",
                    if branch == 0 || branch == 3 {
                        HashMap::from([(7, invalid.clone())])
                    } else {
                        HashMap::new()
                    },
                );
                vars.set_edge_map(
                    "delta",
                    if branch == 1 {
                        HashMap::from([(7, invalid.clone())])
                    } else {
                        HashMap::new()
                    },
                );
                vars.set_scalar("scalar", invalid.clone());
                vars.set_edge_map("capacity", HashMap::new());
                let mut scope = StepScope::new(&selected, &mut vars);
                let result = match branch {
                    0 | 1 => FlowUpdateStep::new("flow", "delta", "result")
                        .apply(&mut Context::new(), &mut scope),
                    2 => FlowUpdateStep::new("flow", "scalar", "result")
                        .apply(&mut Context::new(), &mut scope),
                    _ => ResidualCapacityStep::new("capacity", "flow", "result")
                        .apply(&mut Context::new(), &mut scope),
                };
                assert!(result.is_err(), "branch {branch}");
                assert_eq!(scope.variables().edge_map("result").unwrap(), &old);
            }
        }
    }

    #[test]
    fn union_updates_and_positive_residuals_keep_sparse_edge_identity() {
        use AlgorithmParamValue as V;
        let selected = selection();
        let mut vars = StepVariables::default();
        vars.set_edge_map("flow", HashMap::from([(2, V::Int(3)), (7, V::Int(0))]));
        vars.set_edge_map("delta", HashMap::from([(2, V::Int(4)), (9, V::Float(1.5))]));
        vars.set_edge_map("capacity", HashMap::from([(2, V::Int(10)), (7, V::Int(0))]));
        let mut scope = StepScope::new(&selected, &mut vars);
        FlowUpdateStep::new("flow", "delta", "updated")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        assert_eq!(
            scope.variables().edge_map("updated").unwrap(),
            &HashMap::from([(2, V::Int(7)), (7, V::Int(0)), (9, V::Float(1.5))])
        );
        ResidualCapacityStep::new("capacity", "updated", "result")
            .apply(&mut Context::new(), &mut scope)
            .unwrap();
        assert_eq!(
            scope.variables().edge_map("result").unwrap(),
            &HashMap::from([(2, V::Int(3)), (9, V::Float(1.5))])
        );
        let original = scope.variables().edge_map("result").unwrap().clone();
        let mut cancelled = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(FlowUpdateStep::new("flow", "delta", "result")
            .apply(&mut cancelled, &mut scope)
            .is_err());
        assert!(ResidualCapacityStep::new("capacity", "updated", "result")
            .apply(&mut cancelled, &mut scope)
            .is_err());
        assert_eq!(scope.variables().edge_map("result").unwrap(), &original);
    }
}
