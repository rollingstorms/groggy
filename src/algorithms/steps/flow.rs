//! Network flow primitives for max-flow/min-cut algorithms.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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
                if !flow_map.contains_key(&edge) {
                    result.insert(edge, delta_val.clone());
                }
            }
        } else {
            // Try as scalar
            let delta_scalar = scope.variables().scalar(&self.delta)?;
            // Uniform update: flow[e] += delta for all edges
            for (&edge, flow_val) in flow_map.iter() {
                let new_flow = add_flow_values(flow_val, delta_scalar, &self.flow, &self.delta)?;
                result.insert(edge, new_flow);
            }
        }

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
/// For undirected graphs or when backward edges exist, this handles both
/// forward and reverse directions.
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("residual_capacity cancelled"));
        }

        let capacity_map = scope.variables().edge_map(&self.capacity)?;
        let flow_map = scope.variables().edge_map(&self.flow)?;

        let mut residual = HashMap::with_capacity(capacity_map.len());

        // Compute residual capacity: residual[e] = capacity[e] - flow[e]
        for (&edge, cap_val) in capacity_map.iter() {
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
            if !capacity_map.contains_key(&edge) && is_positive(flow_val)? {
                residual.insert(edge, flow_val.clone());
            }
        }

        scope
            .variables_mut()
            .set_edge_map(self.target.clone(), residual);
        Ok(())
    }
}

// Helper functions for flow arithmetic

fn add_flow_values(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    match (lhs, rhs) {
        (AlgorithmParamValue::Int(a), AlgorithmParamValue::Int(b)) => {
            Ok(AlgorithmParamValue::Int(a + b))
        }
        (AlgorithmParamValue::Float(a), AlgorithmParamValue::Float(b)) => {
            Ok(AlgorithmParamValue::Float(a + b))
        }
        (AlgorithmParamValue::Int(a), AlgorithmParamValue::Float(b)) => {
            Ok(AlgorithmParamValue::Float(*a as f64 + b))
        }
        (AlgorithmParamValue::Float(a), AlgorithmParamValue::Int(b)) => {
            Ok(AlgorithmParamValue::Float(a + *b as f64))
        }
        _ => Err(anyhow!(
            "cannot add flow values: '{}' and '{}' have incompatible types",
            lhs_name,
            rhs_name
        )),
    }
}

fn subtract_flow_values(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    match (lhs, rhs) {
        (AlgorithmParamValue::Int(a), AlgorithmParamValue::Int(b)) => {
            Ok(AlgorithmParamValue::Int(a - b))
        }
        (AlgorithmParamValue::Float(a), AlgorithmParamValue::Float(b)) => {
            Ok(AlgorithmParamValue::Float(a - b))
        }
        (AlgorithmParamValue::Int(a), AlgorithmParamValue::Float(b)) => {
            Ok(AlgorithmParamValue::Float(*a as f64 - b))
        }
        (AlgorithmParamValue::Float(a), AlgorithmParamValue::Int(b)) => {
            Ok(AlgorithmParamValue::Float(a - *b as f64))
        }
        _ => Err(anyhow!(
            "cannot subtract flow values: '{}' and '{}' have incompatible types",
            lhs_name,
            rhs_name
        )),
    }
}

fn is_positive(val: &AlgorithmParamValue) -> Result<bool> {
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("alias cancelled"));
        }

        // Get the source value (try all possible types)
        let value = {
            let vars = scope.variables();

            // Try each type in order
            if let Ok(map) = vars.node_map(&self.source) {
                super::core::StepValue::NodeMap(map.clone())
            } else if let Ok(col) = vars.node_column(&self.source) {
                super::core::StepValue::NodeColumn(col.clone())
            } else if let Ok(map) = vars.edge_map(&self.source) {
                super::core::StepValue::EdgeMap(map.clone())
            } else if let Ok(val) = vars.scalar(&self.source) {
                super::core::StepValue::Scalar(val.clone())
            } else {
                return Err(anyhow!(
                    "alias: source variable '{}' not found or has unsupported type",
                    self.source
                ));
            }
        };

        // Write to target using the appropriate setter
        let vars_mut = scope.variables_mut();
        match value {
            super::core::StepValue::NodeMap(map) => vars_mut.set_node_map(&self.target, map),
            super::core::StepValue::NodeColumn(col) => vars_mut.set_node_column(&self.target, col),
            super::core::StepValue::EdgeMap(map) => vars_mut.set_edge_map(&self.target, map),
            super::core::StepValue::Scalar(val) => vars_mut.set_scalar(&self.target, val),
            _ => {
                return Err(anyhow!(
                    "alias: unsupported value type for '{}'",
                    self.source
                ))
            }
        }

        Ok(())
    }
}
