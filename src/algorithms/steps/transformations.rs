//! Transformation step primitives that map values.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepInput, StepMetadata, StepScope};
use super::expression::{Expr, ExprContext};

/// Callback trait for node-wise mapping steps.
pub trait NodeMapFn: Send + Sync {
    fn map(&self, node: NodeId, input: &StepInput<'_>) -> Result<AlgorithmParamValue>;
}

impl<F> NodeMapFn for F
where
    F: for<'a> Fn(NodeId, &StepInput<'a>) -> Result<AlgorithmParamValue> + Send + Sync,
{
    fn map(&self, node: NodeId, input: &StepInput<'_>) -> Result<AlgorithmParamValue> {
        (self)(node, input)
    }
}

/// Transform node values using a callback.
///
/// **Note**: This step is designed for **programmatic Rust use only**.
/// It cannot be registered in the step registry because it requires an
/// `Arc<dyn NodeMapFn>` callback, which cannot be serialized in a `StepSpec`.
///
/// For declarative pipeline building, use arithmetic operations
/// (`core.add`, `core.mul`, etc.) or create custom steps with specific logic.
///
/// # Example (Rust API)
/// ```ignore
/// let mapper = Arc::new(|node: NodeId, input: &StepInput| {
///     // Custom transformation logic
///     Ok(AlgorithmParamValue::Float(node.0 as f64))
/// });
/// let step = MapNodesStep::new("input", "output", mapper);
/// ```
pub struct MapNodesStep {
    source: String,
    target: String,
    mapper: Arc<dyn NodeMapFn>,
}

impl MapNodesStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        mapper: Arc<dyn NodeMapFn>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            mapper,
        }
    }
}

impl Step for MapNodesStep {
    fn id(&self) -> &'static str {
        "core.map_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Map node values using a callback".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let input_map = scope.variables().node_map(&self.source)?.clone();
        let mut result = HashMap::with_capacity(input_map.len());
        let nodes: Vec<NodeId> = scope.subgraph().nodes().iter().copied().collect();

        for node in nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("map_nodes cancelled"));
            }
            let reader = StepInput {
                subgraph: scope.subgraph(),
                variables: scope.variables(),
            };
            let value = if let Some(existing) = input_map.get(&node) {
                self.mapper
                    .map(node, &reader)
                    .unwrap_or_else(|_| existing.clone())
            } else {
                self.mapper.map(node, &reader)?
            };
            result.insert(node, value);
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

/// Transform node values using expressions (serializable version).
///
/// This step uses the expression system to apply transformations
/// that can be serialized in `StepSpec` and used from Python/JSON.
///
/// # Example
/// ```ignore
/// // Double all values
/// MapNodesExprStep::new(
///     "input",
///     "output",
///     Expr::binary(BinaryOp::Mul, Expr::var("value"), Expr::constant(2.0.into()))
/// )
/// ```
pub struct MapNodesExprStep {
    source: String,
    target: String,
    expr: Expr,
}

impl MapNodesExprStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, expr: Expr) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            expr,
        }
    }
}

impl Step for MapNodesExprStep {
    fn id(&self) -> &'static str {
        "core.map_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Map node values using expression".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        // Try to get source map if it exists, otherwise compute for all nodes
        let source_map = scope.variables().node_map(&self.source).ok();
        let mut result = HashMap::new();
        let nodes: Vec<NodeId> = scope.subgraph().nodes().iter().copied().collect();

        for node in nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("map_nodes cancelled"));
            }

            let step_input = StepInput {
                subgraph: scope.subgraph(),
                variables: scope.variables(),
            };

            let expr_ctx = if let Some(ref map) = source_map {
                if let Some(value) = map.get(&node) {
                    ExprContext::with_value(node, &step_input, value)
                } else {
                    ExprContext::new(node, &step_input)
                }
            } else {
                ExprContext::new(node, &step_input)
            };

            let value = self.expr.eval(&expr_ctx)?;
            result.insert(node, value);
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}
