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
        use std::time::Instant;

        let start = Instant::now();

        let input_map = scope.variables().node_map(&self.source)?.clone();

        // STYLE_ALGO: Use ordered_nodes for determinism
        let nodes = scope.subgraph().ordered_nodes();
        ctx.record_stat("map_nodes_callback.count.nodes", nodes.len() as f64);

        let mut result = HashMap::with_capacity(nodes.len());

        // Prepare step input (reused)
        let reader = StepInput {
            subgraph: scope.subgraph(),
            variables: scope.variables(),
        };

        for &node in nodes.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("map_nodes cancelled"));
            }

            let value = if let Some(existing) = input_map.get(&node) {
                self.mapper
                    .map(node, &reader)
                    .unwrap_or_else(|_| existing.clone())
            } else {
                self.mapper.map(node, &reader)?
            };
            result.insert(node, value);
        }

        ctx.record_duration("map_nodes_callback.total", start.elapsed());

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
    /// If true, updates are written back to source immediately (async/in-place updates)
    async_update: bool,
}

impl MapNodesExprStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, expr: Expr) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            expr,
            async_update: false,
        }
    }
    
    pub fn with_async_update(mut self, async_update: bool) -> Self {
        self.async_update = async_update;
        self
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
        use std::time::Instant;

        let start = Instant::now();

        // STYLE_ALGO: Use ordered_nodes for determinism
        let nodes = scope.subgraph().ordered_nodes();
        ctx.record_stat("map_nodes.count.nodes", nodes.len() as f64);

        if self.async_update {
            // Async/in-place update mode: update the source map as we iterate so subsequent
            // nodes in the same pass observe the new values (matches native LPA semantics).
            for &node in nodes.iter() {
                if ctx.is_cancelled() {
                    return Err(anyhow!("map_nodes cancelled"));
                }

                // Evaluate expression against the current variable state.
                let new_value = {
                    let vars_ref = scope.variables();
                    let source_map = vars_ref.node_map(&self.source).map_err(|_| {
                        anyhow!(
                            "async_update requires source map '{}' to exist",
                            self.source
                        )
                    })?;
                    let current_value = source_map.get(&node);
                    let step_input = StepInput {
                        subgraph: scope.subgraph(),
                        variables: vars_ref,
                    };
                    let expr_ctx = if let Some(value) = current_value {
                        ExprContext::with_value(node, &step_input, value)
                    } else {
                        ExprContext::new(node, &step_input)
                    };
                    self.expr.eval(&expr_ctx)?
                };

                // Write the updated value back immediately.
                {
                    let vars_mut = scope.variables_mut();
                    let source_map = vars_mut.node_map_mut(&self.source).map_err(|_| {
                        anyhow!(
                            "async_update requires source map '{}' to exist",
                            self.source
                        )
                    })?;
                    source_map.insert(node, new_value);
                }
            }

            // Mirror the updated source map into the target variable for downstream steps.
            let updated_map = scope
                .variables()
                .node_map(&self.source)?
                .clone();
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), updated_map);

            ctx.record_duration("map_nodes.total", start.elapsed());
        } else {
            // Standard synchronous mode: collect all results then write at end
            
            // Try to get source map if it exists
            let source_map = scope.variables().node_map(&self.source).ok();

            // Pre-allocate result map
            let mut result = HashMap::with_capacity(nodes.len());

            // Prepare step input (reused across iterations)
            let step_input = StepInput {
                subgraph: scope.subgraph(),
                variables: scope.variables(),
            };

            // STYLE_ALGO: Iterate over ordered nodes (cache-friendly, deterministic)
            for &node in nodes.iter() {
                // Check cancellation periodically
                if ctx.is_cancelled() {
                    return Err(anyhow!("map_nodes cancelled"));
                }

                // Create expression context
                let expr_ctx = if let Some(ref map) = source_map {
                    if let Some(value) = map.get(&node) {
                        ExprContext::with_value(node, &step_input, value)
                    } else {
                        ExprContext::new(node, &step_input)
                    }
                } else {
                    ExprContext::new(node, &step_input)
                };

                // Evaluate expression
                let value = self.expr.eval(&expr_ctx)?;
                result.insert(node, value);
            }

            // Record timing
            ctx.record_duration("map_nodes.total", start.elapsed());

            scope
                .variables_mut()
                .set_node_map(self.target.clone(), result);
        }
        
        Ok(())
    }
}

/// Update a target map in-place with values from a source map.
///
/// This step supports "async" update semantics where each node's update
/// is immediately visible to subsequent nodes during traversal.
/// When `ordered` is true, nodes are processed in deterministic order
/// so later nodes see earlier updates - essential for Label Propagation.
///
/// # Parameters
/// - `source`: Map containing new values
/// - `target`: Map to update in place (gets mutated)
/// - `ordered`: If true, process nodes in sorted order for determinism
///
/// # Example
/// Used in LPA to propagate label updates:
/// ```ignore
/// // new_labels = mode(collect_neighbor_values(labels))
/// // labels = update_in_place(new_labels, target=labels, ordered=true)
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UpdateInPlaceStep {
    source: String,
    target: String,
    #[serde(default)]
    ordered: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output: Option<String>,
}

impl UpdateInPlaceStep {
    pub fn new(source: String, target: String, ordered: bool) -> Self {
        Self {
            source,
            target,
            ordered,
            output: None,
        }
    }
    
    pub fn with_output(mut self, output: String) -> Self {
        self.output = Some(output);
        self
    }
}

impl Step for UpdateInPlaceStep {
    fn id(&self) -> &'static str {
        "core.update_in_place"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Update target map in-place with source values".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let start = std::time::Instant::now();

        if ctx.is_cancelled() {
            return Err(anyhow!("core.update_in_place cancelled"));
        }

        // Get source values (the new values to apply) - clone to avoid borrow issues
        let source_map = scope.variables().node_map(&self.source)?.clone();

        // Determine node order
        let nodes: Vec<NodeId> = if self.ordered {
            // Sorted order for deterministic async updates
            let mut node_ids: Vec<_> = source_map.keys().copied().collect();
            node_ids.sort_unstable();
            node_ids
        } else {
            // Arbitrary order (still deterministic per HashMap iteration)
            source_map.keys().copied().collect()
        };

        // Get mutable access to target map and update in place
        let target_map = scope.variables_mut().node_map_mut(&self.target)?;
        for node_id in nodes {
            if let Some(new_value) = source_map.get(&node_id) {
                target_map.insert(node_id, new_value.clone());
            }
        }

        ctx.record_duration("update_in_place.total", start.elapsed());

        // If output is specified and different from target, clone to output
        if let Some(output_var) = &self.output {
            if output_var != &self.target {
                let final_map = scope.variables().node_map(&self.target)?.clone();
                scope.variables_mut().set_node_map(output_var.clone(), final_map);
            }
        }

        Ok(())
    }
}
