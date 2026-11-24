use super::super::{AlgorithmParamValue, Context};
use super::checked_batch::LoweredBatchPlan;
use super::core::{loop_effect, AtomicStepEffect, Step, StepExecutor, StepRegistry, StepScope};
use super::read_scope::{sealed, ReadStep, ReadStepScope};
use crate::algorithms::execution::{BatchExecutor, BatchPlan};
use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;

use serde_json::Value;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

// JIT disabled temporarily due to Cranelift ARM64 compatibility issues
// #[cfg(not(target_arch = "aarch64"))]
// use crate::algorithms::execution::JitManager;
// #[cfg(not(target_arch = "aarch64"))]
// use std::sync::Mutex;

/// Execution statistics for hot-path detection
#[derive(Debug, Default)]
struct LoopExecutionStats {
    /// Number of times this loop has been executed
    execution_count: AtomicU64,
    /// Total iterations executed across all runs
    total_iterations: AtomicU64,
    #[allow(dead_code)]
    /// Whether JIT compilation has been attempted
    jit_attempted: AtomicU64,
}

/// Loop execution step that executes a sequence of steps multiple times.
///
/// This replaces loop unrolling with native iteration, executing the entire
/// loop body in Rust without crossing the FFI boundary for each step.
///
/// Ordinary bodies use cached native steps. Supported numeric batch plans lower
/// to checked kernels; unported plans retain only non-atomic compatibility.
/// JIT execution is disabled on every platform.
pub struct LoopStep {
    /// Number of iterations
    iterations: usize,

    /// Serialized body to avoid eager deserialization / circular registry deps
    body_json: Value,
    body_cache: OnceCell<Vec<Box<dyn Step>>>,
    lowered_cache: OnceCell<Option<LoweredBatchPlan>>,

    /// Optional: variable initialization for loops
    /// Format: Vec<(initial_var, loop_var)>
    /// Example: [("nodes_0", "ranks")] means: before loop, create alias ranks = nodes_0
    loop_vars: Option<Vec<(String, String)>>,

    registry: &'static StepRegistry,
    /// Execution statistics for hot-path detection
    stats: LoopExecutionStats,
    // Cached JIT compiled function (Tier 2) - disabled temporarily
    // Mutex for interior mutability (compilation happens on first hot execution)
    // #[cfg(not(target_arch = "aarch64"))]
    // jit_cache: Mutex<Option<JitManager>>,
}

impl LoopStep {
    /// Hot path threshold - after this many executions, try JIT compilation
    #[allow(dead_code)]
    const HOT_THRESHOLD: u64 = 100;

    /// Minimum iterations to justify JIT compilation overhead
    #[allow(dead_code)]
    const MIN_ITERATIONS_FOR_JIT: usize = 10;

    /// Create a new loop step
    pub fn new(iterations: usize, body_json: Value, registry: &'static StepRegistry) -> Self {
        Self {
            iterations,
            body_json,
            body_cache: OnceCell::new(),
            lowered_cache: OnceCell::new(),
            loop_vars: None,
            registry,
            stats: LoopExecutionStats::default(),
            // #[cfg(not(target_arch = "aarch64"))]
            // jit_cache: Mutex::new(None),
        }
    }

    /// Create a loop step with variable initialization
    pub fn with_loop_vars(
        iterations: usize,
        body_json: Value,
        loop_vars: Vec<(String, String)>,
        registry: &'static StepRegistry,
    ) -> Self {
        Self {
            iterations,
            body_json,
            body_cache: OnceCell::new(),
            lowered_cache: OnceCell::new(),
            loop_vars: Some(loop_vars),
            registry,
            stats: LoopExecutionStats::default(),
            // #[cfg(not(target_arch = "aarch64"))]
            // jit_cache: Mutex::new(None),
        }
    }

    // JIT compilation disabled temporarily
    // /// Check if this loop should attempt JIT compilation
    // #[allow(dead_code)]
    // #[cfg(not(target_arch = "aarch64"))]
    // fn should_attempt_jit(&self, batch_plan: &BatchPlan) -> bool {
    //     false // JIT disabled
    // }

    // /// Try to compile the batch plan to native code (Tier 2)
    // #[cfg(not(target_arch = "aarch64"))]
    // fn try_jit_compile(
    //     &self,
    //     batch_plan: &BatchPlan,
    // ) -> Option<crate::algorithms::execution::jit::CompiledFunction> {
    //     None // JIT disabled
    // }

    // #[cfg(target_arch = "aarch64")]
    // #[allow(dead_code)]
    // fn try_jit_compile(&self, _batch_plan: &BatchPlan) -> Option<()> {
    //     None // ARM64 not supported
    // }

    /// A supplied malformed plan is never replaced silently by another program.
    fn batch_plan(&self) -> Result<Option<BatchPlan>> {
        let Some(wrapped) = self.body_json.get("batch_plan") else {
            return Ok(None);
        };
        let value = wrapped.get("value").unwrap_or(wrapped);
        let plan: BatchPlan = serde_json::from_value(value.clone())
            .map_err(|e| anyhow!("iter.loop invalid batch_plan: {e}"))?;
        plan.validate()?;
        Ok(Some(plan))
    }

    fn lowered_batch(&self) -> Result<Option<&LoweredBatchPlan>> {
        self.lowered_cache
            .get_or_try_init(|| {
                self.batch_plan()?
                    .map(|plan| LoweredBatchPlan::compile(&plan))
                    .transpose()
                    .map(Option::flatten)
            })
            .map(Option::as_ref)
    }

    fn body_steps(&self) -> Result<&Vec<Box<dyn Step>>> {
        self.body_cache.get_or_try_init(|| {
            // The body JSON might be wrapped in AlgorithmParamValue format: {"type": "Json", "value": [...]}
            // Try to unwrap it first
            let body_array = if let Some(wrapped_value) = self.body_json.get("value") {
                // It's wrapped, extract the value
                wrapped_value
                    .as_array()
                    .ok_or_else(|| anyhow!("iter.loop 'body.value' must be an array"))?
            } else {
                // Not wrapped, use directly
                self.body_json
                    .as_array()
                    .ok_or_else(|| anyhow!("iter.loop 'body' must be an array"))?
            };

            let specs = super::registry::parse_body_steps(&AlgorithmParamValue::Json(
                Value::Array(body_array.clone()),
            ))?;
            let mut steps = Vec::with_capacity(specs.len());
            for spec in specs {
                steps.push(self.registry.instantiate(&spec)?);
            }

            Ok(steps)
        })
    }
}

impl Step for LoopStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        if self.body_json.get("batch_plan").is_some() {
            return if self.lowered_batch().is_ok_and(|plan| plan.is_some()) {
                AtomicStepEffect::PendingProperties
            } else {
                AtomicStepEffect::Unsupported
            };
        }
        self.body_steps()
            .map_or(AtomicStepEffect::Unsupported, |steps| loop_effect(steps))
    }

    fn supports_atomic_properties(&self) -> bool {
        if self.body_json.get("batch_plan").is_some() {
            return self.lowered_batch().is_ok_and(|plan| plan.is_some());
        }
        self.body_steps()
            .is_ok_and(|steps| steps.iter().all(|step| step.supports_atomic_properties()))
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        self.lowered_batch()
            .ok()
            .flatten()
            .map(|_| self as &dyn ReadStep)
    }

    fn id(&self) -> &'static str {
        "iter.loop"
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if self.lowered_batch()?.is_some() {
            return self.apply_read(ctx, &mut scope.read_scope());
        }
        let legacy_plan = self.batch_plan()?;
        let body_steps = self.body_steps()?;
        if ctx.property_batch.as_ref().is_some_and(|batch| {
            let effect = self.atomic_effect();
            effect == AtomicStepEffect::Unsupported
                || (batch.has_topology() && effect == AtomicStepEffect::Properties)
        }) {
            return Err(anyhow!(
                "iter.loop does not support this atomic property body"
            ));
        }

        // Before first iteration, create initial variable bindings
        // loop_vars format: [(initial_var, loop_var), ...]
        // e.g., [("nodes_0", "ranks")] means: create alias ranks = nodes_0
        if let Some(loop_vars) = &self.loop_vars {
            for (initial_var, loop_var) in loop_vars {
                let value = scope
                    .variables()
                    .value(initial_var)
                    .cloned()
                    .ok_or_else(|| {
                        anyhow!("Loop variable initialization: '{}' not found", initial_var)
                    })?;
                scope.variables_mut().set_value(loop_var, value);
            }
        }

        // Update statistics
        let _exec_count = self.stats.execution_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.stats
            .total_iterations
            .fetch_add(self.iterations as u64, Ordering::Relaxed);

        // Try tiered execution (if batch plan available)
        if let Some(batch_plan) = legacy_plan {
            let node_count = scope.subgraph().node_count();

            // Tier 2: JIT compilation disabled temporarily
            // #[cfg(not(target_arch = "aarch64"))]
            // {
            //     if let Some(jit_func) = self.try_jit_compile(&batch_plan) {
            //         ...JIT execution code...
            //     }
            // }

            // Tier 1: Batch executor (interpreted execution)
            let mut executor = BatchExecutor::new(node_count);

            return executor
                .execute_with_context(&batch_plan, self.iterations, ctx, scope)
                .map_err(|e| anyhow!("Batch execution failed: {}", e));
        }

        // Each nested frame refreshes reads after staged writes while sharing
        // variables and the owning batch. Certified root topology uses this same path.
        let base = scope.subgraph().clone();
        let mut executor = StepExecutor::for_frame(&base, scope.fixed_selection());
        for iteration in 0..self.iterations {
            if ctx.is_cancelled() {
                return Err(anyhow!("Loop cancelled before iteration {}", iteration + 1));
            }
            // All descendants share the outer run's staging context.
            for (step_idx, step) in body_steps.iter().enumerate() {
                if ctx.is_cancelled() {
                    return Err(anyhow!("Loop cancelled before step {}", step_idx + 1));
                }
                executor
                    .apply(ctx, scope.variables_mut(), step.as_ref())
                    .map_err(|e| {
                        anyhow!(
                            "Loop iteration {}/{}, step {}/{} ({}) failed: {}",
                            iteration + 1,
                            self.iterations,
                            step_idx + 1,
                            body_steps.len(),
                            step.id(),
                            e
                        )
                    })?;
            }
        }

        Ok(())
    }
}

impl sealed::Sealed for LoopStep {}
impl ReadStep for LoopStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("iter.loop cancelled"));
        }
        let plan = self
            .lowered_cache
            .get()
            .and_then(Option::as_ref)
            .ok_or_else(|| anyhow!("iter.loop has no prepared native batch program"))?;
        if let Some(bindings) = &self.loop_vars {
            for (initial, logical) in bindings {
                let value = scope.variables().value(initial).cloned().ok_or_else(|| {
                    anyhow!("Loop variable initialization: '{initial}' not found")
                })?;
                scope.variables_mut().set_value(logical, value);
            }
        }
        self.stats.execution_count.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_iterations
            .fetch_add(self.iterations as u64, Ordering::Relaxed);
        plan.execute(ctx, scope, self.iterations).map(|_| ())
    }
}

/// Deserialize a loop step from a JSON spec.
///
/// Expected format:
/// ```json
/// {
///   "type": "iter.loop",
///   "iterations": 100,
///   "body": [
///     { "type": "core.add", ... },
///     { "type": "core.mul", ... },
///     ...
///   ],
///   "loop_vars": ["ranks", "contrib"]  // optional
/// }
/// ```
#[allow(dead_code)]
pub fn deserialize_loop_step(
    spec: &Value,
    registry: &'static StepRegistry,
) -> Result<Box<dyn Step>> {
    // Extract iterations
    let iterations = usize::try_from(
        spec["iterations"]
            .as_u64()
            .ok_or_else(|| anyhow!("iter.loop requires non-negative 'iterations' field"))?,
    )
    .map_err(|_| anyhow!("iter.loop iterations must fit usize"))?;

    // Capture raw body JSON for lazy instantiation
    let body_json = spec
        .get("body")
        .cloned()
        .unwrap_or_else(|| Value::Array(vec![]));

    // Optional: extract loop vars (format: [["initial", "logical"], ...])
    let loop_vars = spec.get("loop_vars").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| {
                // Each element can be either a string (old format) or [initial, logical] pair
                if let Some(pair) = v.as_array() {
                    if pair.len() == 2 {
                        if let (Some(initial), Some(logical)) = (pair[0].as_str(), pair[1].as_str())
                        {
                            return Some((initial.to_string(), logical.to_string()));
                        }
                    }
                }
                None
            })
            .collect::<Vec<_>>()
    });

    // Create the loop step
    let loop_step = if let Some(vars) = loop_vars {
        LoopStep::with_loop_vars(iterations, body_json, vars, registry)
    } else {
        LoopStep::new(iterations, body_json, registry)
    };

    Ok(Box::new(loop_step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::steps::core::global_step_registry;

    #[test]
    fn atomic_capability_checks_descendants_and_alternate_batch_programs() {
        crate::algorithms::steps::ensure_core_steps_registered();
        let registry = global_step_registry();
        let supported = serde_json::json!([
            {"id": "iter.loop", "params": {"iterations": 0, "body": [
                {"id": "core.init_nodes", "params": {"target": "x", "value": 1}}
            ]}}
        ]);
        assert!(LoopStep::new(2, supported, registry).supports_atomic_properties());
        for body in [
            serde_json::json!([{"id": "core.execution_block", "params": {"mode": "streaming", "target": "unsupported", "options": {}, "body": {"nodes": []}}}]),
            serde_json::json!([{"params": {}}]),
            serde_json::json!({"value": [], "batch_plan": {}}),
        ] {
            assert!(!LoopStep::new(0, body, registry).supports_atomic_properties());
        }
    }

    #[test]
    fn cancelled_loop_checks_even_an_empty_body() {
        use crate::ownership::OwnerCell;
        use crate::{api::graph::Graph, subgraphs::Subgraph};
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let graph = Arc::new(OwnerCell::new(Graph::new_publishing()));
        let subgraph = Subgraph::from_nodes(graph, Default::default(), "cancel".into())
            .expect("empty selection");
        let mut variables = super::super::core::StepVariables::default();
        let mut scope = StepScope::new(&subgraph, &mut variables);
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        let step = LoopStep::new(2, serde_json::json!([]), global_step_registry());
        let error = step.apply(&mut ctx, &mut scope).expect_err("cancelled");
        assert!(error.to_string().contains("cancelled before iteration 1"));
    }

    #[test]
    fn malformed_supplied_plans_never_fall_back_even_for_zero_iterations() {
        use crate::{subgraphs::Subgraph, Graph};
        let owner = Graph::new().into_shared();
        let selected =
            Subgraph::from_nodes(owner, Default::default(), "invalid".into()).expect("selection");
        for plan in [
            serde_json::json!({}),
            serde_json::json!({"instructions":[{"type":"load_scalar","dst":1,"value":1.}],"slot_count":1,"carried_slots":[]}),
            serde_json::json!({"instructions":[],"slot_count":2,"carried_slots":[[0,1],[1,1]]}),
            serde_json::json!({"instructions":[{"type":"clip","dst":0,"src":0,"min":2.,"max":1.}],"slot_count":1,"carried_slots":[]}),
        ] {
            let step = LoopStep::new(
                0,
                serde_json::json!({"value":[],"batch_plan":plan}),
                global_step_registry(),
            );
            assert!(!step.supports_atomic_properties());
            let mut variables = super::super::core::StepVariables::default();
            let error = step.apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut variables),
            );
            assert!(error.is_err(), "malformed plan silently fell back");
        }
        let plan_only = LoopStep::new(
            0,
            serde_json::json!({"batch_plan":{"instructions":[],"slot_count":0,"carried_slots":[]}}),
            global_step_registry(),
        );
        assert!(plan_only.supports_atomic_properties());
        plan_only
            .apply(
                &mut Context::new(),
                &mut StepScope::new(&selected, &mut Default::default()),
            )
            .expect("plan-only zero iteration");
    }

    #[test]
    fn test_loop_step_creation() {
        let loop_step = LoopStep::new(10, Value::Array(vec![]), global_step_registry());
        assert_eq!(loop_step.iterations, 10);
        assert!(loop_step.body_steps().unwrap().is_empty());
    }

    #[test]
    fn test_loop_step_with_vars() {
        let loop_step = LoopStep::with_loop_vars(
            10,
            Value::Array(vec![]),
            vec![("nodes_0".to_string(), "ranks".to_string())],
            global_step_registry(),
        );
        assert_eq!(loop_step.iterations, 10);
        assert!(loop_step.loop_vars.is_some());
        assert_eq!(loop_step.loop_vars.as_ref().unwrap().len(), 1);
    }
}
