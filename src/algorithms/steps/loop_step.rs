use super::super::{AlgorithmParams, Context};
use super::core::{Step, StepRegistry, StepScope, StepSpec};
use crate::algorithms::execution::{BatchExecutor, BatchPlan};
use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;

use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

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
/// ## Execution Tiers
/// 1. **Step-by-step**: Initial execution, collects statistics
/// 2. **Batch Executor (Tier 1)**: Interpreted batch plan execution
/// 3. **JIT (Tier 2)**: Native machine code (hot paths only, x86_64 only)
pub struct LoopStep {
    /// Number of iterations
    iterations: usize,

    /// Serialized body to avoid eager deserialization / circular registry deps
    body_json: Value,
    body_cache: OnceCell<Vec<Box<dyn Step>>>,

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

    /// Try to deserialize body as a BatchPlan (if it's been pre-compiled)
    fn try_batch_plan(&self) -> Option<BatchPlan> {
        // The batch_plan is at the top level of body_json
        let bp_wrapped = self.body_json.get("batch_plan")?;

        // It might be wrapped in AlgorithmParamValue format: {"type": "Json", "value": {...}}
        let bp_val = if let Some(inner) = bp_wrapped.get("value") {
            inner // Unwrap
        } else {
            bp_wrapped // Already unwrapped
        };

        match serde_json::from_value::<BatchPlan>(bp_val.clone()) {
            Ok(plan) => {
                if std::env::var("GROGGY_DEBUG_BATCH").is_ok() {
                    eprintln!("[LOOP_STEP] ✅ Successfully deserialized BatchPlan!");
                    eprintln!(
                        "[LOOP_STEP]    {} instructions, {} slots",
                        plan.instructions.len(),
                        plan.slot_count
                    );
                }
                Some(plan)
            }
            Err(e) => {
                if std::env::var("GROGGY_DEBUG_BATCH").is_ok() {
                    eprintln!("[LOOP_STEP] ⚠️  BatchPlan deserialization failed: {}", e);
                }
                None
            }
        }
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

            let mut steps = Vec::with_capacity(body_array.len());
            for (idx, step_value) in body_array.iter().enumerate() {
                // The step is in StepSpec format, but params are raw JSON values
                // We need to manually construct the StepSpec
                let step_id = step_value["id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("iter.loop body[{}] missing 'id' field", idx))?;

                // Extract params as a JSON object
                let params_obj = step_value
                    .get("params")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| anyhow!("iter.loop body[{}] has invalid 'params'", idx))?;

                // Convert params from raw JSON to AlgorithmParams
                let params = AlgorithmParams::from_json_map(params_obj)?;

                let spec = StepSpec {
                    id: step_id.to_string(),
                    params,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                };

                let step = self.registry.instantiate(&spec)?;
                steps.push(step);
            }
            Ok(steps)
        })
    }
}

impl Step for LoopStep {
    fn id(&self) -> &'static str {
        "iter.loop"
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let body_steps = self.body_steps()?;

        // Before first iteration, create initial variable bindings
        // loop_vars format: [(initial_var, loop_var), ...]
        // e.g., [("nodes_0", "ranks")] means: create alias ranks = nodes_0
        if let Some(loop_vars) = &self.loop_vars {
            for (initial_var, loop_var) in loop_vars {
                // Use AliasStep logic to copy the variable
                let value = {
                    let vars = scope.variables();

                    if let Ok(map) = vars.node_map(initial_var) {
                        super::core::StepValue::NodeMap(map.clone())
                    } else if let Ok(col) = vars.node_column(initial_var) {
                        super::core::StepValue::NodeColumn(col.clone())
                    } else if let Ok(map) = vars.edge_map(initial_var) {
                        super::core::StepValue::EdgeMap(map.clone())
                    } else if let Ok(val) = vars.scalar(initial_var) {
                        super::core::StepValue::Scalar(val.clone())
                    } else {
                        return Err(anyhow!(
                            "Loop variable initialization: '{}' not found",
                            initial_var
                        ));
                    }
                };

                let vars_mut = scope.variables_mut();
                match value {
                    super::core::StepValue::NodeMap(map) => vars_mut.set_node_map(loop_var, map),
                    super::core::StepValue::NodeColumn(col) => {
                        vars_mut.set_node_column(loop_var, col)
                    }
                    super::core::StepValue::EdgeMap(map) => vars_mut.set_edge_map(loop_var, map),
                    super::core::StepValue::Scalar(val) => vars_mut.set_scalar(loop_var, val),
                    _ => {
                        return Err(anyhow!(
                            "Loop variable '{}' has unsupported type",
                            initial_var
                        ))
                    }
                }
            }
        }

        // Update statistics
        let _exec_count = self.stats.execution_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.stats
            .total_iterations
            .fetch_add(self.iterations as u64, Ordering::Relaxed);

        // Try tiered execution (if batch plan available)
        if let Some(batch_plan) = self.try_batch_plan() {
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
                .execute(&batch_plan, self.iterations, scope)
                .map_err(|e| anyhow!("Batch execution failed: {}", e));
        }

        // Fallback: Execute the loop body for the specified number of iterations
        for iteration in 0..self.iterations {
            // Execute each step in the body
            for (step_idx, step) in body_steps.iter().enumerate() {
                step.apply(ctx, scope).map_err(|e| {
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
    let iterations = spec["iterations"]
        .as_u64()
        .ok_or_else(|| anyhow!("iter.loop requires 'iterations' field"))?
        as usize;

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
