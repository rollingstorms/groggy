use super::super::Context;
use super::core::{Step, StepRegistry, StepScope, StepSpec};
use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use serde_json::Value;

/// Loop execution step that executes a sequence of steps multiple times.
///
/// This replaces loop unrolling with native iteration, executing the entire
/// loop body in Rust without crossing the FFI boundary for each step.
pub struct LoopStep {
    /// Number of iterations
    iterations: usize,

    /// Serialized body to avoid eager deserialization / circular registry deps
    body_json: Value,
    body_cache: OnceCell<Vec<Box<dyn Step>>>,

    /// Optional: track which variables are modified in the loop
    /// (for future optimizations)
    #[allow(dead_code)]
    loop_vars: Option<Vec<String>>,

    registry: &'static StepRegistry,
}

impl LoopStep {
    /// Create a new loop step
    pub fn new(iterations: usize, body_json: Value, registry: &'static StepRegistry) -> Self {
        Self {
            iterations,
            body_json,
            body_cache: OnceCell::new(),
            loop_vars: None,
            registry,
        }
    }

    /// Create a loop step with tracked variables
    pub fn with_loop_vars(
        iterations: usize,
        body_json: Value,
        loop_vars: Vec<String>,
        registry: &'static StepRegistry,
    ) -> Self {
        Self {
            iterations,
            body_json,
            body_cache: OnceCell::new(),
            loop_vars: Some(loop_vars),
            registry,
        }
    }

    fn body_steps(&self) -> Result<&Vec<Box<dyn Step>>> {
        self.body_cache.get_or_try_init(|| {
            let body_array = self
                .body_json
                .as_array()
                .ok_or_else(|| anyhow!("iter.loop 'body' must be an array"))?;

            let mut steps = Vec::with_capacity(body_array.len());
            for (idx, step_value) in body_array.iter().enumerate() {
                let spec: StepSpec = serde_json::from_value(step_value.clone()).map_err(|e| {
                    anyhow!(
                        "iter.loop body[{}] failed to deserialize StepSpec: {}",
                        idx,
                        e
                    )
                })?;
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

        // Execute the loop body for the specified number of iterations
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

    // Optional: extract loop vars
    let loop_vars = spec["loop_vars"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
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
            vec!["ranks".to_string(), "contrib".to_string()],
            global_step_registry(),
        );
        assert_eq!(loop_step.iterations, 10);
        assert!(loop_step.loop_vars.is_some());
        assert_eq!(loop_step.loop_vars.as_ref().unwrap().len(), 2);
    }
}
