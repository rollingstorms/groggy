use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{anyhow, Result};

use serde::Deserialize;

static EXECUTION_COUNTER: AtomicU64 = AtomicU64::new(0);

use crate::subgraphs::Subgraph;

use super::pipeline::AlgorithmSpec;
use super::registry::Registry;
use super::steps::{
    ensure_core_steps_registered, global_step_registry, PipelineValidator, SchemaRegistry, Step,
    StepScope, StepSpec, StepVariables, ValidationReport,
};
use super::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, AlgorithmParams, Context, CostHint,
};

/// Register builder-related algorithms (custom step pipelines).
pub fn register_algorithms(registry: &Registry) -> Result<()> {
    registry.register_factory("builder.step_pipeline", |spec| {
        let algorithm = StepPipelineAlgorithm::try_from_spec(spec)?;
        Ok(Box::new(algorithm) as Box<dyn Algorithm>)
    })
}

/// Get the global schema registry if schemas have been registered.
/// Returns None if schemas haven't been initialized yet.
fn get_schema_registry() -> Option<SchemaRegistry> {
    // For now, schemas are optional. In the future, this could return
    // a lazily-initialized global registry similar to the step registry.
    None
}

/// Validate a pipeline without executing it.
/// Returns a validation report with errors and warnings.
pub fn validate_pipeline(steps: &[StepSpec]) -> ValidationReport {
    let schema_registry = get_schema_registry().unwrap_or_default();
    let validator = PipelineValidator::new(&schema_registry);
    validator.validate(steps)
}

struct StepPipelineAlgorithm {
    display_name: String,
    steps: Vec<Box<dyn Step>>,
}

impl StepPipelineAlgorithm {
    fn try_from_spec(spec: &AlgorithmSpec) -> Result<Self> {
        ensure_core_steps_registered();

        let definition = extract_definition(spec)?;

        // Build step specs for validation and instantiation
        let mut step_specs = Vec::with_capacity(definition.steps.len());
        for raw_step in &definition.steps {
            let params = convert_params(raw_step.params.clone())?;
            let step_spec = StepSpec {
                id: raw_step.id.clone(),
                params,
                inputs: raw_step.inputs.clone(),
                outputs: raw_step.outputs.clone(),
            };
            step_specs.push(step_spec);
        }

        // Optional: validate pipeline if schema registry is available
        // This provides early error detection before instantiation
        if let Some(schema_registry) = get_schema_registry() {
            let validator = PipelineValidator::new(&schema_registry);
            let report = validator.validate(&step_specs);

            if !report.is_valid() {
                return Err(anyhow!("Pipeline validation failed:\n{}", report.format()));
            }

            // Log warnings if any
            if !report.warnings.is_empty() {
                eprintln!("Pipeline validation warnings:\n{}", report.format());
            }
        }

        // Instantiate steps
        let registry = global_step_registry();
        let mut instantiated_steps = Vec::with_capacity(step_specs.len());
        for step_spec in step_specs {
            let step = registry.instantiate(&step_spec)?;
            instantiated_steps.push(step);
        }

        let name = if !definition.name.is_empty() {
            definition.name
        } else {
            spec.params
                .get_text("name")
                .unwrap_or("Custom Step Pipeline")
                .to_string()
        };

        Ok(Self {
            display_name: name,
            steps: instantiated_steps,
        })
    }
}

impl Algorithm for StepPipelineAlgorithm {
    fn id(&self) -> &'static str {
        "builder.step_pipeline"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: self.id().to_string(),
            name: self.display_name.clone(),
            description: format!(
                "Custom algorithm composed from {} step primitives",
                self.steps.len()
            ),
            version: "0.1.0".to_string(),
            supports_cancellation: true,
            cost_hint: CostHint::Linear,
            parameters: Vec::new(),
        }
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let execution_id = EXECUTION_COUNTER.fetch_add(1, Ordering::SeqCst);
        if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
            eprintln!(
                "[exec_{}] Pipeline '{}' starting with {} steps",
                execution_id,
                self.display_name,
                self.steps.len()
            );
        }

        let mut variables = StepVariables::default();

        if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
            eprintln!(
                "[exec_{}] Initial variables size: {}",
                execution_id,
                variables.count()
            );
        }

        for (index, step) in self.steps.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("builder pipeline cancelled before step {index}"));
            }

            if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
                eprintln!(
                    "[exec_{}] Step {} ({}): variables before = {}",
                    execution_id,
                    index,
                    step.id(),
                    variables.count()
                );
            }

            ctx.begin_step(index, step.id());
            let result = {
                let mut scope = StepScope::new(&subgraph, &mut variables);
                step.apply(ctx, &mut scope)
            };
            ctx.finish_step();
            result?;

            if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
                eprintln!(
                    "[exec_{}] Step {} ({}): variables after = {}",
                    execution_id,
                    index,
                    step.id(),
                    variables.count()
                );
            }
        }

        if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
            eprintln!(
                "[exec_{}] Pipeline complete, final variables: {}",
                execution_id,
                variables.count()
            );
        }

        Ok(subgraph)
    }
}

#[derive(Debug, Deserialize)]
struct StepPipelineDefinition {
    #[serde(default)]
    name: String,
    steps: Vec<RawStep>,
}

#[derive(Debug, Deserialize)]
struct RawStep {
    id: String,
    #[serde(default)]
    params: HashMap<String, serde_json::Value>,
    #[serde(default)]
    inputs: Vec<String>,
    #[serde(default)]
    outputs: Vec<String>,
}

fn extract_definition(spec: &AlgorithmSpec) -> Result<StepPipelineDefinition> {
    let steps_value = spec
        .params
        .get("steps")
        .ok_or_else(|| anyhow!("builder.step_pipeline requires a 'steps' parameter"))?;

    let json_value = match steps_value {
        AlgorithmParamValue::Json(value) => value.clone(),
        AlgorithmParamValue::Text(text) => serde_json::from_str::<serde_json::Value>(text)?,
        other => {
            return Err(anyhow!(
                "builder.step_pipeline expected JSON for 'steps', found {:?}",
                other
            ))
        }
    };

    if json_value.is_array() {
        // Backwards compatibility: allow passing steps array directly.
        Ok(StepPipelineDefinition {
            name: spec.params.get_text("name").unwrap_or_default().to_string(),
            steps: serde_json::from_value(json_value)?,
        })
    } else {
        Ok(serde_json::from_value(json_value)?)
    }
}

fn convert_params(raw: HashMap<String, serde_json::Value>) -> Result<AlgorithmParams> {
    let mut params = AlgorithmParams::new();
    for (key, value) in raw {
        let param = json_to_param(value)?;
        params.insert(key, param);
    }
    Ok(params)
}

fn json_to_param(value: serde_json::Value) -> Result<AlgorithmParamValue> {
    Ok(match value {
        serde_json::Value::Null => AlgorithmParamValue::None,
        serde_json::Value::Bool(b) => AlgorithmParamValue::Bool(b),
        serde_json::Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                AlgorithmParamValue::Int(i)
            } else if let Some(f) = num.as_f64() {
                AlgorithmParamValue::Float(f)
            } else {
                return Err(anyhow!("unsupported number format"));
            }
        }
        serde_json::Value::String(s) => AlgorithmParamValue::Text(s),
        serde_json::Value::Array(items) => infer_array_param(items)?,
        serde_json::Value::Object(map) => AlgorithmParamValue::Json(serde_json::Value::Object(map)),
    })
}

fn infer_array_param(items: Vec<serde_json::Value>) -> Result<AlgorithmParamValue> {
    if items.is_empty() {
        return Ok(AlgorithmParamValue::Json(serde_json::Value::Array(
            Vec::new(),
        )));
    }

    if items.iter().all(|v| v.is_boolean()) {
        let values = items
            .into_iter()
            .map(|v| v.as_bool().unwrap())
            .collect::<Vec<bool>>();
        return Ok(AlgorithmParamValue::BoolList(values));
    }

    if items.iter().all(|v| v.is_string()) {
        let values = items
            .into_iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<String>>();
        return Ok(AlgorithmParamValue::TextList(values));
    }

    if items.iter().all(|v| v.is_number() && v.as_i64().is_some()) {
        let values = items
            .into_iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<i64>>();
        return Ok(AlgorithmParamValue::IntList(values));
    }

    if items.iter().all(|v| v.is_number()) {
        let values = items
            .into_iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<f64>>();
        return Ok(AlgorithmParamValue::FloatList(values));
    }

    Ok(AlgorithmParamValue::Json(serde_json::Value::Array(items)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn json_array_inference_detects_ints() {
        let value = serde_json::json!([1, 2, 3]);
        let param = json_to_param(value).unwrap();
        assert_eq!(param, AlgorithmParamValue::IntList(vec![1, 2, 3]));
    }

    #[test]
    fn json_array_inference_falls_back_to_json() {
        let value = serde_json::json!([1, "two", true]);
        let param = json_to_param(value).unwrap();
        assert!(matches!(param, AlgorithmParamValue::Json(_)));
    }

    #[test]
    fn definition_parses_steps_array_only() {
        let spec = AlgorithmSpec {
            id: "builder.step_pipeline".into(),
            params: {
                let mut params = AlgorithmParams::new();
                params.insert(
                    "steps",
                    AlgorithmParamValue::Json(serde_json::json!([
                        {"id": "core.init_nodes", "params": {"target": "x"}}
                    ])),
                );
                params
            },
        };

        let def = extract_definition(&spec).unwrap();
        assert_eq!(def.steps.len(), 1);
    }

    #[test]
    fn register_algorithm_allows_instantiation() {
        ensure_core_steps_registered();

        let registry = Registry::default();
        register_algorithms(&registry).unwrap();

        let mut params = AlgorithmParams::new();
        params.insert(
            "steps",
            AlgorithmParamValue::Json(serde_json::json!([
                {
                    "id": "core.init_nodes",
                    "params": {"target": "labels", "value": 1}
                },
                {
                    "id": "core.attach_node_attr",
                    "params": {"source": "labels", "attr": "label"}
                }
            ])),
        );

        let spec = AlgorithmSpec {
            id: "builder.step_pipeline".to_string(),
            params,
        };

        let algo = registry.instantiate(&spec).unwrap();

        let mut graph = Graph::new();
        let node = graph.add_node();
        let nodes: HashSet<_> = [node].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, sg).unwrap();
        assert!(result.has_node(node));
    }
}
