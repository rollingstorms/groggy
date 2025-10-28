use std::fmt;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::subgraphs::Subgraph;

use super::registry::Registry;
use super::{Algorithm, AlgorithmMetadata, AlgorithmParams, Context};

/// Serializable description of a single algorithm invocation inside a pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmSpec {
    pub id: String,
    #[serde(default)]
    pub params: AlgorithmParams,
}

impl AlgorithmSpec {
    /// Retrieve a typed parameter.
    pub fn param(&self, key: &str) -> Option<&crate::algorithms::AlgorithmParamValue> {
        self.params.get(key)
    }
}

/// High-level pipeline specification exchanged with the FFI and DSL layers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PipelineSpec {
    #[serde(default)]
    pub steps: Vec<AlgorithmSpec>,
}

impl PipelineSpec {
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Builder that collects algorithm specs before instantiation.
#[derive(Clone, Debug, Default)]
pub struct PipelineBuilder {
    specs: Vec<AlgorithmSpec>,
}

impl PipelineBuilder {
    /// Start an empty builder.
    pub fn new() -> Self {
        Self { specs: Vec::new() }
    }

    /// Add an algorithm with a configuration closure.
    pub fn with_algorithm<F>(mut self, id: impl Into<String>, configure: F) -> Self
    where
        F: FnOnce(&mut AlgorithmParams),
    {
        let mut params = AlgorithmParams::new();
        configure(&mut params);
        self.specs.push(AlgorithmSpec {
            id: id.into(),
            params,
        });
        self
    }

    /// Import builder state from a serialized spec.
    pub fn from_spec(spec: PipelineSpec) -> Self {
        Self { specs: spec.steps }
    }

    /// Return the accumulated spec for inspection or serialization.
    pub fn as_spec(&self) -> PipelineSpec {
        PipelineSpec {
            steps: self.specs.clone(),
        }
    }

    /// Validate and instantiate the pipeline using the registry.
    pub fn build(self, registry: &Registry) -> Result<Pipeline> {
        crate::algorithms::ensure_algorithms_registered();
        validate_specs(&self.specs, registry)?;
        let mut steps = Vec::with_capacity(self.specs.len());

        for (index, spec) in self.specs.into_iter().enumerate() {
            let algorithm = registry.instantiate(&spec).map_err(|err| {
                anyhow!("failed to instantiate step {} ({}): {err}", index, spec.id)
            })?;
            let metadata = algorithm.metadata();
            steps.push(PipelineStep {
                id: spec.id,
                metadata,
                algorithm,
            });
        }

        Ok(Pipeline { steps })
    }
}

fn validate_specs(specs: &[AlgorithmSpec], registry: &Registry) -> Result<()> {
    if specs.is_empty() {
        return Err(anyhow!(PipelineValidationError::Empty));
    }

    for (index, spec) in specs.iter().enumerate() {
        if spec.id.trim().is_empty() {
            return Err(anyhow!(PipelineValidationError::MissingIdentifier {
                index
            }));
        }

        if !registry.contains(&spec.id) {
            return Err(anyhow!(PipelineValidationError::UnknownAlgorithm {
                index,
                id: spec.id.clone(),
            }));
        }
    }

    Ok(())
}

/// Fully instantiated pipeline ready for execution.
pub struct Pipeline {
    steps: Vec<PipelineStep>,
}

impl Pipeline {
    /// Execute the pipeline sequentially.
    pub fn run(&self, ctx: &mut Context, mut subgraph: Subgraph) -> Result<Subgraph> {
        let pipeline_start = Instant::now();
        for (idx, step) in self.steps.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("pipeline cancelled before step {idx}"));
            }
            ctx.begin_step(idx, step.id.as_str());
            let timer_label = format!("algorithm.{}", step.id);
            let step_start = Instant::now();
            subgraph = step.algorithm.execute(ctx, subgraph)?;
            let step_elapsed = step_start.elapsed();
            ctx.record_duration(timer_label, step_elapsed);
            ctx.finish_step();
        }
        let elapsed = pipeline_start.elapsed();
        ctx.record_duration("pipeline.run", elapsed);
        Ok(subgraph)
    }

    /// Export the pipeline metadata (useful for introspection).
    pub fn metadata(&self) -> Vec<&AlgorithmMetadata> {
        self.steps.iter().map(|step| &step.metadata).collect()
    }
}

struct PipelineStep {
    id: String,
    metadata: AlgorithmMetadata,
    algorithm: Box<dyn Algorithm>,
}

#[derive(Debug)]
pub enum PipelineValidationError {
    Empty,
    MissingIdentifier { index: usize },
    UnknownAlgorithm { index: usize, id: String },
}

impl fmt::Display for PipelineValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineValidationError::Empty => write!(f, "pipeline must contain at least one step"),
            PipelineValidationError::MissingIdentifier { index } => {
                write!(f, "pipeline step {index} is missing an identifier")
            }
            PipelineValidationError::UnknownAlgorithm { index, id } => {
                write!(
                    f,
                    "pipeline step {index} references unknown algorithm '{id}'"
                )
            }
        }
    }
}

impl std::error::Error for PipelineValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Algorithm;
    use crate::api::graph::Graph;
    use crate::subgraphs::Subgraph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    use crate::algorithms::registry::Registry;
    use crate::algorithms::{AlgorithmParams, Context};

    struct IdentityAlgorithm;

    impl Algorithm for IdentityAlgorithm {
        fn id(&self) -> &'static str {
            "identity"
        }

        fn execute(&self, _ctx: &mut Context, subgraph: Subgraph) -> anyhow::Result<Subgraph> {
            Ok(subgraph)
        }
    }

    #[test]
    fn pipeline_runs_registered_algorithm() {
        let registry = Registry::default();
        registry
            .register_factory("identity", |_spec| Ok(Box::new(IdentityAlgorithm)))
            .expect("register");

        let builder = PipelineBuilder::new().with_algorithm("identity", |_| {});
        let pipeline = builder.build(&registry).expect("build pipeline");

        let mut graph = Graph::new();
        let node = graph.add_node();
        let mut nodes = HashSet::new();
        nodes.insert(node);
        let sg =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".to_string()).unwrap();

        let mut ctx = Context::new();
        let result = pipeline.run(&mut ctx, sg).expect("run");
        assert!(result.has_node(node));
    }

    #[test]
    fn builder_validates_unknown_algorithm() {
        let registry = Registry::default();
        let builder = PipelineBuilder::new().with_algorithm("missing", |_| {});
        let err = builder.build(&registry).err().expect("expected error");
        assert!(err.to_string().contains("unknown algorithm"));
    }

    #[test]
    fn pipeline_spec_roundtrip() {
        let spec = PipelineSpec {
            steps: vec![AlgorithmSpec {
                id: "demo".to_string(),
                params: AlgorithmParams::new(),
            }],
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        let decoded: PipelineSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.steps.len(), 1);
        assert_eq!(decoded.steps[0].id, "demo");
    }

    #[test]
    fn validation_rejects_missing_identifier() {
        let registry = Registry::default();
        let builder = PipelineBuilder::new().with_algorithm("   ", |_| {});
        let err = builder.build(&registry).err().expect("expected error");
        assert!(err.to_string().contains("missing an identifier"));
    }
}
