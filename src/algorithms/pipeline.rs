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
    /// Execute sequentially with the existing per-algorithm publication policy.
    pub fn run(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        self.run_steps(ctx, subgraph, None)
    }

    /// Stage all certified property algorithms and accept exactly once on success.
    /// Unsupported effects reject before any execution. Earlier staged properties
    /// are visible to later algorithms; errors/cancellation discard the outer batch.
    pub fn run_atomic_properties(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        if ctx.property_batch.is_some() {
            return Err(anyhow!(
                "Nested outer atomic pipeline execution is unsupported"
            ));
        }
        for (index, step) in self.steps.iter().enumerate() {
            if !step.algorithm.supports_atomic_properties() {
                return Err(anyhow!(
                    "Pipeline step {index} ({}) does not support atomic property execution",
                    step.id
                ));
            }
        }
        let source = subgraph.graph();
        let staged = ctx.persist_results();
        if staged {
            let graph = source.borrow();
            let head = graph.current_state_id();
            graph.validate_write_head(head)?;
            ctx.property_batch = Some(super::property_batch::PropertyBatch::new(
                graph.write_coordinate(),
            ));
        }
        let previous_outputs = ctx.take_outputs();
        let result = self
            .run_steps(ctx, subgraph, Some(&source))
            .and_then(|result| {
                if ctx.is_cancelled() {
                    return Err(anyhow!("pipeline cancelled before publication"));
                }
                if staged {
                    let pending = ctx
                        .property_batch
                        .take()
                        .ok_or_else(|| anyhow!("Atomic pipeline lost its property batch"))?;
                    pending.accept(&mut source.borrow_mut())?;
                }
                Ok(result)
            });
        // Child algorithms borrow the outer batch; they never accept or discard it.
        // Always clear it, including execution, validation and publication errors.
        if staged {
            ctx.property_batch = None;
        }
        let outputs = ctx.take_outputs();
        ctx.outputs = previous_outputs;
        if result.is_ok() {
            ctx.outputs.extend(outputs);
        }
        result
    }

    fn run_steps(
        &self,
        ctx: &mut Context,
        mut subgraph: Subgraph,
        source: Option<&std::rc::Rc<std::cell::RefCell<crate::Graph>>>,
    ) -> Result<Subgraph> {
        let pipeline_start = Instant::now();
        for (idx, step) in self.steps.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("pipeline cancelled before step {idx}"));
            }
            ctx.begin_step(idx, step.id.as_str());
            let timer_label = format!("algorithm.{}", step.id);
            let step_start = Instant::now();
            let result = step.algorithm.execute(ctx, subgraph);
            ctx.record_duration(timer_label, step_start.elapsed());
            ctx.finish_step();
            subgraph = result?;
            if source.is_some_and(|owner| !std::rc::Rc::ptr_eq(owner, &subgraph.graph())) {
                return Err(anyhow!("Atomic pipeline changed graph owner at step {idx}"));
            }
        }
        ctx.record_duration("pipeline.run", pipeline_start.elapsed());
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

        let mut graph = Graph::new_manual();
        let node = graph.add_node().unwrap();
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

#[cfg(test)]
mod atomic_property_tests {
    use super::*;
    use crate::algorithms::AlgorithmOutput;
    use crate::{AttrValue, Graph};
    use std::{
        cell::RefCell,
        collections::HashMap,
        rc::Rc,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    struct Stage {
        name: &'static str,
        fail: bool,
        cancel: Option<Arc<AtomicBool>>,
    }
    impl Algorithm for Stage {
        fn id(&self) -> &'static str {
            "test.stage"
        }
        fn supports_atomic_properties(&self) -> bool {
            true
        }
        fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
            ctx.property_batch.as_mut().expect("outer staging").stage(
                true,
                HashMap::from([(self.name.into(), vec![(0, AttrValue::Int(7))])]),
            )?;
            ctx.add_output("produced", AlgorithmOutput::NodeIntegers(vec![(0, 7)]));
            if let Some(token) = &self.cancel {
                token.store(true, Ordering::Relaxed);
            }
            if self.fail {
                return Err(anyhow!("late failure"));
            }
            Ok(subgraph)
        }
    }
    struct Unsupported(Arc<AtomicBool>);
    impl Algorithm for Unsupported {
        fn id(&self) -> &'static str {
            "test.unsupported"
        }
        fn execute(&self, _: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
            self.0.store(true, Ordering::Relaxed);
            Ok(subgraph)
        }
    }
    struct OtherOwner;
    impl Algorithm for OtherOwner {
        fn id(&self) -> &'static str {
            "test.owner"
        }
        fn supports_atomic_properties(&self) -> bool {
            true
        }
        fn execute(&self, _: &mut Context, _: Subgraph) -> Result<Subgraph> {
            let mut graph = Graph::new_publishing();
            graph.insert_nodes(1)?;
            Ok(Subgraph::from_nodes(
                Rc::new(RefCell::new(graph)),
                [0].into_iter().collect(),
                "foreign".into(),
            )?)
        }
    }
    struct SwitchBranch;
    impl Algorithm for SwitchBranch {
        fn id(&self) -> &'static str {
            "test.switch_branch"
        }
        fn supports_atomic_properties(&self) -> bool {
            true
        }
        fn execute(&self, _: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
            subgraph
                .graph()
                .borrow_mut()
                .checkout_branch("other".into())?;
            Ok(subgraph)
        }
    }
    fn pipeline(algorithms: Vec<Box<dyn Algorithm>>) -> Pipeline {
        Pipeline {
            steps: algorithms
                .into_iter()
                .map(|algorithm| PipelineStep {
                    id: algorithm.id().into(),
                    metadata: algorithm.metadata(),
                    algorithm,
                })
                .collect(),
        }
    }
    fn fixture() -> (Rc<RefCell<Graph>>, Subgraph) {
        let mut graph = Graph::new_publishing();
        graph.insert_nodes(1).unwrap();
        let graph = Rc::new(RefCell::new(graph));
        let selected =
            Subgraph::from_nodes(graph.clone(), [0].into_iter().collect(), "selected".into())
                .unwrap();
        (graph, selected)
    }
    fn stage(name: &'static str, fail: bool) -> Box<dyn Algorithm> {
        Box::new(Stage {
            name,
            fail,
            cancel: None,
        })
    }
    #[test]
    fn outer_failure_discards_earlier_properties_outputs_and_allows_context_reuse() {
        let (graph, selected) = fixture();
        let before = graph.borrow().current_state_id();
        let mut ctx = Context::new();
        ctx.add_output("previous", AlgorithmOutput::Histogram(vec![3]));
        assert!(pipeline(vec![stage("a", false), stage("b", true)])
            .run_atomic_properties(&mut ctx, selected.clone())
            .is_err());
        assert!(ctx.property_batch.is_none());
        assert!(ctx.active_step.is_none());
        assert_eq!(graph.borrow().current_state_id(), before);
        assert_eq!(graph.borrow().get_node_attr(0, &"a".into()).unwrap(), None);
        assert_eq!(ctx.outputs.len(), 1);
        assert!(ctx.outputs.contains_key("previous"));
        pipeline(vec![stage("a", false), stage("b", false)])
            .run_atomic_properties(&mut ctx, selected)
            .unwrap();
        assert_eq!(graph.borrow().current_state_id(), before + 1);
        assert!(ctx.outputs.contains_key("previous") && ctx.outputs.contains_key("produced"));
    }
    #[test]
    fn unsupported_outer_algorithm_rejects_before_any_prefix_executes() {
        let (graph, selected) = fixture();
        let called = Arc::new(AtomicBool::new(false));
        let mut ctx = Context::new();
        let error = pipeline(vec![
            stage("a", false),
            Box::new(Unsupported(called.clone())),
        ])
        .run_atomic_properties(&mut ctx, selected)
        .unwrap_err();
        assert!(error.to_string().contains("step 1"));
        assert!(!called.load(Ordering::Relaxed));
        assert_eq!(graph.borrow().current_state_id(), 1);
        assert!(ctx.property_batch.is_none());
        assert_eq!(graph.borrow().get_node_attr(0, &"a".into()).unwrap(), None);
    }
    #[test]
    fn cancellation_after_last_algorithm_discards_staging_without_partial_publication() {
        let (graph, selected) = fixture();
        let token = Arc::new(AtomicBool::new(false));
        let mut ctx = Context::with_cancel_token(token.clone());
        let pipeline = pipeline(vec![
            stage("a", false),
            Box::new(Stage {
                name: "b",
                fail: false,
                cancel: Some(token),
            }),
        ]);
        assert!(pipeline.run_atomic_properties(&mut ctx, selected).is_err());
        assert_eq!(graph.borrow().current_state_id(), 1);
        assert!(!graph.borrow().has_uncommitted_changes());
        assert!(ctx.property_batch.is_none());
        assert!(ctx.outputs.is_empty());
    }
    #[test]
    fn atomic_owner_changes_and_final_managed_property_validation_discard_prefixes() {
        let (graph, selected) = fixture();
        let mut ctx = Context::new();
        assert!(pipeline(vec![stage("a", false), Box::new(OtherOwner)])
            .run_atomic_properties(&mut ctx, selected.clone())
            .is_err());
        assert_eq!(graph.borrow().current_state_id(), 1);
        assert!(ctx.property_batch.is_none());
        assert!(
            pipeline(vec![stage("a", false), stage("entity_type", false)])
                .run_atomic_properties(&mut ctx, selected)
                .is_err()
        );
        assert_eq!(graph.borrow().current_state_id(), 1);
        assert_eq!(graph.borrow().get_node_attr(0, &"a".into()).unwrap(), None);
        assert!(ctx.property_batch.is_none() && ctx.outputs.is_empty());
    }

    #[test]
    fn same_head_branch_switch_discards_staging_and_outputs_then_context_can_be_reused() {
        let (graph, selected) = fixture();
        graph.borrow_mut().create_branch("other".into()).unwrap();
        let head = graph.borrow().current_state_id();
        let mut ctx = Context::new();
        ctx.add_output("previous", AlgorithmOutput::Histogram(vec![3]));
        let error = pipeline(vec![stage("a", false), Box::new(SwitchBranch)])
            .run_atomic_properties(&mut ctx, selected.clone())
            .unwrap_err();
        assert!(error.to_string().contains("coordinate"));
        assert_eq!(graph.borrow().current_state_id(), head);
        assert!(!graph.borrow().has_uncommitted_changes());
        assert_eq!(graph.borrow().get_node_attr(0, &"a".into()).unwrap(), None);
        assert!(ctx.property_batch.is_none() && ctx.active_step.is_none());
        assert_eq!(ctx.outputs.len(), 1);
        assert!(ctx.outputs.contains_key("previous"));
        // A fresh invocation intentionally targets the now-selected branch.
        pipeline(vec![stage("a", false)])
            .run_atomic_properties(&mut ctx, selected)
            .unwrap();
        assert_eq!(graph.borrow().current_state_id(), head + 1);
    }
}
