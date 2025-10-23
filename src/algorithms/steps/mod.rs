//! Step primitives used by the pipeline interpreter.
//!
//! Steps are intentionally fine-grained so that Python builders and future
//! planners can assemble bespoke algorithms while still executing entirely in
//! Rust. Phase 1 provides a small core set focused on node transformations and
//! attribute attachment.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, NodeId};

use super::{AlgorithmParamValue, AlgorithmParams, Context, CostHint};

/// Trait implemented by all step primitives.
pub trait Step: Send + Sync {
    fn id(&self) -> &'static str;
    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: String::new(),
            cost_hint: CostHint::Unknown,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()>;
}

/// Metadata available for discovery and validation.
#[derive(Clone, Debug)]
pub struct StepMetadata {
    pub id: String,
    pub description: String,
    pub cost_hint: CostHint,
}

/// Serializable specification for step instantiation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepSpec {
    pub id: String,
    #[serde(default)]
    pub params: AlgorithmParams,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
}

/// Runtime value representation for variables inside a step pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum StepValue {
    /// Mapping from node id → value.
    NodeMap(HashMap<NodeId, AlgorithmParamValue>),
    /// Scalar helper value.
    Scalar(AlgorithmParamValue),
}

impl StepValue {
    fn expect_node_map(&self, name: &str) -> Result<&HashMap<NodeId, AlgorithmParamValue>> {
        match self {
            StepValue::NodeMap(map) => Ok(map),
            _ => Err(anyhow!("variable '{name}' is not a node map")),
        }
    }

    fn expect_scalar(&self, name: &str) -> Result<&AlgorithmParamValue> {
        match self {
            StepValue::Scalar(value) => Ok(value),
            _ => Err(anyhow!("variable '{name}' is not a scalar")),
        }
    }
}

/// Mutable variable storage passed between steps.
#[derive(Default, Clone, Debug)]
pub struct StepVariables {
    values: HashMap<String, StepValue>,
}

impl StepVariables {
    pub fn set_node_map(
        &mut self,
        name: impl Into<String>,
        map: HashMap<NodeId, AlgorithmParamValue>,
    ) {
        self.values.insert(name.into(), StepValue::NodeMap(map));
    }

    pub fn node_map(&self, name: &str) -> Result<&HashMap<NodeId, AlgorithmParamValue>> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| value.expect_node_map(name))
    }

    pub fn set_scalar(&mut self, name: impl Into<String>, value: AlgorithmParamValue) {
        self.values.insert(name.into(), StepValue::Scalar(value));
    }

    pub fn scalar(&self, name: &str) -> Result<&AlgorithmParamValue> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| value.expect_scalar(name))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
}

/// Execution scope passed to each step.
pub struct StepScope<'a> {
    subgraph: &'a Subgraph,
    variables: &'a mut StepVariables,
}

impl<'a> StepScope<'a> {
    pub fn new(subgraph: &'a Subgraph, variables: &'a mut StepVariables) -> Self {
        Self {
            subgraph,
            variables,
        }
    }

    pub fn subgraph(&self) -> &Subgraph {
        self.subgraph
    }

    pub fn variables(&self) -> &StepVariables {
        self.variables
    }

    pub fn variables_mut(&mut self) -> &mut StepVariables {
        self.variables
    }
}

/// Context provided to node mapping callbacks.
pub struct StepInput<'a> {
    pub subgraph: &'a Subgraph,
    pub variables: &'a StepVariables,
}

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

/// Initialize node-level variable with a constant value.
pub struct InitNodesStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitNodesStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitNodesStep {
    fn id(&self) -> &'static str {
        "core.init_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize a node variable with a constant value".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for &node in scope.subgraph().nodes() {
            map.insert(node, self.value.clone());
        }
        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Transform node values using a callback.
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

/// Attach node attributes using a node map variable.
pub struct AttachNodeAttrStep {
    source: String,
    attr: AttrName,
}

impl AttachNodeAttrStep {
    pub fn new(source: impl Into<String>, attr: AttrName) -> Self {
        Self {
            source: source.into(),
            attr,
        }
    }
}

impl Step for AttachNodeAttrStep {
    fn id(&self) -> &'static str {
        "core.attach_node_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Persist a node map as a graph attribute".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        let mut attrs = HashMap::new();
        let mut values = Vec::with_capacity(map.len());

        for (node, value) in map.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("attach_node_attr cancelled"));
            }
            if let Some(attr_value) = value.as_attr_value() {
                values.push((*node, attr_value));
            } else {
                return Err(anyhow!(
                    "variable '{}' contains unsupported attribute type",
                    self.source
                ));
            }
        }

        if values.is_empty() {
            return Ok(());
        }

        attrs.insert(self.attr.clone(), values);
        scope
            .subgraph()
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to set node attributes: {err}"))?;
        Ok(())
    }
}

/// Reduce node values to a scalar statistic.
pub struct ReduceNodeValuesStep {
    source: String,
    target: String,
    reducer: Reduction,
}

impl ReduceNodeValuesStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, reducer: Reduction) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            reducer,
        }
    }
}

impl Step for ReduceNodeValuesStep {
    fn id(&self) -> &'static str {
        "core.reduce_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Aggregate node values into a scalar".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("reduce_nodes cancelled"));
        }
        let value = self.reducer.apply(map)?;
        scope.variables_mut().set_scalar(self.target.clone(), value);
        Ok(())
    }
}

/// Supported reduction strategies.
#[derive(Clone, Copy, Debug)]
pub enum Reduction {
    Sum,
    Min,
    Max,
    Mean,
}

impl Reduction {
    fn apply(&self, map: &HashMap<NodeId, AlgorithmParamValue>) -> Result<AlgorithmParamValue> {
        let mut values = Vec::new();
        for value in map.values() {
            match value {
                AlgorithmParamValue::Int(v) => values.push(*v as f64),
                AlgorithmParamValue::Float(v) => values.push(*v),
                _ => {
                    return Err(anyhow!(
                        "reduction requires numeric values, found {:?}",
                        value
                    ))
                }
            }
        }
        if values.is_empty() {
            return Ok(AlgorithmParamValue::None);
        }

        let result = match self {
            Reduction::Sum => values.iter().sum(),
            Reduction::Min => values.iter().copied().reduce(f64::min).unwrap_or(0.0),
            Reduction::Max => values.iter().copied().reduce(f64::max).unwrap_or(0.0),
            Reduction::Mean => {
                let sum: f64 = values.iter().sum();
                sum / values.len() as f64
            }
        };

        Ok(AlgorithmParamValue::Float(result))
    }
}

/// Factory signature for steps.
pub type StepFactory = dyn Fn(&StepSpec) -> Result<Box<dyn Step>> + Send + Sync;

struct StepEntry {
    factory: Arc<StepFactory>,
    metadata: StepMetadata,
}

/// Registry for step primitives.
#[derive(Default)]
pub struct StepRegistry {
    entries: RwLock<HashMap<String, StepEntry>>,
}

impl StepRegistry {
    pub fn register<F>(&self, id: &str, metadata: StepMetadata, factory: F) -> Result<()>
    where
        F: Fn(&StepSpec) -> Result<Box<dyn Step>> + Send + Sync + 'static,
    {
        let mut guard = self.entries.write().expect("step registry poisoned");
        if guard.contains_key(id) {
            return Err(anyhow!("step '{id}' already registered"));
        }
        guard.insert(
            id.to_string(),
            StepEntry {
                factory: Arc::new(factory),
                metadata,
            },
        );
        Ok(())
    }

    pub fn instantiate(&self, spec: &StepSpec) -> Result<Box<dyn Step>> {
        let guard = self.entries.read().expect("step registry poisoned");
        let entry = guard
            .get(&spec.id)
            .ok_or_else(|| anyhow!("step '{}' is not registered", spec.id))?;
        (entry.factory)(spec)
    }

    pub fn contains(&self, id: &str) -> bool {
        let guard = self.entries.read().expect("step registry poisoned");
        guard.contains_key(id)
    }

    pub fn metadata(&self, id: &str) -> Option<StepMetadata> {
        let guard = self.entries.read().expect("step registry poisoned");
        guard.get(id).map(|entry| entry.metadata.clone())
    }
}

static GLOBAL_STEP_REGISTRY: OnceLock<StepRegistry> = OnceLock::new();

pub fn global_step_registry() -> &'static StepRegistry {
    GLOBAL_STEP_REGISTRY.get_or_init(StepRegistry::default)
}

/// Register the core steps that ship with the engine.
pub fn register_core_steps(registry: &StepRegistry) -> Result<()> {
    registry.register(
        "core.init_nodes",
        StepMetadata {
            id: "core.init_nodes".to_string(),
            description: "Initialize node variable with constant".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.init_nodes requires 'target' param"))?;
            let value = spec
                .params
                .get("value")
                .cloned()
                .unwrap_or(AlgorithmParamValue::None);
            Ok(Box::new(InitNodesStep::new(target.to_string(), value)))
        },
    )?;

    registry.register(
        "core.attach_node_attr",
        StepMetadata {
            id: "core.attach_node_attr".to_string(),
            description: "Persist node map as attribute".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.attach_node_attr requires 'source' param"))?;
            let attr_name = spec
                .params
                .get_text("attr")
                .ok_or_else(|| anyhow!("core.attach_node_attr requires 'attr' param"))?;
            Ok(Box::new(AttachNodeAttrStep::new(
                source.to_string(),
                AttrName::from(attr_name.to_string()),
            )))
        },
    )?;

    registry.register(
        "core.reduce_nodes",
        StepMetadata {
            id: "core.reduce_nodes".to_string(),
            description: "Aggregate node map".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.reduce_nodes requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.reduce_nodes requires 'target' param"))?;
            let reducer = match spec.params.get_text("reducer") {
                Some("sum") | None => Reduction::Sum,
                Some("min") => Reduction::Min,
                Some("max") => Reduction::Max,
                Some("mean") => Reduction::Mean,
                Some(other) => return Err(anyhow!("unknown reducer '{other}'")),
            };
            Ok(Box::new(ReduceNodeValuesStep::new(
                source.to_string(),
                target.to_string(),
                reducer,
            )))
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::types::{AttrName, AttrValue, NodeId};
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    fn sample_subgraph() -> (Subgraph, Vec<NodeId>) {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let nodes = vec![a, b];
        let set: HashSet<NodeId> = nodes.iter().copied().collect();
        let sg =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), set, "test".to_string()).unwrap();
        (sg, nodes)
    }

    #[test]
    fn init_nodes_sets_values() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = InitNodesStep::new("labels", AlgorithmParamValue::Int(1));
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let map = scope.variables().node_map("labels").unwrap();
        for node in nodes {
            assert_eq!(map.get(&node), Some(&AlgorithmParamValue::Int(1)));
        }
    }

    #[test]
    fn attach_node_attr_persists_values() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "scores",
            nodes
                .iter()
                .map(|&node| (node, AlgorithmParamValue::Float(0.5)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let attr_name = AttrName::from("score".to_string());
        let step = AttachNodeAttrStep::new("scores", attr_name.clone());
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let graph = sg.graph();
        let graph_ref = graph.borrow();
        for node in nodes {
            let attr = graph_ref.get_node_attr(node, &attr_name).unwrap().unwrap();
            assert_eq!(attr, AttrValue::Float(0.5));
        }
    }

    #[test]
    fn reduce_node_values_sum() {
        let (sg, nodes) = sample_subgraph();
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "scores",
            nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AlgorithmParamValue::Int((idx + 1) as i64)))
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = ReduceNodeValuesStep::new("scores", "total", Reduction::Sum);
        step.apply(&mut Context::new(), &mut scope).unwrap();

        let value = scope.variables().scalar("total").unwrap();
        assert_eq!(value, &AlgorithmParamValue::Float(3.0));
    }
}
