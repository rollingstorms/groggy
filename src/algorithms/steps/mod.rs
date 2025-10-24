//! Step primitives used by the pipeline interpreter.
//!
//! Steps are intentionally fine-grained so that Python builders and future
//! planners can assemble bespoke algorithms while still executing entirely in
//! Rust. Phase 1 provides a small core set focused on node transformations and
//! attribute attachment.

pub mod temporal;

use std::collections::HashMap;
use std::sync::{Arc, Once, OnceLock, RwLock};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, EdgeId, NodeId};

use super::{AlgorithmParamValue, AlgorithmParams, Context, CostHint};

pub use temporal::{
    AggregateFunction, DiffEdgesStep, DiffNodesStep, MarkChangedNodesStep, TemporalFilterStep,
    TemporalPredicate, WindowAggregateStep,
};

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
    /// Mapping from edge id → value.
    EdgeMap(HashMap<EdgeId, AlgorithmParamValue>),
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

    fn expect_edge_map(&self, name: &str) -> Result<&HashMap<EdgeId, AlgorithmParamValue>> {
        match self {
            StepValue::EdgeMap(map) => Ok(map),
            _ => Err(anyhow!("variable '{name}' is not an edge map")),
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

    pub fn set_edge_map(
        &mut self,
        name: impl Into<String>,
        map: HashMap<EdgeId, AlgorithmParamValue>,
    ) {
        self.values.insert(name.into(), StepValue::EdgeMap(map));
    }

    pub fn scalar(&self, name: &str) -> Result<&AlgorithmParamValue> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| value.expect_scalar(name))
    }

    pub fn edge_map(&self, name: &str) -> Result<&HashMap<EdgeId, AlgorithmParamValue>> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| value.expect_edge_map(name))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
}

fn value_as_f64(value: &AlgorithmParamValue, context: &str) -> Result<f64> {
    match value {
        AlgorithmParamValue::Float(v) => Ok(*v),
        AlgorithmParamValue::Int(v) => Ok(*v as f64),
        other => Err(anyhow!(
            "expected numeric value for {context}, found {:?}",
            other
        )),
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

    pub fn node_ids(&self) -> impl Iterator<Item = &NodeId> {
        self.subgraph.node_set().iter()
    }

    pub fn edge_ids(&self) -> impl Iterator<Item = &EdgeId> {
        self.subgraph.edge_set().iter()
    }
}

/// Context provided to node mapping callbacks.
pub struct StepInput<'a> {
    pub subgraph: &'a Subgraph,
    pub variables: &'a StepVariables,
}

pub struct StepEdgeInput<'a> {
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

pub trait EdgeMapFn: Send + Sync {
    fn map(&self, edge: EdgeId, input: &StepEdgeInput<'_>) -> Result<AlgorithmParamValue>;
}

impl<F> EdgeMapFn for F
where
    F: for<'a> Fn(EdgeId, &StepEdgeInput<'a>) -> Result<AlgorithmParamValue> + Send + Sync,
{
    fn map(&self, edge: EdgeId, input: &StepEdgeInput<'_>) -> Result<AlgorithmParamValue> {
        (self)(edge, input)
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

/// Load an existing node attribute into a step variable.
pub struct LoadNodeAttrStep {
    target: String,
    attr: AttrName,
    default: AlgorithmParamValue,
}

impl LoadNodeAttrStep {
    pub fn new(target: impl Into<String>, attr: AttrName, default: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            attr,
            default,
        }
    }
}

impl Step for LoadNodeAttrStep {
    fn id(&self) -> &'static str {
        "core.load_node_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Load a node attribute into a variable".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().node_set().len());
        for &node in scope.node_ids() {
            let value = scope
                .subgraph()
                .get_node_attribute(node, &self.attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or_else(|| self.default.clone());
            map.insert(node, value);
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

/// Initialize edge-level variable with a constant value.
pub struct InitEdgesStep {
    target: String,
    value: AlgorithmParamValue,
}

impl InitEdgesStep {
    pub fn new(target: impl Into<String>, value: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            value,
        }
    }
}

impl Step for InitEdgesStep {
    fn id(&self) -> &'static str {
        "core.init_edges"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Initialize an edge variable with a constant".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::new();
        for &edge in scope.edge_ids() {
            map.insert(edge, self.value.clone());
        }
        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}

/// Load an existing edge attribute into a variable.
pub struct LoadEdgeAttrStep {
    target: String,
    attr: AttrName,
    default: AlgorithmParamValue,
}

impl LoadEdgeAttrStep {
    pub fn new(target: impl Into<String>, attr: AttrName, default: AlgorithmParamValue) -> Self {
        Self {
            target: target.into(),
            attr,
            default,
        }
    }
}

impl Step for LoadEdgeAttrStep {
    fn id(&self) -> &'static str {
        "core.load_edge_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Load an edge attribute into a variable".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().edge_set().len());
        for &edge in scope.edge_ids() {
            let value = scope
                .subgraph()
                .get_edge_attribute(edge, &self.attr)?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or_else(|| self.default.clone());
            map.insert(edge, value);
        }
        scope.variables_mut().set_edge_map(self.target.clone(), map);
        Ok(())
    }
}

/// Attach edge attributes using an edge map variable.
pub struct AttachEdgeAttrStep {
    source: String,
    attr: AttrName,
}

impl AttachEdgeAttrStep {
    pub fn new(source: impl Into<String>, attr: AttrName) -> Self {
        Self {
            source: source.into(),
            attr,
        }
    }
}

impl Step for AttachEdgeAttrStep {
    fn id(&self) -> &'static str {
        "core.attach_edge_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Persist an edge map as an attribute".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().edge_map(&self.source)?;
        let mut attrs = HashMap::new();
        let mut values = Vec::with_capacity(map.len());

        for (edge, value) in map.iter() {
            if let Some(attr_value) = value.as_attr_value() {
                values.push((*edge, attr_value));
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
            .set_edge_attrs(attrs)
            .map_err(|err| anyhow!("failed to set edge attributes: {err}"))?;
        Ok(())
    }
}

/// Compute node degrees and store them as a node map.
pub struct NodeDegreeStep {
    target: String,
}

impl NodeDegreeStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for NodeDegreeStep {
    fn id(&self) -> &'static str {
        "core.node_degree"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute per-node degree within the subgraph".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let mut map = HashMap::with_capacity(scope.subgraph().node_set().len());
        for &node in scope.node_ids() {
            let degree = scope.subgraph().degree(node)? as i64;
            map.insert(node, AlgorithmParamValue::Int(degree));
        }
        scope.variables_mut().set_node_map(self.target.clone(), map);
        Ok(())
    }
}

/// Normalize numeric node values using a configurable strategy.
pub struct NormalizeNodeValuesStep {
    source: String,
    target: String,
    epsilon: f64,
    method: NormalizeMethod,
}

impl NormalizeNodeValuesStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        method: NormalizeMethod,
        epsilon: f64,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            epsilon,
            method,
        }
    }
}

impl Step for NormalizeNodeValuesStep {
    fn id(&self) -> &'static str {
        "core.normalize_node_values"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Normalize node values using sum/max/minmax".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        let mut normalized = HashMap::with_capacity(map.len());

        match self.method {
            NormalizeMethod::Sum => {
                let mut sum = 0.0;
                for value in map.values() {
                    sum += value_as_f64(value, &self.source)?;
                }

                if sum.abs() <= self.epsilon {
                    return Err(anyhow!(
                        "cannot normalize '{}': total magnitude below epsilon ({})",
                        self.source,
                        self.epsilon
                    ));
                }

                for (&node, value) in map.iter() {
                    let raw = value_as_f64(value, &self.source)?;
                    normalized.insert(node, AlgorithmParamValue::Float(raw / sum));
                }
            }
            NormalizeMethod::Max => {
                let mut max_value = f64::NEG_INFINITY;
                for value in map.values() {
                    let raw = value_as_f64(value, &self.source)?;
                    if raw > max_value {
                        max_value = raw;
                    }
                }

                if max_value.abs() <= self.epsilon {
                    return Err(anyhow!(
                        "cannot normalize '{}': max magnitude below epsilon ({})",
                        self.source,
                        self.epsilon
                    ));
                }

                for (&node, value) in map.iter() {
                    let raw = value_as_f64(value, &self.source)?;
                    normalized.insert(node, AlgorithmParamValue::Float(raw / max_value));
                }
            }
            NormalizeMethod::MinMax => {
                let mut min_value = f64::INFINITY;
                let mut max_value = f64::NEG_INFINITY;
                for value in map.values() {
                    let raw = value_as_f64(value, &self.source)?;
                    if raw < min_value {
                        min_value = raw;
                    }
                    if raw > max_value {
                        max_value = raw;
                    }
                }

                let range = max_value - min_value;
                if range.abs() <= self.epsilon {
                    return Err(anyhow!(
                        "cannot normalize '{}': range below epsilon ({})",
                        self.source,
                        self.epsilon
                    ));
                }

                for (&node, value) in map.iter() {
                    let raw = value_as_f64(value, &self.source)?;
                    let normalized_value = (raw - min_value) / range;
                    normalized.insert(node, AlgorithmParamValue::Float(normalized_value));
                }
            }
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), normalized);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum NormalizeMethod {
    Sum,
    Max,
    MinMax,
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
        "core.load_node_attr",
        StepMetadata {
            id: "core.load_node_attr".to_string(),
            description: "Load node attribute into variable".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let target = spec.params.expect_text("target")?.to_string();
            let attr = spec.params.expect_text("attr")?.to_string();
            let default = spec
                .params
                .get("default")
                .cloned()
                .unwrap_or(AlgorithmParamValue::None);
            Ok(Box::new(LoadNodeAttrStep::new(
                target,
                AttrName::from(attr),
                default,
            )))
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

    registry.register(
        "core.init_edges",
        StepMetadata {
            id: "core.init_edges".to_string(),
            description: "Initialize edge variable with constant".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let target = spec.params.expect_text("target")?.to_string();
            let value = spec
                .params
                .get("value")
                .cloned()
                .unwrap_or(AlgorithmParamValue::None);
            Ok(Box::new(InitEdgesStep::new(target, value)))
        },
    )?;

    registry.register(
        "core.load_edge_attr",
        StepMetadata {
            id: "core.load_edge_attr".to_string(),
            description: "Load edge attribute into variable".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let target = spec.params.expect_text("target")?.to_string();
            let attr = spec.params.expect_text("attr")?.to_string();
            let default = spec
                .params
                .get("default")
                .cloned()
                .unwrap_or(AlgorithmParamValue::None);
            Ok(Box::new(LoadEdgeAttrStep::new(
                target,
                AttrName::from(attr),
                default,
            )))
        },
    )?;

    registry.register(
        "core.attach_edge_attr",
        StepMetadata {
            id: "core.attach_edge_attr".to_string(),
            description: "Persist edge map as attribute".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let attr = spec.params.expect_text("attr")?.to_string();
            Ok(Box::new(AttachEdgeAttrStep::new(
                source,
                AttrName::from(attr),
            )))
        },
    )?;

    registry.register(
        "core.node_degree",
        StepMetadata {
            id: "core.node_degree".to_string(),
            description: "Compute node degrees".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(NodeDegreeStep::new(target)))
        },
    )?;

    registry.register(
        "core.normalize_node_values",
        StepMetadata {
            id: "core.normalize_node_values".to_string(),
            description: "Normalize node map using sum/max/minmax".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let epsilon = spec.params.get_float("epsilon").unwrap_or(1e-9);
            let method = match spec.params.get_text("method") {
                Some("max") => NormalizeMethod::Max,
                Some("minmax") => NormalizeMethod::MinMax,
                Some("sum") | None => NormalizeMethod::Sum,
                Some(other) => return Err(anyhow!("unknown normalization method '{other}'")),
            };
            Ok(Box::new(NormalizeNodeValuesStep::new(
                source, target, method, epsilon,
            )))
        },
    )?;

    // === Temporal Steps ===

    registry.register(
        "temporal.diff_nodes",
        StepMetadata {
            id: "temporal.diff_nodes".to_string(),
            description: "Compute node differences between temporal snapshots".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let before_var = spec.params.get_text("before").map(|s| s.to_string());
            let after_var = spec.params.get_text("after").map(|s| s.to_string());
            let prefix = spec
                .params
                .get_text("output_prefix")
                .unwrap_or("diff")
                .to_string();
            Ok(Box::new(DiffNodesStep::new(before_var, after_var, prefix)))
        },
    )?;

    registry.register(
        "temporal.diff_edges",
        StepMetadata {
            id: "temporal.diff_edges".to_string(),
            description: "Compute edge differences between temporal snapshots".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let before_var = spec.params.get_text("before").map(|s| s.to_string());
            let after_var = spec.params.get_text("after").map(|s| s.to_string());
            let prefix = spec
                .params
                .get_text("output_prefix")
                .unwrap_or("diff")
                .to_string();
            Ok(Box::new(DiffEdgesStep::new(before_var, after_var, prefix)))
        },
    )?;

    registry.register(
        "temporal.window_aggregate",
        StepMetadata {
            id: "temporal.window_aggregate".to_string(),
            description: "Aggregate attribute values over temporal window".to_string(),
            cost_hint: CostHint::Quadratic,
        },
        |spec| {
            let attr_name = spec.params.expect_text("attr")?.to_string();
            let function_str = spec.params.expect_text("function")?;
            let function = AggregateFunction::from_str(function_str)?;
            let output_var = spec.params.expect_text("output")?.to_string();
            let index_var = spec
                .params
                .get_text("index_var")
                .unwrap_or("temporal_index")
                .to_string();
            Ok(Box::new(WindowAggregateStep::new(
                attr_name, function, output_var, index_var,
            )))
        },
    )?;

    registry.register(
        "temporal.filter",
        StepMetadata {
            id: "temporal.filter".to_string(),
            description: "Filter nodes based on temporal properties".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let output_var = spec.params.expect_text("output")?.to_string();
            let predicate_type = spec.params.expect_text("predicate")?;
            
            let predicate = match predicate_type {
                "created_after" => {
                    let commit = spec.params.expect_int("commit")? as u64;
                    TemporalPredicate::CreatedAfter(commit)
                }
                "created_before" => {
                    let commit = spec.params.expect_int("commit")? as u64;
                    TemporalPredicate::CreatedBefore(commit)
                }
                "existed_at" => {
                    let commit = spec.params.expect_int("commit")? as u64;
                    TemporalPredicate::ExistedAt(commit)
                }
                "modified_in_range" => {
                    let start = spec.params.expect_int("start")? as u64;
                    let end = spec.params.expect_int("end")? as u64;
                    TemporalPredicate::ModifiedInRange(start, end)
                }
                other => return Err(anyhow!("Unknown temporal predicate: {}", other)),
            };
            
            Ok(Box::new(TemporalFilterStep::new(predicate, output_var)))
        },
    )?;

    registry.register(
        "temporal.mark_changed_nodes",
        StepMetadata {
            id: "temporal.mark_changed_nodes".to_string(),
            description: "Mark nodes that changed within time window".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let output_var = spec.params.expect_text("output")?.to_string();
            let change_type = spec.params.get_text("change_type").map(|s| s.to_string());
            Ok(Box::new(MarkChangedNodesStep::new(output_var, change_type)))
        },
    )?;

    Ok(())
}

static CORE_STEPS_INIT: Once = Once::new();

/// Ensure that the core step set is registered exactly once.
pub fn ensure_core_steps_registered() {
    CORE_STEPS_INIT.call_once(|| {
        register_core_steps(global_step_registry()).expect("register core steps");
    });
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
    fn load_node_attr_fetches_values() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        graph
            .set_node_attr(a, "score".into(), AttrValue::Float(0.5))
            .unwrap();
        graph
            .set_node_attr(b, "score".into(), AttrValue::Float(1.5))
            .unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = LoadNodeAttrStep::new(
            "scores",
            AttrName::from("score".to_string()),
            AlgorithmParamValue::Float(0.0),
        );
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("scores").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Float(0.5)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Float(1.5)));
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

    #[test]
    fn init_edges_sets_values() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let edge = graph.add_edge(a, b).unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = InitEdgesStep::new("weights", AlgorithmParamValue::Float(1.0));
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().edge_map("weights").unwrap();
        assert_eq!(map.get(&edge), Some(&AlgorithmParamValue::Float(1.0)));
    }

    #[test]
    fn load_edge_attr_fetches_values() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let edge = graph.add_edge(a, b).unwrap();
        graph
            .set_edge_attr(edge, "weight".into(), AttrValue::Float(2.5))
            .unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = LoadEdgeAttrStep::new(
            "weights",
            AttrName::from("weight".to_string()),
            AlgorithmParamValue::Float(0.0),
        );
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().edge_map("weights").unwrap();
        assert_eq!(map.get(&edge), Some(&AlgorithmParamValue::Float(2.5)));
    }

    #[test]
    fn attach_edge_attr_persists_values() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let edge = graph.add_edge(a, b).unwrap();
        let nodes: HashSet<NodeId> = [a, b].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        vars.set_edge_map(
            "weights",
            vec![(edge, AlgorithmParamValue::Float(3.0))]
                .into_iter()
                .collect(),
        );
        let mut scope = StepScope::new(&sg, &mut vars);
        let attr_name = AttrName::from("weight".to_string());
        let step = AttachEdgeAttrStep::new("weights", attr_name.clone());
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let graph = sg.graph();
        let graph_ref = graph.borrow();
        let attr = graph_ref.get_edge_attr(edge, &attr_name).unwrap().unwrap();
        assert_eq!(attr, AttrValue::Float(3.0));
    }

    #[test]
    fn node_degree_step_computes_degrees() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let sg = Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&sg, &mut vars);
        let step = NodeDegreeStep::new("degree");
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("degree").unwrap();
        assert_eq!(map.get(&a), Some(&AlgorithmParamValue::Int(1)));
        assert_eq!(map.get(&b), Some(&AlgorithmParamValue::Int(2)));
        assert_eq!(map.get(&c), Some(&AlgorithmParamValue::Int(1)));
    }

    #[test]
    fn normalize_node_values_step_scales_values() {
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
        let step =
            NormalizeNodeValuesStep::new("scores", "norm_scores", NormalizeMethod::Sum, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("norm_scores").unwrap();
        let total: f64 = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_node_values_step_max_method() {
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
        let step = NormalizeNodeValuesStep::new("scores", "max_scores", NormalizeMethod::Max, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("max_scores").unwrap();
        let max = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_node_values_step_minmax_method() {
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
        let step =
            NormalizeNodeValuesStep::new("scores", "range_scores", NormalizeMethod::MinMax, 1e-9);
        step.apply(&mut Context::new(), &mut scope).unwrap();
        let map = scope.variables().node_map("range_scores").unwrap();
        let min = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .fold(f64::INFINITY, f64::min);
        let max = map
            .values()
            .map(|v| match v {
                AlgorithmParamValue::Float(val) => *val,
                _ => panic!("expected float"),
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((min - 0.0).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ensure_core_steps_registers_once() {
        ensure_core_steps_registered();
        ensure_core_steps_registered();
        let registry = global_step_registry();
        assert!(registry.contains("core.init_nodes"));
        assert!(registry.contains("core.attach_edge_attr"));
    }
}
