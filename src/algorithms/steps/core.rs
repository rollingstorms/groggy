//! Core types and traits for the step system.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::subgraphs::Subgraph;
use crate::temporal::{TemporalIndex, TemporalSnapshot};
use crate::types::{EdgeId, NodeId};

use super::super::{AlgorithmParamValue, AlgorithmParams, Context, CostHint};

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
#[derive(Clone, Debug)]
pub enum StepValue {
    /// Mapping from node id → value.
    NodeMap(HashMap<NodeId, AlgorithmParamValue>),
    /// Mapping from edge id → value.
    EdgeMap(HashMap<EdgeId, AlgorithmParamValue>),
    /// Scalar helper value.
    Scalar(AlgorithmParamValue),
    /// Immutable temporal snapshot cached for reuse.
    Snapshot(TemporalSnapshot),
    /// Shared temporal index handle for cross-step reuse.
    TemporalIndex(Arc<TemporalIndex>),
}

impl StepValue {
    pub(crate) fn expect_node_map(
        &self,
        name: &str,
    ) -> Result<&HashMap<NodeId, AlgorithmParamValue>> {
        match self {
            StepValue::NodeMap(map) => Ok(map),
            _ => Err(anyhow!("variable '{name}' is not a node map")),
        }
    }

    pub(crate) fn expect_edge_map(
        &self,
        name: &str,
    ) -> Result<&HashMap<EdgeId, AlgorithmParamValue>> {
        match self {
            StepValue::EdgeMap(map) => Ok(map),
            _ => Err(anyhow!("variable '{name}' is not an edge map")),
        }
    }

    pub(crate) fn expect_scalar(&self, name: &str) -> Result<&AlgorithmParamValue> {
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

    pub fn set_snapshot(&mut self, name: impl Into<String>, snapshot: TemporalSnapshot) {
        self.values
            .insert(name.into(), StepValue::Snapshot(snapshot));
    }

    pub fn snapshot(&self, name: &str) -> Result<&TemporalSnapshot> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| match value {
                StepValue::Snapshot(snapshot) => Ok(snapshot),
                _ => Err(anyhow!("variable '{name}' is not a temporal snapshot")),
            })
    }

    pub fn set_temporal_index(&mut self, name: impl Into<String>, index: Arc<TemporalIndex>) {
        self.values
            .insert(name.into(), StepValue::TemporalIndex(index));
    }

    pub fn temporal_index(&self, name: &str) -> Result<Arc<TemporalIndex>> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| match value {
                StepValue::TemporalIndex(index) => Ok(Arc::clone(index)),
                _ => Err(anyhow!("variable '{name}' is not a temporal index")),
            })
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

    pub fn iter(&self) -> impl Iterator<Item = (&String, &StepValue)> {
        self.values.iter()
    }
}

/// Mutable scope passed between steps, providing access to the subgraph and variables.
#[derive(Debug)]
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

    pub fn input(&self, _name: &str) -> Result<StepInput<'_>> {
        Ok(StepInput {
            subgraph: self.subgraph,
            variables: self.variables,
        })
    }

    pub fn edge_input(&self, _name: &str) -> Result<StepEdgeInput<'_>> {
        Ok(StepEdgeInput {
            subgraph: self.subgraph,
            variables: self.variables,
        })
    }

    pub fn node_ids(&self) -> impl Iterator<Item = &NodeId> {
        self.subgraph.nodes().iter()
    }

    pub fn edge_ids(&self) -> impl Iterator<Item = &EdgeId> {
        self.subgraph.edges().iter()
    }
}

/// Helper providing access to node-level input data.
#[derive(Debug)]
pub struct StepInput<'a> {
    pub subgraph: &'a Subgraph,
    pub variables: &'a StepVariables,
}

/// Helper providing access to edge-level input data.
#[derive(Debug)]
pub struct StepEdgeInput<'a> {
    pub subgraph: &'a Subgraph,
    pub variables: &'a StepVariables,
}

/// Factory signature for steps.
pub type StepFactory = dyn Fn(&StepSpec) -> Result<Box<dyn Step>> + Send + Sync;

pub(crate) struct StepEntry {
    pub factory: Arc<StepFactory>,
    pub metadata: StepMetadata,
}

/// Global registry of step constructors.
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
