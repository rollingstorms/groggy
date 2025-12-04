//! Core types and traits for the step system.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

static STEP_VARIABLES_COUNTER: AtomicU64 = AtomicU64::new(0);

use crate::subgraphs::Subgraph;
use crate::temporal::{TemporalIndex, TemporalSnapshot};
use crate::traits::SubgraphOperations;
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
#[allow(clippy::large_enum_variant)]
pub enum StepValue {
    /// Mapping from node id → value.
    NodeMap(HashMap<NodeId, AlgorithmParamValue>),
    /// Columnar node storage maintaining deterministic order.
    NodeColumn(NodeColumn),
    /// Mapping from edge id → value.
    EdgeMap(HashMap<EdgeId, AlgorithmParamValue>),
    /// Scalar helper value.
    Scalar(AlgorithmParamValue),
    /// Immutable temporal snapshot cached for reuse.
    Snapshot(TemporalSnapshot),
    /// Shared temporal index handle for cross-step reuse.
    #[allow(clippy::large_enum_variant)]
    TemporalIndex(Arc<TemporalIndex>),
}

#[derive(Clone, Debug)]
pub struct NodeColumn {
    nodes: Vec<NodeId>,
    values: Vec<AlgorithmParamValue>,
    index: HashMap<NodeId, usize>,
}

impl NodeColumn {
    pub fn new(nodes: Vec<NodeId>, values: Vec<AlgorithmParamValue>) -> Self {
        assert_eq!(nodes.len(), values.len());
        let mut index = HashMap::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().copied().enumerate() {
            index.insert(node, idx);
        }
        Self {
            nodes,
            values,
            index,
        }
    }

    pub fn get(&self, node: NodeId) -> Option<&AlgorithmParamValue> {
        self.index.get(&node).and_then(|&idx| self.values.get(idx))
    }

    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &AlgorithmParamValue)> {
        self.nodes.iter().copied().zip(self.values.iter())
    }

    pub fn values_mut(&mut self) -> &mut [AlgorithmParamValue] {
        &mut self.values
    }

    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    pub fn into_pairs(self) -> Vec<(NodeId, AlgorithmParamValue)> {
        self.nodes
            .into_iter()
            .zip(self.values.into_iter())
            .collect()
    }
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
#[derive(Clone, Debug)]
pub struct StepVariables {
    values: HashMap<String, StepValue>,
    neighbor_cache: Option<NeighborCache>,
    instance_id: u64,
}

impl Default for StepVariables {
    fn default() -> Self {
        let instance_id = STEP_VARIABLES_COUNTER.fetch_add(1, Ordering::SeqCst);
        if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
            eprintln!("[StepVariables::{}] Created new instance", instance_id);
        }
        Self {
            values: HashMap::new(),
            neighbor_cache: None,
            instance_id,
        }
    }
}

impl Drop for StepVariables {
    fn drop(&mut self) {
        if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
            eprintln!(
                "[StepVariables::{}] Dropping (had {} variables, neighbor_cache={})",
                self.instance_id,
                self.values.len(),
                self.neighbor_cache.is_some()
            );
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NeighborCache {
    index: HashMap<NodeId, usize>,
    adjacency: Vec<Vec<NodeId>>,
}

impl NeighborCache {
    fn new(nodes: Vec<NodeId>, adjacency: Vec<Vec<NodeId>>) -> Self {
        let mut index = HashMap::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().copied().enumerate() {
            index.insert(node, idx);
        }
        Self { index, adjacency }
    }

    pub fn neighbors(&self, node: NodeId) -> Option<&[NodeId]> {
        self.index
            .get(&node)
            .and_then(|&idx| self.adjacency.get(idx).map(|v| v.as_slice()))
    }
}

impl StepVariables {
    pub fn set_node_map(
        &mut self,
        name: impl Into<String>,
        map: HashMap<NodeId, AlgorithmParamValue>,
    ) {
        self.values.insert(name.into(), StepValue::NodeMap(map));
    }

    pub fn set_node_column(&mut self, name: impl Into<String>, column: NodeColumn) {
        self.values
            .insert(name.into(), StepValue::NodeColumn(column));
    }

    pub fn node_map(&self, name: &str) -> Result<&HashMap<NodeId, AlgorithmParamValue>> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| value.expect_node_map(name))
    }

    pub fn node_column(&self, name: &str) -> Result<&NodeColumn> {
        match self.values.get(name) {
            Some(StepValue::NodeColumn(column)) => Ok(column),
            Some(_) => Err(anyhow!("variable '{name}' is not stored as a node column")),
            None => Err(anyhow!("variable '{name}' not found")),
        }
    }

    pub fn node_column_mut(&mut self, name: &str) -> Result<&mut NodeColumn> {
        match self.values.get_mut(name) {
            Some(StepValue::NodeColumn(column)) => Ok(column),
            Some(_) => Err(anyhow!("variable '{name}' is not stored as a node column")),
            None => Err(anyhow!("variable '{name}' not found")),
        }
    }

    pub fn node_map_mut(
        &mut self,
        name: &str,
    ) -> Result<&mut HashMap<NodeId, AlgorithmParamValue>> {
        match self.values.get_mut(name) {
            Some(StepValue::NodeMap(map)) => Ok(map),
            Some(_) => Err(anyhow!("variable '{name}' is not stored as a node map")),
            None => Err(anyhow!("variable '{name}' not found")),
        }
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

    pub fn count(&self) -> usize {
        self.values.len()
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

    pub(crate) fn neighbor_cache(&mut self) -> Result<&NeighborCache> {
        if self.variables.neighbor_cache.is_none() {
            let nodes: Vec<NodeId> = self.subgraph.nodes().iter().copied().collect();
            if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
                eprintln!(
                    "[StepVariables::{}] Building neighbor_cache for {} nodes: {:?}",
                    self.variables.instance_id,
                    nodes.len(),
                    if nodes.len() <= 10 {
                        format!("{:?}", nodes)
                    } else {
                        format!("[{} nodes]", nodes.len())
                    }
                );
            }
            let mut adjacency = Vec::with_capacity(nodes.len());
            for &node in &nodes {
                let neighbors = self.subgraph.neighbors(node).map_err(|err| anyhow!(err))?;
                adjacency.push(neighbors);
            }
            self.variables.neighbor_cache = Some(NeighborCache::new(nodes, adjacency));
        } else if std::env::var("GROGGY_DEBUG_PIPELINE").is_ok() {
            eprintln!(
                "[StepVariables::{}] Reusing existing neighbor_cache",
                self.variables.instance_id
            );
        }
        Ok(self.variables.neighbor_cache.as_ref().unwrap())
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
