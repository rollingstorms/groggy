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
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};

use super::super::{AlgorithmParamValue, AlgorithmParams, Context, CostHint};

/// Conservative native execution capabilities used before a builder runs.
/// Custom declarations are trusted; this is not a callback effect sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicStepEffect {
    /// No certified atomic execution contract.
    Unsupported,
    /// Existing property-overlay calculations/attachments; reads base topology.
    Properties,
    /// Reads/attaches properties through a captured pending selection.
    PendingProperties,
    /// Produces selected immutable children, including pending property-only values.
    PendingSelections,
    /// Explicit source-bound property attachment; requires a persistent batch.
    StagedProperties,
    /// Reads immutable supplied/retained inputs, never working topology/properties.
    ImmutableInputs,
    /// Creates an independent read-only value from the local pending batch.
    StagedCapture,
    /// Stages topology in the local batch without mutating the owner.
    Topology,
}

/// Conservative composition for fixed child frames: topology stays outside them.
pub(crate) fn nested_effect(steps: &[Box<dyn Step>]) -> AtomicStepEffect {
    composed_effect(steps, false)
}

/// Root sequential loops may stage topology when every repeated read is pending-safe.
pub(crate) fn loop_effect(steps: &[Box<dyn Step>]) -> AtomicStepEffect {
    composed_effect(steps, true)
}

fn composed_effect(steps: &[Box<dyn Step>], topology: bool) -> AtomicStepEffect {
    let mut result = AtomicStepEffect::ImmutableInputs;
    for step in steps {
        match step.atomic_effect() {
            AtomicStepEffect::Unsupported => return AtomicStepEffect::Unsupported,
            AtomicStepEffect::Topology => {
                if !topology || result == AtomicStepEffect::Properties {
                    return AtomicStepEffect::Unsupported;
                }
                result = AtomicStepEffect::Topology;
            }
            AtomicStepEffect::Properties => {
                if matches!(
                    result,
                    AtomicStepEffect::StagedProperties | AtomicStepEffect::Topology
                ) {
                    return AtomicStepEffect::Unsupported;
                }
                result = AtomicStepEffect::Properties;
            }
            AtomicStepEffect::StagedCapture | AtomicStepEffect::StagedProperties => {
                if result == AtomicStepEffect::Properties {
                    return AtomicStepEffect::Unsupported;
                }
                if result != AtomicStepEffect::Topology {
                    result = AtomicStepEffect::StagedProperties;
                }
            }
            AtomicStepEffect::PendingSelections => {
                if matches!(
                    result,
                    AtomicStepEffect::ImmutableInputs | AtomicStepEffect::PendingProperties
                ) {
                    result = AtomicStepEffect::PendingSelections;
                }
            }
            AtomicStepEffect::PendingProperties => {
                if result == AtomicStepEffect::ImmutableInputs {
                    result = AtomicStepEffect::PendingProperties;
                }
            }
            AtomicStepEffect::ImmutableInputs => {}
        }
    }
    result
}

/// One fixed selected execution frame. All levels of the step interpreter use
/// the same pending-read refresh boundary rather than holding stale loop pins.
pub(crate) struct StepExecutor<'a> {
    base: &'a Subgraph,
    fixed: bool,
    pending: Option<(u64, Subgraph)>,
}
impl<'a> StepExecutor<'a> {
    pub(crate) fn new(base: &'a Subgraph) -> Self {
        Self {
            base,
            fixed: false,
            pending: None,
        }
    }
    pub(crate) fn for_frame(base: &'a Subgraph, fixed: bool) -> Self {
        Self {
            base,
            fixed,
            pending: None,
        }
    }

    pub(crate) fn apply(
        &mut self,
        ctx: &mut Context,
        variables: &mut StepVariables,
        step: &dyn Step,
    ) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("step execution cancelled"));
        }
        let effect = step.atomic_effect();
        if let Some(batch) = ctx.property_batch.as_ref() {
            if effect == AtomicStepEffect::Unsupported
                || (batch.has_topology() && effect == AtomicStepEffect::Properties)
            {
                return Err(anyhow!(
                    "Step '{}' does not support the current atomic scope",
                    step.id()
                ));
            }
        }
        let source = if matches!(
            effect,
            AtomicStepEffect::PendingProperties
                | AtomicStepEffect::PendingSelections
                | AtomicStepEffect::StagedProperties
        ) {
            if let Some(batch) = ctx.property_batch.as_mut().filter(|batch| {
                batch.has_topology()
                    || effect == AtomicStepEffect::PendingSelections
                    || self.base.frozen_view().is_some()
            }) {
                if batch.capture_is_current(self.base) {
                    self.base
                } else {
                    let revision = batch.read_revision();
                    if self
                        .pending
                        .as_ref()
                        .is_none_or(|(old, _)| *old != revision)
                    {
                        let view = if self.fixed {
                            batch.capture_scoped(self.base, true)?
                        } else {
                            batch.capture(self.base)?
                        };
                        self.pending = Some((
                            revision,
                            Subgraph::from_frozen_view(
                                self.base.graph(),
                                view,
                                "pending execution".into(),
                            ),
                        ));
                    }
                    &self
                        .pending
                        .as_ref()
                        .ok_or_else(|| anyhow!("pending execution scope was not installed"))?
                        .1
                }
            } else {
                self.base
            }
        } else {
            self.base
        };
        let mut scope = StepScope::for_frame(source, variables, self.fixed);
        step.apply(ctx, &mut scope)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("step execution cancelled"));
        }
        Ok(())
    }
}

/// Ordered checked adjacency for selected directional step reads.
pub(crate) struct SelectedCsr {
    pub nodes: Arc<[NodeId]>,
    pub positions: HashMap<NodeId, usize>,
    pub csr: crate::state::topology::Csr,
}

/// Trait implemented by all step primitives.
pub trait Step: Send + Sync {
    /// Effect classification for local atomic execution. Existing property
    /// certification remains conservative for extensions and nested executors.
    fn atomic_effect(&self) -> AtomicStepEffect {
        if self.supports_atomic_properties() {
            AtomicStepEffect::Properties
        } else {
            AtomicStepEffect::Unsupported
        }
    }

    /// Whether graph reads/writes follow the property-only atomic scope contract.
    /// Custom/unsupported steps reject before an atomic run executes any step.
    fn supports_atomic_properties(&self) -> bool {
        false
    }

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
    /// Immutable graph selection retaining its exact captured source.
    GraphView(crate::GraphView),
    /// Immutable intermediate table; transformations create another owned table.
    Table(Arc<crate::storage::table::BaseTable>),
    /// Mapping from node id → value.
    NodeMap(HashMap<NodeId, AlgorithmParamValue>),
    /// Columnar node storage maintaining deterministic order.
    NodeColumn(NodeColumn),
    /// Mapping from edge id → value.
    EdgeMap(HashMap<EdgeId, AlgorithmParamValue>),
    /// Collection of subgraphs (sampling output).
    SubgraphArray(Vec<Subgraph>),
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

    pub(crate) fn expect_subgraph_array(&self, name: &str) -> Result<&Vec<Subgraph>> {
        match self {
            StepValue::SubgraphArray(array) => Ok(array),
            _ => Err(anyhow!("variable '{name}' is not a subgraph array")),
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
    /// Inspect a typed intermediate without guessing its representation.
    pub fn value(&self, name: &str) -> Option<&StepValue> {
        self.values.get(name)
    }

    /// Install a completely prepared typed value. Replacing a name is explicit.
    pub fn set_value(&mut self, name: impl Into<String>, value: StepValue) {
        self.values.insert(name.into(), value);
    }

    /// Retrieve an immutable captured graph; wrong-type inputs fail explicitly.
    pub fn graph_view(&self, name: &str) -> Result<&crate::GraphView> {
        match self.values.get(name) {
            Some(StepValue::GraphView(view)) => Ok(view),
            Some(_) => Err(anyhow!("variable '{name}' is not a graph view")),
            None => Err(anyhow!("variable '{name}' not found")),
        }
    }

    /// Retrieve an immutable table intermediate, retaining its graph provenance.
    pub fn table(&self, name: &str) -> Result<&crate::storage::table::BaseTable> {
        match self.values.get(name) {
            Some(StepValue::Table(table)) => Ok(table),
            Some(_) => Err(anyhow!("variable '{name}' is not a table")),
            None => Err(anyhow!("variable '{name}' not found")),
        }
    }

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

    pub fn set_subgraph_array(&mut self, name: impl Into<String>, array: Vec<Subgraph>) {
        self.values
            .insert(name.into(), StepValue::SubgraphArray(array));
    }

    pub fn subgraph_array(&self, name: &str) -> Result<&Vec<Subgraph>> {
        self.values
            .get(name)
            .ok_or_else(|| anyhow!("variable '{name}' not found"))
            .and_then(|value| value.expect_subgraph_array(name))
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
    fixed: bool,
}

impl<'a> StepScope<'a> {
    pub fn new(subgraph: &'a Subgraph, variables: &'a mut StepVariables) -> Self {
        Self {
            subgraph,
            variables,
            fixed: false,
        }
    }

    pub(crate) fn for_frame(
        subgraph: &'a Subgraph,
        variables: &'a mut StepVariables,
        fixed: bool,
    ) -> Self {
        Self {
            subgraph,
            variables,
            fixed,
        }
    }
    pub(crate) fn fixed_selection(&self) -> bool {
        self.fixed
    }

    pub fn subgraph(&self) -> &Subgraph {
        self.subgraph
    }

    /// Build checked adjacency from the selected source, never retained owner
    /// pool endpoints when a frozen source supplies topology.
    pub(crate) fn selected_csr(
        &self,
        ctx: &Context,
        direction: super::direction::NeighborDirection,
    ) -> Result<SelectedCsr> {
        use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
        if ctx.is_cancelled() {
            return Err(anyhow!("selected topology cancelled"));
        }
        self.subgraph.validate_selected_read()?;
        let nodes = self.subgraph.ordered_nodes();
        let positions: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(row, &node)| (node, row))
            .collect();
        let edges = self.subgraph.ordered_edges();
        let mapping = direction.edge_mapping();
        let mut failure = None;
        let mut csr = Csr::default();
        build_csr_from_edges_with_scratch(
            &mut csr,
            nodes.len(),
            edges.iter().copied(),
            |node| positions.get(&node).copied(),
            |edge| {
                if ctx.is_cancelled() {
                    failure = Some(anyhow!("selected topology cancelled"));
                    return None;
                }
                match self.subgraph.edge_endpoints(edge) {
                    Ok(pair) => Some(mapping(pair)),
                    Err(error) => {
                        failure = Some(error.into());
                        None
                    }
                }
            },
            CsrOptions {
                add_reverse_edges: direction.add_reverse_edges(),
                sort_neighbors: true,
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(SelectedCsr {
            nodes,
            positions,
            csr,
        })
    }

    /// Read through the shared staged node-property overlay.
    pub fn node_attribute(
        &self,
        ctx: &Context,
        id: NodeId,
        name: &AttrName,
    ) -> crate::GraphResult<Option<AttrValue>> {
        ctx.node_attribute(self.subgraph, id, name)
    }
    /// Read through the shared staged edge-property overlay.
    pub fn edge_attribute(
        &self,
        ctx: &Context,
        id: EdgeId,
        name: &AttrName,
    ) -> crate::GraphResult<Option<AttrValue>> {
        ctx.edge_attribute(self.subgraph, id, name)
    }
    /// Validate and stage/publish a node column, or return an owned output.
    pub fn attach_node_column(
        &self,
        ctx: &mut Context,
        name: AttrName,
        values: Vec<(NodeId, AttrValue)>,
    ) -> Result<()> {
        self.validate_attachment_targets(true, values.iter().map(|(id, _)| *id))?;
        ctx.attach_node_column(self.subgraph, name, values)
    }
    /// Attach node columns through the canonical Context property boundary.
    pub fn attach_node_attrs(
        &self,
        ctx: &mut Context,
        attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> Result<()> {
        self.validate_attachment_targets(true, attrs.values().flatten().map(|(id, _)| *id))?;
        ctx.attach_node_attrs(self.subgraph, attrs)
    }
    /// Attach edge columns through the canonical Context property boundary.
    pub fn attach_edge_attrs(
        &self,
        ctx: &mut Context,
        attrs: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> Result<()> {
        self.validate_attachment_targets(false, attrs.values().flatten().map(|(id, _)| *id))?;
        ctx.attach_edge_attrs(self.subgraph, attrs)
    }

    /// Attach explicitly timed columns through the same fixed-frame validation.
    pub(crate) fn attach_attrs_at(
        &self,
        ctx: &mut Context,
        nodes: bool,
        attrs: HashMap<AttrName, Vec<(usize, AttrValue)>>,
        at: crate::TimestampMicros,
    ) -> Result<()> {
        self.validate_attachment_targets(nodes, attrs.values().flatten().map(|(id, _)| *id))?;
        ctx.attach_attrs_at(self.subgraph, nodes, attrs, at)
    }

    // Root execution keeps its established explicit owner-ID attachment contract.
    // Fixed children must constrain writes as well as reads, including live
    // property-overlay frames and transient owned outputs.
    fn validate_attachment_targets(
        &self,
        nodes: bool,
        targets: impl Iterator<Item = usize>,
    ) -> Result<()> {
        if self.fixed {
            let selected = if nodes {
                self.subgraph.nodes()
            } else {
                self.subgraph.edges()
            };
            for id in targets {
                if !selected.contains(&id) {
                    return Err(anyhow!(
                        "{} attachment target {id} is outside the fixed execution frame",
                        if nodes { "Node" } else { "Edge" }
                    ));
                }
            }
        }
        Ok(())
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

#[cfg(test)]
mod fixed_frame_tests {
    use super::*;
    use crate::{algorithms::property_batch::PropertyBatch, Graph};

    #[test]
    fn fixed_frames_reject_whole_foreign_columns_in_every_attachment_mode() {
        for mode in 0..3 {
            for kind in 0..3 {
                let mut graph = Graph::new_publishing();
                graph.insert_nodes(3).unwrap();
                graph.add_edges(&[(0, 1), (0, 1)]).unwrap();
                let head = graph.current_state_id();
                let owner = graph.into_shared();
                let selected =
                    Subgraph::new(owner.clone(), [0, 1].into(), [0].into(), "child".into());
                let mut variables = StepVariables::default();
                let scope = StepScope::for_frame(&selected, &mut variables, true);
                let mut ctx = Context::new();
                ctx.set_persist_results(mode != 0);
                if mode == 2 {
                    ctx.property_batch = Some(PropertyBatch::for_selection(
                        owner.borrow().write_coordinate(),
                        &selected,
                    ));
                }
                let result = match kind {
                    0 => scope.attach_node_column(
                        &mut ctx,
                        "x".into(),
                        vec![(0, AttrValue::Int(1)), (2, AttrValue::Int(2))],
                    ),
                    1 => scope.attach_node_attrs(
                        &mut ctx,
                        HashMap::from([(
                            "x".into(),
                            vec![(0, AttrValue::Int(1)), (2, AttrValue::Int(2))],
                        )]),
                    ),
                    _ => scope.attach_edge_attrs(
                        &mut ctx,
                        HashMap::from([(
                            "x".into(),
                            vec![(0, AttrValue::Int(1)), (1, AttrValue::Int(2))],
                        )]),
                    ),
                };
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("outside the fixed execution frame"));
                assert!(ctx.take_outputs().is_empty());
                if let Some(batch) = &ctx.property_batch {
                    assert!(batch.get(kind != 2, 0, &"x".into()).is_none());
                }
                assert_eq!(owner.borrow().current_state_id(), head);
                assert_eq!(owner.borrow().get_node_attr(0, &"x".into()).unwrap(), None);
                assert_eq!(owner.borrow().get_edge_attr(0, &"x".into()).unwrap(), None);
            }
        }
    }
}
