//! Sampling step primitives.
//!
//! This module provides steps for:
//! - Random node/edge sampling with reproducible seeds
//! - Fraction-based and count-based sampling
//! - Reservoir sampling for streaming contexts
//!
//! All sampling operations support seeded RNG for reproducibility.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{EdgeId, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope, StepVariables};

/// Specification for sampling: either a fraction or absolute count.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SampleSpec {
    /// Sample a fraction of elements (0.0 to 1.0)
    Fraction { fraction: f64 },
    /// Sample an exact count of elements
    Count { count: usize },
}

impl SampleSpec {
    /// Calculate the number of elements to sample given a total count.
    fn calculate_sample_size(&self, total: usize) -> Result<usize> {
        match self {
            SampleSpec::Fraction { fraction } => {
                if !fraction.is_finite() || *fraction < 0.0 || *fraction > 1.0 {
                    bail!("Fraction must be between 0.0 and 1.0, got {}", fraction);
                }
                Ok((total as f64 * fraction).round() as usize)
            }
            SampleSpec::Count { count } => {
                if *count > total {
                    bail!("Cannot sample {} elements from {} total", count, total);
                }
                Ok(*count)
            }
        }
    }
}

/// Stable selected IDs, checked against the active owner and endpoint closure.
fn checked_population(ctx: &Context, scope: &StepScope<'_>, edges: bool) -> Result<Vec<usize>> {
    check_cancelled(ctx)?;
    let selection = scope.subgraph();
    let graph = selection.graph();
    let graph = graph.borrow();
    for &node in selection.ordered_nodes().iter() {
        check_cancelled(ctx)?;
        if !graph.has_node(node) {
            bail!("Sampling node {} is inactive", node);
        }
    }
    if !edges {
        return Ok(selection.ordered_nodes().to_vec());
    }
    let ids = selection.ordered_edges().to_vec();
    for &edge in &ids {
        check_cancelled(ctx)?;
        if !graph.has_edge(edge) {
            bail!("Sampling edge {} is inactive", edge);
        }
        let (source, target) = graph.edge_endpoints(edge)?;
        if !selection.node_set().contains(&source) || !selection.node_set().contains(&target) {
            bail!("Sampling edge {} endpoint is outside selection", edge);
        }
    }
    Ok(ids)
}

fn check_cancelled(ctx: &Context) -> Result<()> {
    if ctx.is_cancelled() {
        bail!("sampling cancelled");
    }
    Ok(())
}

/// Algorithm R over stable IDs; one local RNG, bounded by available candidates.
fn sample_mask(
    ctx: &Context,
    ids: &[usize],
    count: usize,
    seed: Option<u64>,
) -> Result<HashMap<usize, AlgorithmParamValue>> {
    check_cancelled(ctx)?;
    let count = count.min(ids.len());
    let mut rng = seed.map(fastrand::Rng::with_seed).unwrap_or_default();
    let mut reservoir = Vec::new();
    reservoir.try_reserve_exact(count)?;
    if count != 0 {
        for (index, &id) in ids.iter().enumerate() {
            check_cancelled(ctx)?;
            if index < count {
                reservoir.push(id);
            } else {
                let replacement = rng.usize(0..=index);
                if replacement < count {
                    reservoir[replacement] = id;
                }
            }
        }
    }
    let mut result = HashMap::new();
    result.try_reserve(reservoir.len())?;
    for id in reservoir {
        check_cancelled(ctx)?;
        result.insert(id, AlgorithmParamValue::Int(1));
    }
    check_cancelled(ctx)?;
    Ok(result)
}

/// Sample nodes randomly from the subgraph.
///
/// Returns a map with sampled nodes marked as 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleNodesStep {
    spec: SampleSpec,
    seed: Option<u64>,
    target: String,
}

impl SampleNodesStep {
    pub fn new(spec: SampleSpec, seed: Option<u64>, target: impl Into<String>) -> Self {
        Self {
            spec,
            seed,
            target: target.into(),
        }
    }
}

impl Step for SampleNodesStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.sample_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Sample nodes randomly with optional seed".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let ids = checked_population(ctx, scope, false)?;
        let count = self.spec.calculate_sample_size(ids.len())?;
        let result = sample_mask(ctx, &ids, count, self.seed)?;
        check_cancelled(ctx)?;
        scope.variables_mut().set_node_map(&self.target, result);
        Ok(())
    }
}

/// Sample edges randomly from the subgraph.
///
/// Returns a map with sampled edges marked as 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleEdgesStep {
    spec: SampleSpec,
    seed: Option<u64>,
    target: String,
}

impl SampleEdgesStep {
    pub fn new(spec: SampleSpec, seed: Option<u64>, target: impl Into<String>) -> Self {
        Self {
            spec,
            seed,
            target: target.into(),
        }
    }
}

impl Step for SampleEdgesStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.sample_edges"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Sample edges randomly with optional seed".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let ids = checked_population(ctx, scope, true)?;
        let count = self.spec.calculate_sample_size(ids.len())?;
        let result = sample_mask(ctx, &ids, count, self.seed)?;
        check_cancelled(ctx)?;
        scope.variables_mut().set_edge_map(&self.target, result);
        Ok(())
    }
}

/// Invocation-bounded selected topology; never induce from the whole owner.
struct SelectionTopology {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    rows: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,
    endpoints: HashMap<EdgeId, (NodeId, NodeId)>,
}
impl SelectionTopology {
    fn prepare(ctx: &Context, scope: &StepScope<'_>) -> Result<Self> {
        let edges = checked_population(ctx, scope, true)?;
        let nodes = scope.subgraph().ordered_nodes().to_vec();
        let mut rows: HashMap<_, Vec<_>> = nodes.iter().map(|&node| (node, Vec::new())).collect();
        let mut endpoints = HashMap::with_capacity(edges.len());
        let graph = scope.subgraph().graph();
        let graph = graph.borrow();
        for &edge in &edges {
            check_cancelled(ctx)?;
            let (source, target) = graph.edge_endpoints(edge)?;
            rows.get_mut(&source)
                .ok_or_else(|| anyhow::anyhow!("Missing selected source"))?
                .push((target, edge));
            if graph.is_undirected() && source != target {
                rows.get_mut(&target)
                    .ok_or_else(|| anyhow::anyhow!("Missing selected target"))?
                    .push((source, edge));
            }
            endpoints.insert(edge, (source, target));
        }
        for row in rows.values_mut() {
            row.sort_unstable();
        }
        Ok(Self {
            nodes,
            edges,
            rows,
            endpoints,
        })
    }

    fn validate(&self, ctx: &Context, scope: &StepScope<'_>, selector: &Subgraph) -> Result<()> {
        check_cancelled(ctx)?;
        if !Rc::ptr_eq(&scope.subgraph().graph(), &selector.graph()) {
            bail!("Subgraph selector belongs to a different graph owner");
        }
        for &node in selector.node_set() {
            check_cancelled(ctx)?;
            if !self.rows.contains_key(&node) {
                bail!(
                    "Subgraph selector node {} is outside active selection",
                    node
                );
            }
        }
        for &edge in selector.edge_set() {
            check_cancelled(ctx)?;
            let (source, target) = self.endpoints.get(&edge).ok_or_else(|| {
                anyhow::anyhow!(
                    "Subgraph selector edge {} is outside active selection",
                    edge
                )
            })?;
            if !selector.node_set().contains(source) || !selector.node_set().contains(target) {
                bail!(
                    "Subgraph selector edge {} endpoint is outside its node selection",
                    edge
                );
            }
        }
        Ok(())
    }

    fn induced(&self, ctx: &Context, nodes: &HashSet<NodeId>) -> Result<HashSet<EdgeId>> {
        let mut edges = HashSet::new();
        for node in nodes {
            check_cancelled(ctx)?;
            let row = self
                .rows
                .get(node)
                .ok_or_else(|| anyhow::anyhow!("Node {} is outside active selection", node))?;
            for &(neighbor, edge) in row {
                check_cancelled(ctx)?;
                if nodes.contains(&neighbor) {
                    edges.insert(edge);
                }
            }
        }
        Ok(edges)
    }
}

fn install_subgraphs(
    ctx: &Context,
    scope: &mut StepScope<'_>,
    target: &str,
    outputs: Vec<Subgraph>,
) -> Result<()> {
    check_cancelled(ctx)?;
    scope.variables_mut().set_subgraph_array(target, outputs);
    Ok(())
}

/// Iterate nodes and emit one subgraph per node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterateNodesStep {
    target: String,
}

impl IterateNodesStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for IterateNodesStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "sample.iterate_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Create one subgraph per node in the input".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let topology = SelectionTopology::prepare(ctx, scope)?;
        let graph = scope.subgraph().graph();
        let mut outputs = Vec::with_capacity(topology.nodes.len());
        for &node in &topology.nodes {
            check_cancelled(ctx)?;
            let nodes = HashSet::from([node]);
            let edges = topology.induced(ctx, &nodes)?;
            outputs.push(Subgraph::new(
                graph.clone(),
                nodes,
                edges,
                "sample_iterate_nodes".into(),
            ));
        }
        install_subgraphs(ctx, scope, &self.target, outputs)
    }
}

/// Iterate edges and emit one subgraph per edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterateEdgesStep {
    target: String,
}

impl IterateEdgesStep {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

impl Step for IterateEdgesStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "sample.iterate_edges"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Create one subgraph per edge in the input".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let topology = SelectionTopology::prepare(ctx, scope)?;
        let graph = scope.subgraph().graph();
        let mut outputs = Vec::with_capacity(topology.edges.len());
        for &edge in &topology.edges {
            check_cancelled(ctx)?;
            let (source, target) = topology.endpoints[&edge];
            outputs.push(Subgraph::new(
                graph.clone(),
                HashSet::from([source, target]),
                HashSet::from([edge]),
                "sample_iterate_edges".into(),
            ));
        }
        install_subgraphs(ctx, scope, &self.target, outputs)
    }
}

/// Expand each subgraph in an array to its k-hop neighborhood.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborsStep {
    source: String,
    hops: usize,
    target: String,
}

impl NeighborsStep {
    pub fn new(source: impl Into<String>, hops: usize, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            hops,
            target: target.into(),
        }
    }
}

impl Step for NeighborsStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "sample.neighbors"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Expand each seed subgraph to its k-hop neighborhood".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let topology = SelectionTopology::prepare(ctx, scope)?;
        let seeds: Vec<HashSet<NodeId>> =
            if let Ok(seeds) = scope.variables().subgraph_array(&self.source) {
                for seed in seeds {
                    topology.validate(ctx, scope, seed)?;
                }
                seeds.iter().map(|seed| seed.node_set().clone()).collect()
            } else if let Ok(nodes) = scope.variables().node_map(&self.source) {
                let mut ids: Vec<_> = nodes.keys().copied().collect();
                ids.sort_unstable();
                let mut seeds = Vec::with_capacity(ids.len());
                for node in ids {
                    check_cancelled(ctx)?;
                    if !topology.rows.contains_key(&node) {
                        bail!("Neighborhood node {} is outside active selection", node);
                    }
                    seeds.push(HashSet::from([node]));
                }
                seeds
            } else {
                let edges = scope.variables().edge_map(&self.source)?;
                let mut ids: Vec<_> = edges.keys().copied().collect();
                ids.sort_unstable();
                let mut seeds = Vec::with_capacity(ids.len());
                for edge in ids {
                    check_cancelled(ctx)?;
                    let &(source, target) = topology.endpoints.get(&edge).ok_or_else(|| {
                        anyhow::anyhow!("Neighborhood edge {} is outside active selection", edge)
                    })?;
                    seeds.push(HashSet::from([source, target]));
                }
                seeds
            };
        let graph = scope.subgraph().graph();
        let mut outputs = Vec::with_capacity(seeds.len());
        for mut visited in seeds {
            check_cancelled(ctx)?;
            let mut frontier: Vec<_> = visited.iter().copied().collect();
            frontier.sort_unstable();
            for _ in 0..self.hops {
                if frontier.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                for node in frontier {
                    check_cancelled(ctx)?;
                    for &(neighbor, _) in &topology.rows[&node] {
                        check_cancelled(ctx)?;
                        if visited.insert(neighbor) {
                            next.push(neighbor);
                        }
                    }
                }
                next.sort_unstable();
                frontier = next;
            }
            let edges = topology.induced(ctx, &visited)?;
            outputs.push(Subgraph::new(
                graph.clone(),
                visited,
                edges,
                format!("sample_neighbors_hops_{}", self.hops),
            ));
        }
        install_subgraphs(ctx, scope, &self.target, outputs)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmitMode {
    PerSeed,
    Unified,
}

/// Emit subgraphs from a selection or existing subgraph array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmitSubgraphsStep {
    source: String,
    target: String,
    mode: EmitMode,
    induced: bool,
}

impl EmitSubgraphsStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        mode: EmitMode,
        induced: bool,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            mode,
            induced,
        }
    }
}

impl Step for EmitSubgraphsStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "sample.emit_subgraphs"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Emit subgraphs from a selection".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let topology = SelectionTopology::prepare(ctx, scope)?;
        let graph = scope.subgraph().graph();
        let mut selections: Vec<(HashSet<NodeId>, HashSet<EdgeId>)> = Vec::new();
        if let Ok(array) = scope.variables().subgraph_array(&self.source) {
            for selector in array {
                topology.validate(ctx, scope, selector)?;
                selections.push((selector.node_set().clone(), selector.edge_set().clone()));
            }
            if matches!(self.mode, EmitMode::Unified) && !selections.is_empty() {
                let mut nodes = HashSet::new();
                let mut edges = HashSet::new();
                for (part_nodes, part_edges) in selections {
                    nodes.extend(part_nodes);
                    edges.extend(part_edges);
                }
                selections = vec![(nodes, edges)];
            }
        } else if let Ok(map) = scope.variables().node_map(&self.source) {
            let mut nodes: Vec<_> = map.keys().copied().collect();
            nodes.sort_unstable();
            for node in &nodes {
                check_cancelled(ctx)?;
                if !topology.rows.contains_key(node) {
                    bail!("Emission node {} is outside active selection", node);
                }
            }
            if matches!(self.mode, EmitMode::Unified) {
                selections.push((nodes.into_iter().collect(), HashSet::new()));
            } else {
                selections.extend(
                    nodes
                        .into_iter()
                        .map(|node| (HashSet::from([node]), HashSet::new())),
                );
            }
        } else {
            let map = scope.variables().edge_map(&self.source)?;
            let mut ids: Vec<_> = map.keys().copied().collect();
            ids.sort_unstable();
            let mut nodes = HashSet::new();
            let mut edges = HashSet::new();
            for edge in ids {
                check_cancelled(ctx)?;
                let &(source, target) = topology.endpoints.get(&edge).ok_or_else(|| {
                    anyhow::anyhow!("Emission edge {} is outside active selection", edge)
                })?;
                if matches!(self.mode, EmitMode::PerSeed) {
                    selections.push((HashSet::from([source, target]), HashSet::from([edge])));
                } else {
                    nodes.extend([source, target]);
                    edges.insert(edge);
                }
            }
            if matches!(self.mode, EmitMode::Unified) {
                selections.push((nodes, edges));
            }
        }
        let mut outputs = Vec::with_capacity(selections.len());
        for (nodes, original_edges) in selections {
            check_cancelled(ctx)?;
            let edges = if self.induced {
                topology.induced(ctx, &nodes)?
            } else {
                original_edges
            };
            outputs.push(Subgraph::new(
                graph.clone(),
                nodes,
                edges,
                "sample_emit_subgraphs".into(),
            ));
        }
        install_subgraphs(ctx, scope, &self.target, outputs)
    }
}

/// Map a sub-pipeline over each subgraph in a subgraph array.
pub struct ForEachSubgraphStep {
    source: String,
    target: String,
    body_steps: Vec<Box<dyn Step>>,
}

impl ForEachSubgraphStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        body_steps: Vec<Box<dyn Step>>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            body_steps,
        }
    }
}

impl Step for ForEachSubgraphStep {
    fn id(&self) -> &'static str {
        "sample.for_each"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Map a sub-pipeline over subgraphs".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let inputs = scope.variables().subgraph_array(&self.source)?;
        let mut outputs = Vec::with_capacity(inputs.len());

        for sg in inputs {
            let mut vars = StepVariables::default();
            for step in &self.body_steps {
                let mut inner_scope = StepScope::new(sg, &mut vars);
                step.apply(ctx, &mut inner_scope)?;
            }

            if let Ok(out_array) = vars.subgraph_array("__sample_output__") {
                if let Some(first) = out_array.first() {
                    outputs.push(first.clone());
                    continue;
                }
            }

            outputs.push(sg.clone());
        }

        scope
            .variables_mut()
            .set_subgraph_array(self.target.clone(), outputs);
        Ok(())
    }
}

/// Reservoir sample from a streaming source.
///
/// Samples k elements from a node or edge map variable,
/// maintaining uniform probability even when total size is unknown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirSampleStep {
    source: String,
    k: usize,
    seed: Option<u64>,
    target: String,
    entity_type: EntityType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Nodes,
    Edges,
}

impl ReservoirSampleStep {
    pub fn new(
        source: impl Into<String>,
        k: usize,
        seed: Option<u64>,
        target: impl Into<String>,
        entity_type: EntityType,
    ) -> Self {
        Self {
            source: source.into(),
            k,
            seed,
            target: target.into(),
            entity_type,
        }
    }
}

impl Step for ReservoirSampleStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.reservoir_sample"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Reservoir sample k elements from a map variable".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let edges = matches!(self.entity_type, EntityType::Edges);
        let population = checked_population(ctx, scope, edges)?;
        let source = if edges {
            scope.variables().edge_map(&self.source)?
        } else {
            scope.variables().node_map(&self.source)?
        };
        let mut ids: Vec<_> = source.keys().copied().collect();
        ids.sort_unstable();
        // Validate every source ID even for k=0; never silently drop foreign rows.
        let mut eligible = population.iter().peekable();
        for &id in &ids {
            check_cancelled(ctx)?;
            while eligible.peek().is_some_and(|&&candidate| candidate < id) {
                eligible.next();
            }
            if eligible.peek().copied().copied() != Some(id) {
                bail!("Reservoir source ID {} is outside active selection", id);
            }
        }
        let result = sample_mask(ctx, &ids, self.k, self.seed)?;
        check_cancelled(ctx)?;
        if edges {
            scope.variables_mut().set_edge_map(&self.target, result);
        } else {
            scope.variables_mut().set_node_map(&self.target, result);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_spec_fraction() {
        let spec = SampleSpec::Fraction { fraction: 0.5 };
        assert_eq!(spec.calculate_sample_size(100).unwrap(), 50);
        assert_eq!(spec.calculate_sample_size(10).unwrap(), 5);
    }

    #[test]
    fn test_sample_spec_count() {
        let spec = SampleSpec::Count { count: 10 };
        assert_eq!(spec.calculate_sample_size(100).unwrap(), 10);
        assert_eq!(spec.calculate_sample_size(10).unwrap(), 10);
    }

    #[test]
    fn test_sample_spec_invalid_fraction() {
        let spec = SampleSpec::Fraction { fraction: 1.5 };
        assert!(spec.calculate_sample_size(100).is_err());
    }

    #[test]
    fn test_sample_spec_count_too_large() {
        let spec = SampleSpec::Count { count: 150 };
        assert!(spec.calculate_sample_size(100).is_err());
    }

    #[test]
    fn test_sample_reproducibility() {
        // Same seed should produce same sample
        let spec1 = SampleSpec::Fraction { fraction: 0.5 };
        let spec2 = SampleSpec::Fraction { fraction: 0.5 };

        // This test would need a full integration test setup
        // Just verify specs are equal for now
        assert!(matches!(spec1, SampleSpec::Fraction { fraction: 0.5 }));
        assert!(matches!(spec2, SampleSpec::Fraction { fraction: 0.5 }));
    }
}

#[cfg(test)]
mod checked_tests {
    use super::*;
    use crate::Graph;
    use std::{
        cell::RefCell,
        rc::Rc,
        sync::{atomic::AtomicBool, Arc},
    };

    fn selection() -> Subgraph {
        let mut graph = Graph::new_directed();
        graph.add_nodes(6);
        for (source, target) in [(1, 3), (1, 3), (3, 3), (5, 1), (0, 1)] {
            graph.add_edge(source, target).expect("edge");
        }
        Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            [1, 3, 5].into_iter().collect(),
            "selection".into(),
        )
        .expect("selection")
    }

    #[test]
    fn seeded_sampling_is_repeatable_isolated_and_keeps_selected_entity_identity() {
        let selection = selection();
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&selection, &mut vars);
        let step = SampleNodesStep::new(SampleSpec::Count { count: 2 }, Some(19), "sample");
        fastrand::seed(43);
        let expected = fastrand::u64(..);
        fastrand::seed(43);
        step.apply(&mut Context::new(), &mut scope).expect("sample");
        assert_eq!(
            fastrand::u64(..),
            expected,
            "seeded sampler must not reseed or consume shared RNG"
        );
        let first = scope
            .variables()
            .node_map("sample")
            .expect("sample")
            .clone();
        step.apply(&mut Context::new(), &mut scope).expect("repeat");
        assert_eq!(
            scope.variables().node_map("sample").expect("sample"),
            &first
        );
        assert_eq!(first.len(), 2);
        assert!(first.keys().all(|id| [1, 3, 5].contains(id)));
        SampleEdgesStep::new(SampleSpec::Fraction { fraction: 1.0 }, Some(19), "edges")
            .apply(&mut Context::new(), &mut scope)
            .expect("edges");
        let edges = scope.variables().edge_map("edges").expect("edges");
        assert_eq!(edges.len(), 4);
        assert!(!edges.contains_key(&4));
        assert!(edges.contains_key(&0) && edges.contains_key(&1) && edges.contains_key(&2));
    }

    #[test]
    fn reservoir_order_is_stable_caps_allocation_and_validates_even_zero_requests() {
        let selection = selection();
        let mut vars = StepVariables::default();
        let source: HashMap<_, _> = [5, 1, 3]
            .into_iter()
            .map(|id| (id, AlgorithmParamValue::Int(0)))
            .collect();
        vars.set_node_map("source", source.clone());
        let mut scope = StepScope::new(&selection, &mut vars);
        let step = ReservoirSampleStep::new("source", 2, Some(7), "sample", EntityType::Nodes);
        step.apply(&mut Context::new(), &mut scope).expect("sample");
        let expected = scope
            .variables()
            .node_map("sample")
            .expect("sample")
            .clone();
        scope.variables_mut().set_node_map(
            "source",
            [1, 3, 5]
                .into_iter()
                .map(|id| (id, AlgorithmParamValue::None))
                .collect(),
        );
        step.apply(&mut Context::new(), &mut scope).expect("repeat");
        assert_eq!(
            scope.variables().node_map("sample").expect("repeat"),
            &expected
        );
        ReservoirSampleStep::new("source", usize::MAX, Some(7), "source", EntityType::Nodes)
            .apply(&mut Context::new(), &mut scope)
            .expect("oversized cap");
        assert_eq!(
            scope.variables().node_map("source").expect("source").len(),
            3
        );
        scope
            .variables_mut()
            .set_node_map("source", HashMap::from([(0, AlgorithmParamValue::Int(1))]));
        assert!(
            ReservoirSampleStep::new("source", 0, Some(7), "sample", EntityType::Nodes)
                .apply(&mut Context::new(), &mut scope)
                .is_err()
        );
        assert_eq!(
            scope.variables().node_map("sample").expect("unchanged"),
            &expected
        );
    }

    #[test]
    fn invalid_fraction_inactive_selection_and_cancellation_never_replace_outputs() {
        for fraction in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert!(SampleSpec::Fraction { fraction }
                .calculate_sample_size(0)
                .is_err());
        }
        let selection = selection();
        let mut vars = StepVariables::default();
        let original = HashMap::from([(1, AlgorithmParamValue::Int(9))]);
        vars.set_node_map("sample", original.clone());
        let mut scope = StepScope::new(&selection, &mut vars);
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        assert!(
            SampleNodesStep::new(SampleSpec::Count { count: 0 }, None, "sample")
                .apply(&mut ctx, &mut scope)
                .is_err()
        );
        assert_eq!(
            scope.variables().node_map("sample").expect("unchanged"),
            &original
        );
        selection
            .graph()
            .borrow_mut()
            .remove_node(5)
            .expect("remove");
        assert!(
            SampleNodesStep::new(SampleSpec::Count { count: 0 }, None, "sample")
                .apply(&mut Context::new(), &mut scope)
                .is_err()
        );
        assert_eq!(
            scope.variables().node_map("sample").expect("unchanged"),
            &original
        );
    }
}

#[cfg(test)]
mod subgraph_sampling_tests {
    use super::*;
    use crate::Graph;
    use std::{
        cell::RefCell,
        sync::{atomic::AtomicBool, Arc},
    };

    fn selection(directed: bool) -> Subgraph {
        let mut graph = if directed {
            Graph::new_directed()
        } else {
            Graph::new_undirected()
        };
        graph.add_nodes(6);
        for (u, v) in [(1, 3), (1, 3), (3, 3), (5, 1), (0, 1), (1, 1)] {
            graph.add_edge(u, v).expect("edge");
        }
        Subgraph::new(
            Rc::new(RefCell::new(graph)),
            HashSet::from([1, 3, 5]),
            HashSet::from([0, 2, 3]),
            "bounded".into(),
        )
    }
    fn contents(array: &[Subgraph]) -> Vec<(Vec<NodeId>, Vec<EdgeId>)> {
        array
            .iter()
            .map(|sg| (sg.ordered_nodes().to_vec(), sg.ordered_edges().to_vec()))
            .collect()
    }

    #[test]
    fn iteration_is_ordered_and_never_reintroduces_unselected_owner_edges() {
        let selected = selection(true);
        let mut vars = StepVariables::default();
        let mut scope = StepScope::new(&selected, &mut vars);
        IterateNodesStep::new("nodes")
            .apply(&mut Context::new(), &mut scope)
            .expect("nodes");
        assert_eq!(
            contents(scope.variables().subgraph_array("nodes").expect("nodes")),
            [(vec![1], vec![]), (vec![3], vec![2]), (vec![5], vec![])]
        );
        IterateEdgesStep::new("edges")
            .apply(&mut Context::new(), &mut scope)
            .expect("edges");
        assert_eq!(
            contents(scope.variables().subgraph_array("edges").expect("edges")),
            [
                (vec![1, 3], vec![0]),
                (vec![3], vec![2]),
                (vec![1, 5], vec![3])
            ]
        );
    }

    #[test]
    fn neighborhoods_are_outgoing_or_undirected_and_induced_only_within_invocation() {
        for directed in [false, true] {
            let selected = selection(directed);
            let mut vars = StepVariables::default();
            let seed = Subgraph::new(
                selected.graph(),
                HashSet::from([1]),
                HashSet::new(),
                "seed".into(),
            );
            vars.set_subgraph_array("seeds", vec![seed]);
            let mut scope = StepScope::new(&selected, &mut vars);
            NeighborsStep::new("seeds", 0, "zero")
                .apply(&mut Context::new(), &mut scope)
                .expect("zero");
            assert_eq!(
                contents(scope.variables().subgraph_array("zero").expect("zero")),
                [(vec![1], vec![])]
            );
            NeighborsStep::new("seeds", 1, "one")
                .apply(&mut Context::new(), &mut scope)
                .expect("one");
            let expected = if directed {
                (vec![1, 3], vec![0, 2])
            } else {
                (vec![1, 3, 5], vec![0, 2, 3])
            };
            assert_eq!(
                contents(scope.variables().subgraph_array("one").expect("one")),
                [expected]
            );
            NeighborsStep::new("seeds", usize::MAX, "all")
                .apply(&mut Context::new(), &mut scope)
                .expect("saturates");
            assert!(scope.variables().subgraph_array("all").expect("all")[0]
                .node_set()
                .is_subset(selected.node_set()));
        }
    }

    #[test]
    fn emission_honors_induced_for_each_source_and_mode() {
        let selected = selection(true);
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "nodes",
            HashMap::from([
                (1, AlgorithmParamValue::Int(0)),
                (3, AlgorithmParamValue::None),
            ]),
        );
        vars.set_edge_map("edges", HashMap::from([(0, AlgorithmParamValue::Int(0))]));
        vars.set_subgraph_array(
            "array",
            vec![Subgraph::new(
                selected.graph(),
                HashSet::from([1, 3]),
                HashSet::from([0]),
                "seed".into(),
            )],
        );
        let mut scope = StepScope::new(&selected, &mut vars);
        for induced in [false, true] {
            for source in ["nodes", "edges", "array"] {
                for mode in [EmitMode::PerSeed, EmitMode::Unified] {
                    EmitSubgraphsStep::new(source, "result", mode, induced)
                        .apply(&mut Context::new(), &mut scope)
                        .expect("emit");
                    let output =
                        contents(scope.variables().subgraph_array("result").expect("result"));
                    if source == "nodes" && matches!(mode, EmitMode::PerSeed) {
                        assert_eq!(
                            output,
                            [
                                (vec![1], vec![]),
                                (vec![3], if induced { vec![2] } else { vec![] })
                            ]
                        );
                    } else {
                        let edges = if induced {
                            vec![0, 2]
                        } else if source == "nodes" {
                            vec![]
                        } else {
                            vec![0]
                        };
                        assert_eq!(output, [(vec![1, 3], edges)]);
                    }
                }
            }
        }
    }

    #[test]
    fn foreign_owner_escaping_or_malformed_seeds_and_cancellation_preserve_outputs() {
        let selected = selection(true);
        let mut vars = StepVariables::default();
        vars.set_subgraph_array("result", vec![selected.clone()]);
        let original = contents(vars.subgraph_array("result").expect("result"));
        let foreign = selection(true);
        let bad_seeds = [
            foreign,
            Subgraph::new(
                selected.graph(),
                HashSet::from([0]),
                HashSet::new(),
                "outside".into(),
            ),
            Subgraph::new(
                selected.graph(),
                HashSet::from([1]),
                HashSet::from([0]),
                "malformed".into(),
            ),
        ];
        for seed in bad_seeds {
            vars.set_subgraph_array("seeds", vec![seed]);
            let mut scope = StepScope::new(&selected, &mut vars);
            assert!(NeighborsStep::new("seeds", 0, "result")
                .apply(&mut Context::new(), &mut scope)
                .is_err());
            assert!(
                EmitSubgraphsStep::new("seeds", "result", EmitMode::PerSeed, false)
                    .apply(&mut Context::new(), &mut scope)
                    .is_err()
            );
            assert_eq!(
                contents(
                    scope
                        .variables()
                        .subgraph_array("result")
                        .expect("unchanged")
                ),
                original
            );
        }
        let mut ctx = Context::with_cancel_token(Arc::new(AtomicBool::new(true)));
        let mut scope = StepScope::new(&selected, &mut vars);
        assert!(IterateEdgesStep::new("result")
            .apply(&mut ctx, &mut scope)
            .is_err());
        assert_eq!(
            contents(
                scope
                    .variables()
                    .subgraph_array("result")
                    .expect("unchanged")
            ),
            original
        );
    }

    #[test]
    fn empty_map_and_empty_array_unified_conventions_remain_explicit() {
        let selected = selection(true);
        let mut vars = StepVariables::default();
        vars.set_node_map("map", HashMap::new());
        vars.set_subgraph_array("array", vec![]);
        let mut scope = StepScope::new(&selected, &mut vars);
        for source in ["map", "array"] {
            EmitSubgraphsStep::new(source, "result", EmitMode::Unified, true)
                .apply(&mut Context::new(), &mut scope)
                .expect("empty");
            assert_eq!(
                scope
                    .variables()
                    .subgraph_array("result")
                    .expect("result")
                    .len(),
                usize::from(source == "map")
            );
        }
    }
}
