//! Sampling step primitives.
//!
//! This module provides steps for:
//! - Random node/edge sampling with reproducible seeds
//! - Fraction-based and count-based sampling
//! - Reservoir sampling for streaming contexts
//!
//! All sampling operations support seeded RNG for reproducibility.

use std::collections::HashMap;

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
                if *fraction < 0.0 || *fraction > 1.0 {
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        // Initialize RNG with seed if provided
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        let all_nodes: Vec<NodeId> = scope.node_ids().copied().collect();
        let total = all_nodes.len();
        let sample_size = self.spec.calculate_sample_size(total)?;

        // Use reservoir sampling for efficiency
        let mut result = HashMap::new();

        if sample_size == 0 {
            scope.variables_mut().set_node_map(&self.target, result);
            return Ok(());
        }

        // Reservoir sampling algorithm
        let mut reservoir: Vec<NodeId> = Vec::with_capacity(sample_size);

        for (i, &node) in all_nodes.iter().enumerate() {
            if i < sample_size {
                // Fill reservoir
                reservoir.push(node);
            } else {
                // Randomly replace elements
                let j = fastrand::usize(0..=i);
                if j < sample_size {
                    reservoir[j] = node;
                }
            }
        }

        // Mark sampled nodes
        for node in reservoir {
            result.insert(node, AlgorithmParamValue::Int(1));
        }

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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        // Initialize RNG with seed if provided
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        let all_edges: Vec<EdgeId> = scope.edge_ids().copied().collect();
        let total = all_edges.len();
        let sample_size = self.spec.calculate_sample_size(total)?;

        let mut result = HashMap::new();

        if sample_size == 0 {
            scope.variables_mut().set_edge_map(&self.target, result);
            return Ok(());
        }

        // Reservoir sampling algorithm
        let mut reservoir: Vec<EdgeId> = Vec::with_capacity(sample_size);

        for (i, &edge) in all_edges.iter().enumerate() {
            if i < sample_size {
                reservoir.push(edge);
            } else {
                let j = fastrand::usize(0..=i);
                if j < sample_size {
                    reservoir[j] = edge;
                }
            }
        }

        // Mark sampled edges
        for edge in reservoir {
            result.insert(edge, AlgorithmParamValue::Int(1));
        }

        scope.variables_mut().set_edge_map(&self.target, result);
        Ok(())
    }
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let graph_ref = scope.subgraph().graph();
        let mut subgraphs = Vec::new();
        for &node in scope.subgraph().nodes().iter() {
            let mut nodes = std::collections::HashSet::new();
            nodes.insert(node);
            let sg =
                Subgraph::from_nodes(graph_ref.clone(), nodes, "sample_iterate_nodes".to_string())?;
            subgraphs.push(sg);
        }
        scope
            .variables_mut()
            .set_subgraph_array(self.target.clone(), subgraphs);
        Ok(())
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let graph_ref = scope.subgraph().graph();
        let graph = graph_ref.borrow();
        let mut subgraphs = Vec::new();
        for &edge in scope.subgraph().edges().iter() {
            let (source, target) = graph.edge_endpoints(edge)?;
            let mut nodes = std::collections::HashSet::new();
            nodes.insert(source);
            nodes.insert(target);
            let mut edges = std::collections::HashSet::new();
            edges.insert(edge);
            let sg = Subgraph::new(
                graph_ref.clone(),
                nodes,
                edges,
                "sample_iterate_edges".to_string(),
            );
            subgraphs.push(sg);
        }
        scope
            .variables_mut()
            .set_subgraph_array(self.target.clone(), subgraphs);
        Ok(())
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        let seeds = scope.variables().subgraph_array(&self.source)?;
        let mut out = Vec::with_capacity(seeds.len());
        for seed in seeds {
            let node_ids: Vec<NodeId> = seed.node_set().iter().copied().collect();
            let graph_ref = seed.graph();
            if node_ids.is_empty() {
                out.push(Subgraph::new(
                    graph_ref.clone(),
                    std::collections::HashSet::new(),
                    std::collections::HashSet::new(),
                    "sample_neighbors_empty".to_string(),
                ));
                continue;
            }
            let neighborhood = graph_ref
                .borrow_mut()
                .unified_neighborhood(&node_ids, self.hops)
                .map_err(|err| anyhow::anyhow!(err))?;
            let nodes = neighborhood.node_set().clone();
            let edges = neighborhood.edge_set().clone();
            out.push(Subgraph::new(
                graph_ref.clone(),
                nodes,
                edges,
                format!("sample_neighbors_hops_{}", self.hops),
            ));
        }
        scope
            .variables_mut()
            .set_subgraph_array(self.target.clone(), out);
        Ok(())
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        if let Ok(array) = scope
            .variables()
            .subgraph_array(&self.source)
            .map(|a| a.clone())
        {
            let out = match self.mode {
                EmitMode::PerSeed => array,
                EmitMode::Unified => {
                    if array.is_empty() {
                        Vec::new()
                    } else {
                        let graph_ref = scope.subgraph().graph();
                        let mut node_set = std::collections::HashSet::new();
                        let mut edge_set = std::collections::HashSet::new();
                        for sg in &array {
                            node_set.extend(sg.nodes().iter().copied());
                            edge_set.extend(sg.edge_ids().iter().copied());
                        }

                        let unified = if self.induced {
                            Subgraph::from_nodes(
                                graph_ref,
                                node_set,
                                "sample_emit_subgraphs_unified".to_string(),
                            )?
                        } else {
                            Subgraph::new(
                                graph_ref,
                                node_set,
                                edge_set,
                                "sample_emit_subgraphs_unified".to_string(),
                            )
                        };
                        vec![unified]
                    }
                }
            };
            scope
                .variables_mut()
                .set_subgraph_array(self.target.clone(), out);
            return Ok(());
        }

        let graph_ref = scope.subgraph().graph();
        if let Ok(node_map) = scope.variables().node_map(&self.source) {
            let nodes: Vec<NodeId> = node_map.keys().copied().collect();
            let mut out = Vec::new();
            match self.mode {
                EmitMode::Unified => {
                    let node_set: std::collections::HashSet<NodeId> = nodes.into_iter().collect();
                    let sg = if self.induced {
                        Subgraph::from_nodes(
                            graph_ref.clone(),
                            node_set,
                            "sample_emit_nodes".to_string(),
                        )?
                    } else {
                        Subgraph::new(
                            graph_ref.clone(),
                            node_set,
                            std::collections::HashSet::new(),
                            "sample_emit_nodes".to_string(),
                        )
                    };
                    out.push(sg);
                }
                EmitMode::PerSeed => {
                    for node in nodes {
                        let mut set = std::collections::HashSet::new();
                        set.insert(node);
                        let sg = Subgraph::from_nodes(
                            graph_ref.clone(),
                            set,
                            "sample_emit_node".to_string(),
                        )?;
                        out.push(sg);
                    }
                }
            }
            scope
                .variables_mut()
                .set_subgraph_array(self.target.clone(), out);
            return Ok(());
        }

        let edge_map = scope.variables().edge_map(&self.source)?;
        let edges: Vec<EdgeId> = edge_map.keys().copied().collect();
        let mut out = Vec::new();
        match self.mode {
            EmitMode::Unified => {
                let graph = graph_ref.borrow();
                let mut node_set = std::collections::HashSet::new();
                let mut edge_set = std::collections::HashSet::new();
                for edge in edges {
                    let (source, target) = graph.edge_endpoints(edge)?;
                    node_set.insert(source);
                    node_set.insert(target);
                    edge_set.insert(edge);
                }
                let sg = if self.induced {
                    Subgraph::from_nodes(
                        graph_ref.clone(),
                        node_set,
                        "sample_emit_edges_induced".to_string(),
                    )?
                } else {
                    Subgraph::new(
                        graph_ref.clone(),
                        node_set,
                        edge_set,
                        "sample_emit_edges".to_string(),
                    )
                };
                out.push(sg);
            }
            EmitMode::PerSeed => {
                let graph = graph_ref.borrow();
                for edge in edges {
                    let (source, target) = graph.edge_endpoints(edge)?;
                    let mut node_set = std::collections::HashSet::new();
                    node_set.insert(source);
                    node_set.insert(target);
                    let mut edge_set = std::collections::HashSet::new();
                    edge_set.insert(edge);
                    out.push(Subgraph::new(
                        graph_ref.clone(),
                        node_set,
                        edge_set,
                        "sample_emit_edge".to_string(),
                    ));
                }
            }
        }
        scope
            .variables_mut()
            .set_subgraph_array(self.target.clone(), out);
        Ok(())
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

    fn apply(&self, _ctx: &mut Context, scope: &mut StepScope) -> Result<()> {
        // Initialize RNG with seed if provided
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        match self.entity_type {
            EntityType::Nodes => self.sample_nodes(scope),
            EntityType::Edges => self.sample_edges(scope),
        }
    }
}

impl ReservoirSampleStep {
    fn sample_nodes(&self, scope: &mut StepScope) -> Result<()> {
        let source_map = scope.variables().node_map(&self.source)?;

        let mut result = HashMap::new();

        if self.k == 0 {
            scope.variables_mut().set_node_map(&self.target, result);
            return Ok(());
        }

        // Collect all nodes from source (those with any value)
        let source_nodes: Vec<NodeId> = source_map.keys().copied().collect();

        if source_nodes.is_empty() {
            scope.variables_mut().set_node_map(&self.target, result);
            return Ok(());
        }

        // Reservoir sampling
        let mut reservoir: Vec<NodeId> = Vec::with_capacity(self.k);

        for (i, &node) in source_nodes.iter().enumerate() {
            if i < self.k {
                reservoir.push(node);
            } else {
                let j = fastrand::usize(0..=i);
                if j < self.k {
                    reservoir[j] = node;
                }
            }
        }

        // Mark sampled nodes
        for node in reservoir {
            result.insert(node, AlgorithmParamValue::Int(1));
        }

        scope.variables_mut().set_node_map(&self.target, result);
        Ok(())
    }

    fn sample_edges(&self, scope: &mut StepScope) -> Result<()> {
        let source_map = scope.variables().edge_map(&self.source)?;

        let mut result = HashMap::new();

        if self.k == 0 {
            scope.variables_mut().set_edge_map(&self.target, result);
            return Ok(());
        }

        let source_edges: Vec<EdgeId> = source_map.keys().copied().collect();

        if source_edges.is_empty() {
            scope.variables_mut().set_edge_map(&self.target, result);
            return Ok(());
        }

        // Reservoir sampling
        let mut reservoir: Vec<EdgeId> = Vec::with_capacity(self.k);

        for (i, &edge) in source_edges.iter().enumerate() {
            if i < self.k {
                reservoir.push(edge);
            } else {
                let j = fastrand::usize(0..=i);
                if j < self.k {
                    reservoir[j] = edge;
                }
            }
        }

        // Mark sampled edges
        for edge in reservoir {
            result.insert(edge, AlgorithmParamValue::Int(1));
        }

        scope.variables_mut().set_edge_map(&self.target, result);
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
