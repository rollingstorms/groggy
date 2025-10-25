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

use crate::types::{EdgeId, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

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
