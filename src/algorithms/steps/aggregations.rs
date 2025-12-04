//! Aggregation and reduction step primitives.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};
use super::direction::NeighborDirection;

/// Extract numeric values from a map, returning f64 values.
fn extract_numeric_values(map: &HashMap<NodeId, AlgorithmParamValue>) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for value in map.values() {
        match value {
            AlgorithmParamValue::Int(v) => values.push(*v as f64),
            AlgorithmParamValue::Float(v) => values.push(*v),
            _ => {
                return Err(anyhow!(
                    "operation requires numeric values, found {:?}",
                    value
                ))
            }
        }
    }
    Ok(values)
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
    pub(crate) fn apply(
        &self,
        map: &HashMap<NodeId, AlgorithmParamValue>,
    ) -> Result<AlgorithmParamValue> {
        let values = extract_numeric_values(map)?;
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

/// Compute standard deviation (sample standard deviation) of node values.
pub struct StdDevStep {
    source: String,
    target: String,
}

impl StdDevStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

impl Step for StdDevStep {
    fn id(&self) -> &'static str {
        "core.std"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute standard deviation of node values".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("std cancelled"));
        }

        let values = extract_numeric_values(map)?;
        if values.is_empty() {
            scope
                .variables_mut()
                .set_scalar(self.target.clone(), AlgorithmParamValue::None);
            return Ok(());
        }

        // Compute mean
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;

        // Compute variance (sample variance: divide by n-1)
        let variance = if values.len() > 1 {
            let sum_sq_diff: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
            sum_sq_diff / (values.len() - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Float(std_dev));
        Ok(())
    }
}

/// Compute median value of node values.
pub struct MedianStep {
    source: String,
    target: String,
}

impl MedianStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

impl Step for MedianStep {
    fn id(&self) -> &'static str {
        "core.median"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute median value of node values".to_string(),
            cost_hint: CostHint::Linearithmic, // O(n log n) due to sorting
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("median cancelled"));
        }

        let mut values = extract_numeric_values(map)?;
        if values.is_empty() {
            scope
                .variables_mut()
                .set_scalar(self.target.clone(), AlgorithmParamValue::None);
            return Ok(());
        }

        // Sort values for median computation
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if values.len() % 2 == 0 {
            let mid = values.len() / 2;
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[values.len() / 2]
        };

        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Float(median));
        Ok(())
    }
}

/// Find the mode (most common value) in node values.
pub struct ModeStep {
    source: String,
    target: String,
}

impl ModeStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

impl Step for ModeStep {
    fn id(&self) -> &'static str {
        "core.mode"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Find the most common value in node values".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("mode cancelled"));
        }

        if map.is_empty() {
            scope
                .variables_mut()
                .set_scalar(self.target.clone(), AlgorithmParamValue::None);
            return Ok(());
        }

        // Count occurrences of each value
        let mut counts: HashMap<String, usize> = HashMap::new();
        for value in map.values() {
            let key = format!("{:?}", value);
            *counts.entry(key).or_insert(0) += 1;
        }

        // Find the value with the highest count
        let (mode_key, _) = counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .ok_or_else(|| anyhow!("mode computation failed"))?;

        // Find and clone the first value that matches (for type preservation)
        let mode_value = map
            .values()
            .find(|value| format!("{:?}", value) == *mode_key)
            .cloned()
            .unwrap_or(AlgorithmParamValue::None);

        scope
            .variables_mut()
            .set_scalar(self.target.clone(), mode_value);
        Ok(())
    }
}

/// Compute the q-th quantile of node values.
pub struct QuantileStep {
    source: String,
    q: f64,
    target: String,
}

impl QuantileStep {
    pub fn new(source: impl Into<String>, q: f64, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            q,
            target: target.into(),
        }
    }
}

impl Step for QuantileStep {
    fn id(&self) -> &'static str {
        "core.quantile"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Compute {}-quantile of node values", self.q),
            cost_hint: CostHint::Linearithmic, // O(n log n) due to sorting
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("quantile cancelled"));
        }

        if self.q < 0.0 || self.q > 1.0 {
            return Err(anyhow!("quantile q must be in [0, 1], got {}", self.q));
        }

        let mut values = extract_numeric_values(map)?;
        if values.is_empty() {
            scope
                .variables_mut()
                .set_scalar(self.target.clone(), AlgorithmParamValue::None);
            return Ok(());
        }

        // Sort values for quantile computation
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Linear interpolation method
        let index = self.q * (values.len() - 1) as f64;
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        let quantile = if lower_idx == upper_idx {
            values[lower_idx]
        } else {
            let weight = index - lower_idx as f64;
            values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight
        };

        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Float(quantile));
        Ok(())
    }
}

/// Compute Shannon entropy of node values.
pub struct EntropyStep {
    source: String,
    target: String,
}

impl EntropyStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
        }
    }
}

impl Step for EntropyStep {
    fn id(&self) -> &'static str {
        "core.entropy"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Compute Shannon entropy of node values".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("entropy cancelled"));
        }

        if map.is_empty() {
            scope
                .variables_mut()
                .set_scalar(self.target.clone(), AlgorithmParamValue::Float(0.0));
            return Ok(());
        }

        // Count occurrences of each unique value
        let mut counts: HashMap<String, usize> = HashMap::new();
        for value in map.values() {
            let key = format!("{:?}", value);
            *counts.entry(key).or_insert(0) += 1;
        }

        // Compute entropy: H = -Σ p(x) * log2(p(x))
        let total = map.len() as f64;
        let entropy: f64 = counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        scope
            .variables_mut()
            .set_scalar(self.target.clone(), AlgorithmParamValue::Float(entropy));
        Ok(())
    }
}

/// Compute histogram (binned counts) of node values.
pub struct HistogramStep {
    source: String,
    bins: usize,
    target: String,
}

impl HistogramStep {
    pub fn new(source: impl Into<String>, bins: usize, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            bins,
            target: target.into(),
        }
    }
}

impl Step for HistogramStep {
    fn id(&self) -> &'static str {
        "core.histogram"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Compute histogram with {} bins", self.bins),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let map = scope.variables().node_map(&self.source)?;
        if ctx.is_cancelled() {
            return Err(anyhow!("histogram cancelled"));
        }

        if self.bins == 0 {
            return Err(anyhow!("histogram bins must be > 0"));
        }

        let values = extract_numeric_values(map)?;
        if values.is_empty() {
            // Return empty histogram as node map
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), HashMap::new());
            return Ok(());
        }

        // Find min and max
        let min = values.iter().copied().reduce(f64::min).unwrap_or(0.0);
        let max = values.iter().copied().reduce(f64::max).unwrap_or(0.0);

        // Handle edge case where all values are the same
        if (max - min).abs() < 1e-10 {
            let mut result = HashMap::new();
            result.insert(0, AlgorithmParamValue::Int(values.len() as i64));
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), result);
            return Ok(());
        }

        // Initialize bins
        let mut bin_counts = vec![0i64; self.bins];
        let bin_width = (max - min) / self.bins as f64;

        // Assign values to bins
        for value in values {
            let mut bin_idx = ((value - min) / bin_width).floor() as usize;
            // Handle edge case where value == max
            if bin_idx >= self.bins {
                bin_idx = self.bins - 1;
            }
            bin_counts[bin_idx] += 1;
        }

        // Convert to node map (using bin index as NodeId)
        let mut result = HashMap::new();
        for (idx, count) in bin_counts.iter().enumerate() {
            if *count > 0 {
                result.insert(idx, AlgorithmParamValue::Int(*count));
            }
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

/// Aggregation type for neighbor operations.
#[derive(Clone, Copy, Debug)]
pub enum NeighborAggType {
    Sum,
    Mean,
    Mode,
    Min,
    Max,
}

/// Aggregate values from neighbors for each node.
///
/// This step follows STYLE_ALGO pattern with CSR optimization for efficient
/// neighbor access. It's the foundation for PageRank and LPA builder patterns.
///
/// # Example
/// ```ignore
/// // PageRank iteration: sum(ranks[neighbors(node)])
/// let step = NeighborAggregationStep::new("ranks", "neighbor_sum", NeighborAggType::Sum);
/// // Weighted: sum(ranks[neighbors] * weights[neighbors])
/// let step = NeighborAggregationStep::new("ranks", "weighted_sum", NeighborAggType::Sum)
///     .with_weights("inv_degrees");
/// ```
pub struct NeighborAggregationStep {
    source: String,
    target: String,
    agg_type: NeighborAggType,
    weights: Option<String>,
    direction: NeighborDirection,
}

impl NeighborAggregationStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        agg_type: NeighborAggType,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            agg_type,
            weights: None,
            direction: NeighborDirection::default(), // Default to incoming for PageRank/LPA
        }
    }

    pub fn with_weights(mut self, weights: impl Into<String>) -> Self {
        self.weights = Some(weights.into());
        self
    }

    #[allow(dead_code)]
    pub fn with_direction(mut self, direction: NeighborDirection) -> Self {
        self.direction = direction;
        self
    }
}

impl Step for NeighborAggregationStep {
    fn id(&self) -> &'static str {
        "core.neighbor_agg"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Aggregate neighbor values using {:?}", self.agg_type),
            cost_hint: CostHint::Linear, // O(n + m) with CSR
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        use std::time::Instant;

        let start = Instant::now();

        let subgraph = scope.subgraph();
        let nodes = subgraph.ordered_nodes();
        ctx.record_stat("neighbor_agg.count.nodes", nodes.len() as f64);

        // Build CSR with specified direction
        let mut node_to_idx = HashMap::new();
        for (idx, &node_id) in nodes.iter().enumerate() {
            node_to_idx.insert(node_id, idx);
        }

        let mut csr = Csr::default();
        {
            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            let pool = graph_ref.pool();
            let edges = subgraph.edges();

            let _build_time = match self.direction {
                NeighborDirection::In => build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| node_to_idx.get(&nid).copied(),
                    |eid| {
                        pool.get_edge_endpoints(eid)
                            .map(|(source, target)| (target, source))
                    },
                    CsrOptions {
                        add_reverse_edges: false,
                        sort_neighbors: false,
                    },
                ),
                NeighborDirection::Out => build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| node_to_idx.get(&nid).copied(),
                    |eid| pool.get_edge_endpoints(eid),
                    CsrOptions {
                        add_reverse_edges: false,
                        sort_neighbors: false,
                    },
                ),
                NeighborDirection::Undirected => build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| node_to_idx.get(&nid).copied(),
                    |eid| pool.get_edge_endpoints(eid),
                    CsrOptions {
                        add_reverse_edges: true,
                        sort_neighbors: false,
                    },
                ),
            };
        }

        // Get source values
        let source_map = scope.variables().node_map(&self.source)?;

        // Get optional weights
        let weights_map = if let Some(ref weights_name) = self.weights {
            Some(scope.variables().node_map(weights_name)?)
        } else {
            None
        };

        // Pre-allocate result map
        let mut result = HashMap::with_capacity(nodes.len());

        for (idx, &node) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("neighbor_agg cancelled"));
            }

            let neighbor_indices = csr.neighbors(idx);

            // Aggregate neighbor values
            let agg_value = match self.agg_type {
                NeighborAggType::Sum => {
                    let sum: f64 = if let Some(weights) = weights_map {
                        // Weighted sum: sum(values[neighbor] * weights[neighbor])
                        neighbor_indices
                            .iter()
                            .filter_map(|&nbr_idx| {
                                let nbr_node = nodes[nbr_idx];
                                let value = source_map.get(&nbr_node).and_then(|v| match v {
                                    AlgorithmParamValue::Float(f) => Some(*f),
                                    AlgorithmParamValue::Int(i) => Some(*i as f64),
                                    _ => None,
                                })?;
                                let weight = weights.get(&nbr_node).and_then(|w| match w {
                                    AlgorithmParamValue::Float(f) => Some(*f),
                                    AlgorithmParamValue::Int(i) => Some(*i as f64),
                                    _ => None,
                                })?;
                                Some(value * weight)
                            })
                            .sum()
                    } else {
                        // Unweighted sum: sum(values[neighbor])
                        neighbor_indices
                            .iter()
                            .filter_map(|&nbr_idx| {
                                let nbr_node = nodes[nbr_idx];
                                source_map.get(&nbr_node).and_then(|v| match v {
                                    AlgorithmParamValue::Float(f) => Some(*f),
                                    AlgorithmParamValue::Int(i) => Some(*i as f64),
                                    _ => None,
                                })
                            })
                            .sum()
                    };
                    AlgorithmParamValue::Float(sum)
                }

                NeighborAggType::Mean => {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for &nbr_idx in neighbor_indices {
                        let nbr_node = nodes[nbr_idx];
                        if let Some(value) = source_map.get(&nbr_node) {
                            match value {
                                AlgorithmParamValue::Float(f) => {
                                    sum += f;
                                    count += 1;
                                }
                                AlgorithmParamValue::Int(i) => {
                                    sum += *i as f64;
                                    count += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                    if count > 0 {
                        AlgorithmParamValue::Float(sum / count as f64)
                    } else {
                        AlgorithmParamValue::Float(0.0)
                    }
                }

                NeighborAggType::Min => {
                    let min = neighbor_indices
                        .iter()
                        .filter_map(|&nbr_idx| {
                            let nbr_node = nodes[nbr_idx];
                            source_map.get(&nbr_node).and_then(|v| match v {
                                AlgorithmParamValue::Float(f) => Some(*f),
                                AlgorithmParamValue::Int(i) => Some(*i as f64),
                                _ => None,
                            })
                        })
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    min.map(AlgorithmParamValue::Float)
                        .unwrap_or(AlgorithmParamValue::None)
                }

                NeighborAggType::Max => {
                    let max = neighbor_indices
                        .iter()
                        .filter_map(|&nbr_idx| {
                            let nbr_node = nodes[nbr_idx];
                            source_map.get(&nbr_node).and_then(|v| match v {
                                AlgorithmParamValue::Float(f) => Some(*f),
                                AlgorithmParamValue::Int(i) => Some(*i as f64),
                                _ => None,
                            })
                        })
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    max.map(AlgorithmParamValue::Float)
                        .unwrap_or(AlgorithmParamValue::None)
                }

                NeighborAggType::Mode => {
                    // Count occurrences of each value
                    let mut counts: HashMap<String, usize> = HashMap::new();
                    for &nbr_idx in neighbor_indices {
                        let nbr_node = nodes[nbr_idx];
                        if let Some(value) = source_map.get(&nbr_node) {
                            let key = format!("{:?}", value);
                            *counts.entry(key).or_insert(0) += 1;
                        }
                    }

                    // Find most common value
                    if let Some((mode_key, _)) = counts.iter().max_by_key(|(_, &count)| count) {
                        // Find the original value
                        neighbor_indices
                            .iter()
                            .filter_map(|&nbr_idx| {
                                let nbr_node = nodes[nbr_idx];
                                source_map.get(&nbr_node)
                            })
                            .find(|value| format!("{:?}", value) == *mode_key)
                            .cloned()
                            .unwrap_or(AlgorithmParamValue::None)
                    } else {
                        AlgorithmParamValue::None
                    }
                }
            };

            result.insert(node, agg_value);
        }

        ctx.record_duration("neighbor_agg.total", start.elapsed());

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_dev() {
        let mut map = HashMap::new();
        map.insert(0, AlgorithmParamValue::Float(2.0));
        map.insert(1, AlgorithmParamValue::Float(4.0));
        map.insert(2, AlgorithmParamValue::Float(4.0));
        map.insert(3, AlgorithmParamValue::Float(4.0));
        map.insert(4, AlgorithmParamValue::Float(5.0));
        map.insert(5, AlgorithmParamValue::Float(5.0));
        map.insert(6, AlgorithmParamValue::Float(7.0));
        map.insert(7, AlgorithmParamValue::Float(9.0));

        let values = extract_numeric_values(&map).unwrap();
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Expected std dev is ~2.138
        assert!((std_dev - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_median_odd() {
        let mut map = HashMap::new();
        map.insert(0, AlgorithmParamValue::Float(1.0));
        map.insert(1, AlgorithmParamValue::Float(3.0));
        map.insert(2, AlgorithmParamValue::Float(5.0));

        let mut values = extract_numeric_values(&map).unwrap();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = values[values.len() / 2];
        assert_eq!(median, 3.0);
    }

    #[test]
    fn test_median_even() {
        let mut map = HashMap::new();
        map.insert(0, AlgorithmParamValue::Float(1.0));
        map.insert(1, AlgorithmParamValue::Float(2.0));
        map.insert(2, AlgorithmParamValue::Float(3.0));
        map.insert(3, AlgorithmParamValue::Float(4.0));

        let mut values = extract_numeric_values(&map).unwrap();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = values.len() / 2;
        let median = (values[mid - 1] + values[mid]) / 2.0;
        assert_eq!(median, 2.5);
    }

    #[test]
    fn test_quantile() {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, AlgorithmParamValue::Float(i as f64));
        }

        let mut values = extract_numeric_values(&map).unwrap();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Test 0.25 quantile (25th percentile)
        let q = 0.25;
        let index = q * (values.len() - 1) as f64;
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;
        let quantile = if lower_idx == upper_idx {
            values[lower_idx]
        } else {
            let weight = index - lower_idx as f64;
            values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight
        };

        assert!((quantile - 24.75).abs() < 0.01);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution should have maximum entropy
        let mut map = HashMap::new();
        map.insert(0, AlgorithmParamValue::Int(1));
        map.insert(1, AlgorithmParamValue::Int(2));
        map.insert(2, AlgorithmParamValue::Int(3));
        map.insert(3, AlgorithmParamValue::Int(4));

        let mut counts: HashMap<String, usize> = HashMap::new();
        for value in map.values() {
            let key = format!("{:?}", value);
            *counts.entry(key).or_insert(0) += 1;
        }

        let total = map.len() as f64;
        let entropy: f64 = counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        // log2(4) = 2.0
        assert!((entropy - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_bins() {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, AlgorithmParamValue::Float(i as f64));
        }

        let values = extract_numeric_values(&map).unwrap();
        let bins = 10;
        let min = 0.0;
        let max = 99.0;
        let bin_width = (max - min) / bins as f64;

        let mut bin_counts = vec![0i64; bins];
        for value in values {
            let mut bin_idx = ((value - min) / bin_width).floor() as usize;
            if bin_idx >= bins {
                bin_idx = bins - 1;
            }
            bin_counts[bin_idx] += 1;
        }

        // Each bin should have ~10 elements
        for count in bin_counts {
            assert!((9..=11).contains(&count));
        }
    }

    #[test]
    fn test_neighbor_agg_type() {
        // Just verify the enum and struct exist
        let _agg_type = NeighborAggType::Sum;
        let _step = NeighborAggregationStep::new("src", "tgt", NeighborAggType::Mean);
        // Actual integration test will be in Python or higher-level Rust test
    }
}
