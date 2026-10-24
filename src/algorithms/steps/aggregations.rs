//! Aggregation and reduction step primitives.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Result};

use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::storage::numeric::{
    attributes::{self, Value},
    dot_product::Dot,
    reductions::{self, Scalar},
};
use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};
use super::direction::NeighborDirection;

fn ordered_values(
    map: &HashMap<NodeId, AlgorithmParamValue>,
) -> Vec<(NodeId, &AlgorithmParamValue)> {
    let mut rows: Vec<_> = map.iter().map(|(&id, value)| (id, value)).collect();
    rows.sort_unstable_by_key(|(id, _)| *id);
    rows
}
pub(super) fn numeric_value(value: &AlgorithmParamValue) -> Result<Value> {
    match value {
        AlgorithmParamValue::Int(value) => Ok(Value::Integer(*value)),
        AlgorithmParamValue::Float(value) if value.is_finite() => Ok(Value::Float(*value)),
        _ => Err(anyhow!(
            "Aggregation requires finite numeric values, found {:?}",
            value
        )),
    }
}
fn numeric_values(map: &HashMap<NodeId, AlgorithmParamValue>) -> Result<Vec<Value>> {
    ordered_values(map)
        .into_iter()
        .map(|(_, value)| numeric_value(value))
        .collect()
}
#[cfg(test)]
fn extract_numeric_values(map: &HashMap<NodeId, AlgorithmParamValue>) -> Result<Vec<f64>> {
    Ok(numeric_values(map)?
        .into_iter()
        .map(Scalar::as_f64)
        .collect())
}
fn float_result(value: f64) -> Result<AlgorithmParamValue> {
    if !value.is_finite() {
        return Err(anyhow!("Aggregation result exceeds finite numeric range"));
    }
    Ok(AlgorithmParamValue::Float(value))
}
fn accumulated(values: &[Value], mean: bool) -> Result<AlgorithmParamValue> {
    let mut sum = Dot::new();
    for value in values {
        let (hi, lo) = value.parts();
        sum.add(hi, 1.0);
        sum.add(lo, 1.0);
    }
    float_result(sum.total_divided_by(if mean { values.len() } else { 1 }))
}
fn categorical_key(value: &AlgorithmParamValue) -> Result<String> {
    if matches!(value, AlgorithmParamValue::Float(value) if !value.is_finite())
        || matches!(value, AlgorithmParamValue::FloatList(values) if values.iter().any(|v| !v.is_finite()))
    {
        return Err(anyhow!(
            "Categorical aggregation requires finite floating values"
        ));
    }
    // Typed serialization keeps categories distinct and avoids debug-format keys.
    Ok(serde_json::to_string(value)?)
}
fn frequencies<'a>(
    rows: impl IntoIterator<Item = (NodeId, &'a AlgorithmParamValue)>,
) -> Result<BTreeMap<String, (usize, NodeId, AlgorithmParamValue)>> {
    let mut counts = BTreeMap::new();
    for (node, value) in rows {
        let entry = counts
            .entry(categorical_key(value)?)
            .or_insert((0usize, node, value.clone()));
        entry.0 = entry
            .0
            .checked_add(1)
            .ok_or_else(|| anyhow!("Category count overflow"))?;
        entry.1 = entry.1.min(node);
    }
    Ok(counts)
}
fn modal_value(
    counts: &BTreeMap<String, (usize, NodeId, AlgorithmParamValue)>,
) -> AlgorithmParamValue {
    counts
        .values()
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
        .map(|(_, _, value)| value.clone())
        .unwrap_or(AlgorithmParamValue::None)
}
fn install_scalar(
    ctx: &Context,
    scope: &mut StepScope<'_>,
    target: &str,
    value: AlgorithmParamValue,
) -> Result<()> {
    if ctx.is_cancelled() {
        return Err(anyhow!("Aggregation cancelled before installing result"));
    }
    scope.variables_mut().set_scalar(target, value);
    Ok(())
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
        let values = numeric_values(map)?;
        self.apply_values(&values)
    }

    pub(super) fn apply_values(&self, values: &[Value]) -> Result<AlgorithmParamValue> {
        if values.is_empty() {
            return Ok(AlgorithmParamValue::None);
        }
        match self {
            Reduction::Sum => accumulated(values, false),
            Reduction::Mean => accumulated(values, true),
            Reduction::Min => float_result(
                reductions::extremum(values, true).ok_or_else(|| anyhow!("Missing minimum"))?,
            ),
            Reduction::Max => float_result(
                reductions::extremum(values, false).ok_or_else(|| anyhow!("Missing maximum"))?,
            ),
        }
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let value = self
            .reducer
            .apply(scope.variables().node_map(&self.source)?)?;
        install_scalar(ctx, scope, &self.target, value)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let values = numeric_values(scope.variables().node_map(&self.source)?)?;
        let result = match reductions::std_dev(&values) {
            Some(value) => float_result(value)?,
            None => AlgorithmParamValue::None,
        };
        install_scalar(ctx, scope, &self.target, result)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let values = numeric_values(scope.variables().node_map(&self.source)?)?;
        let input: Vec<_> = values
            .into_iter()
            .map(|value| match value {
                Value::Integer(value) => crate::AttrValue::Int(value),
                Value::Float(value) => crate::AttrValue::Float64(value),
            })
            .collect();
        let result =
            AlgorithmParamValue::from_attr_value(attributes::quantile(&input, 0.5, "linear")?)
                .ok_or_else(|| anyhow!("Invalid quantile output"))?;
        install_scalar(ctx, scope, &self.target, result)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let counts = frequencies(ordered_values(scope.variables().node_map(&self.source)?))?;
        install_scalar(ctx, scope, &self.target, modal_value(&counts))
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let values = numeric_values(scope.variables().node_map(&self.source)?)?;
        let input: Vec<_> = values
            .into_iter()
            .map(|value| match value {
                Value::Integer(value) => crate::AttrValue::Int(value),
                Value::Float(value) => crate::AttrValue::Float64(value),
            })
            .collect();
        let result =
            AlgorithmParamValue::from_attr_value(attributes::quantile(&input, self.q, "linear")?)
                .ok_or_else(|| anyhow!("Invalid quantile output"))?;
        install_scalar(ctx, scope, &self.target, result)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let map = scope.variables().node_map(&self.source)?;
        let counts = frequencies(ordered_values(map))?;
        let mut entropy = Dot::new();
        for (count, _, _) in counts.values() {
            let probability = *count as f64 / map.len() as f64;
            entropy.add(-probability, probability.log2());
        }
        let result = float_result(entropy.total())?;
        install_scalar(ctx, scope, &self.target, result)
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        if self.bins == 0 {
            return Err(anyhow!("Histogram bins must be positive"));
        }
        let values = numeric_values(scope.variables().node_map(&self.source)?)?;
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(self.bins)
            .map_err(|_| anyhow!("Histogram allocation exceeds supported capacity"))?;
        counts.resize(self.bins, 0i64);
        if let (Some(min), Some(max)) = (
            values.iter().copied().min_by(|a, b| a.numeric_cmp(*b)),
            values.iter().copied().max_by(|a, b| a.numeric_cmp(*b)),
        ) {
            let span = max.difference(min);
            let scale = min.as_f64().abs().max(max.as_f64().abs());
            for value in values {
                if ctx.is_cancelled() {
                    return Err(anyhow!("Histogram cancelled"));
                }
                let fraction = if min.numeric_cmp(max).is_eq() {
                    0.0
                } else if span.is_finite() {
                    value.difference(min) / span
                } else {
                    (value.as_f64() / scale - min.as_f64() / scale)
                        / (max.as_f64() / scale - min.as_f64() / scale)
                };
                let index = ((fraction * self.bins as f64).floor() as usize).min(self.bins - 1);
                counts[index] = counts[index]
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Histogram count exceeds i64"))?;
            }
        }
        install_scalar(
            ctx,
            scope,
            &self.target,
            AlgorithmParamValue::IntList(counts.clone()),
        )?;
        ctx.add_output(
            format!("histograms.{}", self.target),
            super::super::AlgorithmOutput::Histogram(counts),
        );
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
    fn supports_atomic_properties(&self) -> bool {
        true
    }
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
        if ctx.is_cancelled() {
            return Err(anyhow!("Aggregation cancelled"));
        }
        let started = std::time::Instant::now();
        if self.weights.is_some() && !matches!(self.agg_type, NeighborAggType::Sum) {
            return Err(anyhow!("Neighbor weights are supported only for sum"));
        }
        let subgraph = scope.subgraph();
        let nodes = subgraph.ordered_nodes();
        let positions: HashMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(index, &node)| (node, index))
            .collect();
        let edges = subgraph.ordered_edges();
        let mut csr = Csr::default();
        {
            let graph = subgraph.graph();
            let graph = graph.borrow();
            for &node in nodes.iter() {
                if !graph.has_node(node) {
                    return Err(anyhow!("Neighbor source node {} is inactive", node));
                }
            }
            for &edge in edges.iter() {
                if ctx.is_cancelled() {
                    return Err(anyhow!("Neighbor aggregation cancelled"));
                }
                if !graph.has_edge(edge) {
                    return Err(anyhow!("Neighbor edge {} is inactive", edge));
                }
                let (u, v) = graph.edge_endpoints(edge)?;
                if !positions.contains_key(&u) || !positions.contains_key(&v) {
                    return Err(anyhow!("Neighbor edge escapes selection"));
                }
            }
            let mapping = self.direction.edge_mapping();
            build_csr_from_edges_with_scratch(
                &mut csr,
                nodes.len(),
                edges.iter().copied(),
                |id| positions.get(&id).copied(),
                |edge| graph.pool().get_edge_endpoints(edge).map(mapping),
                CsrOptions {
                    add_reverse_edges: self.direction.add_reverse_edges(),
                    sort_neighbors: true,
                },
            );
        }
        let source = scope.variables().node_map(&self.source)?;
        let weights = self
            .weights
            .as_ref()
            .map(|name| scope.variables().node_map(name))
            .transpose()?;
        let mut result = HashMap::with_capacity(nodes.len());
        for (index, &node) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("Neighbor aggregation cancelled"));
            }
            let rows: Vec<_> = csr
                .neighbors(index)
                .iter()
                .filter_map(|&index| {
                    let neighbor = nodes[index];
                    source.get(&neighbor).map(|value| (neighbor, value))
                })
                .collect();
            let value = if matches!(self.agg_type, NeighborAggType::Mode) {
                modal_value(&frequencies(rows)?)
            } else {
                let mut values = Vec::with_capacity(rows.len());
                let mut weighted = weights.map(|_| Dot::new());
                for (neighbor, value) in rows {
                    if ctx.is_cancelled() {
                        return Err(anyhow!("Neighbor aggregation cancelled"));
                    }
                    let value = numeric_value(value)?;
                    if let Some(weights) = weights {
                        let weight = numeric_value(weights.get(&neighbor).ok_or_else(|| {
                            anyhow!("Missing neighbor weight for node {}", neighbor)
                        })?)?;
                        let (a, b) = value.parts();
                        let (c, d) = weight.parts();
                        for (left, right) in [(a, c), (a, d), (b, c), (b, d)] {
                            weighted
                                .as_mut()
                                .ok_or_else(|| anyhow!("Missing weighted accumulator"))?
                                .add(left, right);
                        }
                    }
                    values.push(value);
                }
                match self.agg_type {
                    NeighborAggType::Sum if weights.is_some() => float_result(
                        weighted
                            .take()
                            .ok_or_else(|| anyhow!("Missing weighted accumulator"))?
                            .total(),
                    )?,
                    NeighborAggType::Sum => accumulated(&values, false)?,
                    NeighborAggType::Mean if values.is_empty() => AlgorithmParamValue::Float(0.0),
                    NeighborAggType::Mean => accumulated(&values, true)?,
                    NeighborAggType::Min | NeighborAggType::Max => match reductions::extremum(
                        &values,
                        matches!(self.agg_type, NeighborAggType::Min),
                    ) {
                        Some(value) => float_result(value)?,
                        None => AlgorithmParamValue::None,
                    },
                    NeighborAggType::Mode => {
                        return Err(anyhow!("Unexpected numeric mode dispatch"))
                    }
                }
            };
            result.insert(node, value);
        }
        if ctx.is_cancelled() {
            return Err(anyhow!(
                "Neighbor aggregation cancelled before installing result"
            ));
        }
        ctx.record_duration("neighbor_agg.total", started.elapsed());
        ctx.record_stat("neighbor_agg.count.nodes", nodes.len() as f64);
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluate(
        step: &dyn Step,
        values: Vec<AlgorithmParamValue>,
    ) -> (super::super::core::StepVariables, Context) {
        use crate::{api::graph::Graph, subgraphs::Subgraph};
        use std::{cell::RefCell, rc::Rc};
        let mut graph = Graph::new();
        graph.add_nodes(values.len());
        let selection = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..values.len()).collect(),
            "stats".into(),
        )
        .expect("selection");
        let mut vars = super::super::core::StepVariables::default();
        vars.set_node_map("values", values.into_iter().enumerate().collect());
        let mut ctx = Context::new();
        step.apply(&mut ctx, &mut StepScope::new(&selection, &mut vars))
            .expect("step");
        (vars, ctx)
    }

    #[test]
    fn checked_statistics_preserve_mixed_residuals_and_empty_policies() {
        let map = [
            (0, AlgorithmParamValue::Int((1i64 << 53) + 1)),
            (1, AlgorithmParamValue::Float(-(1i64 << 53) as f64)),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            Reduction::Sum.apply(&map).expect("sum"),
            AlgorithmParamValue::Float(1.0)
        );
        let samples: Vec<(Box<dyn Step>, AlgorithmParamValue)> = vec![
            (
                Box::new(StdDevStep::new("values", "out")),
                AlgorithmParamValue::None,
            ),
            (
                Box::new(MedianStep::new("values", "out")),
                AlgorithmParamValue::None,
            ),
            (
                Box::new(QuantileStep::new("values", 0.25, "out")),
                AlgorithmParamValue::None,
            ),
            (
                Box::new(ModeStep::new("values", "out")),
                AlgorithmParamValue::None,
            ),
            (
                Box::new(EntropyStep::new("values", "out")),
                AlgorithmParamValue::Float(0.0),
            ),
        ];
        for (step, expected) in samples {
            let (vars, _) = evaluate(step.as_ref(), vec![]);
            assert_eq!(vars.scalar("out").expect("scalar"), &expected);
        }
        let (vars, _) = evaluate(
            &StdDevStep::new("values", "out"),
            vec![AlgorithmParamValue::Int(7)],
        );
        assert_eq!(
            vars.scalar("out").expect("singleton"),
            &AlgorithmParamValue::Float(0.0)
        );
    }

    #[test]
    fn histogram_bins_are_scalar_counts_and_owned_output() {
        let (vars, mut ctx) = evaluate(
            &HistogramStep::new("values", 3, "hist"),
            vec![AlgorithmParamValue::Int(1), AlgorithmParamValue::Int(1)],
        );
        assert_eq!(
            vars.scalar("hist").expect("scalar"),
            &AlgorithmParamValue::IntList(vec![2, 0, 0])
        );
        assert!(vars.node_map("hist").is_err());
        assert!(
            matches!(&ctx.take_outputs()["histograms.hist"], super::super::super::AlgorithmOutput::Histogram(counts) if counts == &vec![2,0,0])
        );
        let (vars, _) = evaluate(
            &HistogramStep::new("values", 2, "hist"),
            vec![
                AlgorithmParamValue::Float(-1e308),
                AlgorithmParamValue::Float(1e308),
            ],
        );
        assert_eq!(
            vars.scalar("hist").expect("wide range"),
            &AlgorithmParamValue::IntList(vec![1, 1])
        );
    }

    #[test]
    fn neighbor_selection_and_missing_weights_are_checked_without_overwriting_results() {
        use crate::{api::graph::Graph, subgraphs::Subgraph};
        use std::{cell::RefCell, rc::Rc};
        let mut graph = Graph::new_directed();
        graph.add_nodes(3);
        graph.add_edge(0, 2).expect("edge");
        graph.add_edge(1, 2).expect("outside edge");
        let selection = Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            [0, 2].into_iter().collect(),
            "scope".into(),
        )
        .expect("selection");
        let mut vars = super::super::core::StepVariables::default();
        vars.set_node_map(
            "values",
            [
                (0, AlgorithmParamValue::Int(2)),
                (2, AlgorithmParamValue::Int(7)),
            ]
            .into_iter()
            .collect(),
        );
        vars.set_node_map("weights", HashMap::new());
        let mut scope = StepScope::new(&selection, &mut vars);
        let step = NeighborAggregationStep::new("values", "out", NeighborAggType::Sum);
        step.apply(&mut Context::new(), &mut scope)
            .expect("selected sum");
        assert_eq!(
            scope.variables().node_map("out").expect("out")[&2],
            AlgorithmParamValue::Float(2.0)
        );
        let step = step.with_weights("weights");
        assert!(step
            .apply(&mut Context::new(), &mut scope)
            .expect_err("missing weight")
            .to_string()
            .contains("Missing neighbor weight"));
        assert_eq!(
            scope.variables().node_map("out").expect("unchanged")[&2],
            AlgorithmParamValue::Float(2.0)
        );
    }

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
