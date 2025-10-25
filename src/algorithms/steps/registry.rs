//! Registration of core step primitives.

use std::sync::Once;

use anyhow::{anyhow, Result};

use crate::types::AttrName;

use super::super::{AlgorithmParamValue, CostHint};
use super::aggregations::{
    EntropyStep, HistogramStep, MedianStep, ModeStep, QuantileStep, ReduceNodeValuesStep,
    Reduction, StdDevStep,
};
use super::arithmetic::{read_binary_operands, BinaryArithmeticStep, BinaryOperation};
use super::attributes::{
    AttachEdgeAttrStep, AttachNodeAttrStep, EdgeWeightScaleStep, LoadEdgeAttrStep, LoadNodeAttrStep,
};
use super::community::{CommunitySeedStep, LabelPropagateStep, ModularityGainStep, SeedStrategy};
use super::core::{global_step_registry, StepMetadata, StepRegistry};
use super::expression::Expr;
use super::filtering::{
    FilterEdgesByAttrStep, FilterNodesByAttrStep, Predicate, SortNodesByAttrStep, SortOrder,
    TopKStep,
};
use super::flow::{FlowUpdateStep, ResidualCapacityStep};
use super::init::{InitEdgesStep, InitNodesStep};
use super::normalization::{
    ClipValuesStep, NormalizeMethod, NormalizeNodeValuesStep, NormalizeValuesStep, StandardizeStep,
};
use super::pathfinding::{KShortestPathsStep, RandomWalkStep, ShortestPathMapStep};
use super::structural::{
    EdgeWeightSumStep, KCoreMarkStep, NodeDegreeStep, TriangleCountStep, WeightedDegreeStep,
};
use super::temporal::{
    AggregateFunction, DiffEdgesStep, DiffNodesStep, MarkChangedNodesStep, TemporalFilterStep,
    TemporalPredicate, WindowAggregateStep,
};
use super::transformations::MapNodesExprStep;

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
        "core.load_attr",
        StepMetadata {
            id: "core.load_attr".to_string(),
            description: "Load node attribute into variable".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let target = spec
                .params
                .get_text("target")
                .or_else(|| spec.params.get_text("output"))
                .ok_or_else(|| anyhow!("core.load_attr requires 'target' or 'output' param"))?
                .to_string();
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
        "core.add",
        StepMetadata {
            id: "core.add".to_string(),
            description: BinaryOperation::Add.description().to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let (left, right, target) = read_binary_operands(spec, "core.add")?;
            Ok(Box::new(BinaryArithmeticStep::new(
                "core.add",
                left,
                right,
                target,
                BinaryOperation::Add,
            )))
        },
    )?;

    registry.register(
        "core.sub",
        StepMetadata {
            id: "core.sub".to_string(),
            description: BinaryOperation::Sub.description().to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let (left, right, target) = read_binary_operands(spec, "core.sub")?;
            Ok(Box::new(BinaryArithmeticStep::new(
                "core.sub",
                left,
                right,
                target,
                BinaryOperation::Sub,
            )))
        },
    )?;

    registry.register(
        "core.mul",
        StepMetadata {
            id: "core.mul".to_string(),
            description: BinaryOperation::Mul.description().to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let (left, right, target) = read_binary_operands(spec, "core.mul")?;
            Ok(Box::new(BinaryArithmeticStep::new(
                "core.mul",
                left,
                right,
                target,
                BinaryOperation::Mul,
            )))
        },
    )?;

    registry.register(
        "core.div",
        StepMetadata {
            id: "core.div".to_string(),
            description: BinaryOperation::Div.description().to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let (left, right, target) = read_binary_operands(spec, "core.div")?;
            Ok(Box::new(BinaryArithmeticStep::new(
                "core.div",
                left,
                right,
                target,
                BinaryOperation::Div,
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
        "core.std",
        StepMetadata {
            id: "core.std".to_string(),
            description: "Compute standard deviation of node values".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.std requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.std requires 'target' param"))?;
            Ok(Box::new(StdDevStep::new(
                source.to_string(),
                target.to_string(),
            )))
        },
    )?;

    registry.register(
        "core.median",
        StepMetadata {
            id: "core.median".to_string(),
            description: "Compute median value of node values".to_string(),
            cost_hint: CostHint::Linearithmic,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.median requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.median requires 'target' param"))?;
            Ok(Box::new(MedianStep::new(
                source.to_string(),
                target.to_string(),
            )))
        },
    )?;

    registry.register(
        "core.mode",
        StepMetadata {
            id: "core.mode".to_string(),
            description: "Find most common value in node values".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.mode requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.mode requires 'target' param"))?;
            Ok(Box::new(ModeStep::new(
                source.to_string(),
                target.to_string(),
            )))
        },
    )?;

    registry.register(
        "core.quantile",
        StepMetadata {
            id: "core.quantile".to_string(),
            description: "Compute q-th quantile of node values".to_string(),
            cost_hint: CostHint::Linearithmic,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.quantile requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.quantile requires 'target' param"))?;
            let q = match spec.params.get("q") {
                Some(AlgorithmParamValue::Float(v)) => *v,
                Some(AlgorithmParamValue::Int(v)) => *v as f64,
                _ => return Err(anyhow!("core.quantile requires 'q' float param")),
            };
            Ok(Box::new(QuantileStep::new(
                source.to_string(),
                q,
                target.to_string(),
            )))
        },
    )?;

    registry.register(
        "core.entropy",
        StepMetadata {
            id: "core.entropy".to_string(),
            description: "Compute Shannon entropy of node values".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.entropy requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.entropy requires 'target' param"))?;
            Ok(Box::new(EntropyStep::new(
                source.to_string(),
                target.to_string(),
            )))
        },
    )?;

    registry.register(
        "core.histogram",
        StepMetadata {
            id: "core.histogram".to_string(),
            description: "Compute histogram of node values with specified bins".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec
                .params
                .get_text("source")
                .ok_or_else(|| anyhow!("core.histogram requires 'source' param"))?;
            let target = spec
                .params
                .get_text("target")
                .ok_or_else(|| anyhow!("core.histogram requires 'target' param"))?;
            let bins = match spec.params.get("bins") {
                Some(AlgorithmParamValue::Int(v)) => *v as usize,
                _ => return Err(anyhow!("core.histogram requires 'bins' integer param")),
            };
            Ok(Box::new(HistogramStep::new(
                source.to_string(),
                bins,
                target.to_string(),
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
        "core.weighted_degree",
        StepMetadata {
            id: "core.weighted_degree".to_string(),
            description: "Compute weighted node degrees".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let weight_attr = spec.params.expect_text("weight_attr")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(WeightedDegreeStep::new(
                AttrName::from(weight_attr),
                target,
            )))
        },
    )?;

    registry.register(
        "core.k_core_mark",
        StepMetadata {
            id: "core.k_core_mark".to_string(),
            description: "Mark nodes in k-core".to_string(),
            cost_hint: CostHint::Quadratic,
        },
        |spec| {
            let k = spec.params.expect_int("k")? as usize;
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(KCoreMarkStep::new(k, target)))
        },
    )?;

    registry.register(
        "core.triangle_count",
        StepMetadata {
            id: "core.triangle_count".to_string(),
            description: "Count triangles per node".to_string(),
            cost_hint: CostHint::Quadratic,
        },
        |spec| {
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(TriangleCountStep::new(target)))
        },
    )?;

    registry.register(
        "core.edge_weight_sum",
        StepMetadata {
            id: "core.edge_weight_sum".to_string(),
            description: "Sum edge weights incident to nodes".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let weight_attr = spec.params.expect_text("weight_attr")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(EdgeWeightSumStep::new(
                AttrName::from(weight_attr),
                target,
            )))
        },
    )?;

    registry.register(
        "core.edge_weight_scale",
        StepMetadata {
            id: "core.edge_weight_scale".to_string(),
            description: "Scale edge weights by factor".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let attr = spec.params.expect_text("attr")?.to_string();
            let factor = spec.params.get_float("factor").unwrap_or(1.0);
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(EdgeWeightScaleStep::new(
                AttrName::from(attr),
                factor,
                target,
            )))
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

    registry.register(
        "core.normalize_values",
        StepMetadata {
            id: "core.normalize_values".to_string(),
            description: "Normalize values (node or edge map) using sum/max/minmax".to_string(),
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
            Ok(Box::new(NormalizeValuesStep::new(
                source, target, method, epsilon,
            )))
        },
    )?;

    registry.register(
        "core.standardize",
        StepMetadata {
            id: "core.standardize".to_string(),
            description: "Z-score standardization (mean=0, std=1)".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let epsilon = spec.params.get_float("epsilon").unwrap_or(1e-9);
            Ok(Box::new(StandardizeStep::new(source, target, epsilon)))
        },
    )?;

    registry.register(
        "core.clip",
        StepMetadata {
            id: "core.clip".to_string(),
            description: "Clip values to range [min, max]".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let min = spec.params.get_float("min").unwrap_or(f64::NEG_INFINITY);
            let max = spec.params.get_float("max").unwrap_or(f64::INFINITY);
            Ok(Box::new(ClipValuesStep::new(source, target, min, max)))
        },
    )?;

    registry.register(
        "core.map_nodes",
        StepMetadata {
            id: "core.map_nodes".to_string(),
            description: "Map node values using expression".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();

            // Parse expression from params
            // Convert AlgorithmParamValue to serde_json::Value for parsing
            let expr_value = spec
                .params
                .get("expr")
                .ok_or_else(|| anyhow!("'expr' parameter is required"))?;

            // Convert to JSON value for serde deserialization
            let json_value = serde_json::to_value(expr_value)
                .map_err(|e| anyhow!("failed to convert expr to JSON: {}", e))?;

            let expr: Expr = serde_json::from_value(json_value)
                .map_err(|e| anyhow!("failed to parse expression: {}", e))?;

            Ok(Box::new(MapNodesExprStep::new(source, target, expr)))
        },
    )?;

    // === Filtering & Ordering Steps ===

    registry.register(
        "core.sort_nodes_by_attr",
        StepMetadata {
            id: "core.sort_nodes_by_attr".to_string(),
            description: "Sort nodes by attribute value".to_string(),
            cost_hint: CostHint::Linearithmic,
        },
        |spec| {
            let attr = spec.params.expect_text("attr")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let order = match spec.params.get_text("order") {
                Some("desc") | Some("descending") => SortOrder::Descending,
                Some("asc") | Some("ascending") | None => SortOrder::Ascending,
                Some(other) => return Err(anyhow!("unknown sort order: {}", other)),
            };
            Ok(Box::new(SortNodesByAttrStep::new(attr, order, target)))
        },
    )?;

    registry.register(
        "core.filter_nodes_by_attr",
        StepMetadata {
            id: "core.filter_nodes_by_attr".to_string(),
            description: "Filter nodes by attribute predicate".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let attr = spec.params.expect_text("attr")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();

            // Parse predicate from params
            let pred_value = spec
                .params
                .get("predicate")
                .ok_or_else(|| anyhow!("'predicate' parameter is required"))?;
            let json_value = serde_json::to_value(pred_value)
                .map_err(|e| anyhow!("failed to convert predicate to JSON: {}", e))?;
            let predicate: Predicate = serde_json::from_value(json_value)
                .map_err(|e| anyhow!("failed to parse predicate: {}", e))?;

            Ok(Box::new(FilterNodesByAttrStep::new(
                attr, predicate, target,
            )))
        },
    )?;

    registry.register(
        "core.filter_edges_by_attr",
        StepMetadata {
            id: "core.filter_edges_by_attr".to_string(),
            description: "Filter edges by attribute predicate".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let attr = spec.params.expect_text("attr")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();

            // Parse predicate from params
            let pred_value = spec
                .params
                .get("predicate")
                .ok_or_else(|| anyhow!("'predicate' parameter is required"))?;
            let json_value = serde_json::to_value(pred_value)
                .map_err(|e| anyhow!("failed to convert predicate to JSON: {}", e))?;
            let predicate: Predicate = serde_json::from_value(json_value)
                .map_err(|e| anyhow!("failed to parse predicate: {}", e))?;

            Ok(Box::new(FilterEdgesByAttrStep::new(
                attr, predicate, target,
            )))
        },
    )?;

    registry.register(
        "core.top_k",
        StepMetadata {
            id: "core.top_k".to_string(),
            description: "Select top-k values from node map".to_string(),
            cost_hint: CostHint::Linearithmic,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let k = spec
                .params
                .get_int("k")
                .ok_or_else(|| anyhow!("'k' parameter is required"))?;
            if k < 0 {
                return Err(anyhow!("k must be non-negative, got {}", k));
            }
            let order = match spec.params.get_text("order") {
                Some("asc") | Some("ascending") => SortOrder::Ascending,
                Some("desc") | Some("descending") | None => SortOrder::Descending,
                Some(other) => return Err(anyhow!("unknown sort order: {}", other)),
            };
            Ok(Box::new(TopKStep::new(source, k as usize, target, order)))
        },
    )?;

    // === Sampling Steps ===

    registry.register(
        "core.sample_nodes",
        StepMetadata {
            id: "core.sample_nodes".to_string(),
            description: "Sample nodes randomly with optional seed".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            use super::sampling::{SampleNodesStep, SampleSpec};

            let target = spec.params.expect_text("target")?.to_string();
            let seed = spec.params.get_int("seed").map(|s| s as u64);

            // Parse sampling spec
            let sample_spec = if let Some(fraction) = spec.params.get_float("fraction") {
                SampleSpec::Fraction { fraction }
            } else if let Some(count) = spec.params.get_int("count") {
                if count < 0 {
                    return Err(anyhow!("count must be non-negative, got {}", count));
                }
                SampleSpec::Count {
                    count: count as usize,
                }
            } else {
                return Err(anyhow!(
                    "Either 'fraction' or 'count' parameter is required"
                ));
            };

            Ok(Box::new(SampleNodesStep::new(sample_spec, seed, target)))
        },
    )?;

    registry.register(
        "core.sample_edges",
        StepMetadata {
            id: "core.sample_edges".to_string(),
            description: "Sample edges randomly with optional seed".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            use super::sampling::{SampleEdgesStep, SampleSpec};

            let target = spec.params.expect_text("target")?.to_string();
            let seed = spec.params.get_int("seed").map(|s| s as u64);

            let sample_spec = if let Some(fraction) = spec.params.get_float("fraction") {
                SampleSpec::Fraction { fraction }
            } else if let Some(count) = spec.params.get_int("count") {
                if count < 0 {
                    return Err(anyhow!("count must be non-negative, got {}", count));
                }
                SampleSpec::Count {
                    count: count as usize,
                }
            } else {
                return Err(anyhow!(
                    "Either 'fraction' or 'count' parameter is required"
                ));
            };

            Ok(Box::new(SampleEdgesStep::new(sample_spec, seed, target)))
        },
    )?;

    registry.register(
        "core.reservoir_sample",
        StepMetadata {
            id: "core.reservoir_sample".to_string(),
            description: "Reservoir sample k elements from a map variable".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            use super::sampling::{EntityType, ReservoirSampleStep};

            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let k = spec
                .params
                .get_int("k")
                .ok_or_else(|| anyhow!("'k' parameter is required"))?;
            if k < 0 {
                return Err(anyhow!("k must be non-negative, got {}", k));
            }
            let seed = spec.params.get_int("seed").map(|s| s as u64);

            let entity_type = match spec.params.get_text("entity_type") {
                Some("nodes") | None => EntityType::Nodes,
                Some("edges") => EntityType::Edges,
                Some(other) => {
                    return Err(anyhow!(
                        "unknown entity_type: {}, expected 'nodes' or 'edges'",
                        other
                    ))
                }
            };

            Ok(Box::new(ReservoirSampleStep::new(
                source,
                k as usize,
                seed,
                target,
                entity_type,
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

    // Pathfinding steps (Section 1.8)
    registry.register(
        "core.shortest_path_map",
        StepMetadata {
            id: "core.shortest_path_map".to_string(),
            description: "Single-source shortest paths using BFS or Dijkstra".to_string(),
            cost_hint: CostHint::Linearithmic,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let output = spec.params.expect_text("output")?.to_string();
            let weight_attr = spec
                .params
                .get_text("weight_attr")
                .map(|s| AttrName::from(s.to_string()));
            Ok(Box::new(ShortestPathMapStep::new(
                source,
                weight_attr,
                output,
            )))
        },
    )?;

    registry.register(
        "core.k_shortest_paths",
        StepMetadata {
            id: "core.k_shortest_paths".to_string(),
            description: "K-shortest paths using Yen's algorithm".to_string(),
            cost_hint: CostHint::Quadratic,
        },
        |spec| {
            let source = spec.params.expect_text("source")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            let k = spec.params.expect_int("k")? as usize;
            let output = spec.params.expect_text("output")?.to_string();
            let weight_attr = spec
                .params
                .get_text("weight_attr")
                .map(|s| AttrName::from(s.to_string()));
            Ok(Box::new(KShortestPathsStep::new(
                source,
                target,
                k,
                weight_attr,
                output,
            )))
        },
    )?;

    registry.register(
        "core.random_walk",
        StepMetadata {
            id: "core.random_walk".to_string(),
            description: "Random walk sequences with optional restart and weighted transitions"
                .to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let start_nodes = spec.params.expect_text("start_nodes")?.to_string();
            let length = spec.params.expect_int("length")? as usize;
            let output = spec.params.expect_text("output")?.to_string();
            let restart_prob = spec.params.get_float("restart_prob").unwrap_or(0.0);
            let weight_attr = spec
                .params
                .get_text("weight_attr")
                .map(|s| AttrName::from(s.to_string()));
            let seed = spec.params.get_int("seed").map(|i| i as u64);
            Ok(Box::new(RandomWalkStep::new(
                start_nodes,
                length,
                restart_prob,
                weight_attr,
                seed,
                output,
            )))
        },
    )?;

    // Community detection helpers
    registry.register(
        "core.community_seed",
        StepMetadata {
            id: "core.community_seed".to_string(),
            description: "Initialize community labels using a seeding strategy".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let strategy_str = spec.params.expect_text("strategy")?;
            let strategy = SeedStrategy::from_str(strategy_str)?;
            let target = spec.params.expect_text("target")?.to_string();
            let k = spec.params.get_int("k").map(|i| i as usize);
            let seed = spec.params.get_int("seed").map(|i| i as u64);
            Ok(Box::new(CommunitySeedStep::new(strategy, target, k, seed)))
        },
    )?;

    registry.register(
        "core.modularity_gain",
        StepMetadata {
            id: "core.modularity_gain".to_string(),
            description: "Compute modularity change for each node relative to current partition"
                .to_string(),
            cost_hint: CostHint::Quadratic,
        },
        |spec| {
            let partition = spec.params.expect_text("partition")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(ModularityGainStep::new(partition, target)))
        },
    )?;

    registry.register(
        "core.label_propagate_step",
        StepMetadata {
            id: "core.label_propagate_step".to_string(),
            description: "Single iteration of label propagation: adopt most common neighbor label"
                .to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let labels = spec.params.expect_text("labels")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(LabelPropagateStep::new(labels, target)))
        },
    )?;

    // Flow and capacity steps (Section 1.10)
    registry.register(
        "core.flow_update",
        StepMetadata {
            id: "core.flow_update".to_string(),
            description: "Update flow values along edges (flow + delta)".to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let flow = spec.params.expect_text("flow")?.to_string();
            let delta = spec.params.expect_text("delta")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(FlowUpdateStep::new(flow, delta, target)))
        },
    )?;

    registry.register(
        "core.residual_capacity",
        StepMetadata {
            id: "core.residual_capacity".to_string(),
            description: "Compute residual capacity (capacity - flow) for max-flow algorithms"
                .to_string(),
            cost_hint: CostHint::Linear,
        },
        |spec| {
            let capacity = spec.params.expect_text("capacity")?.to_string();
            let flow = spec.params.expect_text("flow")?.to_string();
            let target = spec.params.expect_text("target")?.to_string();
            Ok(Box::new(ResidualCapacityStep::new(capacity, flow, target)))
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
