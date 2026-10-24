//! Transformation step primitives that map values.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::arithmetic::ModeTieBreak;
use super::core::{Step, StepInput, StepMetadata, StepScope};
use super::expression::{Expr, ExprContext};
use super::filtering::compare_values;

fn selected_nodes(subgraph: &crate::subgraphs::Subgraph, ctx: &Context) -> Result<Vec<NodeId>> {
    if ctx.is_cancelled() {
        return Err(anyhow!("transformation cancelled"));
    }
    let nodes = subgraph.ordered_nodes().to_vec();
    let graph = subgraph.graph();
    let graph = graph.borrow();
    for &node in &nodes {
        if ctx.is_cancelled() {
            return Err(anyhow!("transformation cancelled"));
        }
        if !graph.has_node(node) {
            return Err(anyhow!("Transformation node {} is inactive", node));
        }
    }
    Ok(nodes)
}

/// Checked selected outgoing adjacency; undirected edges contribute both ways,
/// with a stored self-loop contributing once. Preserve parallel incidences.
pub(super) fn selected_neighbors(
    subgraph: &crate::subgraphs::Subgraph,
    ctx: &Context,
) -> Result<(Vec<NodeId>, HashMap<NodeId, Vec<NodeId>>)> {
    let nodes = selected_nodes(subgraph, ctx)?;
    let mut rows: HashMap<_, Vec<_>> = nodes.iter().map(|&node| (node, Vec::new())).collect();
    let graph = subgraph.graph();
    let graph = graph.borrow();
    for &edge in subgraph.ordered_edges().iter() {
        if ctx.is_cancelled() {
            return Err(anyhow!("transformation cancelled"));
        }
        if !graph.has_edge(edge) {
            return Err(anyhow!("Transformation edge {} is inactive", edge));
        }
        let (source, target) = graph.edge_endpoints(edge)?;
        if !rows.contains_key(&source) || !rows.contains_key(&target) {
            return Err(anyhow!(
                "Transformation edge {} endpoint is outside selection",
                edge
            ));
        }
        rows.get_mut(&source)
            .ok_or_else(|| anyhow!("Missing selected endpoint"))?
            .push(target);
        if graph.is_undirected() && source != target {
            rows.get_mut(&target)
                .ok_or_else(|| anyhow!("Missing selected endpoint"))?
                .push(source);
        }
    }
    for row in rows.values_mut() {
        row.sort_unstable();
    }
    Ok((nodes, rows))
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

/// Transform node values using a callback.
///
/// **Note**: This step is designed for **programmatic Rust use only**.
/// It cannot be registered in the step registry because it requires an
/// `Arc<dyn NodeMapFn>` callback, which cannot be serialized in a `StepSpec`.
///
/// For declarative pipeline building, use arithmetic operations
/// (`core.add`, `core.mul`, etc.) or create custom steps with specific logic.
///
/// # Example (Rust API)
/// ```ignore
/// let mapper = Arc::new(|node: NodeId, input: &StepInput| {
///     // Custom transformation logic
///     Ok(AlgorithmParamValue::Float(node.0 as f64))
/// });
/// let step = MapNodesStep::new("input", "output", mapper);
/// ```
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
        use std::time::Instant;

        let start = Instant::now();

        scope.variables().node_map(&self.source)?;

        // STYLE_ALGO: Use ordered_nodes for determinism
        let nodes = selected_nodes(scope.subgraph(), ctx)?;
        ctx.record_stat("map_nodes_callback.count.nodes", nodes.len() as f64);

        let mut result = HashMap::with_capacity(nodes.len());

        // Prepare step input (reused)
        let reader = StepInput {
            subgraph: scope.subgraph(),
            variables: scope.variables(),
        };

        for &node in nodes.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("map_nodes cancelled"));
            }

            let value = self.mapper.map(node, &reader)?;
            result.insert(node, value);
        }

        if ctx.is_cancelled() {
            return Err(anyhow!("map_nodes cancelled"));
        }
        ctx.record_duration("map_nodes_callback.total", start.elapsed());

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

/// Update target labels in-place using neighbor mode (async label propagation semantics).
pub struct NeighborModeUpdateStep {
    target: String,
    include_self: bool,
    tie_break: ModeTieBreak,
    ordered: bool,
    output: Option<String>,
}

impl NeighborModeUpdateStep {
    pub fn new(
        target: impl Into<String>,
        include_self: bool,
        tie_break: ModeTieBreak,
        ordered: bool,
    ) -> Self {
        Self {
            target: target.into(),
            include_self,
            tie_break,
            ordered,
            output: None,
        }
    }

    pub fn with_output(mut self, output: Option<String>) -> Self {
        self.output = output;
        self
    }
}

impl Step for NeighborModeUpdateStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.neighbor_mode_update"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Update labels in-place using neighbor mode with deterministic ordering"
                .to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.neighbor_mode_update cancelled"));
        }

        let (nodes, neighbors) = selected_neighbors(scope.subgraph(), ctx)?;
        // Work privately: traversal remains asynchronous, but failures never install
        // a partially updated variable. Both ordering modes use stable ID order.
        let mut target_map = scope.variables().node_map(&self.target)?.clone();
        let iteration_nodes = nodes;
        let _ordered = self.ordered;
        let mut counts: HashMap<String, (usize, AlgorithmParamValue)> = HashMap::new();
        let mut order_keys: Vec<String> = Vec::new();

        for node_id in iteration_nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("core.neighbor_mode_update cancelled"));
            }

            counts.clear();
            order_keys.clear();

            // Helper closure to insert value into counts map.
            let mut record_value = |value: &AlgorithmParamValue| -> Result<()> {
                super::expression::validate_value(value)?;
                let key = serde_json::to_string(value)?;
                order_keys.push(key.clone());
                counts
                    .entry(key)
                    .and_modify(|(count, _)| *count += 1)
                    .or_insert((1, value.clone()));
                Ok(())
            };

            if self.include_self {
                if let Some(value) = target_map.get(&node_id) {
                    record_value(value)?;
                }
            }

            for neighbor_node in &neighbors[&node_id] {
                if let Some(value) = target_map.get(neighbor_node) {
                    record_value(value)?;
                }
            }

            if counts.is_empty() {
                continue;
            }

            let max_freq = counts.values().map(|(count, _)| *count).max().unwrap_or(0);

            let mut candidates: Vec<(String, AlgorithmParamValue)> = counts
                .iter()
                .filter(|(_, (count, _))| *count == max_freq)
                .map(|(key, (_, value))| (key.clone(), value.clone()))
                .collect();

            if !matches!(self.tie_break, ModeTieBreak::Keep) {
                for (_, value) in &candidates {
                    if compare_values(value, &candidates[0].1).is_none() {
                        return Err(anyhow!(
                            "Neighbor mode ties require comparable scalar labels"
                        ));
                    }
                }
            }
            let selected = match self.tie_break {
                ModeTieBreak::Keep => {
                    let mut chosen: Option<AlgorithmParamValue> = None;
                    for key in &order_keys {
                        if let Some((_, value)) = candidates
                            .iter()
                            .find(|(candidate_key, _)| candidate_key == key)
                        {
                            chosen = Some(value.clone());
                            break;
                        }
                    }
                    chosen.unwrap_or_else(|| candidates[0].1.clone())
                }
                ModeTieBreak::Lowest => {
                    candidates.sort_by(|(ak, a), (bk, b)| {
                        compare_values(a, b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| ak.cmp(bk))
                    });
                    candidates[0].1.clone()
                }
                ModeTieBreak::Highest => {
                    candidates.sort_by(|(ak, a), (bk, b)| {
                        compare_values(b, a)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| ak.cmp(bk))
                    });
                    candidates[0].1.clone()
                }
            };

            target_map.insert(node_id, selected);
        }

        if ctx.is_cancelled() {
            return Err(anyhow!("neighbor mode cancelled"));
        }
        if let Some(output_var) = &self.output {
            if output_var != &self.target {
                scope
                    .variables_mut()
                    .set_node_map(output_var.clone(), target_map.clone());
            }
        }
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), target_map);

        Ok(())
    }
}

/// Transform node values using expressions (serializable version).
///
/// This step uses the expression system to apply transformations
/// that can be serialized in `StepSpec` and used from Python/JSON.
///
/// # Example
/// ```ignore
/// // Double all values
/// MapNodesExprStep::new(
///     "input",
///     "output",
///     Expr::binary(BinaryOp::Mul, Expr::var("value"), Expr::constant(2.0.into()))
/// )
/// ```
pub struct MapNodesExprStep {
    source: String,
    target: String,
    expr: Expr,
    /// If true, updates are written back to source immediately (async/in-place updates)
    async_update: bool,
}

impl MapNodesExprStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, expr: Expr) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            expr,
            async_update: false,
        }
    }

    pub fn with_async_update(mut self, async_update: bool) -> Self {
        self.async_update = async_update;
        self
    }
}

impl Step for MapNodesExprStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.map_nodes"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Map node values using expression".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let start = std::time::Instant::now();
        if ctx.is_cancelled() {
            return Err(anyhow!("map_nodes cancelled"));
        }
        let nodes = selected_nodes(scope.subgraph(), ctx)?;
        let neighbors = if self.expr.needs_neighbors() {
            Some(selected_neighbors(scope.subgraph(), ctx)?.1)
        } else {
            None
        };
        ctx.record_stat("map_nodes.count.nodes", nodes.len() as f64);
        // Clone only the async source map once. Expressions resolve this local
        // override while other variables remain borrowed from the original scope.
        let mut working = if self.async_update {
            Some(scope.variables().node_map(&self.source)?.clone())
        } else {
            None
        };
        let mut result = if self.async_update {
            HashMap::new()
        } else {
            HashMap::with_capacity(nodes.len())
        };
        for node in nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("map_nodes cancelled"));
            }
            let variables = scope.variables();
            let input = StepInput {
                subgraph: scope.subgraph(),
                variables,
            };
            let source_map = working
                .as_ref()
                .or_else(|| variables.node_map(&self.source).ok());
            let value = source_map.and_then(|map| map.get(&node));
            let expr_ctx = if let Some(value) = value {
                ExprContext::with_value(node, &input, value)
            } else {
                ExprContext::new(node, &input)
            };
            let value = self.expr.eval_scoped(
                &expr_ctx,
                ctx.property_batch.as_ref(),
                neighbors.as_ref().map(|rows| rows[&node].as_slice()),
                working.as_ref().map(|map| (self.source.as_str(), map)),
            )?;
            if let Some(working) = &mut working {
                working.insert(node, value);
            } else {
                result.insert(node, value);
            }
        }
        if ctx.is_cancelled() {
            return Err(anyhow!("map_nodes cancelled"));
        }
        if let Some(updated) = working {
            if self.source != self.target {
                scope
                    .variables_mut()
                    .set_node_map(self.source.clone(), updated.clone());
            }
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), updated);
        } else {
            scope
                .variables_mut()
                .set_node_map(self.target.clone(), result);
        }
        ctx.record_duration("map_nodes.total", start.elapsed());
        Ok(())
    }
}

/// Update a target map in-place with values from a source map.
///
/// This step supports "async" update semantics where each node's update
/// is immediately visible to subsequent nodes during traversal.
/// When `ordered` is true, nodes are processed in deterministic order
/// so later nodes see earlier updates - essential for Label Propagation.
///
/// # Parameters
/// - `source`: Map containing new values
/// - `target`: Map to update in place (gets mutated)
/// - `ordered`: If true, process nodes in sorted order for determinism
///
/// # Example
/// Used in LPA to propagate label updates:
/// ```ignore
/// // new_labels = mode(collect_neighbor_values(labels))
/// // labels = update_in_place(new_labels, target=labels, ordered=true)
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UpdateInPlaceStep {
    source: String,
    target: String,
    #[serde(default)]
    ordered: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output: Option<String>,
}

impl UpdateInPlaceStep {
    pub fn new(source: String, target: String, ordered: bool) -> Self {
        Self {
            source,
            target,
            ordered,
            output: None,
        }
    }

    pub fn with_output(mut self, output: String) -> Self {
        self.output = Some(output);
        self
    }
}

impl Step for UpdateInPlaceStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.update_in_place"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Update target map in-place with source values".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        let start = std::time::Instant::now();

        if ctx.is_cancelled() {
            return Err(anyhow!("core.update_in_place cancelled"));
        }

        // Get source values (the new values to apply) - clone to avoid borrow issues
        let source_map = scope.variables().node_map(&self.source)?.clone();

        let selected = selected_nodes(scope.subgraph(), ctx)?;
        let selected: std::collections::HashSet<_> = selected.into_iter().collect();
        if source_map.keys().any(|node| !selected.contains(node)) {
            return Err(anyhow!(
                "update_in_place source node is outside active selection"
            ));
        }
        let mut nodes: Vec<_> = source_map.keys().copied().collect();
        nodes.sort_unstable();
        let _ordered = self.ordered;
        let mut target_map = scope.variables().node_map(&self.target)?.clone();
        for node_id in nodes {
            if ctx.is_cancelled() {
                return Err(anyhow!("update_in_place cancelled"));
            }
            if let Some(new_value) = source_map.get(&node_id) {
                target_map.insert(node_id, new_value.clone());
            }
        }
        if ctx.is_cancelled() {
            return Err(anyhow!("update_in_place cancelled"));
        }
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), target_map);

        ctx.record_duration("update_in_place.total", start.elapsed());

        // If output is specified and different from target, clone to output
        if let Some(output_var) = &self.output {
            if output_var != &self.target {
                let final_map = scope.variables().node_map(&self.target)?.clone();
                scope
                    .variables_mut()
                    .set_node_map(output_var.clone(), final_map);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::steps::{expression::BinaryOp, StepVariables},
        subgraphs::Subgraph,
        Graph,
    };
    use std::{cell::RefCell, rc::Rc};

    fn path() -> Subgraph {
        let mut graph = Graph::new_undirected();
        graph.add_nodes(3);
        graph.add_edge(0, 1).expect("edge");
        graph.add_edge(1, 2).expect("edge");
        Subgraph::from_nodes(
            Rc::new(RefCell::new(graph)),
            (0..3).collect(),
            "path".into(),
        )
        .expect("selection")
    }

    #[test]
    fn asynchronous_expression_preserves_visibility_without_partial_install_on_error() {
        let selection = path();
        let original = HashMap::from([
            (0, AlgorithmParamValue::Int(1)),
            (1, AlgorithmParamValue::Int(2)),
            (2, AlgorithmParamValue::Int(3)),
        ]);
        let expression = Expr::Call {
            func: "sum".into(),
            args: vec![Expr::Call {
                func: "neighbor_values".into(),
                args: vec![Expr::var("values")],
            }],
        };
        for (asynchronous, expected) in [(false, [2, 4, 2]), (true, [2, 5, 5])] {
            let mut vars = StepVariables::default();
            vars.set_node_map("values", original.clone());
            let mut scope = StepScope::new(&selection, &mut vars);
            MapNodesExprStep::new("values", "result", expression.clone())
                .with_async_update(asynchronous)
                .apply(&mut Context::new(), &mut scope)
                .expect("map");
            for (node, expected) in expected.into_iter().enumerate() {
                assert_eq!(
                    scope.variables().node_map("result").expect("result")[&node],
                    AlgorithmParamValue::Int(expected)
                );
            }
        }
        let mut vars = StepVariables::default();
        vars.set_node_map("values", original.clone());
        vars.set_node_map("result", original.clone());
        let mut scope = StepScope::new(&selection, &mut vars);
        let invalid = Expr::binary(
            BinaryOp::Div,
            Expr::constant(AlgorithmParamValue::Int(1)),
            Expr::binary(
                BinaryOp::Sub,
                Expr::var("value"),
                Expr::constant(AlgorithmParamValue::Int(2)),
            ),
        );
        assert!(MapNodesExprStep::new("values", "result", invalid)
            .with_async_update(true)
            .apply(&mut Context::new(), &mut scope)
            .is_err());
        assert_eq!(
            scope.variables().node_map("values").expect("source"),
            &original
        );
        assert_eq!(
            scope.variables().node_map("result").expect("target"),
            &original
        );
    }

    #[test]
    fn callbacks_propagate_errors_and_remain_uncertified() {
        let selection = path();
        let mut vars = StepVariables::default();
        vars.set_node_map("values", HashMap::from([(0, AlgorithmParamValue::Int(1))]));
        let mut scope = StepScope::new(&selection, &mut vars);
        let callback: Arc<dyn NodeMapFn> =
            Arc::new(|_: NodeId, _: &StepInput<'_>| Err(anyhow!("callback failed")));
        let step = MapNodesStep::new("values", "result", callback);
        assert!(!step.supports_atomic_properties());
        assert!(step.apply(&mut Context::new(), &mut scope).is_err());
        assert!(scope.variables().node_map("result").is_err());
    }

    #[test]
    fn cancellation_and_foreign_update_sources_leave_variables_unchanged() {
        let selection = path();
        let mut vars = StepVariables::default();
        let original = HashMap::from([(0, AlgorithmParamValue::Int(7))]);
        vars.set_node_map("values", original.clone());
        vars.set_node_map(
            "foreign",
            HashMap::from([(99, AlgorithmParamValue::Int(9))]),
        );
        let mut scope = StepScope::new(&selection, &mut vars);
        assert!(
            UpdateInPlaceStep::new("foreign".into(), "values".into(), false)
                .apply(&mut Context::new(), &mut scope)
                .is_err()
        );
        let mut ctx =
            Context::with_cancel_token(Arc::new(std::sync::atomic::AtomicBool::new(true)));
        assert!(
            NeighborModeUpdateStep::new("values", true, ModeTieBreak::Lowest, true)
                .apply(&mut ctx, &mut scope)
                .is_err()
        );
        assert_eq!(
            scope.variables().node_map("values").expect("source"),
            &original
        );
    }
}
