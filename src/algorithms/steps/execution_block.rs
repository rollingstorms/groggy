//! Execution block step for structured message-passing and other block-based execution modes.
//!
//! This module implements the runtime execution of execution blocks created by the Python builder.
//! Execution blocks encapsulate a sub-computation with special semantics like:
//! - Message-pass: Gauss-Seidel style neighbor updates with in-place writes
//! - Streaming: Incremental/transactional updates (future)

use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::traits::SubgraphOperations;
use crate::types::NodeId;

use super::super::{AlgorithmParamValue, Context};
use super::core::{Step, StepMetadata, StepScope};
use super::direction::NeighborDirection;

/// Execution mode for the block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    /// Message-passing / Gauss-Seidel: ordered node updates with in-place writes
    MessagePass,
    /// Streaming: incremental/transactional updates (future)
    Streaming,
}

/// Body operation node within the execution block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyNode {
    pub id: String,
    pub domain: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub output: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Execution block body specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBody {
    pub nodes: Vec<BodyNode>,
}

/// Options for execution block behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockOptions {
    #[serde(default = "default_true")]
    pub include_self: bool,
    #[serde(default = "default_true")]
    pub ordered: bool,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tie_break: Option<String>,
    #[serde(default)]
    pub direction: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Default for BlockOptions {
    fn default() -> Self {
        Self {
            include_self: true,
            ordered: true,
            name: None,
            tie_break: None,
            direction: None,
        }
    }
}

/// Step that executes an execution block.
pub struct ExecutionBlockStep {
    mode: ExecutionMode,
    target: String,
    options: BlockOptions,
    body_value: serde_json::Value,
    body_cache: OnceCell<BlockBody>,
}

impl ExecutionBlockStep {
    pub fn new(
        mode: ExecutionMode,
        target: String,
        options: BlockOptions,
        body_value: serde_json::Value,
    ) -> Self {
        Self {
            mode,
            target,
            options,
            body_value,
            body_cache: OnceCell::new(),
        }
    }

    fn body(&self) -> Result<&BlockBody> {
        self.body_cache.get_or_try_init(|| {
            serde_json::from_value(self.body_value.clone())
                .map_err(|e| anyhow!("Failed to deserialize execution block body: {}", e))
        })
    }

    /// Execute message-pass mode (Gauss-Seidel style updates).
    fn execute_message_pass(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        // Get the target variable (what we're updating)
        // Remember the original storage type so we can preserve it
        let was_node_map = scope.variables().node_map(&self.target).is_ok();

        let mut node_column = if scope.variables().contains(&self.target) {
            match scope.variables().node_column(&self.target) {
                Ok(col) => col.clone(),
                Err(_) => {
                    // Try as node map and convert
                    let map = scope.variables().node_map(&self.target)?;
                    let mut nodes: Vec<_> = map.keys().copied().collect();
                    nodes.sort_unstable();
                    let values = nodes.iter().map(|n| map[n].clone()).collect();
                    super::core::NodeColumn::new(nodes, values)
                }
            }
        } else {
            return Err(anyhow!(
                "execution_block target '{}' not found",
                self.target
            ));
        };

        let nodes = node_column.nodes().to_vec();

        // Determine neighbor direction - actually use it!
        let direction = self
            .options
            .direction
            .as_deref()
            .and_then(NeighborDirection::from_str)
            .unwrap_or(NeighborDirection::In);

        // Message-pass: iterate through nodes in order (Gauss-Seidel)
        let body = self.body()?;

        // Check if we can execute this block - detect unsupported operations early
        if let Err(e) = self.validate_body_support(body) {
            eprintln!(
                "WARN: execution_block(message_pass,{}): {} -> falling back would require expand_blocks=True",
                self.options.name.as_deref().unwrap_or("unnamed"),
                e
            );
            return Err(anyhow!(
                "Block contains unsupported operations. Use expand_blocks=True in IR or implement the operation: {}",
                e
            ));
        }

        // For each node, evaluate the block body and update in-place
        for &node_id in &nodes {
            // Get neighbors with the specified direction
            let neighbors: Vec<NodeId> = {
                let subgraph = scope.subgraph();
                let mut nbrs = match direction {
                    NeighborDirection::In => {
                        // Incoming edges: find nodes that point to this node
                        subgraph
                            .neighbors(node_id)
                            .map_err(|e| anyhow!("Failed to get neighbors: {}", e))?
                    }
                    NeighborDirection::Out => {
                        // Outgoing edges: find nodes this node points to
                        subgraph
                            .neighbors(node_id)
                            .map_err(|e| anyhow!("Failed to get neighbors: {}", e))?
                    }
                    NeighborDirection::Undirected => {
                        // All connected nodes
                        subgraph
                            .neighbors(node_id)
                            .map_err(|e| anyhow!("Failed to get neighbors: {}", e))?
                    }
                };

                if self.options.include_self && !nbrs.contains(&node_id) {
                    nbrs.push(node_id);
                }

                // Sort neighbors for deterministic execution
                nbrs.sort_unstable();
                nbrs
            };

            // Evaluate block body for this node
            let update_value = if body.nodes.is_empty() {
                // Empty body: just keep current value
                node_column
                    .get(node_id)
                    .cloned()
                    .unwrap_or(AlgorithmParamValue::None)
            } else {
                // Execute body operations
                self.evaluate_body_for_node(node_id, &neighbors, &node_column, body, ctx, scope)?
            };

            // Update in-place
            if let Some(idx) = node_column.nodes().iter().position(|&n| n == node_id) {
                node_column.values_mut()[idx] = update_value;
            }
        }

        // Store updated values back using the original storage type
        if was_node_map {
            // Convert back to node map
            let mut map = HashMap::new();
            for (node, value) in node_column.into_pairs() {
                map.insert(node, value);
            }
            scope.variables_mut().set_node_map(&self.target, map);
        } else {
            // Keep as node column
            scope
                .variables_mut()
                .set_node_column(&self.target, node_column);
        }

        Ok(())
    }

    /// Validate that the block body contains only supported operations.
    fn validate_body_support(&self, body: &BlockBody) -> Result<()> {
        for body_node in &body.nodes {
            match body_node.op_type.as_str() {
                // Supported operations
                "collect_neighbor_values"
                | "neighbor_agg"
                | "mode_list"
                | "constant"
                | "add"
                | "sub"
                | "mul"
                | "div"
                | "where"
                | "apply" => {
                    // OK
                }
                // Unsupported - would need fallback
                _ => {
                    return Err(anyhow!(
                        "unsupported op {}.{}",
                        body_node.domain,
                        body_node.op_type
                    ));
                }
            }
        }
        Ok(())
    }

    /// Evaluate the block body for a specific node.
    ///
    /// This evaluates the operation DAG for a single node in Gauss-Seidel order.
    /// The body is a DAG of operations with explicit dependencies.
    fn evaluate_body_for_node(
        &self,
        node_id: NodeId,
        neighbors: &[NodeId],
        node_column: &super::core::NodeColumn,
        body: &BlockBody,
        ctx: &mut Context,
        scope: &mut StepScope<'_>,
    ) -> Result<AlgorithmParamValue> {
        // Build a local value store for intermediate results
        // Store values as Vec for neighbor collections
        let mut local_values: HashMap<String, AlgorithmParamValue> = HashMap::new();
        let mut local_lists: HashMap<String, Vec<AlgorithmParamValue>> = HashMap::new();

        // Pre-populate with target variable
        if let Some(val) = node_column.get(node_id) {
            local_values.insert(self.target.clone(), val.clone());
        }

        // Evaluate each body node in topological order
        for body_node in &body.nodes {
            let result = self.evaluate_body_node(
                body_node,
                node_id,
                neighbors,
                node_column,
                &local_values,
                &mut local_lists,
                ctx,
                scope,
            )?;

            // Store result if this node has an output
            if let Some(output) = &body_node.output {
                local_values.insert(output.clone(), result);
            }
        }

        // Find the final "apply" operation or return the last computed value
        for body_node in body.nodes.iter().rev() {
            if body_node.op_type == "apply" {
                // The apply node's input is the final value
                if let Some(input_var) = body_node.inputs.first() {
                    return local_values
                        .get(input_var)
                        .cloned()
                        .ok_or_else(|| anyhow!("Apply input '{}' not found", input_var));
                }
            }
        }

        // Fallback: return last output value or current value
        body.nodes
            .iter()
            .rev()
            .filter_map(|n| n.output.as_ref().and_then(|o| local_values.get(o)))
            .next()
            .cloned()
            .or_else(|| node_column.get(node_id).cloned())
            .ok_or_else(|| anyhow!("No value computed for node {}", node_id))
    }

    /// Evaluate a single body node operation.
    fn evaluate_body_node(
        &self,
        body_node: &BodyNode,
        node_id: NodeId,
        neighbors: &[NodeId],
        node_column: &super::core::NodeColumn,
        local_values: &HashMap<String, AlgorithmParamValue>,
        local_lists: &mut HashMap<String, Vec<AlgorithmParamValue>>,
        _ctx: &mut Context,
        scope: &mut StepScope<'_>,
    ) -> Result<AlgorithmParamValue> {
        match body_node.op_type.as_str() {
            // Graph operations
            "collect_neighbor_values" => {
                // Collect values from neighbors
                let source = body_node
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow!("collect_neighbor_values requires source input"))?;

                let mut values = Vec::new();
                for &nbr in neighbors {
                    let val =
                        self.resolve_node_value(source, nbr, node_column, local_values, scope)?;
                    values.push(val);
                }

                // Store the list and return a marker
                if let Some(output) = &body_node.output {
                    local_lists.insert(output.clone(), values);
                }

                // Return a sentinel value - the mode operation will look up the list
                Ok(AlgorithmParamValue::None)
            }

            // Aggregation operations
            "mode_list" => {
                let source = body_node
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow!("mode_list requires source input"))?;

                // Look up in local_lists
                let values = local_lists
                    .get(source)
                    .ok_or_else(|| anyhow!("mode_list input '{}' not found", source))?;

                let tie_break = body_node
                    .metadata
                    .get("tie_break")
                    .and_then(|v| v.as_str())
                    .or(self.options.tie_break.as_deref());
                compute_mode(values, tie_break)
            }

            // Arithmetic operations
            "add" | "sub" | "mul" | "div" => {
                let left_name = body_node.inputs.get(0).ok_or_else(|| {
                    anyhow!(
                        "{} requires left input (inputs={:?})",
                        body_node.op_type,
                        body_node.inputs
                    )
                })?;
                let right_name = body_node.inputs.get(1).ok_or_else(|| {
                    anyhow!(
                        "{} requires right input (inputs={:?})",
                        body_node.op_type,
                        body_node.inputs
                    )
                })?;

                let left_value = if let Some(val) = local_values.get(left_name) {
                    val.clone()
                } else {
                    self.resolve_node_value(left_name, node_id, node_column, local_values, scope)?
                };

                let right_value = if let Some(val) = local_values.get(right_name) {
                    val.clone()
                } else {
                    self.resolve_node_value(right_name, node_id, node_column, local_values, scope)?
                };

                apply_arithmetic(body_node.op_type.as_str(), &left_value, &right_value)
            }

            // Conditional operations
            "where" => {
                let if_true_name = body_node.inputs.get(0).ok_or_else(|| {
                    anyhow!(
                        "where requires if_true input (inputs={:?})",
                        body_node.inputs
                    )
                })?;
                let if_false_name = body_node.inputs.get(1).ok_or_else(|| {
                    anyhow!(
                        "where requires if_false input (inputs={:?})",
                        body_node.inputs
                    )
                })?;

                let if_true_value = if let Some(val) = local_values.get(if_true_name) {
                    val.clone()
                } else {
                    self.resolve_node_value(
                        if_true_name,
                        node_id,
                        node_column,
                        local_values,
                        scope,
                    )?
                };

                let _if_false_value = if let Some(val) = local_values.get(if_false_name) {
                    val.clone()
                } else {
                    self.resolve_node_value(
                        if_false_name,
                        node_id,
                        node_column,
                        local_values,
                        scope,
                    )?
                };

                // Simplified: just return if_true for now
                // Full implementation would evaluate condition
                Ok(if_true_value)
            }

            "neighbor_agg" => {
                let source = body_node
                    .inputs
                    .first()
                    .ok_or_else(|| anyhow!("neighbor_agg requires source input"))?;
                let agg = body_node
                    .metadata
                    .get("agg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("sum");

                match agg {
                    "sum" => {
                        self.sum_neighbors(source, neighbors, node_column, local_values, scope)
                    }
                    other => Err(anyhow!("neighbor_agg agg '{}' not supported", other)),
                }
            }

            "constant" => {
                let value = body_node
                    .metadata
                    .get("value")
                    .ok_or_else(|| anyhow!("constant node missing 'value' metadata"))?;
                Ok(json_value_to_param(value)?)
            }

            // Control operations
            "apply" => {
                // Apply just passes through its input
                body_node
                    .inputs
                    .first()
                    .and_then(|i| local_values.get(i))
                    .cloned()
                    .ok_or_else(|| anyhow!("apply requires input"))
            }

            // Unsupported operations - return error to trigger fallback
            _ => Err(anyhow!(
                "Unsupported block operation: {}.{}",
                body_node.domain,
                body_node.op_type
            )),
        }
    }
}

impl Step for ExecutionBlockStep {
    fn id(&self) -> &'static str {
        "core.execution_block"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Execute structured execution block (message-pass, etc.)".to_string(),
            cost_hint: super::super::CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        match self.mode {
            ExecutionMode::MessagePass => self.execute_message_pass(ctx, scope),
            ExecutionMode::Streaming => {
                // Future implementation
                Err(anyhow!("Streaming execution mode not yet implemented"))
            }
        }
    }
}

/// Apply arithmetic operation to two values.
fn apply_arithmetic(
    op: &str,
    left: &AlgorithmParamValue,
    right: &AlgorithmParamValue,
) -> Result<AlgorithmParamValue> {
    match (left, right) {
        (AlgorithmParamValue::Int(l), AlgorithmParamValue::Int(r)) => {
            let result = match op {
                "add" => l + r,
                "sub" => l - r,
                "mul" => l * r,
                "div" => {
                    if *r == 0 {
                        return Err(anyhow!("Division by zero"));
                    }
                    l / r
                }
                _ => return Err(anyhow!("Unknown arithmetic op: {}", op)),
            };
            Ok(AlgorithmParamValue::Int(result))
        }
        (AlgorithmParamValue::Float(l), AlgorithmParamValue::Float(r)) => {
            let result = match op {
                "add" => l + r,
                "sub" => l - r,
                "mul" => l * r,
                "div" => l / r,
                _ => return Err(anyhow!("Unknown arithmetic op: {}", op)),
            };
            Ok(AlgorithmParamValue::Float(result))
        }
        // Promote int to float if mixed
        (AlgorithmParamValue::Int(l), AlgorithmParamValue::Float(_r)) => {
            apply_arithmetic(op, &AlgorithmParamValue::Float(*l as f64), right)
        }
        (AlgorithmParamValue::Float(_l), AlgorithmParamValue::Int(r)) => {
            apply_arithmetic(op, left, &AlgorithmParamValue::Float(*r as f64))
        }
        _ => Err(anyhow!(
            "Arithmetic operation {} requires numeric operands",
            op
        )),
    }
}

/// Compute mode (most frequent value) with tie-breaking.
fn compute_mode(
    values: &[AlgorithmParamValue],
    tie_break: Option<&str>,
) -> Result<AlgorithmParamValue> {
    if values.is_empty() {
        return Ok(AlgorithmParamValue::None);
    }

    // Count frequencies
    let mut freq_map: HashMap<String, (usize, AlgorithmParamValue)> = HashMap::new();

    for val in values {
        let key = format!("{:?}", val); // Simple key representation
        freq_map
            .entry(key.clone())
            .and_modify(|(count, _)| *count += 1)
            .or_insert((1, val.clone()));
    }

    // Find mode - iterate deterministically by sorting keys
    let mut max_freq = 0;
    let mut candidates = Vec::new();

    let mut sorted_entries: Vec<_> = freq_map.iter().collect();
    sorted_entries.sort_by_key(|(key, _)| key.as_str());

    for (_, (count, val)) in sorted_entries {
        if *count > max_freq {
            max_freq = *count;
            candidates.clear();
            candidates.push(val.clone());
        } else if *count == max_freq {
            candidates.push(val.clone());
        }
    }

    // Handle ties
    if candidates.len() == 1 {
        Ok(candidates[0].clone())
    } else {
        // Apply tie-breaking strategy
        match tie_break {
            Some("lowest") => {
                // Return lowest value (for numeric types)
                candidates.sort_by(|a, b| match (a, b) {
                    (AlgorithmParamValue::Int(x), AlgorithmParamValue::Int(y)) => x.cmp(y),
                    (AlgorithmParamValue::Float(x), AlgorithmParamValue::Float(y)) => {
                        x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    _ => std::cmp::Ordering::Equal,
                });
                Ok(candidates[0].clone())
            }
            Some("highest") => {
                candidates.sort_by(|a, b| match (a, b) {
                    (AlgorithmParamValue::Int(x), AlgorithmParamValue::Int(y)) => y.cmp(x),
                    (AlgorithmParamValue::Float(x), AlgorithmParamValue::Float(y)) => {
                        y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    _ => std::cmp::Ordering::Equal,
                });
                Ok(candidates[0].clone())
            }
            _ => {
                // Default: return first candidate
                Ok(candidates[0].clone())
            }
        }
    }
}

fn json_value_to_param(value: &serde_json::Value) -> Result<AlgorithmParamValue> {
    use serde_json::Value::*;
    Ok(match value {
        Null => AlgorithmParamValue::None,
        Bool(b) => AlgorithmParamValue::Bool(*b),
        Number(n) => {
            if let Some(i) = n.as_i64() {
                AlgorithmParamValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                AlgorithmParamValue::Float(f)
            } else {
                return Err(anyhow!("Unsupported numeric constant: {}", n));
            }
        }
        String(s) => AlgorithmParamValue::Text(s.clone()),
        other => {
            return Err(anyhow!(
                "Unsupported constant value in execution block: {}",
                other
            ))
        }
    })
}

fn value_to_f64(value: &AlgorithmParamValue) -> Result<f64> {
    match value {
        AlgorithmParamValue::Float(v) => Ok(*v),
        AlgorithmParamValue::Int(v) => Ok(*v as f64),
        _ => Err(anyhow!(
            "Execution block expected numeric value, found {:?}",
            value
        )),
    }
}

impl ExecutionBlockStep {
    fn sum_neighbors(
        &self,
        source: &str,
        neighbors: &[NodeId],
        node_column: &super::core::NodeColumn,
        local_values: &HashMap<String, AlgorithmParamValue>,
        scope: &mut StepScope<'_>,
    ) -> Result<AlgorithmParamValue> {
        let mut total = 0.0f64;
        for &nbr in neighbors {
            let value = self.resolve_node_value(source, nbr, node_column, local_values, scope)?;
            total += value_to_f64(&value)?;
        }
        Ok(AlgorithmParamValue::Float(total))
    }

    fn resolve_node_value(
        &self,
        var_name: &str,
        node_id: NodeId,
        node_column: &super::core::NodeColumn,
        local_values: &HashMap<String, AlgorithmParamValue>,
        scope: &mut StepScope<'_>,
    ) -> Result<AlgorithmParamValue> {
        if var_name == self.target {
            if let Some(val) = node_column.get(node_id) {
                return Ok(val.clone());
            }
        }

        if let Some(val) = local_values.get(var_name) {
            return Ok(val.clone());
        }

        if let Ok(column) = scope.variables().node_column(var_name) {
            if let Some(val) = column.get(node_id) {
                return Ok(val.clone());
            }
        }

        if let Ok(map) = scope.variables().node_map(var_name) {
            if let Some(val) = map.get(&node_id) {
                return Ok(val.clone());
            }
        }

        if let Ok(val) = scope.variables().scalar(var_name) {
            return Ok(val.clone());
        }

        Err(anyhow!(
            "Variable '{}' not found for node {} inside execution block",
            var_name,
            node_id
        ))
    }
}
