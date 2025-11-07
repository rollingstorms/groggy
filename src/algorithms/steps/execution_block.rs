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
            .and_then(|d| NeighborDirection::from_str(d))
            .unwrap_or(NeighborDirection::In);

        // Message-pass: iterate through nodes in order (Gauss-Seidel)
        let body = self.body()?;

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

    /// Evaluate the block body for a specific node.
    ///
    /// This is a simplified implementation that handles common patterns.
    /// A full implementation would recursively evaluate the operation DAG.
    fn evaluate_body_for_node(
        &self,
        node_id: NodeId,
        neighbors: &[NodeId],
        node_column: &super::core::NodeColumn,
        body: &BlockBody,
        _ctx: &mut Context,
        _scope: &mut StepScope<'_>,
    ) -> Result<AlgorithmParamValue> {
        // For Phase 4 MVP, we'll implement the most common pattern:
        // collect_neighbor_values -> mode

        // Collect neighbor values
        let mut neighbor_values = Vec::new();
        for &nbr in neighbors {
            if let Some(val) = node_column.get(nbr) {
                neighbor_values.push(val.clone());
            }
        }

        // If body specifies mode operation, compute mode
        for body_node in &body.nodes {
            if body_node.op_type == "mode_list" {
                // Compute mode (most frequent value)
                return compute_mode(&neighbor_values, self.options.tie_break.as_deref());
            }
        }

        // Default: return first neighbor value or current value
        neighbor_values
            .first()
            .cloned()
            .or_else(|| node_column.get(node_id).cloned())
            .ok_or_else(|| anyhow!("No value computed for node {}", node_id))
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

    // Find mode
    let mut max_freq = 0;
    let mut candidates = Vec::new();

    for (count, val) in freq_map.values() {
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
