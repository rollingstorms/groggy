//! Batch execution engine for efficient loop interpretation.
//!
//! The BatchExecutor interprets BatchPlans in native Rust code, eliminating
//! per-step FFI overhead and enabling tight loop execution.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::algorithms::steps::StepScope;
use crate::algorithms::AlgorithmParamValue;
use crate::types::NodeId;

use super::batch_plan::{AggregateOp, BatchInstruction, BatchPlan, Direction, SlotId, TieBreak};

/// Batch executor for efficient loop body interpretation.
///
/// Maintains a register file (slots) and executes instructions sequentially
/// without variable lookups or per-step overhead.
pub struct BatchExecutor {
    /// Register file: temporary storage for intermediate values (float vectors)
    slots: Vec<Vec<f64>>,

    /// Number of nodes (for validation and initialization)
    node_count: usize,
}

impl BatchExecutor {
    /// Create a new batch executor for a given node count
    pub fn new(node_count: usize) -> Self {
        Self {
            slots: Vec::new(),
            node_count,
        }
    }

    /// Execute a batch plan for a fixed number of iterations
    ///
    /// This is the main entry point for loop execution. It:
    /// 1. Allocates slots (registers) based on plan.slot_count
    /// 2. For each iteration:
    ///    a. Executes all instructions in sequence
    ///    b. Copies carried variables for next iteration
    /// 3. Returns control to caller
    pub fn execute(
        &mut self,
        plan: &BatchPlan,
        iterations: usize,
        scope: &mut StepScope,
    ) -> Result<()> {
        // Validate plan before execution
        plan.validate()?;

        // DEBUG: Print plan details on first execution
        if std::env::var("GROGGY_DEBUG_BATCH").is_ok() {
            eprintln!(
                "[BATCH_EXECUTOR] Plan: {} instructions, {} slots, {} carried",
                plan.instructions.len(),
                plan.slot_count,
                plan.carried_slots.len()
            );
            eprintln!("[BATCH_EXECUTOR] Carried slots: {:?}", plan.carried_slots);
            for (idx, instr) in plan.instructions.iter().enumerate() {
                eprintln!("[BATCH_EXECUTOR]   Instruction {}: {:?}", idx, instr);
            }
        }

        // Allocate slots (registers)
        self.allocate_slots(plan.slot_count)?;

        // Execute loop iterations
        for iter in 0..iterations {
            // Execute instruction sequence
            for (idx, instr) in plan.instructions.iter().enumerate() {
                self.execute_instruction(instr, scope).map_err(|e| {
                    anyhow!(
                        "Iteration {}/{}, instruction {}/{}: {}",
                        iter + 1,
                        iterations,
                        idx + 1,
                        plan.instructions.len(),
                        e
                    )
                })?;
            }

            // Copy carried variables for next iteration (phi nodes)
            if iter < iterations - 1 {
                // Don't copy on last iteration
                for (from_slot, to_slot) in &plan.carried_slots {
                    self.copy_slot(*from_slot, *to_slot)?;
                }
            }
        }

        Ok(())
    }

    /// Allocate slots (registers) for execution
    fn allocate_slots(&mut self, count: usize) -> Result<()> {
        self.slots.clear();
        self.slots.resize(count, vec![0.0; self.node_count]);
        Ok(())
    }

    /// Allocate slots for JIT execution (public for LoopStep)
    /// This is separate from execute() to allow JIT to access slot pointers
    pub fn allocate_slots_for_jit(&mut self, count: usize) -> Result<()> {
        self.allocate_slots(count)
    }

    /// Get mutable pointers to slot data for JIT execution
    ///
    /// # Safety
    /// The returned pointers are valid only as long as:
    /// 1. The BatchExecutor is not moved
    /// 2. No other methods that reallocate slots are called
    /// 3. The slot count matches what was allocated
    pub fn get_slot_pointers(&mut self) -> Vec<*mut f64> {
        self.slots.iter_mut().map(|v| v.as_mut_ptr()).collect()
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        instr: &BatchInstruction,
        scope: &mut StepScope,
    ) -> Result<()> {
        match instr {
            BatchInstruction::LoadNodeProp { dst, var_name } => {
                self.load_node_prop(*dst, var_name, scope)
            }
            BatchInstruction::StoreNodeProp { src, var_name } => {
                self.store_node_prop(*src, var_name, scope)
            }
            BatchInstruction::LoadScalar { dst, value } => self.load_scalar(*dst, *value),
            BatchInstruction::Add { dst, lhs, rhs } => {
                self.arithmetic_op(*dst, *lhs, *rhs, |a, b| a + b)
            }
            BatchInstruction::Sub { dst, lhs, rhs } => {
                self.arithmetic_op(*dst, *lhs, *rhs, |a, b| a - b)
            }
            BatchInstruction::Mul { dst, lhs, rhs } => {
                self.arithmetic_op(*dst, *lhs, *rhs, |a, b| a * b)
            }
            BatchInstruction::Div { dst, lhs, rhs } => self.arithmetic_op(
                *dst,
                *lhs,
                *rhs,
                |a, b| {
                    if b.abs() < 1e-9 {
                        0.0
                    } else {
                        a / b
                    }
                },
            ),
            BatchInstruction::NeighborAggregate {
                dst,
                src,
                operation,
                direction,
            } => self.neighbor_aggregate(*dst, *src, *operation, *direction, scope),
            BatchInstruction::NeighborMode {
                dst,
                src,
                tie_break,
                direction,
            } => self.neighbor_mode(*dst, *src, *tie_break, *direction, scope),
            BatchInstruction::FusedNeighborMulAgg {
                dst,
                src,
                multiplier,
                operation,
                direction,
            } => {
                self.fused_neighbor_mul_agg(*dst, *src, *multiplier, *operation, *direction, scope)
            }
            BatchInstruction::FusedMADD { dst, a, b, c } => self.fused_madd(*dst, *a, *b, *c),
            BatchInstruction::FusedAXPY { dst, alpha, x, y } => {
                self.fused_axpy(*dst, *alpha, *x, *y)
            }
        }
    }

    /// Load a node property from the graph into a slot
    fn load_node_prop(&mut self, dst: SlotId, var_name: &str, scope: &mut StepScope) -> Result<()> {
        // Use deterministic ordered node list
        let nodes = scope.subgraph().ordered_nodes();
        let dst_vec = &mut self.slots[dst];

        // Prefer NodeColumn for fast access
        // NodeColumn already has O(1) lookup via internal HashMap
        if let Ok(column) = scope.variables().node_column(var_name) {
            for (i, &node) in nodes.iter().enumerate() {
                let value = column.get(node).ok_or_else(|| {
                    anyhow!(
                        "LoadNodeProp: node {} missing in column '{}'",
                        node,
                        var_name
                    )
                })?;
                dst_vec[i] = value_to_f64(value)?;
            }
            return Ok(());
        }

        // Fallback to node map
        if let Ok(map) = scope.variables().node_map(var_name) {
            for (i, &node) in nodes.iter().enumerate() {
                let value = map.get(&node).ok_or_else(|| {
                    anyhow!("LoadNodeProp: node {} missing in map '{}'", node, var_name)
                })?;
                dst_vec[i] = value_to_f64(value)?;
            }
            return Ok(());
        }

        Err(anyhow!(
            "LoadNodeProp: variable '{}' is not stored as a node column or node map",
            var_name
        ))
    }

    /// Store a slot value back to a node property in the graph
    fn store_node_prop(
        &mut self,
        src: SlotId,
        var_name: &str,
        scope: &mut StepScope,
    ) -> Result<()> {
        // Get source slot and use deterministic ordered node list (same as load)
        let src_vec = self.slots[src].clone();
        let nodes = scope.subgraph().ordered_nodes();
        let debug_values = std::env::var("GROGGY_DEBUG_BATCH_VALUES").is_ok();

        // Try to update existing column
        let wrote_column = if let Ok(column) = scope.variables_mut().node_column_mut(var_name) {
            let col_nodes = column.nodes();

            // Check if column order matches subgraph ordered_nodes
            let can_use_direct = col_nodes.len() == nodes.len()
                && col_nodes.iter().zip(nodes.iter()).all(|(a, b)| a == b);

            if can_use_direct {
                // Fast path: direct index-to-index write (no HashMap needed!)
                let values_mut = column.values_mut();
                for (i, &value) in src_vec.iter().enumerate() {
                    values_mut[i] = AlgorithmParamValue::Float(value);
                }
            } else {
                // Fallback: build mapping (rare case where orderings differ)
                let mut node_to_col_idx = HashMap::new();
                for (col_idx, &node) in col_nodes.iter().enumerate() {
                    node_to_col_idx.insert(node, col_idx);
                }

                let values_mut = column.values_mut();
                for (slot_idx, &node) in nodes.iter().enumerate() {
                    if let Some(&col_idx) = node_to_col_idx.get(&node) {
                        values_mut[col_idx] = AlgorithmParamValue::Float(src_vec[slot_idx]);
                    } else {
                        return Err(anyhow!(
                            "StoreNodeProp: node {} missing in column '{}'",
                            node,
                            var_name
                        ));
                    }
                }
            }
            true
        } else {
            // Create new node map
            let mut map = HashMap::new();
            for (i, &node) in nodes.iter().enumerate() {
                map.insert(node, AlgorithmParamValue::Float(src_vec[i]));
            }
            scope.variables_mut().set_node_map(var_name, map);
            false
        };

        if debug_values {
            let sample_len = src_vec.len().min(5);
            let sample: Vec<f64> = src_vec.iter().copied().take(sample_len).collect();
            eprintln!(
                "[BATCH_EXECUTOR] store '{}' sample ({} nodes, column={}) {:?}",
                var_name,
                src_vec.len(),
                wrote_column,
                sample
            );
        }

        Ok(())
    }

    /// Load a scalar value (broadcast to all nodes) into a slot
    fn load_scalar(&mut self, dst: SlotId, value: f64) -> Result<()> {
        self.slots[dst].fill(value);
        Ok(())
    }

    /// Generic arithmetic operation on two slots
    fn arithmetic_op<F>(&mut self, dst: SlotId, lhs: SlotId, rhs: SlotId, op: F) -> Result<()>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Clone source data to avoid borrow checker issues
        let lhs_data = self.get_float_vec(lhs)?.clone();
        let rhs_data = self.get_float_vec(rhs)?.clone();
        let node_count = self.node_count;

        let dst_vec = self.get_float_vec_mut(dst)?;
        for i in 0..node_count {
            dst_vec[i] = op(lhs_data[i], rhs_data[i]);
        }

        Ok(())
    }

    /// Aggregate neighbor values
    fn neighbor_aggregate(
        &mut self,
        dst: SlotId,
        src: SlotId,
        operation: AggregateOp,
        _direction: Direction,
        scope: &mut StepScope,
    ) -> Result<()> {
        // Use deterministic ordered node list (same as load/store)
        let nodes = scope.subgraph().ordered_nodes();
        let neighbor_cache = scope.neighbor_cache()?;

        // Build neighbor list
        let neighbor_lists: Vec<Vec<NodeId>> = nodes
            .iter()
            .map(|&node| neighbor_cache.neighbors(node).unwrap_or(&[]).to_vec())
            .collect();

        // Now work with self's slots
        let src_vec = self.get_float_vec(src)?.clone();
        let dst_vec = self.get_float_vec_mut(dst)?;

        // Build node index for fast lookup
        let mut node_index = HashMap::new();
        for (i, &node) in nodes.iter().enumerate() {
            node_index.insert(node, i);
        }

        // Aggregate neighbors for each node
        for (i, neighbors) in neighbor_lists.iter().enumerate() {
            if neighbors.is_empty() {
                dst_vec[i] = 0.0;
                continue;
            }

            let mut agg_value = match operation {
                AggregateOp::Sum => 0.0,
                AggregateOp::Mean => 0.0,
                AggregateOp::Min => f64::INFINITY,
                AggregateOp::Max => f64::NEG_INFINITY,
            };

            let mut count = 0;
            for &neighbor in neighbors.iter() {
                if let Some(&neighbor_idx) = node_index.get(&neighbor) {
                    let neighbor_value = src_vec[neighbor_idx];
                    match operation {
                        AggregateOp::Sum | AggregateOp::Mean => agg_value += neighbor_value,
                        AggregateOp::Min => agg_value = agg_value.min(neighbor_value),
                        AggregateOp::Max => agg_value = agg_value.max(neighbor_value),
                    }
                    count += 1;
                }
            }

            dst_vec[i] = match operation {
                AggregateOp::Mean => {
                    if count > 0 {
                        agg_value / count as f64
                    } else {
                        0.0
                    }
                }
                _ => agg_value,
            };
        }

        Ok(())
    }

    /// Compute mode of neighbor values
    fn neighbor_mode(
        &mut self,
        dst: SlotId,
        src: SlotId,
        tie_break: TieBreak,
        _direction: Direction,
        scope: &mut StepScope,
    ) -> Result<()> {
        // Use deterministic ordered node list (same as load/store)
        let nodes = scope.subgraph().ordered_nodes();
        let neighbor_cache = scope.neighbor_cache()?;
        let neighbor_lists: Vec<Vec<NodeId>> = nodes
            .iter()
            .map(|&node| neighbor_cache.neighbors(node).unwrap_or(&[]).to_vec())
            .collect();

        // Work with slots
        let src_vec = self.get_float_vec(src)?.clone();
        let dst_vec = self.get_float_vec_mut(dst)?;

        // Build node index
        let mut node_index = HashMap::new();
        for (i, &node) in nodes.iter().enumerate() {
            node_index.insert(node, i);
        }

        // Compute mode for each node
        for (i, neighbors) in neighbor_lists.iter().enumerate() {
            if neighbors.is_empty() {
                dst_vec[i] = src_vec[i]; // Keep own value if no neighbors
                continue;
            }

            // Count frequency of each value
            let mut frequency: HashMap<i64, (f64, usize)> = HashMap::new();
            for &neighbor in neighbors {
                if let Some(&neighbor_idx) = node_index.get(&neighbor) {
                    let value = src_vec[neighbor_idx];
                    let key = (value * 1000.0) as i64; // Discretize for counting
                    frequency
                        .entry(key)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert((value, 1));
                }
            }

            // Find mode
            let mut mode_value = src_vec[i];
            let mut max_count = 0;

            for (_key, (value, count)) in frequency.iter() {
                let should_update = match tie_break {
                    TieBreak::Lowest => {
                        *count > max_count || (*count == max_count && *value < mode_value)
                    }
                    TieBreak::Highest => {
                        *count > max_count || (*count == max_count && *value > mode_value)
                    }
                    TieBreak::First => *count > max_count,
                };

                if should_update {
                    mode_value = *value;
                    max_count = *count;
                }
            }

            dst_vec[i] = mode_value;
        }

        Ok(())
    }

    /// Fused neighbor multiply-aggregate operation
    fn fused_neighbor_mul_agg(
        &mut self,
        dst: SlotId,
        src: SlotId,
        multiplier: SlotId,
        operation: AggregateOp,
        _direction: Direction,
        scope: &mut StepScope,
    ) -> Result<()> {
        // Use deterministic ordered node list (same as load/store)
        let nodes = scope.subgraph().ordered_nodes();
        let neighbor_cache = scope.neighbor_cache()?;
        let neighbor_lists: Vec<Vec<NodeId>> = nodes
            .iter()
            .map(|&node| neighbor_cache.neighbors(node).unwrap_or(&[]).to_vec())
            .collect();

        // Work with slots
        let src_vec = self.get_float_vec(src)?.clone();
        let mult_vec = self.get_float_vec(multiplier)?.clone();
        let dst_vec = self.get_float_vec_mut(dst)?;

        // Build node index
        let mut node_index = HashMap::new();
        for (i, &node) in nodes.iter().enumerate() {
            node_index.insert(node, i);
        }

        // For each node: aggregate(neighbors[src] * mult)
        for (i, neighbors) in neighbor_lists.iter().enumerate() {
            let mult = mult_vec[i];

            if neighbors.is_empty() {
                dst_vec[i] = 0.0;
                continue;
            }

            let mut agg = match operation {
                AggregateOp::Sum | AggregateOp::Mean => 0.0,
                AggregateOp::Min => f64::INFINITY,
                AggregateOp::Max => f64::NEG_INFINITY,
            };

            let mut count = 0;
            for &neighbor in neighbors.iter() {
                if let Some(&neighbor_idx) = node_index.get(&neighbor) {
                    let neighbor_value = src_vec[neighbor_idx] * mult;
                    match operation {
                        AggregateOp::Sum | AggregateOp::Mean => agg += neighbor_value,
                        AggregateOp::Min => agg = agg.min(neighbor_value),
                        AggregateOp::Max => agg = agg.max(neighbor_value),
                    }
                    count += 1;
                }
            }

            dst_vec[i] = match operation {
                AggregateOp::Mean => {
                    if count > 0 {
                        agg / count as f64
                    } else {
                        0.0
                    }
                }
                _ => agg,
            };
        }

        Ok(())
    }

    /// Fused multiply-add: dst = a * b + c
    fn fused_madd(&mut self, dst: SlotId, a: SlotId, b: SlotId, c: SlotId) -> Result<()> {
        let a_data = self.get_float_vec(a)?.clone();
        let b_data = self.get_float_vec(b)?.clone();
        let c_data = self.get_float_vec(c)?.clone();
        let node_count = self.node_count;

        let dst_vec = self.get_float_vec_mut(dst)?;
        for i in 0..node_count {
            dst_vec[i] = a_data[i] * b_data[i] + c_data[i];
        }

        Ok(())
    }

    /// Fused axpy: dst = alpha * x + y
    fn fused_axpy(&mut self, dst: SlotId, alpha: SlotId, x: SlotId, y: SlotId) -> Result<()> {
        let alpha_data = self.get_float_vec(alpha)?.clone();
        let x_data = self.get_float_vec(x)?.clone();
        let y_data = self.get_float_vec(y)?.clone();
        let node_count = self.node_count;

        let dst_vec = self.get_float_vec_mut(dst)?;
        for i in 0..node_count {
            dst_vec[i] = alpha_data[i] * x_data[i] + y_data[i];
        }

        Ok(())
    }

    /// Copy one slot to another (for loop-carried variables)
    fn copy_slot(&mut self, from: SlotId, to: SlotId) -> Result<()> {
        let from_data = self.get_float_vec(from)?.clone();
        let to_vec = self.get_float_vec_mut(to)?;
        to_vec.copy_from_slice(&from_data);
        Ok(())
    }

    /// Helper: get immutable reference to float vector in a slot
    fn get_float_vec(&self, slot: SlotId) -> Result<&Vec<f64>> {
        Ok(&self.slots[slot])
    }

    /// Helper: get mutable reference to float vector in a slot
    fn get_float_vec_mut(&mut self, slot: SlotId) -> Result<&mut Vec<f64>> {
        Ok(&mut self.slots[slot])
    }
}

fn value_to_f64(value: &AlgorithmParamValue) -> Result<f64> {
    match value {
        AlgorithmParamValue::Float(v) => Ok(*v),
        AlgorithmParamValue::Int(v) => Ok(*v as f64),
        _ => Err(anyhow!(
            "BatchExecutor expected numeric value, found {:?}",
            value
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_executor_creation() {
        let executor = BatchExecutor::new(10);
        assert_eq!(executor.node_count, 10);
    }

    #[test]
    #[ignore] // TODO: Fix test - needs proper subgraph setup and Graph API
    fn test_arithmetic_operations() {
        // TODO: Need proper Graph -> Subgraph conversion for test setup
        // Placeholder test - requires Graph API changes to create proper subgraph
    }
}
