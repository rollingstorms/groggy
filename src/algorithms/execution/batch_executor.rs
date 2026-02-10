//! Batch execution engine for efficient loop interpretation.
//!
//! The BatchExecutor interprets BatchPlans in native Rust code, eliminating
//! per-step FFI overhead and enabling tight loop execution.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;

use crate::algorithms::steps::StepScope;
use crate::algorithms::AlgorithmParamValue;
use crate::types::AttrName;
use crate::types::NodeId;

use super::batch_plan::{AggregateOp, BatchInstruction, BatchPlan, Direction, SlotId, TieBreak};

/// Batch executor for efficient loop body interpretation.
///
/// Maintains a register file (slots) and executes instructions sequentially
/// without variable lookups or per-step overhead.
pub struct BatchExecutor {
    /// Register file: temporary storage for intermediate values (float vectors)
    slots: Vec<SlotData>,

    /// Number of nodes (for validation and initialization)
    node_count: usize,
}

#[derive(Clone, Debug)]
enum SlotData {
    FloatVec(Vec<f64>),
    IntVec(Vec<i64>),
    BoolVec(Vec<bool>),
    FloatScalar(f64),
    IntScalar(i64),
    BoolScalar(bool),
    ListVec(Vec<Vec<f64>>),
}

struct BatchExecContext {
    nodes: Arc<[NodeId]>,
    node_index: Option<HashMap<NodeId, usize>>,
    neighbor_lists: Option<Vec<Vec<NodeId>>>,
    neighbor_indices: Option<Vec<Vec<usize>>>,
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

        let context = self.build_context(plan, scope)?;

        // Allocate slots (registers)
        self.allocate_slots(plan.slot_count)?;

        // Execute loop iterations
        for iter in 0..iterations {
            // Execute instruction sequence
            for (idx, instr) in plan.instructions.iter().enumerate() {
                self.execute_instruction(instr, scope, &context).map_err(|e| {
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
        self.slots.reserve(count);
        for _ in 0..count {
            self.slots
                .push(SlotData::FloatVec(vec![0.0; self.node_count]));
        }
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
        for slot in 0..self.slots.len() {
            let converted = self.to_f64_vec(slot).unwrap_or_else(|_| vec![0.0; self.node_count]);
            self.slots[slot] = SlotData::FloatVec(converted);
        }
        self.slots
            .iter_mut()
            .map(|v| match v {
                SlotData::FloatVec(vec) => vec.as_mut_ptr(),
                _ => std::ptr::null_mut(),
            })
            .collect()
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        instr: &BatchInstruction,
        scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        match instr {
            BatchInstruction::LoadNodeProp { dst, var_name } => {
                self.load_node_prop(*dst, var_name, scope, context)
            }
            BatchInstruction::StoreNodeProp { src, var_name } => {
                self.store_node_prop(*src, var_name, scope, context)
            }
            BatchInstruction::LoadScalar { dst, value } => self.load_scalar(*dst, *value),
            BatchInstruction::BroadcastScalar { dst, scalar } => {
                self.broadcast_scalar(*dst, *scalar)
            }
            BatchInstruction::InitNodes { dst, value } => self.init_nodes(*dst, *value),
            BatchInstruction::LoadNodeAttr {
                dst,
                attr_name,
                default,
            } => self.load_node_attr(*dst, attr_name, *default, scope, context),
            BatchInstruction::LoadEdgeAttr {
                dst,
                attr_name,
                default,
            } => self.load_edge_attr(*dst, attr_name, *default, scope),
            BatchInstruction::NodeDegree { dst } => self.node_degree(*dst, scope, context),
            BatchInstruction::GraphNodeCount { dst } => self.graph_node_count(*dst, scope),
            BatchInstruction::GraphEdgeCount { dst } => self.graph_edge_count(*dst, scope),
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
            BatchInstruction::Min { dst, lhs, rhs } => {
                self.arithmetic_op(*dst, *lhs, *rhs, |a, b| a.min(b))
            }
            BatchInstruction::Max { dst, lhs, rhs } => {
                self.arithmetic_op(*dst, *lhs, *rhs, |a, b| a.max(b))
            }
            BatchInstruction::Abs { dst, src } => self.unary_op(*dst, *src, |v| v.abs()),
            BatchInstruction::Clip { dst, src, min, max } => {
                self.unary_op(*dst, *src, |v| v.max(*min).min(*max))
            }
            BatchInstruction::Recip { dst, src, epsilon } => {
                self.unary_op(*dst, *src, |v| 1.0 / (v + *epsilon))
            }
            BatchInstruction::Sqrt { dst, src } => self.unary_op(*dst, *src, |v| v.sqrt()),
            BatchInstruction::Exp { dst, src } => self.unary_op(*dst, *src, |v| v.exp()),
            BatchInstruction::Log { dst, src } => self.unary_op(*dst, *src, |v| v.ln()),
            BatchInstruction::Pow { dst, base, exp } => {
                self.arithmetic_op(*dst, *base, *exp, |a, b| a.powf(b))
            }
            BatchInstruction::Compare { dst, lhs, op, rhs } => {
                self.compare_op(*dst, *lhs, *rhs, *op)
            }
            BatchInstruction::Where {
                dst,
                condition,
                if_true,
                if_false,
            } => self.where_op(*dst, *condition, *if_true, *if_false),
            BatchInstruction::ReduceScalar { dst, src, operation } => {
                self.reduce_scalar(*dst, *src, *operation)
            }
            BatchInstruction::Normalize {
                dst,
                src,
                method,
                epsilon,
            } => self.normalize(*dst, *src, *method, *epsilon),
            BatchInstruction::UpdateInPlace {
                source,
                target,
                ordered: _,
            } => self.update_in_place(*source, *target),
            BatchInstruction::NeighborModeUpdate {
                target,
                include_self,
                tie_break,
                ordered: _,
            } => self.neighbor_mode_update(*target, *include_self, *tie_break, scope, context),
            BatchInstruction::CollectNeighborValues {
                dst,
                src,
                include_self,
            } => self.collect_neighbor_values(*dst, *src, *include_self, scope, context),
            BatchInstruction::ModeList { dst, src, tie_break } => {
                self.mode_list(*dst, *src, *tie_break)
            }
            BatchInstruction::NeighborAggregate {
                dst,
                src,
                operation,
                direction,
            } => self.neighbor_aggregate(*dst, *src, *operation, *direction, scope, context),
            BatchInstruction::NeighborMode {
                dst,
                src,
                tie_break,
                direction,
            } => self.neighbor_mode(*dst, *src, *tie_break, *direction, scope, context),
            BatchInstruction::FusedNeighborMulAgg {
                dst,
                src,
                multiplier,
                operation,
                direction,
            } => {
                self.fused_neighbor_mul_agg(
                    *dst,
                    *src,
                    *multiplier,
                    *operation,
                    *direction,
                    scope,
                    context,
                )
            }
            BatchInstruction::FusedMADD { dst, a, b, c } => self.fused_madd(*dst, *a, *b, *c),
            BatchInstruction::FusedAXPY { dst, alpha, x, y } => {
                self.fused_axpy(*dst, *alpha, *x, *y)
            }
        }
    }

    /// Load a node property from the graph into a slot
    fn load_node_prop(
        &mut self,
        dst: SlotId,
        var_name: &str,
        scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        // Use deterministic ordered node list
        let nodes = &context.nodes;

        // Prefer NodeColumn for fast access
        // NodeColumn already has O(1) lookup via internal HashMap
        if let Ok(column) = scope.variables().node_column(var_name) {
            let mut int_vec: Vec<i64> = Vec::with_capacity(nodes.len());
            let mut float_vec: Vec<f64> = Vec::with_capacity(nodes.len());
            let mut use_float = false;
            for &node in nodes.iter() {
                let value = column.get(node).ok_or_else(|| {
                    anyhow!(
                        "LoadNodeProp: node {} missing in column '{}'",
                        node,
                        var_name
                    )
                })?;
                match value {
                    AlgorithmParamValue::Int(v) => {
                        if use_float {
                            float_vec.push(*v as f64);
                        } else {
                            int_vec.push(*v);
                        }
                    }
                    AlgorithmParamValue::Float(v) => {
                        if !use_float {
                            float_vec = int_vec.iter().map(|v| *v as f64).collect();
                            int_vec.clear();
                            use_float = true;
                        }
                        float_vec.push(*v);
                    }
                    _ => {
                        return Err(anyhow!(
                            "LoadNodeProp: non-numeric value in column '{}'",
                            var_name
                        ))
                    }
                }
            }
            if use_float {
                self.slots[dst] = SlotData::FloatVec(float_vec);
            } else {
                self.slots[dst] = SlotData::IntVec(int_vec);
            }
            return Ok(());
        }

        // Fallback to node map
        if let Ok(map) = scope.variables().node_map(var_name) {
            let mut int_vec: Vec<i64> = Vec::with_capacity(nodes.len());
            let mut float_vec: Vec<f64> = Vec::with_capacity(nodes.len());
            let mut use_float = false;
            for &node in nodes.iter() {
                let value = map.get(&node).ok_or_else(|| {
                    anyhow!("LoadNodeProp: node {} missing in map '{}'", node, var_name)
                })?;
                match value {
                    AlgorithmParamValue::Int(v) => {
                        if use_float {
                            float_vec.push(*v as f64);
                        } else {
                            int_vec.push(*v);
                        }
                    }
                    AlgorithmParamValue::Float(v) => {
                        if !use_float {
                            float_vec = int_vec.iter().map(|v| *v as f64).collect();
                            int_vec.clear();
                            use_float = true;
                        }
                        float_vec.push(*v);
                    }
                    _ => {
                        return Err(anyhow!(
                            "LoadNodeProp: non-numeric value in map '{}'",
                            var_name
                        ))
                    }
                }
            }
            if use_float {
                self.slots[dst] = SlotData::FloatVec(float_vec);
            } else {
                self.slots[dst] = SlotData::IntVec(int_vec);
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
        context: &BatchExecContext,
    ) -> Result<()> {
        // Get source slot and use deterministic ordered node list (same as load)
        let src_slot = self.slots[src].clone();
        let nodes = &context.nodes;
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
                match &src_slot {
                    SlotData::FloatVec(values) => {
                        for (i, &value) in values.iter().enumerate() {
                            values_mut[i] = AlgorithmParamValue::Float(value);
                        }
                    }
                    SlotData::IntVec(values) => {
                        for (i, &value) in values.iter().enumerate() {
                            values_mut[i] = AlgorithmParamValue::Int(value);
                        }
                    }
                    SlotData::FloatScalar(value) => {
                        for value_mut in values_mut.iter_mut() {
                            *value_mut = AlgorithmParamValue::Float(*value);
                        }
                    }
                    SlotData::IntScalar(value) => {
                        for value_mut in values_mut.iter_mut() {
                            *value_mut = AlgorithmParamValue::Int(*value);
                        }
                    }
                    SlotData::BoolVec(_) | SlotData::BoolScalar(_) => {
                        return Err(anyhow!(
                            "StoreNodeProp: boolean slot not supported for '{}'",
                            var_name
                        ));
                    }
                }
            } else {
                // Fallback: build mapping (rare case where orderings differ)
                let mut node_to_col_idx = HashMap::new();
                for (col_idx, &node) in col_nodes.iter().enumerate() {
                    node_to_col_idx.insert(node, col_idx);
                }

                let values_mut = column.values_mut();
                match &src_slot {
                    SlotData::FloatVec(values) => {
                        for (slot_idx, &node) in nodes.iter().enumerate() {
                            if let Some(&col_idx) = node_to_col_idx.get(&node) {
                                values_mut[col_idx] = AlgorithmParamValue::Float(values[slot_idx]);
                            } else {
                                return Err(anyhow!(
                                    "StoreNodeProp: node {} missing in column '{}'",
                                    node,
                                    var_name
                                ));
                            }
                        }
                    }
                    SlotData::IntVec(values) => {
                        for (slot_idx, &node) in nodes.iter().enumerate() {
                            if let Some(&col_idx) = node_to_col_idx.get(&node) {
                                values_mut[col_idx] = AlgorithmParamValue::Int(values[slot_idx]);
                            } else {
                                return Err(anyhow!(
                                    "StoreNodeProp: node {} missing in column '{}'",
                                    node,
                                    var_name
                                ));
                            }
                        }
                    }
                    SlotData::FloatScalar(value) => {
                        for (slot_idx, &node) in nodes.iter().enumerate() {
                            if let Some(&col_idx) = node_to_col_idx.get(&node) {
                                values_mut[col_idx] = AlgorithmParamValue::Float(*value);
                            } else {
                                return Err(anyhow!(
                                    "StoreNodeProp: node {} missing in column '{}'",
                                    node,
                                    var_name
                                ));
                            }
                        }
                    }
                    SlotData::IntScalar(value) => {
                        for (slot_idx, &node) in nodes.iter().enumerate() {
                            if let Some(&col_idx) = node_to_col_idx.get(&node) {
                                values_mut[col_idx] = AlgorithmParamValue::Int(*value);
                            } else {
                                return Err(anyhow!(
                                    "StoreNodeProp: node {} missing in column '{}'",
                                    node,
                                    var_name
                                ));
                            }
                        }
                    }
                    SlotData::BoolVec(_) | SlotData::BoolScalar(_) => {
                        return Err(anyhow!(
                            "StoreNodeProp: boolean slot not supported for '{}'",
                            var_name
                        ));
                    }
                }
            }
            true
        } else {
            // Create new node map
            let mut map = HashMap::new();
            match &src_slot {
                SlotData::FloatVec(values) => {
                    for (i, &node) in nodes.iter().enumerate() {
                        map.insert(node, AlgorithmParamValue::Float(values[i]));
                    }
                }
                SlotData::IntVec(values) => {
                    for (i, &node) in nodes.iter().enumerate() {
                        map.insert(node, AlgorithmParamValue::Int(values[i]));
                    }
                }
                SlotData::FloatScalar(value) => {
                    for &node in nodes.iter() {
                        map.insert(node, AlgorithmParamValue::Float(*value));
                    }
                }
                SlotData::IntScalar(value) => {
                    for &node in nodes.iter() {
                        map.insert(node, AlgorithmParamValue::Int(*value));
                    }
                }
                SlotData::BoolVec(_) | SlotData::BoolScalar(_) => {
                    return Err(anyhow!(
                        "StoreNodeProp: boolean slot not supported for '{}'",
                        var_name
                    ));
                }
            }
            scope.variables_mut().set_node_map(var_name, map);
            false
        };

        if debug_values {
            let sample_len = match &src_slot {
                SlotData::FloatVec(values) => values.len().min(5),
                SlotData::IntVec(values) => values.len().min(5),
                SlotData::FloatScalar(_) => 1,
                SlotData::IntScalar(_) => 1,
                SlotData::BoolVec(values) => values.len().min(5),
                SlotData::BoolScalar(_) => 1,
            };
            let sample: Vec<f64> = match &src_slot {
                SlotData::FloatVec(values) => values.iter().copied().take(sample_len).collect(),
                SlotData::IntVec(values) => values
                    .iter()
                    .take(sample_len)
                    .map(|v| *v as f64)
                    .collect(),
                SlotData::FloatScalar(value) => vec![*value],
                SlotData::IntScalar(value) => vec![*value as f64],
                SlotData::BoolVec(values) => values
                    .iter()
                    .take(sample_len)
                    .map(|v| if *v { 1.0 } else { 0.0 })
                    .collect(),
                SlotData::BoolScalar(value) => vec![if *value { 1.0 } else { 0.0 }],
            };
            eprintln!(
                "[BATCH_EXECUTOR] store '{}' sample ({} nodes, column={}) {:?}",
                var_name,
                match &src_slot {
                    SlotData::FloatVec(values) => values.len(),
                    SlotData::IntVec(values) => values.len(),
                    SlotData::FloatScalar(_) => 1,
                    SlotData::IntScalar(_) => 1,
                    SlotData::BoolVec(values) => values.len(),
                    SlotData::BoolScalar(_) => 1,
                },
                wrote_column,
                sample
            );
        }

        Ok(())
    }

    /// Load a scalar value (broadcast to all nodes) into a slot
    fn load_scalar(&mut self, dst: SlotId, value: f64) -> Result<()> {
        self.slots[dst] = SlotData::FloatScalar(value);
        Ok(())
    }

    fn init_nodes(&mut self, dst: SlotId, value: f64) -> Result<()> {
        self.slots[dst] = SlotData::FloatVec(vec![value; self.node_count]);
        Ok(())
    }

    fn load_node_attr(
        &mut self,
        dst: SlotId,
        attr_name: &str,
        default: f64,
        scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let nodes = &context.nodes;
        let mut values = Vec::with_capacity(nodes.len());
        for &node in nodes.iter() {
            let attr = scope
                .subgraph()
                .get_node_attribute(node, &AttrName::from(attr_name.to_string()))?
                .and_then(AlgorithmParamValue::from_attr_value);
            match attr {
                Some(AlgorithmParamValue::Float(v)) => values.push(v),
                Some(AlgorithmParamValue::Int(v)) => values.push(v as f64),
                Some(_) => {
                    return Err(anyhow!(
                        "LoadNodeAttr: non-numeric value for '{}'",
                        attr_name
                    ))
                }
                None => values.push(default),
            }
        }
        self.slots[dst] = SlotData::FloatVec(values);
        Ok(())
    }

    fn load_edge_attr(
        &mut self,
        dst: SlotId,
        attr_name: &str,
        default: f64,
        scope: &mut StepScope,
    ) -> Result<()> {
        let mut map = HashMap::new();
        for edge in scope.edge_ids() {
            let value = scope
                .subgraph()
                .get_edge_attribute(*edge, &AttrName::from(attr_name.to_string()))?
                .and_then(AlgorithmParamValue::from_attr_value)
                .unwrap_or_else(|| AlgorithmParamValue::Float(default));
            map.insert(*edge, value);
        }
        scope
            .variables_mut()
            .set_edge_map(attr_name.to_string(), map);
        self.slots[dst] = SlotData::FloatScalar(default);
        Ok(())
    }

    fn node_degree(
        &mut self,
        dst: SlotId,
        scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let nodes = &context.nodes;
        let neighbor_cache = scope.neighbor_cache()?;
        let mut values = Vec::with_capacity(nodes.len());
        for &node in nodes.iter() {
            let degree = neighbor_cache.neighbors(node).map(|n| n.len()).unwrap_or(0);
            values.push(degree as f64);
        }
        self.slots[dst] = SlotData::FloatVec(values);
        Ok(())
    }

    fn graph_node_count(&mut self, dst: SlotId, scope: &mut StepScope) -> Result<()> {
        self.slots[dst] = SlotData::FloatScalar(scope.subgraph().node_count() as f64);
        Ok(())
    }

    fn graph_edge_count(&mut self, dst: SlotId, scope: &mut StepScope) -> Result<()> {
        self.slots[dst] = SlotData::FloatScalar(scope.subgraph().edge_count() as f64);
        Ok(())
    }

    fn broadcast_scalar(&mut self, dst: SlotId, scalar: SlotId) -> Result<()> {
        match &self.slots[scalar] {
            SlotData::FloatScalar(value) => {
                self.slots[dst] = SlotData::FloatVec(vec![*value; self.node_count]);
            }
            SlotData::IntScalar(value) => {
                self.slots[dst] = SlotData::IntVec(vec![*value; self.node_count]);
            }
            SlotData::BoolScalar(value) => {
                self.slots[dst] = SlotData::BoolVec(vec![*value; self.node_count]);
            }
            SlotData::FloatVec(_) | SlotData::IntVec(_) | SlotData::BoolVec(_) => {
                return Err(anyhow!("BroadcastScalar expects a scalar slot"));
            }
        }
        Ok(())
    }

    /// Generic arithmetic operation on two slots
    fn arithmetic_op<F>(&mut self, dst: SlotId, lhs: SlotId, rhs: SlotId, op: F) -> Result<()>
    where
        F: Fn(f64, f64) -> f64,
    {
        let lhs_data = self.to_f64_vec(lhs)?;
        let rhs_data = self.to_f64_vec(rhs)?;
        let mut out = vec![0.0; self.node_count];
        for i in 0..self.node_count {
            out[i] = op(lhs_data[i], rhs_data[i]);
        }
        self.slots[dst] = SlotData::FloatVec(out);
        Ok(())
    }

    fn unary_op<F>(&mut self, dst: SlotId, src: SlotId, op: F) -> Result<()>
    where
        F: Fn(f64) -> f64,
    {
        let src_data = self.to_f64_vec(src)?;
        let mut out = vec![0.0; self.node_count];
        for i in 0..self.node_count {
            out[i] = op(src_data[i]);
        }
        self.slots[dst] = SlotData::FloatVec(out);
        Ok(())
    }

    fn compare_op(
        &mut self,
        dst: SlotId,
        lhs: SlotId,
        rhs: SlotId,
        op: super::batch_plan::CompareOp,
    ) -> Result<()> {
        let lhs_data = self.to_f64_vec(lhs)?;
        let rhs_data = self.to_f64_vec(rhs)?;
        let mut out = vec![false; self.node_count];
        for i in 0..self.node_count {
            out[i] = match op {
                super::batch_plan::CompareOp::Eq => lhs_data[i] == rhs_data[i],
                super::batch_plan::CompareOp::Ne => lhs_data[i] != rhs_data[i],
                super::batch_plan::CompareOp::Lt => lhs_data[i] < rhs_data[i],
                super::batch_plan::CompareOp::Le => lhs_data[i] <= rhs_data[i],
                super::batch_plan::CompareOp::Gt => lhs_data[i] > rhs_data[i],
                super::batch_plan::CompareOp::Ge => lhs_data[i] >= rhs_data[i],
            };
        }
        self.slots[dst] = SlotData::BoolVec(out);
        Ok(())
    }

    fn where_op(
        &mut self,
        dst: SlotId,
        condition: SlotId,
        if_true: SlotId,
        if_false: SlotId,
    ) -> Result<()> {
        let cond = self.to_bool_vec(condition)?;
        let true_slot = self.slots[if_true].clone();
        let false_slot = self.slots[if_false].clone();

        match (true_slot, false_slot) {
            (SlotData::IntVec(a), SlotData::IntVec(b)) => {
                let mut out = vec![0; self.node_count];
                for i in 0..self.node_count {
                    out[i] = if cond[i] { a[i] } else { b[i] };
                }
                self.slots[dst] = SlotData::IntVec(out);
            }
            (SlotData::IntScalar(a), SlotData::IntScalar(b)) => {
                let mut out = vec![0; self.node_count];
                for i in 0..self.node_count {
                    out[i] = if cond[i] { a } else { b };
                }
                self.slots[dst] = SlotData::IntVec(out);
            }
            _ => {
                let true_vec = self.to_f64_vec_from_slot(true_slot)?;
                let false_vec = self.to_f64_vec_from_slot(false_slot)?;
                let mut out = vec![0.0; self.node_count];
                for i in 0..self.node_count {
                    out[i] = if cond[i] { true_vec[i] } else { false_vec[i] };
                }
                self.slots[dst] = SlotData::FloatVec(out);
            }
        }
        Ok(())
    }

    fn reduce_scalar(&mut self, dst: SlotId, src: SlotId, operation: AggregateOp) -> Result<()> {
        match &self.slots[src] {
            SlotData::IntVec(values) => {
                let scalar = match operation {
                    AggregateOp::Sum => values.iter().sum::<i64>() as f64,
                    AggregateOp::Mean => {
                        if values.is_empty() {
                            0.0
                        } else {
                            values.iter().sum::<i64>() as f64 / values.len() as f64
                        }
                    }
                    AggregateOp::Min => values.iter().copied().min().unwrap_or(0) as f64,
                    AggregateOp::Max => values.iter().copied().max().unwrap_or(0) as f64,
                };
                self.slots[dst] = SlotData::FloatScalar(scalar);
            }
            _ => {
                let values = self.to_f64_vec(src)?;
                let scalar = match operation {
                    AggregateOp::Sum => values.iter().sum::<f64>(),
                    AggregateOp::Mean => {
                        if values.is_empty() {
                            0.0
                        } else {
                            values.iter().sum::<f64>() / values.len() as f64
                        }
                    }
                    AggregateOp::Min => values
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min),
                    AggregateOp::Max => values
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                };
                self.slots[dst] = SlotData::FloatScalar(scalar);
            }
        }
        Ok(())
    }

    fn normalize(
        &mut self,
        dst: SlotId,
        src: SlotId,
        method: AggregateOp,
        epsilon: f64,
    ) -> Result<()> {
        let values = self.to_f64_vec(src)?;
        let denom = match method {
            AggregateOp::Sum => values.iter().sum::<f64>().max(epsilon),
            AggregateOp::Mean => {
                if values.is_empty() {
                    epsilon
                } else {
                    (values.iter().sum::<f64>() / values.len() as f64).max(epsilon)
                }
            }
            AggregateOp::Min => values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
                .max(epsilon),
            AggregateOp::Max => values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .max(epsilon),
        };

        let mut out = vec![0.0; self.node_count];
        for i in 0..self.node_count {
            out[i] = values[i] / denom;
        }
        self.slots[dst] = SlotData::FloatVec(out);
        Ok(())
    }

    fn update_in_place(&mut self, source: SlotId, target: SlotId) -> Result<()> {
        let source_data = self.slots[source].clone();
        self.slots[target] = source_data;
        Ok(())
    }

    fn neighbor_mode_update(
        &mut self,
        target: SlotId,
        include_self: bool,
        tie_break: TieBreak,
        scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let neighbor_indices = context.neighbor_indices.as_ref().ok_or_else(|| {
            anyhow!("Neighbor mode update requested without cached neighbor context")
        })?;

        match self.slots[target].clone() {
            SlotData::IntVec(mut values) => {
                let mut min_label = i64::MAX;
                let mut max_label = i64::MIN;
                for &val in values.iter() {
                    min_label = min_label.min(val);
                    max_label = max_label.max(val);
                }

                let range = max_label.saturating_sub(min_label);
                let use_dense = range >= 0 && range <= 10_000;

                if use_dense {
                    let len = (range + 1) as usize;
                    let mut counts = vec![0usize; len];
                    let mut touched: Vec<usize> = Vec::new();

                    for (i, neighbors) in neighbor_indices.iter().enumerate() {
                        touched.clear();

                        let mut consider = |val: i64| {
                            let idx = (val - min_label) as usize;
                            if counts[idx] == 0 {
                                touched.push(idx);
                            }
                            counts[idx] += 1;
                        };

                        if include_self {
                            consider(values[i]);
                        }
                        for &neighbor_idx in neighbors.iter() {
                            consider(values[neighbor_idx]);
                        }

                        if touched.is_empty() {
                            continue;
                        }

                        let mut mode_value = values[i];
                        let mut max_count = 0usize;
                        for &idx in touched.iter() {
                            let count = counts[idx];
                            let val = min_label + idx as i64;
                            let should_update = match tie_break {
                                TieBreak::Lowest => {
                                    count > max_count || (count == max_count && val < mode_value)
                                }
                                TieBreak::Highest => {
                                    count > max_count || (count == max_count && val > mode_value)
                                }
                                TieBreak::First => count > max_count,
                            };
                            if should_update {
                                mode_value = val;
                                max_count = count;
                            }
                        }

                        for &idx in touched.iter() {
                            counts[idx] = 0;
                        }

                        values[i] = mode_value;
                    }
                } else {
                    let mut frequency: HashMap<i64, usize> = HashMap::new();
                    let mut touched: Vec<i64> = Vec::new();

                    for (i, neighbors) in neighbor_indices.iter().enumerate() {
                        frequency.clear();
                        touched.clear();

                        let mut consider = |val: i64| {
                            let entry = frequency.entry(val).or_insert(0);
                            if *entry == 0 {
                                touched.push(val);
                            }
                            *entry += 1;
                        };

                        if include_self {
                            consider(values[i]);
                        }
                        for &neighbor_idx in neighbors.iter() {
                            consider(values[neighbor_idx]);
                        }

                        if touched.is_empty() {
                            continue;
                        }

                        let mut mode_value = values[i];
                        let mut max_count = 0usize;
                        for &val in touched.iter() {
                            let count = frequency.get(&val).copied().unwrap_or(0);
                            let should_update = match tie_break {
                                TieBreak::Lowest => {
                                    count > max_count || (count == max_count && val < mode_value)
                                }
                                TieBreak::Highest => {
                                    count > max_count || (count == max_count && val > mode_value)
                                }
                                TieBreak::First => count > max_count,
                            };
                            if should_update {
                                mode_value = val;
                                max_count = count;
                            }
                        }

                        values[i] = mode_value;
                    }
                }

                self.slots[target] = SlotData::IntVec(values);
            }
            _ => {
                let mut values = self.to_f64_vec(target)?;
                let mut frequency: HashMap<i64, (f64, usize)> = HashMap::new();
                let mut touched: Vec<i64> = Vec::new();

                for (i, neighbors) in neighbor_indices.iter().enumerate() {
                    frequency.clear();
                    touched.clear();

                    if include_self {
                        let val = values[i];
                        let key = (val * 1000.0) as i64;
                        frequency
                            .entry(key)
                            .and_modify(|(_, count)| *count += 1)
                            .or_insert((val, 1));
                        touched.push(key);
                    }

                    for &neighbor_idx in neighbors.iter() {
                        let val = values[neighbor_idx];
                        let key = (val * 1000.0) as i64;
                        frequency
                            .entry(key)
                            .and_modify(|(_, count)| *count += 1)
                            .or_insert((val, 1));
                        touched.push(key);
                    }

                    if frequency.is_empty() {
                        continue;
                    }

                    let mut mode_value = values[i];
                    let mut max_count = 0usize;
                    for key in touched.iter() {
                        let (val, count) = frequency.get(key).copied().unwrap_or((0.0, 0));
                        let should_update = match tie_break {
                            TieBreak::Lowest => {
                                count > max_count || (count == max_count && val < mode_value)
                            }
                            TieBreak::Highest => {
                                count > max_count || (count == max_count && val > mode_value)
                            }
                            TieBreak::First => count > max_count,
                        };
                        if should_update {
                            mode_value = val;
                            max_count = count;
                        }
                    }
                    values[i] = mode_value;
                }
                self.slots[target] = SlotData::FloatVec(values);
            }
        }

        Ok(())
    }

    fn collect_neighbor_values(
        &mut self,
        dst: SlotId,
        src: SlotId,
        include_self: bool,
        scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let nodes = &context.nodes;
        let neighbor_indices = context.neighbor_indices.as_ref().ok_or_else(|| {
            anyhow!("CollectNeighborValues requested without cached neighbor context")
        })?;

        let src_vec = self.to_f64_vec(src)?;
        let mut lists: Vec<Vec<f64>> = Vec::with_capacity(nodes.len());

        for (i, neighbors) in neighbor_indices.iter().enumerate() {
            let mut values = Vec::with_capacity(neighbors.len() + if include_self { 1 } else { 0 });
            if include_self {
                values.push(src_vec[i]);
            }
            for &neighbor_idx in neighbors.iter() {
                values.push(src_vec[neighbor_idx]);
            }
            lists.push(values);
        }

        self.slots[dst] = SlotData::ListVec(lists);
        Ok(())
    }

    fn mode_list(&mut self, dst: SlotId, src: SlotId, tie_break: TieBreak) -> Result<()> {
        let lists = match &self.slots[src] {
            SlotData::ListVec(values) => values,
            _ => {
                return Err(anyhow!(
                    "ModeList expects list slot at {}, got {:?}",
                    src,
                    self.slots[src]
                ))
            }
        };

        let mut out = vec![0.0; self.node_count];
        for (i, list) in lists.iter().enumerate() {
            if list.is_empty() {
                out[i] = 0.0;
                continue;
            }

            let mut freq: HashMap<u64, (f64, usize)> = HashMap::new();
            for &val in list.iter() {
                let key = val.to_bits();
                freq.entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((val, 1));
            }

            let mut mode_value = list[0];
            let mut max_count = 0usize;
            for (_key, (val, count)) in freq.iter() {
                let should_update = match tie_break {
                    TieBreak::Lowest => {
                        *count > max_count || (*count == max_count && *val < mode_value)
                    }
                    TieBreak::Highest => {
                        *count > max_count || (*count == max_count && *val > mode_value)
                    }
                    TieBreak::First => *count > max_count,
                };
                if should_update {
                    mode_value = *val;
                    max_count = *count;
                }
            }
            out[i] = mode_value;
        }

        self.slots[dst] = SlotData::FloatVec(out);
        Ok(())
    }

    /// Aggregate neighbor values
    fn neighbor_aggregate(
        &mut self,
        dst: SlotId,
        src: SlotId,
        operation: AggregateOp,
        _direction: Direction,
        _scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let nodes = &context.nodes;
        let neighbor_indices = context.neighbor_indices.as_ref().ok_or_else(|| {
            anyhow!("Neighbor aggregation requested without cached neighbor context")
        })?;

        // Now work with self's slots
        let src_vec = self.to_f64_vec(src)?;
        let mut dst_vec = vec![0.0; self.node_count];

        // Aggregate neighbors for each node
        for (i, neighbors) in neighbor_indices.iter().enumerate() {
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
            for &neighbor_idx in neighbors.iter() {
                let neighbor_value = src_vec[neighbor_idx];
                match operation {
                    AggregateOp::Sum | AggregateOp::Mean => agg_value += neighbor_value,
                    AggregateOp::Min => agg_value = agg_value.min(neighbor_value),
                    AggregateOp::Max => agg_value = agg_value.max(neighbor_value),
                }
                count += 1;
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
        self.slots[dst] = SlotData::FloatVec(dst_vec);
        Ok(())
    }

    /// Compute mode of neighbor values
    fn neighbor_mode(
        &mut self,
        dst: SlotId,
        src: SlotId,
        tie_break: TieBreak,
        _direction: Direction,
        _scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let nodes = &context.nodes;
        let neighbor_indices = context.neighbor_indices.as_ref().ok_or_else(|| {
            anyhow!("Neighbor mode requested without cached neighbor context")
        })?;

        // Work with slots
        let src_vec = self.to_f64_vec(src)?;
        let mut dst_vec = vec![0.0; self.node_count];

        // Compute mode for each node
        for (i, neighbors) in neighbor_indices.iter().enumerate() {
            if neighbors.is_empty() {
                dst_vec[i] = src_vec[i]; // Keep own value if no neighbors
                continue;
            }

            // Count frequency of each value
            let mut frequency: HashMap<i64, (f64, usize)> = HashMap::new();
            for &neighbor_idx in neighbors {
                let value = src_vec[neighbor_idx];
                let key = (value * 1000.0) as i64; // Discretize for counting
                frequency
                    .entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((value, 1));
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
        self.slots[dst] = SlotData::FloatVec(dst_vec);
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
        _scope: &mut StepScope,
        context: &BatchExecContext,
    ) -> Result<()> {
        let nodes = &context.nodes;
        let neighbor_indices = context.neighbor_indices.as_ref().ok_or_else(|| {
            anyhow!("Neighbor aggregation requested without cached neighbor context")
        })?;

        // Work with slots
        let src_vec = self.to_f64_vec(src)?;
        let mult_vec = self.to_f64_vec(multiplier)?;
        let mut dst_vec = vec![0.0; self.node_count];

        // For each node: aggregate(neighbors[src] * mult)
        for (i, neighbors) in neighbor_indices.iter().enumerate() {
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
            for &neighbor_idx in neighbors.iter() {
                let neighbor_value = src_vec[neighbor_idx] * mult;
                match operation {
                    AggregateOp::Sum | AggregateOp::Mean => agg += neighbor_value,
                    AggregateOp::Min => agg = agg.min(neighbor_value),
                    AggregateOp::Max => agg = agg.max(neighbor_value),
                }
                count += 1;
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
        self.slots[dst] = SlotData::FloatVec(dst_vec);
        Ok(())
    }

    /// Fused multiply-add: dst = a * b + c
    fn fused_madd(&mut self, dst: SlotId, a: SlotId, b: SlotId, c: SlotId) -> Result<()> {
        let a_data = self.to_f64_vec(a)?;
        let b_data = self.to_f64_vec(b)?;
        let c_data = self.to_f64_vec(c)?;
        let mut out = vec![0.0; self.node_count];
        for i in 0..self.node_count {
            out[i] = a_data[i] * b_data[i] + c_data[i];
        }
        self.slots[dst] = SlotData::FloatVec(out);

        Ok(())
    }

    /// Fused axpy: dst = alpha * x + y
    fn fused_axpy(&mut self, dst: SlotId, alpha: SlotId, x: SlotId, y: SlotId) -> Result<()> {
        let alpha_data = self.to_f64_vec(alpha)?;
        let x_data = self.to_f64_vec(x)?;
        let y_data = self.to_f64_vec(y)?;
        let mut out = vec![0.0; self.node_count];
        for i in 0..self.node_count {
            out[i] = alpha_data[i] * x_data[i] + y_data[i];
        }
        self.slots[dst] = SlotData::FloatVec(out);

        Ok(())
    }

    /// Copy one slot to another (for loop-carried variables)
    fn copy_slot(&mut self, from: SlotId, to: SlotId) -> Result<()> {
        let from_data = self.slots[from].clone();
        self.slots[to] = from_data;
        Ok(())
    }

    /// Helper: get immutable reference to float vector in a slot
    fn to_f64_vec(&self, slot: SlotId) -> Result<Vec<f64>> {
        self.to_f64_vec_from_slot(self.slots.get(slot).ok_or_else(|| {
            anyhow!("slot {} out of bounds for batch executor", slot)
        })?)
    }

    fn to_f64_vec_from_slot(&self, slot: &SlotData) -> Result<Vec<f64>> {
        Ok(match slot {
            SlotData::FloatVec(values) => values.clone(),
            SlotData::IntVec(values) => values.iter().map(|v| *v as f64).collect(),
            SlotData::BoolVec(values) => values.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect(),
            SlotData::FloatScalar(value) => vec![*value; self.node_count],
            SlotData::IntScalar(value) => vec![*value as f64; self.node_count],
            SlotData::BoolScalar(value) => vec![if *value { 1.0 } else { 0.0 }; self.node_count],
        })
    }

    fn to_bool_vec(&self, slot: SlotId) -> Result<Vec<bool>> {
        match self.slots.get(slot).ok_or_else(|| {
            anyhow!("slot {} out of bounds for batch executor", slot)
        })? {
            SlotData::BoolVec(values) => Ok(values.clone()),
            SlotData::BoolScalar(value) => Ok(vec![*value; self.node_count]),
            SlotData::FloatVec(values) => Ok(values.iter().map(|v| *v != 0.0).collect()),
            SlotData::IntVec(values) => Ok(values.iter().map(|v| *v != 0).collect()),
            SlotData::FloatScalar(value) => Ok(vec![*value != 0.0; self.node_count]),
            SlotData::IntScalar(value) => Ok(vec![*value != 0; self.node_count]),
        }
    }

    fn build_context(&mut self, plan: &BatchPlan, scope: &mut StepScope) -> Result<BatchExecContext> {
        let nodes = scope.subgraph().ordered_nodes();
        if nodes.len() != self.node_count {
            return Err(anyhow!(
                "BatchExecutor node_count mismatch: expected {}, got {}",
                self.node_count,
                nodes.len()
            ));
        }

        let needs_neighbors = plan.instructions.iter().any(|instr| {
            matches!(
                instr,
                BatchInstruction::NeighborAggregate { .. }
                    | BatchInstruction::NeighborMode { .. }
                    | BatchInstruction::FusedNeighborMulAgg { .. }
            )
        });

        if needs_neighbors {
            let neighbor_cache = scope.neighbor_cache()?;
            let neighbor_lists: Vec<Vec<NodeId>> = nodes
                .iter()
                .map(|&node| neighbor_cache.neighbors(node).unwrap_or(&[]).to_vec())
                .collect();

            let mut node_index = HashMap::with_capacity(nodes.len());
            for (i, &node) in nodes.iter().enumerate() {
                node_index.insert(node, i);
            }

            let mut neighbor_indices = Vec::with_capacity(neighbor_lists.len());
            for neighbors in neighbor_lists.iter() {
                let mut indices = Vec::with_capacity(neighbors.len());
                for neighbor in neighbors.iter() {
                    if let Some(&idx) = node_index.get(neighbor) {
                        indices.push(idx);
                    }
                }
                neighbor_indices.push(indices);
            }

            Ok(BatchExecContext {
                nodes,
                node_index: Some(node_index),
                neighbor_lists: Some(neighbor_lists),
                neighbor_indices: Some(neighbor_indices),
            })
        } else {
            Ok(BatchExecContext {
                nodes,
                node_index: None,
                neighbor_lists: None,
                neighbor_indices: None,
            })
        }
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
