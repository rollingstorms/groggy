//! Compatibility lowering for the numeric subset of the legacy register plan.
//! There is no arithmetic interpreter here: operations use the checked kernels.
use super::{
    arithmetic::{BinaryArithmeticStep, BinaryOperation, UnaryMathStep},
    core::{AtomicStepEffect, CheckedReadStep, Step, StepValue, StepVariables},
    init::{GraphEdgeCountStep, GraphNodeCountStep, InitNodesStep, InitScalarStep},
    normalization::{ClipValuesStep, NormalizeMethod, NormalizeValuesStep},
    read_scope::{ReadStep, ReadStepScope},
    BroadcastScalarStep, CompareOp, CompareStep, LoadNodeAttrStep, ModeListStep, ModeTieBreak,
    NodeColumn, RecipStep, ReduceScalarStep, ReductionOp, UnaryOp, WhereStep,
};
use crate::algorithms::{
    execution::batch_plan::{
        AggregateOp, BatchInstruction as I, BatchPlan, CompareOp as C, TieBreak,
    },
    AlgorithmParamValue as V, Context,
};
use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

struct Native {
    id: &'static str,
    effect: AtomicStepEffect,
    kernel: Box<dyn ReadStep>,
}
impl Native {
    fn new<S: Step + ReadStep + 'static>(step: S) -> Self {
        Self {
            id: step.id(),
            effect: step.atomic_effect(),
            kernel: Box::new(step),
        }
    }
    fn checked(&self) -> CheckedReadStep<'_> {
        CheckedReadStep::native(self.id, self.effect, self.kernel.as_ref())
    }
}
enum Operation {
    Native(Native),
    Load { slot: String, name: String },
    Store { slot: String, name: String },
}

/// Constructed only from known native kernels, with metadata captured on the owner.
/// Registers belong to a private variable frame, never the caller's namespace.
pub(crate) struct LoweredBatchPlan {
    operations: Vec<Operation>,
    slot_count: usize,
    carried: Vec<(String, String)>,
}
fn slot(index: usize) -> String {
    format!("slot/{index}")
}
impl LoweredBatchPlan {
    /// None means a valid plan contains a family not yet lowered. Malformed plans
    /// are errors even when another instruction would require the legacy path.
    pub(crate) fn compile(plan: &BatchPlan) -> Result<Option<Self>> {
        plan.validate()?;
        let mut operations = Vec::with_capacity(plan.instructions.len());
        let mut supported = true;
        for instruction in &plan.instructions {
            let native = match instruction {
                I::LoadNodeProp { dst, var_name } => {
                    operations.push(Operation::Load {
                        slot: slot(*dst),
                        name: var_name.clone(),
                    });
                    continue;
                }
                I::StoreNodeProp { src, var_name } => {
                    operations.push(Operation::Store {
                        slot: slot(*src),
                        name: var_name.clone(),
                    });
                    continue;
                }
                I::LoadScalar { dst, value } => {
                    Native::new(InitScalarStep::new(slot(*dst), V::Float(*value)))
                }
                I::InitNodes { dst, value } => {
                    Native::new(InitNodesStep::new(slot(*dst), V::Float(*value)))
                }
                I::LoadNodeAttr {
                    dst,
                    attr_name,
                    default,
                } => Native::new(LoadNodeAttrStep::new(
                    slot(*dst),
                    attr_name.clone().into(),
                    V::Float(*default),
                )),
                I::GraphNodeCount { dst } => Native::new(GraphNodeCountStep::new(slot(*dst))),
                I::GraphEdgeCount { dst } => Native::new(GraphEdgeCountStep::new(slot(*dst))),
                I::Add { dst, lhs, rhs }
                | I::Sub { dst, lhs, rhs }
                | I::Mul { dst, lhs, rhs }
                | I::Div { dst, lhs, rhs }
                | I::Min { dst, lhs, rhs }
                | I::Max { dst, lhs, rhs } => {
                    let (id, op) = match instruction {
                        I::Add { .. } => ("core.add", BinaryOperation::Add),
                        I::Sub { .. } => ("core.sub", BinaryOperation::Sub),
                        I::Mul { .. } => ("core.mul", BinaryOperation::Mul),
                        I::Div { .. } => ("core.div", BinaryOperation::Div),
                        I::Min { .. } => ("core.min", BinaryOperation::Min),
                        _ => ("core.max", BinaryOperation::Max),
                    };
                    Native::new(BinaryArithmeticStep::new(
                        id,
                        slot(*lhs),
                        slot(*rhs),
                        slot(*dst),
                        op,
                    ))
                }
                I::Pow { dst, base, exp } => Native::new(BinaryArithmeticStep::new(
                    "core.pow",
                    slot(*base),
                    slot(*exp),
                    slot(*dst),
                    BinaryOperation::Pow,
                )),
                I::Abs { dst, src }
                | I::Sqrt { dst, src }
                | I::Exp { dst, src }
                | I::Log { dst, src } => {
                    let (id, op) = match instruction {
                        I::Abs { .. } => ("core.abs", UnaryOp::Abs),
                        I::Sqrt { .. } => ("core.sqrt", UnaryOp::Sqrt),
                        I::Exp { .. } => ("core.exp", UnaryOp::Exp),
                        _ => ("core.log", UnaryOp::Log),
                    };
                    Native::new(UnaryMathStep::new(id, slot(*src), slot(*dst), op, None)?)
                }
                I::Clip { dst, src, min, max } => {
                    Native::new(ClipValuesStep::new(slot(*src), slot(*dst), *min, *max))
                }
                I::Recip { dst, src, epsilon } => {
                    Native::new(RecipStep::new(slot(*src), slot(*dst), *epsilon))
                }
                I::Compare { dst, lhs, op, rhs } => Native::new(CompareStep::new(
                    slot(*lhs),
                    match op {
                        C::Eq => CompareOp::Eq,
                        C::Ne => CompareOp::Ne,
                        C::Lt => CompareOp::Lt,
                        C::Le => CompareOp::Le,
                        C::Gt => CompareOp::Gt,
                        C::Ge => CompareOp::Ge,
                    },
                    slot(*rhs),
                    slot(*dst),
                )),
                I::Where {
                    dst,
                    condition,
                    if_true,
                    if_false,
                } => Native::new(WhereStep::new(
                    slot(*condition),
                    slot(*if_true),
                    slot(*if_false),
                    slot(*dst),
                )),
                I::ReduceScalar {
                    dst,
                    src,
                    operation,
                } => Native::new(ReduceScalarStep::new(
                    slot(*src),
                    match operation {
                        AggregateOp::Sum => ReductionOp::Sum,
                        AggregateOp::Mean => ReductionOp::Mean,
                        AggregateOp::Min => ReductionOp::Min,
                        AggregateOp::Max => ReductionOp::Max,
                    },
                    slot(*dst),
                )),
                I::ModeList {
                    dst,
                    src,
                    tie_break,
                } => Native::new(ModeListStep::new(
                    slot(*src),
                    slot(*dst),
                    match tie_break {
                        TieBreak::Lowest => ModeTieBreak::Lowest,
                        TieBreak::Highest => ModeTieBreak::Highest,
                        TieBreak::First => ModeTieBreak::Keep,
                    },
                )),
                I::Normalize {
                    dst,
                    src,
                    method,
                    epsilon,
                } if !matches!(method, AggregateOp::Min | AggregateOp::Mean) => {
                    Native::new(NormalizeValuesStep::new(
                        slot(*src),
                        slot(*dst),
                        if matches!(method, AggregateOp::Sum) {
                            NormalizeMethod::Sum
                        } else {
                            NormalizeMethod::Max
                        },
                        *epsilon,
                    ))
                }
                I::BroadcastScalar { dst, scalar } => {
                    // A fresh reference is required even when dst aliases scalar.
                    operations.push(Operation::Native(Native::new(InitNodesStep::new(
                        "broadcast/reference",
                        V::Float(0.),
                    ))));
                    Native::new(BroadcastScalarStep::new(
                        slot(*scalar),
                        "broadcast/reference",
                        slot(*dst),
                    ))
                }
                _ => {
                    supported = false;
                    continue;
                }
            };
            operations.push(Operation::Native(native));
        }
        Ok(supported.then(|| Self {
            operations,
            slot_count: plan.slot_count,
            carried: plan
                .carried_slots
                .iter()
                .map(|(a, b)| (slot(*a), slot(*b)))
                .collect(),
        }))
    }

    pub(crate) fn execute(
        &self,
        ctx: &mut Context,
        scope: &mut ReadStepScope<'_>,
        iterations: usize,
    ) -> Result<StepVariables> {
        if ctx.is_cancelled() {
            bail!("Batch loop cancelled");
        }
        let nodes = scope.ordered_nodes();
        scope.with_executor(|mut executor, outer| {
            let mut locals = StepVariables::default();
            // Legacy registers begin as zero-valued node vectors.
            for index in 0..self.slot_count {
                locals.set_node_map(
                    slot(index),
                    nodes.iter().map(|&id| (id, V::Float(0.))).collect(),
                );
            }
            for iteration in 0..iterations {
                if ctx.is_cancelled() {
                    bail!("Batch loop cancelled before iteration {}", iteration + 1);
                }
                for (index, operation) in self.operations.iter().enumerate() {
                    if ctx.is_cancelled() {
                        bail!("Batch loop cancelled before instruction {}", index + 1);
                    }
                    let result = match operation {
                        Operation::Native(native) => {
                            executor.apply_prepared(ctx, &mut locals, &native.checked())
                        }
                        Operation::Load { slot, name } => {
                            load(outer, &nodes, name).map(|value| locals.set_node_map(slot, value))
                        }
                        Operation::Store { slot, name } => {
                            store(&locals, outer, &nodes, slot, name)
                        }
                    };
                    result.map_err(|e| {
                        anyhow!(
                            "Batch iteration {}, instruction {}: {e}",
                            iteration + 1,
                            index + 1
                        )
                    })?;
                }
                if iteration + 1 < iterations {
                    // Phi assignments are simultaneous: swaps/chains observe the same
                    // iteration-end register file, independent of carry list order.
                    let carried = self
                        .carried
                        .iter()
                        .map(|(a, b)| {
                            locals
                                .value(a)
                                .cloned()
                                .map(|value| (b.clone(), value))
                                .ok_or_else(|| anyhow!("Missing carried slot {a}"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    for (name, value) in carried {
                        locals.set_value(name, value);
                    }
                }
            }
            if ctx.is_cancelled() {
                bail!("Batch loop cancelled");
            }
            Ok(locals)
        })
    }
}
fn load(vars: &StepVariables, nodes: &[usize], name: &str) -> Result<HashMap<usize, V>> {
    nodes
        .iter()
        .map(|&node| {
            let value = match vars.value(name) {
                Some(StepValue::NodeColumn(column)) => column.get(node),
                Some(StepValue::NodeMap(map)) => map.get(&node),
                _ => bail!("LoadNodeProp '{name}' requires a node map or column"),
            }
            .ok_or_else(|| anyhow!("LoadNodeProp '{name}' missing node {node}"))?;
            if !matches!(value, V::Int(_) | V::Float(_)) {
                bail!("LoadNodeProp '{name}' requires numeric values");
            }
            Ok((node, value.clone()))
        })
        .collect()
}
fn store(
    locals: &StepVariables,
    outer: &mut StepVariables,
    nodes: &[usize],
    source: &str,
    name: &str,
) -> Result<()> {
    let values = match locals.value(source) {
        Some(StepValue::NodeMap(map)) => nodes
            .iter()
            .map(|id| {
                map.get(id)
                    .cloned()
                    .ok_or_else(|| anyhow!("StoreNodeProp missing node {id}"))
            })
            .collect::<Result<Vec<_>>>()?,
        Some(StepValue::Scalar(value)) => vec![value.clone(); nodes.len()],
        _ => bail!("StoreNodeProp requires a numeric node map or scalar"),
    };
    if values.iter().any(|v| !matches!(v, V::Int(_) | V::Float(_))) {
        bail!("StoreNodeProp requires numeric values");
    }
    if matches!(outer.value(name), Some(StepValue::NodeColumn(_))) {
        outer.set_node_column(name, NodeColumn::new(nodes.to_vec(), values));
    } else {
        outer.set_node_map(name, nodes.iter().copied().zip(values).collect());
    }
    Ok(())
}
