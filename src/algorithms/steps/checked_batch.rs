//! Compatibility lowering for the numeric subset of the legacy register plan.
//! There is no arithmetic interpreter here: operations use the checked kernels.
use super::{
    aggregations::{NeighborAggType, NeighborAggregationStep},
    arithmetic::{BinaryArithmeticStep, BinaryOperation, UnaryMathStep},
    core::{AtomicStepEffect, CheckedReadStep, Step, StepValue, StepVariables},
    init::{GraphEdgeCountStep, GraphNodeCountStep, InitNodesStep, InitScalarStep},
    normalization::{ClipValuesStep, NormalizeMethod, NormalizeValuesStep},
    read_scope::{ReadStep, ReadStepScope},
    transformations::UpdateInPlaceStep,
    BroadcastScalarStep, CollectNeighborValuesStep, CompareOp, CompareStep, FusedAXPY, FusedMADD,
    FusedNeighborMulAgg, LoadNodeAttrStep, ModeListStep, ModeTieBreak, NeighborDirection,
    NeighborModeUpdateStep, NodeColumn, NodeDegreeStep, RecipStep, ReduceScalarStep, ReductionOp,
    UnaryOp, WhereStep,
};
use crate::algorithms::{
    execution::batch_plan::{
        AggregateOp, BatchInstruction as I, BatchPlan, CompareOp as C, Direction, TieBreak,
    },
    AlgorithmParamValue as V, Context,
};
use anyhow::{anyhow, bail, Result};
use std::{collections::HashMap, sync::Arc};

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
                I::NodeDegree { dst } => Native::new(NodeDegreeStep::new(slot(*dst), None)),
                I::UpdateInPlace {
                    source,
                    target,
                    ordered,
                } => Native::new(UpdateInPlaceStep::new(
                    slot(*source),
                    slot(*target),
                    *ordered,
                )),
                I::NeighborModeUpdate {
                    target,
                    include_self,
                    tie_break,
                    ordered,
                } => Native::new(NeighborModeUpdateStep::new(
                    slot(*target),
                    *include_self,
                    mode_tie(*tie_break),
                    *ordered,
                )),
                I::CollectNeighborValues {
                    dst,
                    src,
                    include_self,
                } => Native::new(CollectNeighborValuesStep::new(
                    slot(*src),
                    slot(*dst),
                    *include_self,
                )),
                I::NeighborAggregate {
                    dst,
                    src,
                    operation,
                    direction,
                } => Native::new(
                    NeighborAggregationStep::new(slot(*src), slot(*dst), aggregate(*operation))
                        .with_direction(neighbor_direction(*direction)),
                ),
                I::NeighborMode {
                    dst,
                    src,
                    tie_break,
                    direction,
                } => {
                    // Keep the collected list separate when dst aliases src.
                    operations.push(Operation::Native(Native::new(
                        CollectNeighborValuesStep::new(slot(*src), "neighbor/mode", false)
                            .with_direction(neighbor_direction(*direction)),
                    )));
                    Native::new(ModeListStep::new(
                        "neighbor/mode",
                        slot(*dst),
                        mode_tie(*tie_break),
                    ))
                }
                I::FusedNeighborMulAgg {
                    dst,
                    src,
                    multiplier,
                    operation,
                    direction,
                } => {
                    if matches!(operation, AggregateOp::Sum) {
                        Native::new(
                            FusedNeighborMulAgg::new(slot(*src), slot(*multiplier), slot(*dst))
                                .with_direction(neighbor_direction(*direction)),
                        )
                    } else {
                        // Only sum has a specialized native fused kernel. Other
                        // reductions compose the same checked primitives, without
                        // introducing a second arithmetic/aggregation interpreter.
                        operations.push(Operation::Native(Native::new(BinaryArithmeticStep::new(
                            "core.mul",
                            slot(*src),
                            slot(*multiplier),
                            "neighbor/product",
                            BinaryOperation::Mul,
                        ))));
                        Native::new(
                            NeighborAggregationStep::new(
                                "neighbor/product",
                                slot(*dst),
                                aggregate(*operation),
                            )
                            .with_direction(neighbor_direction(*direction)),
                        )
                    }
                }
                I::FusedMADD { dst, a, b, c } => {
                    Native::new(FusedMADD::new(slot(*a), slot(*b), slot(*c), slot(*dst)))
                }
                I::FusedAXPY { dst, alpha, x, y } => {
                    operations.push(Operation::Native(Native::new(InitScalarStep::new(
                        "fused/one",
                        V::Float(1.),
                    ))));
                    Native::new(FusedAXPY::new(
                        slot(*alpha),
                        slot(*x),
                        "fused/one",
                        slot(*y),
                        slot(*dst),
                    ))
                }
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
            if self.slot_count > 0 {
                let zero = Arc::new(StepValue::NodeMap(
                    nodes.iter().map(|&id| (id, V::Float(0.))).collect(),
                ));
                for index in 0..self.slot_count {
                    locals.set_shared_value(slot(index), Arc::clone(&zero));
                }
                drop(zero);
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
                        Operation::Load { slot, name } => load(outer, &nodes, name)
                            .map(|value| locals.set_shared_value(slot, value)),
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
                                .shared_value(a)
                                .map(|value| (b.clone(), value))
                                .ok_or_else(|| anyhow!("Missing carried slot {a}"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    for (name, value) in carried {
                        locals.set_shared_value(name, value);
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
fn neighbor_direction(direction: Direction) -> NeighborDirection {
    match direction {
        Direction::In => NeighborDirection::In,
        Direction::Out => NeighborDirection::Out,
        Direction::Undirected => NeighborDirection::Undirected,
    }
}
fn aggregate(op: AggregateOp) -> NeighborAggType {
    match op {
        AggregateOp::Sum => NeighborAggType::Sum,
        AggregateOp::Mean => NeighborAggType::Mean,
        AggregateOp::Min => NeighborAggType::Min,
        AggregateOp::Max => NeighborAggType::Max,
    }
}
fn mode_tie(tie: TieBreak) -> ModeTieBreak {
    match tie {
        TieBreak::Lowest => ModeTieBreak::Lowest,
        TieBreak::Highest => ModeTieBreak::Highest,
        TieBreak::First => ModeTieBreak::Keep,
    }
}
fn load(vars: &StepVariables, nodes: &[usize], name: &str) -> Result<Arc<StepValue>> {
    if let Some(StepValue::NodeMap(map)) = vars.value(name) {
        if map.len() == nodes.len() {
            validate_map(map, nodes, "LoadNodeProp")?;
            return vars
                .shared_value(name)
                .ok_or_else(|| anyhow!("Missing variable '{name}'"));
        }
    }
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
        .collect::<Result<HashMap<usize, V>>>()
        .map(|map| Arc::new(StepValue::NodeMap(map)))
}
fn validate_map(map: &HashMap<usize, V>, nodes: &[usize], operation: &str) -> Result<()> {
    for node in nodes {
        let value = map
            .get(node)
            .ok_or_else(|| anyhow!("{operation} missing node {node}"))?;
        if !matches!(value, V::Int(_) | V::Float(_)) {
            bail!("{operation} requires numeric values");
        }
    }
    Ok(())
}
fn store(
    locals: &StepVariables,
    outer: &mut StepVariables,
    nodes: &[usize],
    source: &str,
    name: &str,
) -> Result<()> {
    if !matches!(outer.value(name), Some(StepValue::NodeColumn(_))) {
        if let Some(StepValue::NodeMap(map)) = locals.value(source) {
            if map.len() == nodes.len() {
                validate_map(map, nodes, "StoreNodeProp")?;
                let value = locals
                    .shared_value(source)
                    .ok_or_else(|| anyhow!("Missing slot '{source}'"))?;
                outer.set_shared_value(name, value);
                return Ok(());
            }
        }
    }
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

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn exact_membership_loads_and_stores_share_and_subset_loads_project() {
        let mut outer = StepVariables::default();
        outer.set_node_map("source", [(3, V::Int(7)), (9, V::Int(11))].into());
        let loaded = load(&outer, &[3, 9], "source").unwrap();
        assert!(Arc::ptr_eq(&loaded, &outer.shared_value("source").unwrap()));
        let mut locals = StepVariables::default();
        locals.set_shared_value("slot", loaded);
        store(&locals, &mut outer, &[3, 9], "slot", "target").unwrap();
        assert!(Arc::ptr_eq(
            &locals.shared_value("slot").unwrap(),
            &outer.shared_value("target").unwrap()
        ));
        outer.node_map_mut("target").unwrap().insert(3, V::Int(13));
        assert_eq!(locals.node_map("slot").unwrap()[&3], V::Int(7));
        let projected = load(&outer, &[9], "source").unwrap();
        assert_eq!(projected.expect_node_map("projected").unwrap().len(), 1);
        assert!(!Arc::ptr_eq(
            &projected,
            &outer.shared_value("source").unwrap()
        ));
        assert!(load(&outer, &[3, 7], "source").is_err());
    }

    #[test]
    fn sharing_preserves_validation_and_column_output_representation() {
        let mut outer = StepVariables::default();
        let mut locals = StepVariables::default();
        locals.set_node_map("slot", [(3, V::Text("bad".into()))].into());
        outer.set_node_map("target", [(3, V::Int(17))].into());
        assert!(store(&locals, &mut outer, &[3], "slot", "target").is_err());
        assert_eq!(outer.node_map("target").unwrap()[&3], V::Int(17));
        locals.set_node_map("slot", [(3, V::Int(19))].into());
        outer.set_node_column("target", NodeColumn::new(vec![3], vec![V::Int(17)]));
        store(&locals, &mut outer, &[3], "slot", "target").unwrap();
        assert_eq!(
            outer.node_column("target").unwrap().get(3),
            Some(&V::Int(19))
        );
        assert!(outer.node_map("target").is_err());
    }
}
