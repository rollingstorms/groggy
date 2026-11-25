//! Arithmetic operation step primitives.

use super::read_scope::{sealed, ReadStep, ReadStepScope};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{anyhow, bail, Result};

use crate::types::{EdgeId, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{AtomicStepEffect, Step, StepMetadata, StepScope, StepSpec};

/// Binary arithmetic operations over scalar/node/edge values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

impl BinaryOperation {
    pub(crate) fn description(self) -> &'static str {
        match self {
            BinaryOperation::Add => "Element-wise addition",
            BinaryOperation::Sub => "Element-wise subtraction",
            BinaryOperation::Mul => "Element-wise multiplication",
            BinaryOperation::Div => "Element-wise division",
            BinaryOperation::Pow => "Checked element-wise power",
            BinaryOperation::Min => "Exact element-wise minimum",
            BinaryOperation::Max => "Exact element-wise maximum",
        }
    }

    fn apply(
        self,
        lhs: &AlgorithmParamValue,
        rhs: &AlgorithmParamValue,
        lhs_name: &str,
        rhs_name: &str,
    ) -> Result<AlgorithmParamValue> {
        match self {
            BinaryOperation::Add => apply_add(lhs, rhs, lhs_name, rhs_name),
            BinaryOperation::Sub => apply_sub(lhs, rhs, lhs_name, rhs_name),
            BinaryOperation::Mul => apply_mul(lhs, rhs, lhs_name, rhs_name),
            BinaryOperation::Div => apply_div(lhs, rhs, lhs_name, rhs_name),
            BinaryOperation::Min | BinaryOperation::Max => {
                NumericValue::from(lhs, lhs_name)?;
                NumericValue::from(rhs, rhs_name)?;
                if [lhs, rhs]
                    .iter()
                    .any(|value| matches!(value, AlgorithmParamValue::Float(v) if !v.is_finite()))
                {
                    bail!("minimum/maximum requires finite numbers");
                }
                let order = super::filtering::compare_values(lhs, rhs)
                    .ok_or_else(|| anyhow!("minimum/maximum requires comparable numbers"))?;
                let left = if self == BinaryOperation::Min {
                    !order.is_gt()
                } else {
                    !order.is_lt()
                };
                Ok(if left { lhs.clone() } else { rhs.clone() })
            }
            BinaryOperation::Pow => {
                super::expression::eval_binary_op(super::expression::BinaryOp::Pow, lhs, rhs)
            }
        }
    }
}

/// Element-wise binary arithmetic on node or edge maps.
pub struct BinaryArithmeticStep {
    id: &'static str,
    left: String,
    right: String,
    target: String,
    op: BinaryOperation,
}

impl BinaryArithmeticStep {
    pub(crate) fn new(
        id: &'static str,
        left: impl Into<String>,
        right: impl Into<String>,
        target: impl Into<String>,
        op: BinaryOperation,
    ) -> Self {
        Self {
            id,
            left: left.into(),
            right: right.into(),
            target: target.into(),
            op,
        }
    }
}

impl Step for BinaryArithmeticStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        self.id
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id.to_string(),
            description: self.op.description().to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for BinaryArithmeticStep {}
impl ReadStep for BinaryArithmeticStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("{} cancelled", self.id));
        }

        let left = resolve_operand(scope, &self.left)?;
        let right = resolve_operand(scope, &self.right)?;

        match (left, right) {
            (Operand::NodeMap(lhs), Operand::NodeMap(rhs)) => {
                let result = combine_maps(lhs, rhs, self.op, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::NodeMap(lhs), Operand::Scalar(rhs)) => {
                let result = combine_map_scalar(lhs, rhs, self.op, &self.left, &self.right, false)?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::Scalar(lhs), Operand::NodeMap(rhs)) => {
                let result = combine_map_scalar(rhs, lhs, self.op, &self.right, &self.left, true)?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::EdgeMap(lhs), Operand::EdgeMap(rhs)) => {
                let result = combine_maps(lhs, rhs, self.op, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::EdgeMap(lhs), Operand::Scalar(rhs)) => {
                let result = combine_map_scalar(lhs, rhs, self.op, &self.left, &self.right, false)?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::Scalar(lhs), Operand::EdgeMap(rhs)) => {
                let result = combine_map_scalar(rhs, lhs, self.op, &self.right, &self.left, true)?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::Scalar(lhs), Operand::Scalar(rhs))
                if matches!(
                    self.op,
                    BinaryOperation::Pow | BinaryOperation::Min | BinaryOperation::Max
                ) =>
            {
                let result = self.op.apply(lhs, rhs, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_scalar(self.target.clone(), result);
                Ok(())
            }
            (Operand::Scalar(_), Operand::Scalar(_)) => {
                Err(anyhow!("{} requires at least one map operand", self.id))
            }
            (Operand::NodeMap(_), Operand::EdgeMap(_))
            | (Operand::EdgeMap(_), Operand::NodeMap(_)) => Err(anyhow!(
                "{} requires operands of the same map type",
                self.id
            )),
        }
    }
}

/// Checked value math using the canonical expression kernels, with preserved map kind.
pub(crate) struct UnaryMathStep {
    id: &'static str,
    source: String,
    target: String,
    operation: super::expression::UnaryOp,
    log_divisor: Option<AlgorithmParamValue>,
}
impl UnaryMathStep {
    pub(crate) fn new(
        id: &'static str,
        source: String,
        target: String,
        operation: super::expression::UnaryOp,
        base: Option<f64>,
    ) -> Result<Self> {
        let log_divisor = base
            .map(|base| {
                if operation != super::expression::UnaryOp::Log
                    || !base.is_finite()
                    || base <= 0.
                    || base == 1.
                {
                    bail!("log base must be finite, positive and different from one");
                }
                super::expression::eval_unary_op(
                    super::expression::UnaryOp::Log,
                    &AlgorithmParamValue::Float(base),
                )
            })
            .transpose()?;
        Ok(Self {
            id,
            source,
            target,
            operation,
            log_divisor,
        })
    }
    fn evaluate(&self, value: &AlgorithmParamValue) -> Result<AlgorithmParamValue> {
        let result = super::expression::eval_unary_op(self.operation, value)?;
        match &self.log_divisor {
            Some(divisor) => super::expression::eval_binary_op(
                super::expression::BinaryOp::Div,
                &result,
                divisor,
            ),
            None => Ok(result),
        }
    }
    fn map<K: Copy + Ord + Eq + Hash>(
        &self,
        values: &HashMap<K, AlgorithmParamValue>,
        ctx: &Context,
    ) -> Result<HashMap<K, AlgorithmParamValue>> {
        let mut keys: Vec<_> = values.keys().copied().collect();
        keys.sort_unstable();
        let mut result = HashMap::with_capacity(keys.len());
        for key in keys {
            if ctx.is_cancelled() {
                bail!("{} cancelled", self.id);
            }
            let value = values
                .get(&key)
                .ok_or_else(|| anyhow!("{} missing input entry", self.id))?;
            result.insert(key, self.evaluate(value)?);
        }
        Ok(result)
    }
}
impl Step for UnaryMathStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn id(&self) -> &'static str {
        self.id
    }
    fn supports_atomic_properties(&self) -> bool {
        true
    }
    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id.into(),
            description: "Checked element-wise scalar/node/edge math".into(),
            cost_hint: CostHint::Linear,
        }
    }
    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for UnaryMathStep {}
impl ReadStep for UnaryMathStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            bail!("{} cancelled", self.id);
        }
        match resolve_operand(scope, &self.source)? {
            Operand::NodeMap(values) => {
                let result = self.map(values, ctx)?;
                if ctx.is_cancelled() {
                    bail!("{} cancelled", self.id);
                }
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
            }
            Operand::EdgeMap(values) => {
                let result = self.map(values, ctx)?;
                if ctx.is_cancelled() {
                    bail!("{} cancelled", self.id);
                }
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
            }
            Operand::Scalar(value) => {
                let result = self.evaluate(value)?;
                if ctx.is_cancelled() {
                    bail!("{} cancelled", self.id);
                }
                scope
                    .variables_mut()
                    .set_scalar(self.target.clone(), result);
            }
        }
        Ok(())
    }
}

enum Operand<'a> {
    NodeMap(&'a HashMap<NodeId, AlgorithmParamValue>),
    EdgeMap(&'a HashMap<EdgeId, AlgorithmParamValue>),
    Scalar(&'a AlgorithmParamValue),
}

fn resolve_operand<'a>(scope: &'a ReadStepScope<'_>, name: &str) -> Result<Operand<'a>> {
    if !scope.variables().contains(name) {
        bail!("variable '{name}' not found");
    }

    if let Ok(map) = scope.variables().node_map(name) {
        return Ok(Operand::NodeMap(map));
    }

    if let Ok(map) = scope.variables().edge_map(name) {
        return Ok(Operand::EdgeMap(map));
    }

    if let Ok(value) = scope.variables().scalar(name) {
        return Ok(Operand::Scalar(value));
    }

    Err(anyhow!(
        "variable '{}' must be a node map, edge map, or scalar",
        name
    ))
}

fn combine_maps<K>(
    left: &HashMap<K, AlgorithmParamValue>,
    right: &HashMap<K, AlgorithmParamValue>,
    op: BinaryOperation,
    left_name: &str,
    right_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + Eq + Hash + Ord + Debug,
{
    if left.is_empty() {
        return Ok(HashMap::new());
    }

    let mut keys: Vec<K> = left.keys().copied().collect();
    keys.sort_unstable();

    let mut result = HashMap::with_capacity(left.len());
    for key in keys {
        let left_value = left.get(&key).expect("key must exist");
        let right_value = right.get(&key).ok_or_else(|| {
            anyhow!(
                "variable '{}' missing entry for id {:?} matching '{}'",
                right_name,
                key,
                left_name
            )
        })?;
        let computed = op.apply(left_value, right_value, left_name, right_name)?;
        result.insert(key, computed);
    }

    for &key in right.keys() {
        if !left.contains_key(&key) {
            bail!(
                "variable '{}' contains entry for id {:?} missing in '{}'",
                right_name,
                key,
                left_name
            );
        }
    }

    Ok(result)
}

fn combine_map_scalar<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    scalar: &AlgorithmParamValue,
    op: BinaryOperation,
    map_name: &str,
    scalar_name: &str,
    scalar_left: bool,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + Eq + Hash + Ord + Debug,
{
    if map.is_empty() {
        return Ok(HashMap::new());
    }

    let mut keys: Vec<K> = map.keys().copied().collect();
    keys.sort_unstable();

    let mut result = HashMap::with_capacity(map.len());
    for key in keys {
        let value = map.get(&key).expect("key must exist");
        let computed = if scalar_left {
            op.apply(scalar, value, scalar_name, map_name)?
        } else {
            op.apply(value, scalar, map_name, scalar_name)?
        };
        result.insert(key, computed);
    }

    Ok(result)
}

pub(crate) fn read_binary_operands(spec: &StepSpec, id: &str) -> Result<(String, String, String)> {
    let left = spec
        .params
        .get_text("left")
        .or_else(|| spec.params.get_text("var1"))
        .or_else(|| spec.params.get_text("lhs"))
        .ok_or_else(|| anyhow!("{id} requires 'left' (alias: 'var1'/'lhs') param"))?
        .to_string();

    let right = spec
        .params
        .get_text("right")
        .or_else(|| spec.params.get_text("var2"))
        .or_else(|| spec.params.get_text("rhs"))
        .ok_or_else(|| anyhow!("{id} requires 'right' (alias: 'var2'/'rhs') param"))?
        .to_string();

    let target = spec
        .params
        .get_text("target")
        .or_else(|| spec.params.get_text("output"))
        .or_else(|| spec.params.get_text("result"))
        .ok_or_else(|| anyhow!("{id} requires 'target' (alias: 'output'/'result') param"))?
        .to_string();

    Ok((left, right, target))
}

#[derive(Clone, Copy, Debug)]
enum NumericValue {
    Int(i64),
    Float(f64),
}

impl NumericValue {
    fn from(value: &AlgorithmParamValue, context: &str) -> Result<Self> {
        match value {
            AlgorithmParamValue::Int(v) => Ok(NumericValue::Int(*v)),
            AlgorithmParamValue::Float(v) => Ok(NumericValue::Float(*v)),
            other => Err(anyhow!(
                "expected numeric value for '{context}', found {:?}",
                other
            )),
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            NumericValue::Int(v) => v as f64,
            NumericValue::Float(v) => v,
        }
    }
}

fn apply_add(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    if let (AlgorithmParamValue::Text(left), AlgorithmParamValue::Text(right)) = (lhs, rhs) {
        let mut combined = String::with_capacity(left.len() + right.len());
        combined.push_str(left);
        combined.push_str(right);
        return Ok(AlgorithmParamValue::Text(combined));
    }

    let left = NumericValue::from(lhs, lhs_name)?;
    let right = NumericValue::from(rhs, rhs_name)?;

    match (left, right) {
        (NumericValue::Int(l), NumericValue::Int(r)) => match l.checked_add(r) {
            Some(sum) => Ok(AlgorithmParamValue::Int(sum)),
            None => Ok(AlgorithmParamValue::Float(l as f64 + r as f64)),
        },
        (l, r) => Ok(AlgorithmParamValue::Float(l.as_f64() + r.as_f64())),
    }
}

fn apply_sub(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    let left = NumericValue::from(lhs, lhs_name)?;
    let right = NumericValue::from(rhs, rhs_name)?;

    match (left, right) {
        (NumericValue::Int(l), NumericValue::Int(r)) => match l.checked_sub(r) {
            Some(diff) => Ok(AlgorithmParamValue::Int(diff)),
            None => Ok(AlgorithmParamValue::Float(l as f64 - r as f64)),
        },
        (l, r) => Ok(AlgorithmParamValue::Float(l.as_f64() - r.as_f64())),
    }
}

fn apply_mul(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    let left = NumericValue::from(lhs, lhs_name)?;
    let right = NumericValue::from(rhs, rhs_name)?;

    match (left, right) {
        (NumericValue::Int(l), NumericValue::Int(r)) => match l.checked_mul(r) {
            Some(prod) => Ok(AlgorithmParamValue::Int(prod)),
            None => Ok(AlgorithmParamValue::Float(l as f64 * r as f64)),
        },
        (l, r) => Ok(AlgorithmParamValue::Float(l.as_f64() * r.as_f64())),
    }
}

fn apply_div(
    lhs: &AlgorithmParamValue,
    rhs: &AlgorithmParamValue,
    lhs_name: &str,
    rhs_name: &str,
) -> Result<AlgorithmParamValue> {
    let left = NumericValue::from(lhs, lhs_name)?;
    let right = NumericValue::from(rhs, rhs_name)?;

    let divisor = right.as_f64();
    if divisor == 0.0 {
        bail!(
            "division by zero: '{}' value is zero while evaluating '{}'",
            rhs_name,
            lhs_name
        );
    }

    Ok(AlgorithmParamValue::Float(left.as_f64() / divisor))
}

/// Element-wise reciprocal (1/x) with safe epsilon handling for near-zero values.
pub struct RecipStep {
    source: String,
    target: String,
    epsilon: f64,
}

impl RecipStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, epsilon: f64) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            epsilon,
        }
    }
}

impl Step for RecipStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.recip"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Element-wise reciprocal (1/x) with safe zero handling".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for RecipStep {}
impl ReadStep for RecipStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.recip cancelled"));
        }

        let operand = resolve_operand(scope, &self.source)?;

        match operand {
            Operand::NodeMap(map) => {
                let result = apply_recip_to_map(map, self.epsilon, &self.source)?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            Operand::EdgeMap(map) => {
                let result = apply_recip_to_map(map, self.epsilon, &self.source)?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            Operand::Scalar(value) => {
                let result = apply_recip_scalar(value, self.epsilon, &self.source)?;
                scope
                    .variables_mut()
                    .set_scalar(self.target.clone(), result);
                Ok(())
            }
        }
    }
}

fn apply_recip_to_map<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    epsilon: f64,
    map_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + Eq + Hash + Ord,
{
    if map.is_empty() {
        return Ok(HashMap::new());
    }

    let mut keys: Vec<K> = map.keys().copied().collect();
    keys.sort_unstable();

    let mut result = HashMap::with_capacity(map.len());
    for key in keys {
        let value = map.get(&key).expect("key must exist");
        let recip = apply_recip_scalar(value, epsilon, map_name)?;
        result.insert(key, recip);
    }

    Ok(result)
}

fn apply_recip_scalar(
    value: &AlgorithmParamValue,
    epsilon: f64,
    context: &str,
) -> Result<AlgorithmParamValue> {
    let num = NumericValue::from(value, context)?;
    let val = num.as_f64();

    // Use max(abs(val), epsilon) to handle near-zero values
    // For zero or near-zero: use epsilon
    // For normal values: use the value itself
    let abs_val = val.abs();
    let safe_val = if abs_val < epsilon { epsilon } else { abs_val };
    let denominator = val.signum() * safe_val;

    Ok(AlgorithmParamValue::Float(1.0 / denominator))
}

/// Comparison operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "eq" => Ok(CompareOp::Eq),
            "ne" => Ok(CompareOp::Ne),
            "lt" => Ok(CompareOp::Lt),
            "le" => Ok(CompareOp::Le),
            "gt" => Ok(CompareOp::Gt),
            "ge" => Ok(CompareOp::Ge),
            _ => Err(anyhow!("unknown comparison operator: {}", s)),
        }
    }

    fn description(self) -> &'static str {
        match self {
            CompareOp::Eq => "Element-wise equality comparison",
            CompareOp::Ne => "Element-wise inequality comparison",
            CompareOp::Lt => "Element-wise less-than comparison",
            CompareOp::Le => "Element-wise less-than-or-equal comparison",
            CompareOp::Gt => "Element-wise greater-than comparison",
            CompareOp::Ge => "Element-wise greater-than-or-equal comparison",
        }
    }

    pub(crate) fn apply_values(
        self,
        left: &AlgorithmParamValue,
        right: &AlgorithmParamValue,
    ) -> Result<bool> {
        NumericValue::from(left, "comparison left")?;
        NumericValue::from(right, "comparison right")?;
        let order = super::filtering::compare_values(left, right)
            .ok_or_else(|| anyhow!("comparison requires numeric values without NaN"))?;
        Ok(match self {
            CompareOp::Eq => order.is_eq(),
            CompareOp::Ne => !order.is_eq(),
            CompareOp::Lt => order.is_lt(),
            CompareOp::Le => !order.is_gt(),
            CompareOp::Gt => order.is_gt(),
            CompareOp::Ge => !order.is_lt(),
        })
    }
}

/// Element-wise comparison producing 0.0/1.0 masks.
pub struct CompareStep {
    left: String,
    op: CompareOp,
    right: String,
    target: String,
    right_is_scalar: bool,
}

impl CompareStep {
    pub fn new(
        left: impl Into<String>,
        op: CompareOp,
        right: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            left: left.into(),
            op,
            right: right.into(),
            target: target.into(),
            right_is_scalar: false,
        }
    }

    pub fn with_scalar_right(mut self, is_scalar: bool) -> Self {
        self.right_is_scalar = is_scalar;
        self
    }
}

impl Step for CompareStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.compare"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: self.op.description().to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for CompareStep {}
impl ReadStep for CompareStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.compare cancelled"));
        }

        let left = resolve_operand(scope, &self.left)?;
        let right = resolve_operand(scope, &self.right)?;

        match (left, right) {
            (Operand::NodeMap(lhs), Operand::NodeMap(rhs)) => {
                let result = compare_maps(lhs, rhs, self.op, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::NodeMap(lhs), Operand::Scalar(rhs)) => {
                let result = compare_map_scalar(lhs, rhs, self.op, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::EdgeMap(lhs), Operand::EdgeMap(rhs)) => {
                let result = compare_maps(lhs, rhs, self.op, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::EdgeMap(lhs), Operand::Scalar(rhs)) => {
                let result = compare_map_scalar(lhs, rhs, self.op, &self.left, &self.right)?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            (Operand::Scalar(_), Operand::NodeMap(_))
            | (Operand::Scalar(_), Operand::EdgeMap(_)) => {
                Err(anyhow!("core.compare requires left operand to be a map"))
            }
            (Operand::Scalar(_), Operand::Scalar(_)) => {
                Err(anyhow!("core.compare requires at least one map operand"))
            }
            (Operand::NodeMap(_), Operand::EdgeMap(_))
            | (Operand::EdgeMap(_), Operand::NodeMap(_)) => Err(anyhow!(
                "core.compare requires operands of the same map type"
            )),
        }
    }
}

fn compare_maps<K>(
    left: &HashMap<K, AlgorithmParamValue>,
    right: &HashMap<K, AlgorithmParamValue>,
    op: CompareOp,
    left_name: &str,
    right_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + Eq + Hash + Ord + Debug,
{
    if left.is_empty() {
        return Ok(HashMap::new());
    }

    let mut keys: Vec<K> = left.keys().copied().collect();
    keys.sort_unstable();

    let mut result = HashMap::with_capacity(left.len());
    for key in keys {
        let left_value = left.get(&key).expect("key must exist");
        let right_value = right.get(&key).ok_or_else(|| {
            anyhow!(
                "variable '{}' missing entry for id {:?} matching '{}'",
                right_name,
                key,
                left_name
            )
        })?;

        let cmp_result = op.apply_values(left_value, right_value)?;
        let mask_value = if cmp_result { 1.0 } else { 0.0 };

        result.insert(key, AlgorithmParamValue::Float(mask_value));
    }

    for &key in right.keys() {
        if !left.contains_key(&key) {
            bail!(
                "variable '{}' contains entry for id {:?} missing in '{}'",
                right_name,
                key,
                left_name
            );
        }
    }

    Ok(result)
}

fn compare_map_scalar<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    scalar: &AlgorithmParamValue,
    op: CompareOp,
    map_name: &str,
    scalar_name: &str,
) -> Result<HashMap<K, AlgorithmParamValue>>
where
    K: Copy + Eq + Hash,
{
    if map.is_empty() {
        return Ok(HashMap::new());
    }

    NumericValue::from(scalar, scalar_name)?;

    let mut result = HashMap::with_capacity(map.len());
    for (&key, value) in map.iter() {
        NumericValue::from(value, map_name)?;
        let cmp_result = op.apply_values(value, scalar)?;
        let mask_value = if cmp_result { 1.0 } else { 0.0 };

        result.insert(key, AlgorithmParamValue::Float(mask_value));
    }

    Ok(result)
}

/// Shared builder-mask rule; keep map and execution-block selection identical.
pub(crate) fn condition_is_true(value: &AlgorithmParamValue, context: &str) -> Result<bool> {
    Ok(NumericValue::from(value, context)?.as_f64().abs() > f64::EPSILON)
}

/// Element-wise conditional selection (where/if-then-else).
pub struct WhereStep {
    condition: String,
    if_true: String,
    if_false: String,
    target: String,
    true_is_scalar: bool,
    false_is_scalar: bool,
}

impl WhereStep {
    pub fn new(
        condition: impl Into<String>,
        if_true: impl Into<String>,
        if_false: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            condition: condition.into(),
            if_true: if_true.into(),
            if_false: if_false.into(),
            target: target.into(),
            true_is_scalar: false,
            false_is_scalar: false,
        }
    }

    pub fn with_scalar_flags(mut self, true_is_scalar: bool, false_is_scalar: bool) -> Self {
        self.true_is_scalar = true_is_scalar;
        self.false_is_scalar = false_is_scalar;
        self
    }
}

impl Step for WhereStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.where"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Element-wise conditional selection".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for WhereStep {}
impl ReadStep for WhereStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.where cancelled"));
        }

        let condition = resolve_operand(scope, &self.condition)?;
        let if_true = resolve_operand(scope, &self.if_true)?;
        let if_false = resolve_operand(scope, &self.if_false)?;

        match condition {
            Operand::NodeMap(cond_map) => {
                let result = where_node_map(
                    cond_map,
                    if_true,
                    if_false,
                    &self.condition,
                    &self.if_true,
                    &self.if_false,
                )?;
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            Operand::EdgeMap(cond_map) => {
                let result = where_edge_map(
                    cond_map,
                    if_true,
                    if_false,
                    &self.condition,
                    &self.if_true,
                    &self.if_false,
                )?;
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            Operand::Scalar(_) => Err(anyhow!("core.where requires condition to be a map")),
        }
    }
}

fn where_node_map(
    condition: &HashMap<NodeId, AlgorithmParamValue>,
    if_true: Operand<'_>,
    if_false: Operand<'_>,
    cond_name: &str,
    _true_name: &str,
    _false_name: &str,
) -> Result<HashMap<NodeId, AlgorithmParamValue>> {
    if condition.is_empty() {
        return Ok(HashMap::new());
    }

    let mut keys: Vec<NodeId> = condition.keys().copied().collect();
    keys.sort_unstable();

    let mut result = HashMap::with_capacity(condition.len());

    for node in keys {
        let cond_val = condition.get(&node).expect("key must exist");
        let is_true = condition_is_true(cond_val, cond_name)?;

        let selected = if is_true {
            match &if_true {
                Operand::NodeMap(map) => map
                    .get(&node)
                    .ok_or_else(|| anyhow!("if_true map missing node {:?}", node))?
                    .clone(),
                Operand::Scalar(val) => (*val).clone(),
                Operand::EdgeMap(_) => {
                    bail!("if_true must be node map when condition is node map")
                }
            }
        } else {
            match &if_false {
                Operand::NodeMap(map) => map
                    .get(&node)
                    .ok_or_else(|| anyhow!("if_false map missing node {:?}", node))?
                    .clone(),
                Operand::Scalar(val) => (*val).clone(),
                Operand::EdgeMap(_) => {
                    bail!("if_false must be node map when condition is node map")
                }
            }
        };

        result.insert(node, selected);
    }

    Ok(result)
}

fn where_edge_map(
    condition: &HashMap<EdgeId, AlgorithmParamValue>,
    if_true: Operand<'_>,
    if_false: Operand<'_>,
    cond_name: &str,
    _true_name: &str,
    _false_name: &str,
) -> Result<HashMap<EdgeId, AlgorithmParamValue>> {
    if condition.is_empty() {
        return Ok(HashMap::new());
    }

    let mut keys: Vec<EdgeId> = condition.keys().copied().collect();
    keys.sort_unstable();

    let mut result = HashMap::with_capacity(condition.len());

    for edge in keys {
        let cond_val = condition.get(&edge).expect("key must exist");
        let is_true = condition_is_true(cond_val, cond_name)?;

        let selected = if is_true {
            match &if_true {
                Operand::EdgeMap(map) => map
                    .get(&edge)
                    .ok_or_else(|| anyhow!("if_true map missing edge {:?}", edge))?
                    .clone(),
                Operand::Scalar(val) => (*val).clone(),
                Operand::NodeMap(_) => {
                    bail!("if_true must be edge map when condition is edge map")
                }
            }
        } else {
            match &if_false {
                Operand::EdgeMap(map) => map
                    .get(&edge)
                    .ok_or_else(|| anyhow!("if_false map missing edge {:?}", edge))?
                    .clone(),
                Operand::Scalar(val) => (*val).clone(),
                Operand::NodeMap(_) => {
                    bail!("if_false must be edge map when condition is edge map")
                }
            }
        };

        result.insert(edge, selected);
    }

    Ok(result)
}

/// Reduction operations for scalar aggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Mean,
    Min,
    Max,
}

impl ReductionOp {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "sum" => Ok(ReductionOp::Sum),
            "mean" => Ok(ReductionOp::Mean),
            "min" => Ok(ReductionOp::Min),
            "max" => Ok(ReductionOp::Max),
            _ => Err(anyhow!("unknown reduction operator: {}", s)),
        }
    }

    fn description(self) -> &'static str {
        match self {
            ReductionOp::Sum => "Sum reduction",
            ReductionOp::Mean => "Mean reduction",
            ReductionOp::Min => "Min reduction",
            ReductionOp::Max => "Max reduction",
        }
    }
}

/// Reduce a node or edge map to a single scalar value.
pub struct ReduceScalarStep {
    source: String,
    op: ReductionOp,
    target: String,
}

impl ReduceScalarStep {
    pub fn new(source: impl Into<String>, op: ReductionOp, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            op,
            target: target.into(),
        }
    }
}

impl Step for ReduceScalarStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.reduce_scalar"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: self.op.description().to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for ReduceScalarStep {}
impl ReadStep for ReduceScalarStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.reduce_scalar cancelled"));
        }

        let operand = resolve_operand(scope, &self.source)?;

        let scalar_value = match operand {
            Operand::NodeMap(map) => reduce_map_to_scalar(map, self.op, &self.source)?,
            Operand::EdgeMap(map) => reduce_map_to_scalar(map, self.op, &self.source)?,
            Operand::Scalar(val) => {
                // If already scalar, just pass it through
                val.clone()
            }
        };

        scope
            .variables_mut()
            .set_scalar(self.target.clone(), scalar_value);
        Ok(())
    }
}

fn reduce_map_to_scalar<K>(
    map: &HashMap<K, AlgorithmParamValue>,
    op: ReductionOp,
    map_name: &str,
) -> Result<AlgorithmParamValue>
where
    K: Eq + Hash + Ord + Copy,
{
    if map.is_empty() {
        return Ok(AlgorithmParamValue::Float(0.0));
    }

    let mut keys: Vec<_> = map.keys().copied().collect();
    keys.sort_unstable();
    let values = keys
        .into_iter()
        .map(|key| {
            super::aggregations::numeric_value(
                map.get(&key)
                    .ok_or_else(|| anyhow!("Missing value in '{}'", map_name))?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let reducer = match op {
        ReductionOp::Sum => super::aggregations::Reduction::Sum,
        ReductionOp::Mean => super::aggregations::Reduction::Mean,
        ReductionOp::Min => super::aggregations::Reduction::Min,
        ReductionOp::Max => super::aggregations::Reduction::Max,
    };
    reducer.apply_values(&values)
}

/// Broadcast a scalar value to all nodes/edges in a reference map.
pub struct BroadcastScalarStep {
    scalar: String,
    reference: String,
    target: String,
}

impl BroadcastScalarStep {
    pub fn new(
        scalar: impl Into<String>,
        reference: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            scalar: scalar.into(),
            reference: reference.into(),
            target: target.into(),
        }
    }
}

impl Step for BroadcastScalarStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.broadcast_scalar"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Broadcast scalar to all nodes/edges".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for BroadcastScalarStep {}
impl ReadStep for BroadcastScalarStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.broadcast_scalar cancelled"));
        }

        // Get the scalar value
        let scalar_operand = resolve_operand(scope, &self.scalar)?;
        let scalar_value = match scalar_operand {
            Operand::Scalar(val) => val.clone(),
            Operand::NodeMap(_) | Operand::EdgeMap(_) => {
                bail!("core.broadcast_scalar requires scalar source, got map")
            }
        };

        // Get the reference map to determine keys
        let reference_operand = resolve_operand(scope, &self.reference)?;

        match reference_operand {
            Operand::NodeMap(ref_map) => {
                let result = broadcast_to_node_map(ref_map, &scalar_value);
                scope
                    .variables_mut()
                    .set_node_map(self.target.clone(), result);
                Ok(())
            }
            Operand::EdgeMap(ref_map) => {
                let result = broadcast_to_edge_map(ref_map, &scalar_value);
                scope
                    .variables_mut()
                    .set_edge_map(self.target.clone(), result);
                Ok(())
            }
            Operand::Scalar(_) => Err(anyhow!(
                "core.broadcast_scalar requires map reference, got scalar"
            )),
        }
    }
}

fn broadcast_to_node_map(
    reference: &HashMap<NodeId, AlgorithmParamValue>,
    scalar: &AlgorithmParamValue,
) -> HashMap<NodeId, AlgorithmParamValue> {
    let mut result = HashMap::with_capacity(reference.len());
    let mut keys: Vec<NodeId> = reference.keys().copied().collect();
    keys.sort_unstable();
    for node_id in keys {
        result.insert(node_id, scalar.clone());
    }
    result
}

fn broadcast_to_edge_map(
    reference: &HashMap<EdgeId, AlgorithmParamValue>,
    scalar: &AlgorithmParamValue,
) -> HashMap<EdgeId, AlgorithmParamValue> {
    let mut result = HashMap::with_capacity(reference.len());
    let mut keys: Vec<EdgeId> = reference.keys().copied().collect();
    keys.sort_unstable();
    for edge_id in keys {
        result.insert(edge_id, scalar.clone());
    }
    result
}

fn value_to_json(value: &AlgorithmParamValue) -> serde_json::Value {
    use serde_json::Value;
    match value {
        AlgorithmParamValue::Int(i) => Value::Number((*i).into()),
        AlgorithmParamValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        AlgorithmParamValue::Bool(b) => Value::Bool(*b),
        AlgorithmParamValue::Text(s) => Value::String(s.clone()),
        AlgorithmParamValue::IntList(v) => {
            Value::Array(v.iter().map(|i| Value::Number((*i).into())).collect())
        }
        AlgorithmParamValue::FloatList(v) => Value::Array(
            v.iter()
                .filter_map(|f| serde_json::Number::from_f64(*f).map(Value::Number))
                .collect(),
        ),
        AlgorithmParamValue::BoolList(v) => {
            Value::Array(v.iter().map(|b| Value::Bool(*b)).collect())
        }
        AlgorithmParamValue::TextList(v) => {
            Value::Array(v.iter().map(|s| Value::String(s.clone())).collect())
        }
        AlgorithmParamValue::Json(j) => j.clone(),
        AlgorithmParamValue::None => Value::Null,
    }
}

/// Collect neighbor values into lists for each node.
///
/// Used for algorithms like LPA that need to examine all neighbor values
/// to find the most common one.
pub struct CollectNeighborValuesStep {
    source: String,
    target: String,
    include_self: bool,
    direction: Option<super::direction::NeighborDirection>,
}

/// Tie-breaking strategy for mode calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeTieBreak {
    Lowest,
    Highest,
    Keep,
}

/// Find the most frequent value in lists (mode).
///
/// For each node's list of values, finds the most common value.
/// Used in LPA to adopt the most common neighbor label.
pub struct ModeListStep {
    source: String,
    target: String,
    tie_break: ModeTieBreak,
}

impl ModeListStep {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        tie_break: ModeTieBreak,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            tie_break,
        }
    }
}

impl Step for ModeListStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::ImmutableInputs
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.mode_list"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Find most frequent value in lists".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for ModeListStep {}
impl ReadStep for ModeListStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.mode_list cancelled"));
        }

        let source_map = scope.variables().node_map(&self.source)?;
        let mut result = HashMap::with_capacity(source_map.len());

        for (&node_id, value) in source_map.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("core.mode_list cancelled"));
            }

            // Keep integer labels typed through counting. The generic JSON path
            // remains authoritative for mixed domains and floating labels.
            let integer_mode = match value {
                AlgorithmParamValue::IntList(values) => {
                    Some(mode_integers(ctx, values.iter().copied(), self.tie_break)?)
                }
                AlgorithmParamValue::Json(serde_json::Value::Array(values))
                    if values.iter().all(|value| value.as_i64().is_some()) =>
                {
                    Some(mode_integers(
                        ctx,
                        values.iter().filter_map(|value| value.as_i64()),
                        self.tie_break,
                    )?)
                }
                _ => None,
            };
            if let Some(value) = integer_mode {
                result.insert(node_id, value);
                continue;
            }
            super::expression::validate_value(value)?;
            // Extract array from JSON value
            let array = match value {
                AlgorithmParamValue::Json(serde_json::Value::Array(arr)) => arr,
                AlgorithmParamValue::IntList(v) => {
                    // Convert to JSON array
                    &v.iter()
                        .map(|i| serde_json::Value::Number((*i).into()))
                        .collect::<Vec<_>>()
                }
                AlgorithmParamValue::FloatList(v) => &v
                    .iter()
                    .filter_map(|f| serde_json::Number::from_f64(*f).map(serde_json::Value::Number))
                    .collect::<Vec<_>>(),
                _ => {
                    return Err(anyhow!(
                        "core.mode_list requires JSON array or list, got {:?}",
                        value
                    ));
                }
            };

            if array.is_empty() {
                // Empty list - keep original value or use None
                result.insert(node_id, AlgorithmParamValue::None);
                continue;
            }

            // Count frequencies
            let mut freq_map: std::collections::HashMap<String, (usize, serde_json::Value)> =
                std::collections::HashMap::new();

            for val in array {
                if ctx.is_cancelled() {
                    return Err(anyhow!("core.mode_list cancelled"));
                }
                let key = serde_json::to_string(val).unwrap_or_default();
                freq_map
                    .entry(key)
                    .and_modify(|(count, _)| *count += 1)
                    .or_insert((1, val.clone()));
            }

            // Find max frequency
            let max_freq = freq_map
                .values()
                .map(|(count, _)| *count)
                .max()
                .unwrap_or(0);

            // Find all values with max frequency
            let mut candidates: Vec<_> = freq_map
                .into_iter()
                .filter(|(_, (count, _))| *count == max_freq)
                .map(|(_, (_, val))| val)
                .collect();

            if !matches!(self.tie_break, ModeTieBreak::Keep) {
                for candidate in &candidates {
                    compare_json_values(candidate, &candidates[0])?;
                }
            }
            // Apply tie-breaking
            let mode_value = match self.tie_break {
                ModeTieBreak::Keep => {
                    // Keep the first occurrence in the original list
                    // Find which candidate appears first in the original array
                    let mut first_candidate = candidates[0].clone();
                    let mut first_pos = usize::MAX;

                    for candidate in &candidates {
                        for (i, val) in array.iter().enumerate() {
                            if val == candidate && i < first_pos {
                                first_pos = i;
                                first_candidate = candidate.clone();
                                break;
                            }
                        }
                    }
                    first_candidate
                }
                ModeTieBreak::Lowest => {
                    // Find the minimum value
                    candidates.sort_by(|a, b| {
                        compare_json_values(a, b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.to_string().cmp(&b.to_string()))
                    });
                    candidates.into_iter().next().unwrap()
                }
                ModeTieBreak::Highest => {
                    // Find the maximum value
                    candidates.sort_by(|a, b| {
                        compare_json_values(b, a)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.to_string().cmp(&b.to_string()))
                    });
                    candidates.into_iter().next().unwrap()
                }
            };

            // Convert JSON value back to AlgorithmParamValue
            let result_value = json_to_algorithm_value(&mode_value);
            result.insert(node_id, result_value);
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

/// Integer specialization of the same mode contract: exact category identity,
/// modal frequency first, then numeric or first-occurrence tie resolution.
fn mode_integers(
    ctx: &Context,
    values: impl Iterator<Item = i64>,
    tie_break: ModeTieBreak,
) -> Result<AlgorithmParamValue> {
    let mut counts: HashMap<i64, (usize, usize)> = HashMap::new();
    for (position, value) in values.enumerate() {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.mode_list cancelled"));
        }
        let entry = counts.entry(value).or_insert((0, position));
        entry.0 = entry
            .0
            .checked_add(1)
            .ok_or_else(|| anyhow!("Mode frequency exceeds usize"))?;
    }
    let best = counts
        .into_iter()
        .max_by(|(left, (lc, lp)), (right, (rc, rp))| {
            lc.cmp(rc).then_with(|| match tie_break {
                ModeTieBreak::Lowest => right.cmp(left),
                ModeTieBreak::Highest => left.cmp(right),
                ModeTieBreak::Keep => rp.cmp(lp),
            })
        });
    Ok(best
        .map(|(value, _)| AlgorithmParamValue::Int(value))
        .unwrap_or(AlgorithmParamValue::None))
}

fn compare_json_values(a: &serde_json::Value, b: &serde_json::Value) -> Result<std::cmp::Ordering> {
    use crate::storage::numeric::scalar::{compare_inputs, NumericInput};
    use serde_json::Value;
    let input = |n: &serde_json::Number| -> Result<NumericInput> {
        if let Some(i) = n.as_i64() {
            Ok(NumericInput::Integer(i128::from(i)))
        } else if let Some(i) = n.as_u64() {
            Ok(NumericInput::Integer(i128::from(i)))
        } else {
            n.as_f64()
                .map(NumericInput::Float)
                .ok_or_else(|| anyhow!("Invalid numeric label"))
        }
    };
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => compare_inputs(input(a)?, input(b)?)
            .ok_or_else(|| anyhow!("Mode ties require ordered numeric labels")),
        (Value::String(a), Value::String(b)) => Ok(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Ok(a.cmp(b)),
        (Value::Null, Value::Null) => Ok(std::cmp::Ordering::Equal),
        _ => Err(anyhow!("Mode ties require comparable scalar labels")),
    }
}

fn json_to_algorithm_value(val: &serde_json::Value) -> AlgorithmParamValue {
    use serde_json::Value;
    match val {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                AlgorithmParamValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                AlgorithmParamValue::Float(f)
            } else {
                AlgorithmParamValue::None
            }
        }
        Value::String(s) => AlgorithmParamValue::Text(s.clone()),
        Value::Bool(b) => AlgorithmParamValue::Bool(*b),
        Value::Null => AlgorithmParamValue::None,
        other => AlgorithmParamValue::Json(other.clone()),
    }
}

impl CollectNeighborValuesStep {
    pub fn new(source: impl Into<String>, target: impl Into<String>, include_self: bool) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            include_self,
            direction: None,
        }
    }
    /// Override graph-default adjacency with an explicit traversal direction.
    pub fn with_direction(mut self, direction: super::direction::NeighborDirection) -> Self {
        self.direction = Some(direction);
        self
    }
}

impl Step for CollectNeighborValuesStep {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.collect_neighbor_values"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Collect neighbor values into lists".to_string(),
            cost_hint: CostHint::Linear,
        }
    }

    fn read_kernel(&self) -> Option<&dyn ReadStep> {
        Some(self)
    }
    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        self.apply_read(ctx, &mut scope.read_scope())
    }
}

impl sealed::Sealed for CollectNeighborValuesStep {}
impl ReadStep for CollectNeighborValuesStep {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.collect_neighbor_values cancelled"));
        }

        // Get ordered nodes and build CSR
        let nodes = scope.ordered_nodes();

        let direction = self.direction.unwrap_or_else(|| {
            if scope.is_directed() {
                super::direction::NeighborDirection::Out
            } else {
                super::direction::NeighborDirection::Undirected
            }
        });
        let topology = scope.selected_csr(ctx, direction)?;
        let csr = &topology.csr;

        // Get source values
        let source_map = scope.variables().node_map(&self.source)?;

        // Collect neighbor values for each node
        let mut result = HashMap::with_capacity(nodes.len());

        for u_idx in 0..csr.node_count() {
            if ctx.is_cancelled() {
                return Err(anyhow!("core.collect_neighbor_values cancelled"));
            }

            let node = nodes[u_idx];
            let nbrs = csr.neighbors(u_idx);

            // Collect neighbor values as JSON array
            let mut values = Vec::with_capacity(nbrs.len() + if self.include_self { 1 } else { 0 });

            // Optionally include self value
            if self.include_self {
                if let Some(self_value) = source_map.get(&node) {
                    super::expression::validate_value(self_value)?;
                    values.push(value_to_json(self_value));
                }
            }

            // Collect neighbor values
            for &nbr_idx in nbrs {
                if ctx.is_cancelled() {
                    return Err(anyhow!("core.collect_neighbor_values cancelled"));
                }
                let nbr_node = nodes[nbr_idx];
                if let Some(value) = source_map.get(&nbr_node) {
                    super::expression::validate_value(value)?;
                    values.push(value_to_json(value));
                }
            }

            result.insert(
                node,
                AlgorithmParamValue::Json(serde_json::Value::Array(values)),
            );
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);
        Ok(())
    }
}

#[cfg(test)]
mod checked_comparison_tests {
    use super::*;
    #[test]
    fn comparisons_preserve_adjacent_integers_and_distinguish_tiny_floats() {
        let low = AlgorithmParamValue::Int(9_007_199_254_740_992);
        let high = AlgorithmParamValue::Int(9_007_199_254_740_993);
        assert!(CompareOp::Lt.apply_values(&low, &high).unwrap());
        assert!(!CompareOp::Eq.apply_values(&low, &high).unwrap());
        assert!(CompareOp::Gt
            .apply_values(&high, &AlgorithmParamValue::Float(9_007_199_254_740_992.))
            .unwrap());
        assert!(!CompareOp::Eq
            .apply_values(
                &AlgorithmParamValue::Float(1e-20),
                &AlgorithmParamValue::Float(0.)
            )
            .unwrap());
        assert!(CompareOp::Eq
            .apply_values(&AlgorithmParamValue::Float(f64::NAN), &low)
            .is_err());
    }
}
