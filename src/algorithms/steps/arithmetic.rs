//! Arithmetic operation step primitives.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{anyhow, bail, Result};

use crate::types::{EdgeId, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope, StepSpec};

/// Binary arithmetic operations (add, sub, mul, div).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOperation {
    pub(crate) fn description(self) -> &'static str {
        match self {
            BinaryOperation::Add => "Element-wise addition",
            BinaryOperation::Sub => "Element-wise subtraction",
            BinaryOperation::Mul => "Element-wise multiplication",
            BinaryOperation::Div => "Element-wise division",
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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

enum Operand<'a> {
    NodeMap(&'a HashMap<NodeId, AlgorithmParamValue>),
    EdgeMap(&'a HashMap<EdgeId, AlgorithmParamValue>),
    Scalar(&'a AlgorithmParamValue),
}

fn resolve_operand<'a>(scope: &'a StepScope<'a>, name: &str) -> Result<Operand<'a>> {
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
    if divisor.abs() <= f64::EPSILON {
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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

    fn apply(self, left: f64, right: f64) -> bool {
        match self {
            CompareOp::Eq => (left - right).abs() < f64::EPSILON,
            CompareOp::Ne => (left - right).abs() >= f64::EPSILON,
            CompareOp::Lt => left < right,
            CompareOp::Le => left <= right,
            CompareOp::Gt => left > right,
            CompareOp::Ge => left >= right,
        }
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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

        let left_num = NumericValue::from(left_value, left_name)?;
        let right_num = NumericValue::from(right_value, right_name)?;

        let cmp_result = op.apply(left_num.as_f64(), right_num.as_f64());
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

    let scalar_num = NumericValue::from(scalar, scalar_name)?;
    let scalar_val = scalar_num.as_f64();

    let mut result = HashMap::with_capacity(map.len());
    for (&key, value) in map.iter() {
        let num = NumericValue::from(value, map_name)?;
        let val = num.as_f64();

        let cmp_result = op.apply(val, scalar_val);
        let mask_value = if cmp_result { 1.0 } else { 0.0 };

        result.insert(key, AlgorithmParamValue::Float(mask_value));
    }

    Ok(result)
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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
        let cond_num = NumericValue::from(cond_val, cond_name)?;
        let is_true = cond_num.as_f64().abs() > f64::EPSILON;

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
        let cond_num = NumericValue::from(cond_val, cond_name)?;
        let is_true = cond_num.as_f64().abs() > f64::EPSILON;

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

    fn apply(self, values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }

        match self {
            ReductionOp::Sum => Some(values.iter().sum()),
            ReductionOp::Mean => Some(values.iter().sum::<f64>() / values.len() as f64),
            ReductionOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min).into(),
            ReductionOp::Max => values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .into(),
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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

    let mut values = Vec::with_capacity(map.len());
    let mut keys: Vec<K> = map.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let value = map.get(&key).expect("key must exist");
        let num = NumericValue::from(value, map_name)?;
        values.push(num.as_f64());
    }

    let result = op.apply(&values).unwrap_or(0.0);
    Ok(AlgorithmParamValue::Float(result))
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("core.mode_list cancelled"));
        }

        let source_map = scope.variables().node_map(&self.source)?;
        let mut result = HashMap::with_capacity(source_map.len());

        for (&node_id, value) in source_map.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("core.mode_list cancelled"));
            }

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
                        compare_json_values(a, b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.into_iter().next().unwrap()
                }
                ModeTieBreak::Highest => {
                    // Find the maximum value
                    candidates.sort_by(|a, b| {
                        compare_json_values(b, a).unwrap_or(std::cmp::Ordering::Equal)
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

fn compare_json_values(a: &serde_json::Value, b: &serde_json::Value) -> Result<std::cmp::Ordering> {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let fa = na.as_f64().unwrap_or(0.0);
            let fb = nb.as_f64().unwrap_or(0.0);
            Ok(fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal))
        }
        (Value::String(sa), Value::String(sb)) => Ok(sa.cmp(sb)),
        (Value::Bool(ba), Value::Bool(bb)) => Ok(ba.cmp(bb)),
        _ => Ok(std::cmp::Ordering::Equal),
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
        }
    }
}

impl Step for CollectNeighborValuesStep {
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

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};

        if ctx.is_cancelled() {
            return Err(anyhow!("core.collect_neighbor_values cancelled"));
        }

        // Get ordered nodes and build CSR
        let subgraph = scope.subgraph();
        let nodes = subgraph.ordered_nodes();

        // Detect if graph is directed
        let graph_rc = subgraph.graph();
        let add_reverse = {
            let graph = graph_rc.borrow();
            graph.is_undirected()
        };

        // Get or build CSR
        let csr = if let Some(cached) = subgraph.csr_cache_get(add_reverse) {
            cached
        } else {
            let mut csr = Csr::default();
            let edges = subgraph.edges();

            let mut node_to_idx = std::collections::HashMap::new();
            for (idx, &node_id) in nodes.iter().enumerate() {
                node_to_idx.insert(node_id, idx);
            }

            {
                let graph = graph_rc.borrow();
                let pool = graph.pool();

                let _build_time = build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| node_to_idx.get(&nid).copied(),
                    |eid| pool.get_edge_endpoints(eid),
                    CsrOptions {
                        add_reverse_edges: add_reverse,
                        sort_neighbors: false,
                    },
                );
            }

            let csr_arc = std::sync::Arc::new(csr);
            subgraph.csr_cache_store(add_reverse, csr_arc.clone());
            csr_arc
        };

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
                    values.push(value_to_json(self_value));
                }
            }

            // Collect neighbor values
            for &nbr_idx in nbrs {
                let nbr_node = nodes[nbr_idx];
                if let Some(value) = source_map.get(&nbr_node) {
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
