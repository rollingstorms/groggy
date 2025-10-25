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
    K: Copy + Eq + Hash + Debug,
{
    if left.is_empty() {
        return Ok(HashMap::new());
    }

    let mut result = HashMap::with_capacity(left.len());
    for (&key, left_value) in left.iter() {
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

    for key in right.keys() {
        if !left.contains_key(key) {
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
    K: Copy + Eq + Hash + Debug,
{
    if map.is_empty() {
        return Ok(HashMap::new());
    }

    let mut result = HashMap::with_capacity(map.len());
    for (&key, value) in map.iter() {
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
