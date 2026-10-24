//! Expression system for map_nodes and other transformation steps.
//!
//! This module provides a serializable expression language that allows
//! declarative transformations without requiring Rust callbacks.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::traits::SubgraphOperations;
use crate::types::{AttrName, NodeId};

use std::collections::{HashMap, HashSet};

use super::super::AlgorithmParamValue;
use super::core::StepInput;

/// Binary operations supported in expressions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
}

/// Unary operations supported in expressions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOp {
    Neg,
    Not,
    Abs,
    Sqrt,
    Log,
    Exp,
    Floor,
    Ceil,
    Round,
}

/// Expression that can be evaluated in the context of a node.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Expr {
    /// Constant value
    Const {
        #[serde(deserialize_with = "deserialize_constant")]
        value: AlgorithmParamValue,
    },

    /// Reference to a variable (from step variables)
    Var { name: String },

    /// Reference to node attribute
    Attr { name: String },

    /// Binary operation
    BinaryOp {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation
    UnaryOp { op: UnaryOp, arg: Box<Expr> },

    /// Function call
    Call { func: String, args: Vec<Expr> },

    /// Conditional expression
    If {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
}

/// Accept the native tagged representation and the literal JSON emitted by the
/// Python expression parser. Keep inference here so FFI only marshals the AST.
fn deserialize_constant<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<AlgorithmParamValue, D::Error> {
    use serde_json::Value;
    use AlgorithmParamValue as V;
    let value = Value::deserialize(deserializer)?;
    let result = match value {
        Value::Object(ref object) if object.contains_key("type") => {
            serde_json::from_value(value).map_err(serde::de::Error::custom)?
        }
        Value::Null => V::None,
        Value::Bool(value) => V::Bool(value),
        Value::String(value) => V::Text(value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                V::Int(value)
            } else if value.is_u64() {
                return Err(serde::de::Error::custom(
                    "expression integer literal exceeds i64",
                ));
            } else {
                V::Float(
                    value
                        .as_f64()
                        .ok_or_else(|| serde::de::Error::custom("invalid expression number"))?,
                )
            }
        }
        Value::Array(values) if values.iter().all(|v| v.as_i64().is_some()) => {
            V::IntList(values.iter().filter_map(Value::as_i64).collect())
        }
        Value::Array(values) if values.iter().all(Value::is_number) => {
            if values.iter().any(|v| v.is_u64() && v.as_i64().is_none()) {
                return Err(serde::de::Error::custom(
                    "expression integer literal exceeds i64",
                ));
            }
            V::FloatList(values.iter().filter_map(Value::as_f64).collect())
        }
        Value::Array(values) if values.iter().all(Value::is_boolean) => {
            V::BoolList(values.iter().filter_map(Value::as_bool).collect())
        }
        Value::Array(values) if values.iter().all(Value::is_string) => V::TextList(
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        ),
        value => V::Json(value),
    };
    validate_value(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

/// Context for evaluating expressions.
pub struct ExprContext<'a> {
    pub node: NodeId,
    pub input: &'a StepInput<'a>,
    pub current_value: Option<&'a AlgorithmParamValue>,
    /// Optional CSR context for optimized neighbor access
    pub csr_ctx: Option<CsrContext<'a>>,
}

/// CSR context for efficient neighbor operations in expressions
pub struct CsrContext<'a> {
    /// Current node's CSR index
    pub node_idx: usize,
    /// Reference to CSR adjacency
    pub csr: &'a crate::state::topology::Csr,
    /// Ordered nodes array (for index->NodeId mapping)
    pub ordered_nodes: &'a [crate::types::NodeId],
}

impl<'a> ExprContext<'a> {
    pub fn new(node: NodeId, input: &'a StepInput<'a>) -> Self {
        Self {
            node,
            input,
            current_value: None,
            csr_ctx: None,
        }
    }

    pub fn with_value(
        node: NodeId,
        input: &'a StepInput<'a>,
        value: &'a AlgorithmParamValue,
    ) -> Self {
        Self {
            node,
            input,
            current_value: Some(value),
            csr_ctx: None,
        }
    }

    /// Create context with CSR optimization (for Phase 3 neighbor aggregation)
    pub fn with_csr(
        node: NodeId,
        node_idx: usize,
        csr: &'a crate::state::topology::Csr,
        ordered_nodes: &'a [crate::types::NodeId],
        input: &'a StepInput<'a>,
    ) -> Self {
        Self {
            node,
            input,
            current_value: None,
            csr_ctx: Some(CsrContext {
                node_idx,
                csr,
                ordered_nodes,
            }),
        }
    }
}

// Keep run-specific overlays private and preserve the public ExprContext shape.
struct Evaluation<'a> {
    context: &'a ExprContext<'a>,
    property_batch: Option<&'a super::super::property_batch::PropertyBatch>,
    neighbors: Option<&'a [NodeId]>,
    source_override: Option<(&'a str, &'a HashMap<NodeId, AlgorithmParamValue>)>,
}
impl<'a> std::ops::Deref for Evaluation<'a> {
    type Target = ExprContext<'a>;
    fn deref(&self) -> &Self::Target {
        self.context
    }
}
impl Evaluation<'_> {
    fn node_map(&self, name: &str) -> Result<&HashMap<NodeId, AlgorithmParamValue>> {
        if let Some((source, values)) = self.source_override {
            if source == name {
                return Ok(values);
            }
        }
        self.context.input.variables.node_map(name)
    }
}

impl Expr {
    /// Evaluate the expression in the given context.
    pub fn eval(&self, ctx: &ExprContext) -> Result<AlgorithmParamValue> {
        self.eval_scoped(ctx, None, None, None)
    }

    pub(super) fn eval_scoped<'a>(
        &self,
        context: &'a ExprContext<'a>,
        property_batch: Option<&'a super::super::property_batch::PropertyBatch>,
        neighbors: Option<&'a [NodeId]>,
        source_override: Option<(&'a str, &'a HashMap<NodeId, AlgorithmParamValue>)>,
    ) -> Result<AlgorithmParamValue> {
        self.eval_context(&Evaluation {
            context,
            property_batch,
            neighbors,
            source_override,
        })
    }

    fn eval_context(&self, ctx: &Evaluation<'_>) -> Result<AlgorithmParamValue> {
        let value = self.eval_inner(ctx)?;
        validate_value(&value)?;
        Ok(value)
    }

    fn eval_inner(&self, ctx: &Evaluation<'_>) -> Result<AlgorithmParamValue> {
        match self {
            Expr::Const { value } => Ok(value.clone()),

            Expr::Var { name } => {
                if name == "value" {
                    // Special case: "value" refers to the current value in the map
                    ctx.current_value
                        .cloned()
                        .ok_or_else(|| anyhow!("no current value available"))
                } else {
                    // Look up in variables
                    let var_map = ctx.node_map(name)?;
                    var_map.get(&ctx.node).cloned().ok_or_else(|| {
                        anyhow!("variable '{}' not found for node {:?}", name, ctx.node)
                    })
                }
            }

            Expr::Attr { name } => {
                let name = AttrName::from(name.clone());
                let value = if let Some(value) = ctx
                    .property_batch
                    .and_then(|batch| batch.get(true, ctx.node, &name))
                {
                    Some(value.clone())
                } else {
                    ctx.input.subgraph.get_node_attribute(ctx.node, &name)?
                };
                value
                    .and_then(AlgorithmParamValue::from_attr_value)
                    .ok_or_else(|| {
                        anyhow!("attribute '{}' not found for node {:?}", name, ctx.node)
                    })
            }

            Expr::BinaryOp { op, left, right } => {
                let left_val = left.eval_context(ctx)?;
                if matches!(
                    (op, &left_val),
                    (BinaryOp::And, AlgorithmParamValue::Bool(false))
                ) {
                    return Ok(AlgorithmParamValue::Bool(false));
                }
                if matches!(
                    (op, &left_val),
                    (BinaryOp::Or, AlgorithmParamValue::Bool(true))
                ) {
                    return Ok(AlgorithmParamValue::Bool(true));
                }
                let right_val = right.eval_context(ctx)?;
                eval_binary_op(*op, &left_val, &right_val)
            }

            Expr::UnaryOp { op, arg } => {
                let arg_val = arg.eval_context(ctx)?;
                eval_unary_op(*op, &arg_val)
            }

            Expr::Call { func, args } => eval_function(func, args, ctx),

            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond_val = condition.eval_context(ctx)?;
                let cond_bool = match cond_val {
                    AlgorithmParamValue::Bool(b) => b,
                    AlgorithmParamValue::Int(i) => i != 0,
                    AlgorithmParamValue::Float(f) => f != 0.0,
                    _ => return Err(anyhow!("condition must evaluate to boolean or number")),
                };

                if cond_bool {
                    then_expr.eval_context(ctx)
                } else {
                    else_expr.eval_context(ctx)
                }
            }
        }
    }

    pub(super) fn needs_neighbors(&self) -> bool {
        match self {
            Self::Call { func, args } => {
                matches!(
                    func.as_str(),
                    "neighbors" | "neighbor_count" | "neighbor_values"
                ) || args.iter().any(Self::needs_neighbors)
            }
            Self::BinaryOp { left, right, .. } => left.needs_neighbors() || right.needs_neighbors(),
            Self::UnaryOp { arg, .. } => arg.needs_neighbors(),
            Self::If {
                condition,
                then_expr,
                else_expr,
            } => {
                condition.needs_neighbors()
                    || then_expr.needs_neighbors()
                    || else_expr.needs_neighbors()
            }
            _ => false,
        }
    }

    /// Helper: Create a constant expression
    pub fn constant(value: AlgorithmParamValue) -> Self {
        Expr::Const { value }
    }

    /// Helper: Create a variable reference
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var { name: name.into() }
    }

    /// Helper: Create an attribute reference
    pub fn attr(name: impl Into<String>) -> Self {
        Expr::Attr { name: name.into() }
    }

    /// Helper: Create a binary operation
    pub fn binary(op: BinaryOp, left: Expr, right: Expr) -> Self {
        Expr::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

/// Expressions reject nonfinite scalar/list values instead of leaking NaN/Inf
/// into later properties. Integers retain checked i64 arithmetic.
pub(super) fn validate_value(value: &AlgorithmParamValue) -> Result<()> {
    match value {
        AlgorithmParamValue::Float(value) if !value.is_finite() => {
            Err(anyhow!("expression requires finite numeric values"))
        }
        AlgorithmParamValue::FloatList(values) if values.iter().any(|v| !v.is_finite()) => {
            Err(anyhow!("expression requires finite numeric values"))
        }
        _ => Ok(()),
    }
}

fn eval_binary_op(
    op: BinaryOp,
    left: &AlgorithmParamValue,
    right: &AlgorithmParamValue,
) -> Result<AlgorithmParamValue> {
    use AlgorithmParamValue::*;
    use BinaryOp as Op;
    validate_value(left)?;
    validate_value(right)?;
    if matches!(op, Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge) {
        let compared = super::filtering::compare_values(left, right);
        let equal = compared.map(|c| c.is_eq()).unwrap_or(left == right);
        return Ok(Bool(match op {
            Op::Eq => equal,
            Op::Ne => !equal,
            _ => {
                let order = compared
                    .ok_or_else(|| anyhow!("expression comparison requires comparable scalars"))?;
                match op {
                    Op::Lt => order.is_lt(),
                    Op::Le => order.is_le(),
                    Op::Gt => order.is_gt(),
                    Op::Ge => order.is_ge(),
                    _ => unreachable!(),
                }
            }
        }));
    }
    if let (Int(a), Int(b)) = (left, right) {
        let checked = match op {
            Op::Add => Some(a.checked_add(*b)),
            Op::Sub => Some(a.checked_sub(*b)),
            Op::Mul => Some(a.checked_mul(*b)),
            Op::Mod => Some(a.checked_rem(*b)),
            Op::Pow if *b >= 0 => Some(
                u32::try_from(*b)
                    .ok()
                    .and_then(|power| a.checked_pow(power)),
            ),
            _ => Option::None,
        };
        if let Some(value) = checked {
            return value
                .map(Int)
                .ok_or_else(|| anyhow!("expression integer overflow or invalid remainder"));
        }
    }
    match (op, left, right) {
        (Op::Add, Text(a), Text(b)) => return Ok(Text(format!("{a}{b}"))),
        (Op::And, Bool(a), Bool(b)) => return Ok(Bool(*a && *b)),
        (Op::Or, Bool(a), Bool(b)) => return Ok(Bool(*a || *b)),
        _ => {}
    }
    let number = |value: &AlgorithmParamValue| match value {
        Int(value) => Ok(*value as f64),
        Float(value) => Ok(*value),
        _ => Err(anyhow!("expression arithmetic requires numbers")),
    };
    let (a, b) = (number(left)?, number(right)?);
    if matches!(op, Op::Div | Op::Mod) && b == 0.0 {
        return Err(anyhow!("division by zero"));
    }
    let result = Float(match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        Op::Mod => a % b,
        Op::Pow => a.powf(b),
        _ => return Err(anyhow!("unsupported expression operator")),
    });
    validate_value(&result)?;
    Ok(result)
}

fn eval_unary_op(op: UnaryOp, arg: &AlgorithmParamValue) -> Result<AlgorithmParamValue> {
    use AlgorithmParamValue::*;
    validate_value(arg)?;
    if let Int(value) = arg {
        match op {
            UnaryOp::Neg => {
                return value
                    .checked_neg()
                    .map(Int)
                    .ok_or_else(|| anyhow!("expression integer overflow"))
            }
            UnaryOp::Abs => {
                return value
                    .checked_abs()
                    .map(Int)
                    .ok_or_else(|| anyhow!("expression integer overflow"))
            }
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round => return Ok(Int(*value)),
            _ => {}
        }
    }
    if let (UnaryOp::Not, Bool(value)) = (op, arg) {
        return Ok(Bool(!value));
    }
    let value = match arg {
        Int(value) => *value as f64,
        Float(value) => *value,
        _ => return Err(anyhow!("expression unary operator requires number")),
    };
    let result = Float(match op {
        UnaryOp::Neg => -value,
        UnaryOp::Abs => value.abs(),
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::Log => value.ln(),
        UnaryOp::Exp => value.exp(),
        UnaryOp::Floor => value.floor(),
        UnaryOp::Ceil => value.ceil(),
        UnaryOp::Round => value.round(),
        _ => return Err(anyhow!("unsupported expression unary operator")),
    });
    validate_value(&result)?;
    Ok(result)
}

fn expression_neighbors(ctx: &Evaluation<'_>) -> Result<Vec<NodeId>> {
    if let Some(neighbors) = ctx.neighbors {
        return Ok(neighbors.to_vec());
    }
    if let Some(csr) = &ctx.csr_ctx {
        return Ok(csr
            .csr
            .neighbors(csr.node_idx)
            .iter()
            .map(|&i| csr.ordered_nodes[i])
            .collect());
    }
    let (_, mut rows) = super::transformations::selected_neighbors(
        ctx.input.subgraph,
        &super::super::Context::default(),
    )?;
    rows.remove(&ctx.node)
        .ok_or_else(|| anyhow!("expression node outside selection"))
}

fn eval_function(func: &str, args: &[Expr], ctx: &Evaluation<'_>) -> Result<AlgorithmParamValue> {
    match func {
        "min" | "max" => {
            if args.len() != 2 {
                return Err(anyhow!("{func} requires 2 arguments"));
            }
            let a = args[0].eval_context(ctx)?;
            let b = args[1].eval_context(ctx)?;
            super::aggregations::numeric_value(&a)?;
            super::aggregations::numeric_value(&b)?;
            let order = super::filtering::compare_values(&a, &b)
                .ok_or_else(|| anyhow!("{func} requires numeric arguments"))?;
            Ok(
                if (func == "min" && order.is_gt()) || (func == "max" && order.is_lt()) {
                    b
                } else {
                    a
                },
            )
        }

        "neighbors" => {
            if !args.is_empty() {
                return Err(anyhow!("neighbors takes no arguments"));
            }
            let nodes = expression_neighbors(ctx)?
                .into_iter()
                .map(|node| i64::try_from(node).map_err(|_| anyhow!("node ID exceeds i64")))
                .collect::<Result<Vec<_>>>()?;
            Ok(AlgorithmParamValue::IntList(nodes))
        }
        "count" => {
            if args.len() != 1 {
                return Err(anyhow!("count requires 1 argument"));
            }
            let length = match args[0].eval_context(ctx)? {
                AlgorithmParamValue::IntList(values) => values.len(),
                AlgorithmParamValue::FloatList(values) => values.len(),
                _ => return Err(anyhow!("count requires numeric array")),
            };
            Ok(AlgorithmParamValue::Int(i64::try_from(length)?))
        }
        "neighbor_count" => {
            if !args.is_empty() {
                return Err(anyhow!("neighbor_count takes no arguments"));
            }
            let degree = expression_neighbors(ctx)?.len();
            Ok(AlgorithmParamValue::Int(degree as i64))
        }

        "neighbor_values" => {
            // Get variable values for current node's neighbors
            if args.len() != 1 {
                return Err(anyhow!("neighbor_values requires 1 argument"));
            }

            // The argument should be a variable reference
            let Expr::Var { name } = &args[0] else {
                return Err(anyhow!("neighbor_values expects a variable name"));
            };

            // Get the variable map
            let var_map = ctx.node_map(name)?;

            // Get neighbors
            let neighbors = expression_neighbors(ctx)?;

            // Collect neighbor values, preserving integer-only collections when possible.
            let mut int_values = Vec::new();
            let mut float_values = Vec::new();
            let mut use_float = false;

            // Preserve one value per distinct selected neighbor, including a stored self-loop.
            let mut seen_neighbors = HashSet::new();
            for neighbor in neighbors {
                if !seen_neighbors.insert(neighbor) {
                    continue;
                }
                if let Some(val) = var_map.get(&neighbor) {
                    match val {
                        AlgorithmParamValue::Float(f) => {
                            if !use_float {
                                use_float = true;
                                if !int_values.is_empty() {
                                    float_values.extend(int_values.iter().map(|i| *i as f64));
                                    int_values.clear();
                                }
                            }
                            float_values.push(*f);
                        }
                        AlgorithmParamValue::Int(i) => {
                            if use_float {
                                float_values.push(*i as f64);
                            } else {
                                int_values.push(*i);
                            }
                        }
                        AlgorithmParamValue::FloatList(list) => {
                            if !use_float {
                                use_float = true;
                                if !int_values.is_empty() {
                                    float_values.extend(int_values.iter().map(|i| *i as f64));
                                    int_values.clear();
                                }
                            }
                            float_values.extend(list.iter().copied());
                        }
                        AlgorithmParamValue::IntList(list) => {
                            if use_float {
                                float_values.extend(list.iter().map(|i| *i as f64));
                            } else {
                                int_values.extend(list.iter().copied());
                            }
                        }
                        _ => {
                            return Err(anyhow!(
                                "neighbor_values requires numeric values or arrays"
                            ))
                        }
                    }
                }
            }

            if use_float {
                if !int_values.is_empty() {
                    float_values.extend(int_values.into_iter().map(|i| i as f64));
                }
                Ok(AlgorithmParamValue::FloatList(float_values))
            } else {
                Ok(AlgorithmParamValue::IntList(int_values))
            }
        }

        "sum" | "mean" => {
            if args.len() != 1 {
                return Err(anyhow!("{func} requires 1 argument"));
            }
            let value = args[0].eval_context(ctx)?;
            if func == "sum" {
                if let AlgorithmParamValue::IntList(values) = &value {
                    let total = values
                        .iter()
                        .try_fold(0i128, |sum, value| sum.checked_add(i128::from(*value)))
                        .ok_or_else(|| anyhow!("expression integer sum overflow"))?;
                    return i64::try_from(total)
                        .map(AlgorithmParamValue::Int)
                        .map_err(|_| anyhow!("expression integer sum overflow"));
                }
                if matches!(
                    value,
                    AlgorithmParamValue::Int(_) | AlgorithmParamValue::Float(_)
                ) {
                    return Ok(value);
                }
            }
            let values = match value {
                AlgorithmParamValue::IntList(values) => values
                    .into_iter()
                    .map(AlgorithmParamValue::Int)
                    .collect::<Vec<_>>(),
                AlgorithmParamValue::FloatList(values) => {
                    values.into_iter().map(AlgorithmParamValue::Float).collect()
                }
                _ => return Err(anyhow!("{func} requires numeric array")),
            };
            if values.is_empty() {
                return Ok(AlgorithmParamValue::Float(0.0));
            }
            let values = values
                .iter()
                .map(super::aggregations::numeric_value)
                .collect::<Result<Vec<_>>>()?;
            super::aggregations::accumulated(&values, func == "mean")
        }

        "mode" => {
            // Find most common value in an array
            if args.len() != 1 {
                return Err(anyhow!("mode requires 1 argument"));
            }

            let values = args[0].eval_context(ctx)?;

            match values {
                AlgorithmParamValue::FloatList(vec) => {
                    if vec.is_empty() {
                        return Ok(AlgorithmParamValue::Float(0.0));
                    }

                    // Count occurrences - use integer representation for HashMap key
                    let mut counts: std::collections::HashMap<i64, usize> =
                        std::collections::HashMap::new();
                    for &val in &vec {
                        // Convert to bits for exact comparison
                        let key = val.to_bits() as i64;
                        *counts.entry(key).or_insert(0) += 1;
                    }

                    // Find most common
                    let mut best_key = 0i64;
                    let mut best_count = 0usize;
                    let mut first = true;
                    for (key, count) in counts.into_iter() {
                        if first
                            || count > best_count
                            || (count == best_count
                                && f64::from_bits(key as u64)
                                    .total_cmp(&f64::from_bits(best_key as u64))
                                    .is_lt())
                        {
                            best_key = key;
                            best_count = count;
                            first = false;
                        }
                    }
                    Ok(AlgorithmParamValue::Float(f64::from_bits(best_key as u64)))
                }
                AlgorithmParamValue::IntList(vec) => {
                    if vec.is_empty() {
                        return Ok(AlgorithmParamValue::Int(0));
                    }

                    // Count occurrences
                    let mut counts: std::collections::HashMap<i64, usize> =
                        std::collections::HashMap::new();
                    for &val in &vec {
                        *counts.entry(val).or_insert(0) += 1;
                    }

                    // Find most common
                    let mut best_val = 0i64;
                    let mut best_count = 0usize;
                    let mut first = true;
                    for (val, count) in counts.into_iter() {
                        if first || count > best_count || (count == best_count && val < best_val) {
                            best_val = val;
                            best_count = count;
                            first = false;
                        }
                    }
                    Ok(AlgorithmParamValue::Int(best_val))
                }
                _ => Err(anyhow!("mode requires numeric array")),
            }
        }

        _ => Err(anyhow!("unknown function: {}", func)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_expr() {
        let expr = Expr::constant(AlgorithmParamValue::Int(42));
        // Would need mock context to test eval
        assert!(matches!(expr, Expr::Const { .. }));
    }

    #[test]
    fn test_binary_op_add_ints() {
        let result = eval_binary_op(
            BinaryOp::Add,
            &AlgorithmParamValue::Int(10),
            &AlgorithmParamValue::Int(32),
        )
        .unwrap();
        assert_eq!(result, AlgorithmParamValue::Int(42));
    }

    #[test]
    fn test_binary_op_mul_floats() {
        let result = eval_binary_op(
            BinaryOp::Mul,
            &AlgorithmParamValue::Float(2.5),
            &AlgorithmParamValue::Float(4.0),
        )
        .unwrap();
        assert_eq!(result, AlgorithmParamValue::Float(10.0));
    }

    #[test]
    fn test_unary_op_neg() {
        let result = eval_unary_op(UnaryOp::Neg, &AlgorithmParamValue::Int(42)).unwrap();
        assert_eq!(result, AlgorithmParamValue::Int(-42));
    }

    #[test]
    fn test_unary_op_sqrt() {
        let result = eval_unary_op(UnaryOp::Sqrt, &AlgorithmParamValue::Float(16.0)).unwrap();
        assert_eq!(result, AlgorithmParamValue::Float(4.0));
    }
}

#[cfg(test)]
mod checked_tests {
    use super::*;
    use AlgorithmParamValue::{Float, Int};
    #[test]
    fn checked_integer_arithmetic_and_mixed_divisors() {
        for (op, left, right) in [
            (BinaryOp::Add, i64::MAX, 1),
            (BinaryOp::Sub, i64::MIN, 1),
            (BinaryOp::Mul, i64::MAX, 2),
            (BinaryOp::Mod, i64::MIN, -1),
            (BinaryOp::Pow, 2, 63),
        ] {
            assert!(eval_binary_op(op, &Int(left), &Int(right)).is_err());
        }
        for op in [UnaryOp::Neg, UnaryOp::Abs] {
            assert!(eval_unary_op(op, &Int(i64::MIN)).is_err());
        }
        for (left, right) in [
            (Int(1), Float(0.0)),
            (Float(1.0), Int(0)),
            (Float(1.0), Float(-0.0)),
        ] {
            assert!(eval_binary_op(BinaryOp::Div, &left, &right).is_err());
        }
        assert_eq!(
            eval_binary_op(BinaryOp::Div, &Float(1e-20), &Float(1e-20)).expect("tiny divisor"),
            Float(1.0)
        );
        assert_eq!(
            eval_binary_op(
                BinaryOp::Gt,
                &Int((1i64 << 53) + 1),
                &Float((1i64 << 53) as f64)
            )
            .expect("exact comparison"),
            AlgorithmParamValue::Bool(true)
        );
        assert!(eval_unary_op(UnaryOp::Sqrt, &Int(-1)).is_err());
    }
}

#[cfg(test)]
mod serialization_tests {
    use super::*;
    #[test]
    fn native_tagged_constants_round_trip_and_python_literals_keep_integer_precision() {
        let original = Expr::constant(AlgorithmParamValue::Int((1i64 << 53) + 1));
        let tagged = serde_json::to_value(&original).expect("encode");
        let roundtrip: Expr = serde_json::from_value(tagged).expect("native decode");
        assert!(
            matches!(roundtrip, Expr::Const { value: AlgorithmParamValue::Int(value) } if value == (1i64 << 53) + 1)
        );
        let literal: Expr = serde_json::from_str(r#"{"type":"const","value":9007199254740993}"#)
            .expect("literal decode");
        assert!(
            matches!(literal, Expr::Const { value: AlgorithmParamValue::Int(value) } if value == (1i64 << 53) + 1)
        );
        assert!(
            serde_json::from_str::<Expr>(r#"{"type":"const","value":18446744073709551615}"#)
                .is_err()
        );
    }
}
