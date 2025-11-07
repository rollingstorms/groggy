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
    Const { value: AlgorithmParamValue },

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

impl Expr {
    /// Evaluate the expression in the given context.
    pub fn eval(&self, ctx: &ExprContext) -> Result<AlgorithmParamValue> {
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
                    let var_map = ctx.input.variables.node_map(name)?;
                    var_map.get(&ctx.node).cloned().ok_or_else(|| {
                        anyhow!("variable '{}' not found for node {:?}", name, ctx.node)
                    })
                }
            }

            Expr::Attr { name } => ctx
                .input
                .subgraph
                .get_node_attribute(ctx.node, &AttrName::from(name.clone()))?
                .and_then(AlgorithmParamValue::from_attr_value)
                .ok_or_else(|| anyhow!("attribute '{}' not found for node {:?}", name, ctx.node)),

            Expr::BinaryOp { op, left, right } => {
                let left_val = left.eval(ctx)?;
                let right_val = right.eval(ctx)?;
                eval_binary_op(*op, &left_val, &right_val)
            }

            Expr::UnaryOp { op, arg } => {
                let arg_val = arg.eval(ctx)?;
                eval_unary_op(*op, &arg_val)
            }

            Expr::Call { func, args } => eval_function(func, args, ctx),

            Expr::If {
                condition,
                then_expr,
                else_expr,
            } => {
                let cond_val = condition.eval(ctx)?;
                let cond_bool = match cond_val {
                    AlgorithmParamValue::Bool(b) => b,
                    AlgorithmParamValue::Int(i) => i != 0,
                    AlgorithmParamValue::Float(f) => f != 0.0,
                    _ => return Err(anyhow!("condition must evaluate to boolean or number")),
                };

                if cond_bool {
                    then_expr.eval(ctx)
                } else {
                    else_expr.eval(ctx)
                }
            }
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

fn eval_binary_op(
    op: BinaryOp,
    left: &AlgorithmParamValue,
    right: &AlgorithmParamValue,
) -> Result<AlgorithmParamValue> {
    use AlgorithmParamValue::*;

    match (op, left, right) {
        // Arithmetic on numbers
        (BinaryOp::Add, Int(a), Int(b)) => Ok(Int(a + b)),
        (BinaryOp::Add, Float(a), Float(b)) => Ok(Float(a + b)),
        (BinaryOp::Add, Int(a), Float(b)) => Ok(Float(*a as f64 + b)),
        (BinaryOp::Add, Float(a), Int(b)) => Ok(Float(a + *b as f64)),

        (BinaryOp::Sub, Int(a), Int(b)) => Ok(Int(a - b)),
        (BinaryOp::Sub, Float(a), Float(b)) => Ok(Float(a - b)),
        (BinaryOp::Sub, Int(a), Float(b)) => Ok(Float(*a as f64 - b)),
        (BinaryOp::Sub, Float(a), Int(b)) => Ok(Float(a - *b as f64)),

        (BinaryOp::Mul, Int(a), Int(b)) => Ok(Int(a * b)),
        (BinaryOp::Mul, Float(a), Float(b)) => Ok(Float(a * b)),
        (BinaryOp::Mul, Int(a), Float(b)) => Ok(Float(*a as f64 * b)),
        (BinaryOp::Mul, Float(a), Int(b)) => Ok(Float(a * (*b as f64))),

        (BinaryOp::Div, Int(a), Int(b)) => {
            if *b == 0 {
                Err(anyhow!("division by zero"))
            } else {
                Ok(Float(*a as f64 / *b as f64))
            }
        }
        (BinaryOp::Div, Float(a), Float(b)) => {
            if b.abs() < f64::EPSILON {
                Err(anyhow!("division by zero"))
            } else {
                Ok(Float(a / b))
            }
        }
        (BinaryOp::Div, Int(a), Float(b)) => Ok(Float(*a as f64 / b)),
        (BinaryOp::Div, Float(a), Int(b)) => Ok(Float(a / *b as f64)),

        // String concatenation
        (BinaryOp::Add, Text(a), Text(b)) => Ok(Text(format!("{}{}", a, b))),

        // Comparisons
        (BinaryOp::Eq, a, b) => Ok(Bool(a == b)),
        (BinaryOp::Ne, a, b) => Ok(Bool(a != b)),

        (BinaryOp::Lt, Int(a), Int(b)) => Ok(Bool(a < b)),
        (BinaryOp::Lt, Float(a), Float(b)) => Ok(Bool(a < b)),
        (BinaryOp::Le, Int(a), Int(b)) => Ok(Bool(a <= b)),
        (BinaryOp::Le, Float(a), Float(b)) => Ok(Bool(a <= b)),
        (BinaryOp::Gt, Int(a), Int(b)) => Ok(Bool(a > b)),
        (BinaryOp::Gt, Float(a), Float(b)) => Ok(Bool(a > b)),
        (BinaryOp::Ge, Int(a), Int(b)) => Ok(Bool(a >= b)),
        (BinaryOp::Ge, Float(a), Float(b)) => Ok(Bool(a >= b)),

        // Logical
        (BinaryOp::And, Bool(a), Bool(b)) => Ok(Bool(*a && *b)),
        (BinaryOp::Or, Bool(a), Bool(b)) => Ok(Bool(*a || *b)),

        _ => Err(anyhow!(
            "unsupported operation: {:?} on {:?} and {:?}",
            op,
            left,
            right
        )),
    }
}

fn eval_unary_op(op: UnaryOp, arg: &AlgorithmParamValue) -> Result<AlgorithmParamValue> {
    use AlgorithmParamValue::*;

    match (op, arg) {
        (UnaryOp::Neg, Int(v)) => Ok(Int(-v)),
        (UnaryOp::Neg, Float(v)) => Ok(Float(-v)),

        (UnaryOp::Not, Bool(v)) => Ok(Bool(!v)),

        (UnaryOp::Abs, Int(v)) => Ok(Int(v.abs())),
        (UnaryOp::Abs, Float(v)) => Ok(Float(v.abs())),

        (UnaryOp::Sqrt, Int(v)) => Ok(Float((*v as f64).sqrt())),
        (UnaryOp::Sqrt, Float(v)) => Ok(Float(v.sqrt())),

        (UnaryOp::Log, Int(v)) => Ok(Float((*v as f64).ln())),
        (UnaryOp::Log, Float(v)) => Ok(Float(v.ln())),

        (UnaryOp::Exp, Int(v)) => Ok(Float((*v as f64).exp())),
        (UnaryOp::Exp, Float(v)) => Ok(Float(v.exp())),

        (UnaryOp::Floor, Float(v)) => Ok(Float(v.floor())),
        (UnaryOp::Ceil, Float(v)) => Ok(Float(v.ceil())),
        (UnaryOp::Round, Float(v)) => Ok(Float(v.round())),

        _ => Err(anyhow!("unsupported operation: {:?} on {:?}", op, arg)),
    }
}

fn eval_function(func: &str, args: &[Expr], ctx: &ExprContext) -> Result<AlgorithmParamValue> {
    match func {
        "min" => {
            if args.len() != 2 {
                return Err(anyhow!("min requires 2 arguments"));
            }
            let a = args[0].eval(ctx)?;
            let b = args[1].eval(ctx)?;

            match (&a, &b) {
                (AlgorithmParamValue::Int(x), AlgorithmParamValue::Int(y)) => {
                    Ok(AlgorithmParamValue::Int(*x.min(y)))
                }
                (AlgorithmParamValue::Float(x), AlgorithmParamValue::Float(y)) => {
                    Ok(AlgorithmParamValue::Float(x.min(*y)))
                }
                _ => Err(anyhow!("min requires numeric arguments")),
            }
        }

        "max" => {
            if args.len() != 2 {
                return Err(anyhow!("max requires 2 arguments"));
            }
            let a = args[0].eval(ctx)?;
            let b = args[1].eval(ctx)?;

            match (&a, &b) {
                (AlgorithmParamValue::Int(x), AlgorithmParamValue::Int(y)) => {
                    Ok(AlgorithmParamValue::Int(*x.max(y)))
                }
                (AlgorithmParamValue::Float(x), AlgorithmParamValue::Float(y)) => {
                    Ok(AlgorithmParamValue::Float(x.max(*y)))
                }
                _ => Err(anyhow!("max requires numeric arguments")),
            }
        }

        "neighbor_count" => {
            if !args.is_empty() {
                return Err(anyhow!("neighbor_count takes no arguments"));
            }
            let degree = ctx.input.subgraph.degree(ctx.node)?;
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
            let var_map = ctx.input.variables.node_map(name)?;

            // Get neighbors
            let neighbors = ctx.input.subgraph.neighbors(ctx.node)?;

            // Collect neighbor values, preserving integer-only collections when possible.
            let mut int_values = Vec::new();
            let mut float_values = Vec::new();
            let mut use_float = false;

            // Collect neighbor values (excluding self; use core.collect_neighbor_values with include_self=True for that)
            let mut seen_neighbors = HashSet::new();
            for neighbor in neighbors {
                if !ctx.input.subgraph.has_edge_between(ctx.node, neighbor)? {
                    continue;
                }
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
                        _ => {}
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

        "sum" => {
            // Sum an array of values
            if args.len() != 1 {
                return Err(anyhow!("sum requires 1 argument"));
            }

            let values = args[0].eval(ctx)?;

            match values {
                AlgorithmParamValue::FloatList(vec) => {
                    let sum: f64 = vec.iter().sum();
                    Ok(AlgorithmParamValue::Float(sum))
                }
                AlgorithmParamValue::IntList(vec) => {
                    let sum: i64 = vec.iter().sum();
                    Ok(AlgorithmParamValue::Int(sum))
                }
                AlgorithmParamValue::Float(f) => Ok(AlgorithmParamValue::Float(f)),
                AlgorithmParamValue::Int(i) => Ok(AlgorithmParamValue::Int(i)),
                _ => Err(anyhow!("sum requires numeric array or value")),
            }
        }

        "mean" => {
            // Calculate mean of an array
            if args.len() != 1 {
                return Err(anyhow!("mean requires 1 argument"));
            }

            let values = args[0].eval(ctx)?;

            match values {
                AlgorithmParamValue::FloatList(vec) => {
                    if vec.is_empty() {
                        return Ok(AlgorithmParamValue::Float(0.0));
                    }
                    let sum: f64 = vec.iter().sum();
                    let mean = sum / vec.len() as f64;
                    Ok(AlgorithmParamValue::Float(mean))
                }
                AlgorithmParamValue::IntList(vec) => {
                    if vec.is_empty() {
                        return Ok(AlgorithmParamValue::Float(0.0));
                    }
                    let sum: i64 = vec.iter().sum();
                    let mean = sum as f64 / vec.len() as f64;
                    Ok(AlgorithmParamValue::Float(mean))
                }
                _ => Err(anyhow!("mean requires numeric array")),
            }
        }

        "mode" => {
            // Find most common value in an array
            if args.len() != 1 {
                return Err(anyhow!("mode requires 1 argument"));
            }

            let values = args[0].eval(ctx)?;

            match values {
                AlgorithmParamValue::FloatList(vec) => {
                    if vec.is_empty() {
                        return Ok(AlgorithmParamValue::Float(0.0));
                    }

                    // Count occurrences - use integer representation for HashMap key
                    let mut counts: HashMap<i64, usize> = HashMap::new();
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
                        if first || count > best_count || (count == best_count && key < best_key) {
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
                    use std::collections::HashMap;
                    let mut counts: HashMap<i64, usize> = HashMap::new();
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
