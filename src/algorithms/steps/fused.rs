//! Fused operation step primitives for performance optimization.
//!
//! These steps combine multiple operations into a single pass to eliminate
//! intermediate allocations and improve cache locality. Primary use case is
//! optimizing iterative graph algorithms like PageRank and LPA.

use super::read_scope::{sealed, ReadStep, ReadStepScope};
use anyhow::{anyhow, Result};
use std::collections::HashMap;

use super::super::{Context, CostHint};
use super::core::{AtomicStepEffect, Step, StepMetadata, StepScope, StepSpec};
use super::direction::NeighborDirection;

/// Fused neighbor aggregation with element-wise multiplication.
///
/// Replaces the pattern:
///   temp = values * scalars
///   result = neighbor_agg(temp)
///
/// With a single pass:
///   result[node] = sum(values[neighbor] * scalars[neighbor] for neighbor in neighbors(node))
///
/// This eliminates one intermediate allocation and combines two full node iterations.
pub struct FusedNeighborMulAgg {
    values: String,
    scalars: String,
    target: String,
    direction: NeighborDirection,
}

impl FusedNeighborMulAgg {
    pub fn new(
        values: impl Into<String>,
        scalars: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            values: values.into(),
            scalars: scalars.into(),
            target: target.into(),
            direction: NeighborDirection::default(),
        }
    }

    pub fn with_direction(mut self, direction: NeighborDirection) -> Self {
        self.direction = direction;
        self
    }

    pub fn from_spec(spec: &StepSpec) -> Result<Self> {
        let values = spec
            .params
            .get_text("values")
            .ok_or_else(|| anyhow!("FusedNeighborMulAgg requires 'values' parameter"))?
            .to_string();
        let scalars = spec
            .params
            .get_text("scalars")
            .ok_or_else(|| anyhow!("FusedNeighborMulAgg requires 'scalars' parameter"))?
            .to_string();
        let target = spec
            .params
            .get_text("target")
            .ok_or_else(|| anyhow!("FusedNeighborMulAgg requires 'target' parameter"))?
            .to_string();

        // Parse optional direction parameter
        let direction = spec
            .params
            .get_text("direction")
            .map(|value| {
                NeighborDirection::from_str(value)
                    .ok_or_else(|| anyhow!("Invalid fused neighbor direction: {}", value))
            })
            .transpose()?
            .unwrap_or_default();

        Ok(Self {
            values,
            scalars,
            target,
            direction,
        })
    }
}

impl Step for FusedNeighborMulAgg {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        // Inputs and outputs are run variables; graph access is topology-only.
        true
    }

    fn id(&self) -> &'static str {
        "graph.fused_neighbor_mul_agg"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Fused neighbor aggregation with element-wise multiplication".to_string(),
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

impl sealed::Sealed for FusedNeighborMulAgg {}
impl ReadStep for FusedNeighborMulAgg {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("FusedNeighborMulAgg cancelled"));
        }

        let selected = scope.selected_csr(ctx, self.direction)?;
        let nodes = &selected.nodes;
        let csr = &selected.csr;

        let values_map = scope.variables().node_map(&self.values)?;
        let scalars_map = scope.variables().node_map(&self.scalars)?;

        // Convert to f64 arrays for fast indexing
        let mut values_arr = vec![0.0; nodes.len()];
        let mut scalars_arr = vec![0.0; nodes.len()];

        for (idx, &node) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("Fused operation cancelled"));
            }
            if let Some(v) = values_map.get(&node) {
                values_arr[idx] = match v {
                    crate::algorithms::AlgorithmParamValue::Float(f) => *f,
                    crate::algorithms::AlgorithmParamValue::Int(i) => *i as f64,
                    _ => return Err(anyhow!("values must be numeric")),
                };
            }
            if let Some(s) = scalars_map.get(&node) {
                scalars_arr[idx] = match s {
                    crate::algorithms::AlgorithmParamValue::Float(f) => *f,
                    crate::algorithms::AlgorithmParamValue::Int(i) => *i as f64,
                    _ => return Err(anyhow!("scalars must be numeric")),
                };
            }
        }

        if values_arr
            .iter()
            .chain(&scalars_arr)
            .any(|v| !v.is_finite())
        {
            return Err(anyhow!("Fused operation requires finite numeric inputs"));
        }
        // Allocate result map
        let mut result = HashMap::with_capacity(nodes.len());

        // Fused kernel: aggregate neighbors with multiplication in single pass
        for (idx, &node) in nodes.iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(anyhow!("Fused operation cancelled"));
            }
            let mut sum = 0.0;

            // Iterate over CSR neighbors and accumulate weighted sum
            for &neighbor_idx in csr.neighbors(idx) {
                if ctx.is_cancelled() {
                    return Err(anyhow!("Fused operation cancelled"));
                }
                // Fused: multiply neighbor's value by its scalar, then accumulate
                sum += values_arr[neighbor_idx] * scalars_arr[neighbor_idx];
            }

            if !sum.is_finite() {
                return Err(anyhow!("Fused neighbor result is not finite"));
            }
            result.insert(node, crate::algorithms::AlgorithmParamValue::Float(sum));
        }

        // Store result
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);

        Ok(())
    }
}

/// Fused AXPY operation: result = a * x + b * y
///
/// Replaces the pattern:
///   temp1 = a * x
///   temp2 = b * y
///   result = temp1 + temp2
///
/// With a single pass that eliminates two intermediate allocations.
pub struct FusedAXPY {
    a: String,
    x: String,
    b: String,
    y: String,
    target: String,
}

impl FusedAXPY {
    pub fn new(
        a: impl Into<String>,
        x: impl Into<String>,
        b: impl Into<String>,
        y: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            a: a.into(),
            x: x.into(),
            b: b.into(),
            y: y.into(),
            target: target.into(),
        }
    }

    pub fn from_spec(spec: &StepSpec) -> Result<Self> {
        let a = spec
            .params
            .get_text("a")
            .ok_or_else(|| anyhow!("FusedAXPY requires 'a' parameter"))?
            .to_string();
        let x = spec
            .params
            .get_text("x")
            .ok_or_else(|| anyhow!("FusedAXPY requires 'x' parameter"))?
            .to_string();
        let b = spec
            .params
            .get_text("b")
            .ok_or_else(|| anyhow!("FusedAXPY requires 'b' parameter"))?
            .to_string();
        let y = spec
            .params
            .get_text("y")
            .ok_or_else(|| anyhow!("FusedAXPY requires 'y' parameter"))?
            .to_string();
        let target = spec
            .params
            .get_text("target")
            .ok_or_else(|| anyhow!("FusedAXPY requires 'target' parameter"))?
            .to_string();

        Ok(Self::new(a, x, b, y, target))
    }
}

impl Step for FusedAXPY {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        // Inputs and outputs are run variables; graph access is topology-only.
        true
    }

    fn id(&self) -> &'static str {
        "core.fused_axpy"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Fused AXPY: result = a * x + b * y".to_string(),
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

impl sealed::Sealed for FusedAXPY {}
impl ReadStep for FusedAXPY {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("FusedAXPY cancelled"));
        }

        scope.validate_selected_read()?;
        let nodes = scope.ordered_nodes();

        // Try to get each variable as node map or scalar
        let a_map = scope.variables().node_map(&self.a).ok();
        let a_scalar = if a_map.is_none() {
            scope
                .variables()
                .scalar(&self.a)
                .ok()
                .and_then(|v| match v {
                    crate::algorithms::AlgorithmParamValue::Float(f) => Some(*f),
                    crate::algorithms::AlgorithmParamValue::Int(i) => Some(*i as f64),
                    _ => None,
                })
        } else {
            None
        };

        let x_map = scope.variables().node_map(&self.x)?;

        let b_map = scope.variables().node_map(&self.b).ok();
        let b_scalar = if b_map.is_none() {
            scope
                .variables()
                .scalar(&self.b)
                .ok()
                .and_then(|v| match v {
                    crate::algorithms::AlgorithmParamValue::Float(f) => Some(*f),
                    crate::algorithms::AlgorithmParamValue::Int(i) => Some(*i as f64),
                    _ => None,
                })
        } else {
            None
        };

        let y_map = scope.variables().node_map(&self.y)?;

        let mut result = std::collections::HashMap::with_capacity(nodes.len());

        for &node in nodes.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("Fused operation cancelled"));
            }
            let x_val = match x_map.get(&node) {
                Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                _ => return Err(anyhow!("x must be numeric for all nodes")),
            };
            let y_val = match y_map.get(&node) {
                Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                _ => return Err(anyhow!("y must be numeric for all nodes")),
            };

            let a_val = if let Some(map) = a_map {
                match map.get(&node) {
                    Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                    Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                    _ => return Err(anyhow!("a must be numeric for all nodes")),
                }
            } else if let Some(scalar) = a_scalar {
                scalar
            } else {
                return Err(anyhow!("a must be either a node map or scalar"));
            };

            let b_val = if let Some(map) = b_map {
                match map.get(&node) {
                    Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                    Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                    _ => return Err(anyhow!("b must be numeric for all nodes")),
                }
            } else if let Some(scalar) = b_scalar {
                scalar
            } else {
                return Err(anyhow!("b must be either a node map or scalar"));
            };

            // Fused computation: a * x + b * y
            if [a_val, x_val, b_val, y_val].iter().any(|v| !v.is_finite()) {
                return Err(anyhow!("Fused operation requires finite numeric inputs"));
            }
            let value = a_val * x_val + b_val * y_val;
            if !value.is_finite() {
                return Err(anyhow!("Fused AXPY result is not finite"));
            }
            result.insert(node, crate::algorithms::AlgorithmParamValue::Float(value));
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);

        Ok(())
    }
}

/// Fused multiply-add: result = a * b + c
///
/// Common pattern in many graph algorithms. Eliminates one intermediate allocation.
pub struct FusedMADD {
    a: String,
    b: String,
    c: String,
    target: String,
}

impl FusedMADD {
    pub fn new(
        a: impl Into<String>,
        b: impl Into<String>,
        c: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            a: a.into(),
            b: b.into(),
            c: c.into(),
            target: target.into(),
        }
    }

    pub fn from_spec(spec: &StepSpec) -> Result<Self> {
        let a = spec
            .params
            .get_text("a")
            .ok_or_else(|| anyhow!("FusedMADD requires 'a' parameter"))?
            .to_string();
        let b = spec
            .params
            .get_text("b")
            .ok_or_else(|| anyhow!("FusedMADD requires 'b' parameter"))?
            .to_string();
        let c = spec
            .params
            .get_text("c")
            .ok_or_else(|| anyhow!("FusedMADD requires 'c' parameter"))?
            .to_string();
        let target = spec
            .params
            .get_text("target")
            .ok_or_else(|| anyhow!("FusedMADD requires 'target' parameter"))?
            .to_string();

        Ok(Self::new(a, b, c, target))
    }
}

impl Step for FusedMADD {
    fn atomic_effect(&self) -> AtomicStepEffect {
        AtomicStepEffect::PendingProperties
    }

    fn supports_atomic_properties(&self) -> bool {
        // Inputs and outputs are run variables; graph access is topology-only.
        true
    }

    fn id(&self) -> &'static str {
        "core.fused_madd"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: "Fused multiply-add: result = a * b + c".to_string(),
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

impl sealed::Sealed for FusedMADD {}
impl ReadStep for FusedMADD {
    fn apply_read(&self, ctx: &mut Context, scope: &mut ReadStepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("FusedMADD cancelled"));
        }

        scope.validate_selected_read()?;
        let nodes = scope.ordered_nodes();

        let a_map = scope.variables().node_map(&self.a)?;
        let b_map = scope.variables().node_map(&self.b)?;
        let c_map = scope.variables().node_map(&self.c)?;

        let mut result = std::collections::HashMap::with_capacity(nodes.len());

        for &node in nodes.iter() {
            if ctx.is_cancelled() {
                return Err(anyhow!("Fused operation cancelled"));
            }
            let a_val = match a_map.get(&node) {
                Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                _ => return Err(anyhow!("a must be numeric for all nodes")),
            };
            let b_val = match b_map.get(&node) {
                Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                _ => return Err(anyhow!("b must be numeric for all nodes")),
            };
            let c_val = match c_map.get(&node) {
                Some(crate::algorithms::AlgorithmParamValue::Float(f)) => *f,
                Some(crate::algorithms::AlgorithmParamValue::Int(i)) => *i as f64,
                _ => return Err(anyhow!("c must be numeric for all nodes")),
            };

            // Fused computation: a * b + c
            if [a_val, b_val, c_val].iter().any(|v| !v.is_finite()) {
                return Err(anyhow!("Fused operation requires finite numeric inputs"));
            }
            let value = a_val * b_val + c_val;
            if !value.is_finite() {
                return Err(anyhow!("Fused MADD result is not finite"));
            }
            result.insert(node, crate::algorithms::AlgorithmParamValue::Float(value));
        }

        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::steps::{StepScope, StepVariables};
    use crate::algorithms::{AlgorithmParamValue, Context};
    use crate::api::graph::Graph;
    use crate::ownership::OwnerCell;
    use crate::subgraphs::Subgraph;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn test_fused_neighbor_mul_agg() {
        // Create simple test graph: 0 -> 1 -> 2
        let mut g = Graph::new_manual();
        let n0 = g.add_node().unwrap();
        let n1 = g.add_node().unwrap();
        let n2 = g.add_node().unwrap();
        g.add_edge(n0, n1).unwrap();
        g.add_edge(n1, n2).unwrap();

        let nodes: HashSet<_> = [n0, n1, n2].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Arc::new(OwnerCell::new(g)), nodes, "test".to_string()).unwrap();

        let mut ctx = Context::default();
        let mut vars = StepVariables::default();

        // Set up test data
        let mut values_map = HashMap::new();
        values_map.insert(n0, AlgorithmParamValue::Float(1.0));
        values_map.insert(n1, AlgorithmParamValue::Float(2.0));
        values_map.insert(n2, AlgorithmParamValue::Float(3.0));
        vars.set_node_map("values", values_map);

        let mut scalars_map = HashMap::new();
        scalars_map.insert(n0, AlgorithmParamValue::Float(0.5));
        scalars_map.insert(n1, AlgorithmParamValue::Float(1.0));
        scalars_map.insert(n2, AlgorithmParamValue::Float(1.5));
        vars.set_node_map("scalars", scalars_map);

        let mut scope = StepScope::new(&subgraph, &mut vars);

        // Execute fused operation
        let step = FusedNeighborMulAgg::new("values", "scalars", "result");
        step.apply(&mut ctx, &mut scope).unwrap();

        // Verify result
        let result = scope.variables().node_map("result").unwrap();
        // Node 0: no incoming edges -> 0.0
        // Node 1: neighbor 0, value=1.0, scalar=0.5 -> 1.0 * 0.5 = 0.5
        // Node 2: neighbor 1, value=2.0, scalar=1.0 -> 2.0 * 1.0 = 2.0
        assert_eq!(
            match result.get(&n0).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            0.0
        );
        assert_eq!(
            match result.get(&n1).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            0.5
        );
        assert_eq!(
            match result.get(&n2).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            2.0
        );
    }

    #[test]
    fn test_fused_axpy_vectors() {
        let mut g = Graph::new_manual();
        let n0 = g.add_node().unwrap();
        let n1 = g.add_node().unwrap();

        let nodes: HashSet<_> = [n0, n1].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Arc::new(OwnerCell::new(g)), nodes, "test".to_string()).unwrap();

        let mut ctx = Context::default();
        let mut vars = StepVariables::default();

        let mut a_map = HashMap::new();
        a_map.insert(n0, AlgorithmParamValue::Float(2.0));
        a_map.insert(n1, AlgorithmParamValue::Float(3.0));
        vars.set_node_map("a", a_map);

        let mut x_map = HashMap::new();
        x_map.insert(n0, AlgorithmParamValue::Float(1.0));
        x_map.insert(n1, AlgorithmParamValue::Float(2.0));
        vars.set_node_map("x", x_map);

        let mut b_map = HashMap::new();
        b_map.insert(n0, AlgorithmParamValue::Float(0.5));
        b_map.insert(n1, AlgorithmParamValue::Float(1.0));
        vars.set_node_map("b", b_map);

        let mut y_map = HashMap::new();
        y_map.insert(n0, AlgorithmParamValue::Float(4.0));
        y_map.insert(n1, AlgorithmParamValue::Float(6.0));
        vars.set_node_map("y", y_map);

        let mut scope = StepScope::new(&subgraph, &mut vars);

        let step = FusedAXPY::new("a", "x", "b", "y", "result");
        step.apply(&mut ctx, &mut scope).unwrap();

        let result = scope.variables().node_map("result").unwrap();
        // result[0] = 2.0 * 1.0 + 0.5 * 4.0 = 2.0 + 2.0 = 4.0
        // result[1] = 3.0 * 2.0 + 1.0 * 6.0 = 6.0 + 6.0 = 12.0
        assert_eq!(
            match result.get(&n0).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            4.0
        );
        assert_eq!(
            match result.get(&n1).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            12.0
        );
    }

    #[test]
    fn test_fused_madd() {
        let mut g = Graph::new_manual();
        let n0 = g.add_node().unwrap();
        let n1 = g.add_node().unwrap();

        let nodes: HashSet<_> = [n0, n1].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Arc::new(OwnerCell::new(g)), nodes, "test".to_string()).unwrap();

        let mut ctx = Context::default();
        let mut vars = StepVariables::default();

        let mut a_map = HashMap::new();
        a_map.insert(n0, AlgorithmParamValue::Float(2.0));
        a_map.insert(n1, AlgorithmParamValue::Float(3.0));
        vars.set_node_map("a", a_map);

        let mut b_map = HashMap::new();
        b_map.insert(n0, AlgorithmParamValue::Float(4.0));
        b_map.insert(n1, AlgorithmParamValue::Float(5.0));
        vars.set_node_map("b", b_map);

        let mut c_map = HashMap::new();
        c_map.insert(n0, AlgorithmParamValue::Float(1.0));
        c_map.insert(n1, AlgorithmParamValue::Float(2.0));
        vars.set_node_map("c", c_map);

        let mut scope = StepScope::new(&subgraph, &mut vars);

        let step = FusedMADD::new("a", "b", "c", "result");
        step.apply(&mut ctx, &mut scope).unwrap();

        let result = scope.variables().node_map("result").unwrap();
        // result[0] = 2.0 * 4.0 + 1.0 = 9.0
        // result[1] = 3.0 * 5.0 + 2.0 = 17.0
        assert_eq!(
            match result.get(&n0).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            9.0
        );
        assert_eq!(
            match result.get(&n1).unwrap() {
                AlgorithmParamValue::Float(f) => *f,
                _ => panic!("expected float"),
            },
            17.0
        );
    }
}
