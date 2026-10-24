//! Filtering and ordering step primitives.
//!
//! This module provides steps for:
//! - Sorting nodes/edges by attribute values
//! - Filtering based on predicates
//! - Selecting top-k elements

use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::storage::numeric::{scalar::compare_inputs, NumericInput};
use crate::types::{AttrName, NodeId};

use super::super::{AlgorithmParamValue, Context, CostHint};
use super::core::{Step, StepMetadata, StepScope};

/// Predicate for filtering operations.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Predicate {
    /// Equal to value
    Eq { value: AlgorithmParamValue },
    /// Not equal to value
    Ne { value: AlgorithmParamValue },
    /// Less than value
    Lt { value: AlgorithmParamValue },
    /// Less than or equal to value
    Le { value: AlgorithmParamValue },
    /// Greater than value
    Gt { value: AlgorithmParamValue },
    /// Greater than or equal to value
    Ge { value: AlgorithmParamValue },
    /// String contains substring
    Contains { substring: String },
    /// Within numeric range [min, max]
    Range { min: f64, max: f64 },
}

impl Predicate {
    pub(crate) fn validate(&self) -> Result<()> {
        if let Self::Range { min, max } = self {
            if min.is_nan() || max.is_nan() || min > max {
                return Err(anyhow!("Predicate range requires ordered, non-NaN bounds"));
            }
        }
        Ok(())
    }

    pub fn eval(&self, value: &AlgorithmParamValue) -> bool {
        match self {
            Predicate::Eq { value: target } => value == target,
            Predicate::Ne { value: target } => value != target,

            Predicate::Lt { value: target } => {
                compare_values(value, target) == Some(Ordering::Less)
            }
            Predicate::Le { value: target } => {
                matches!(
                    compare_values(value, target),
                    Some(Ordering::Less | Ordering::Equal)
                )
            }
            Predicate::Gt { value: target } => {
                compare_values(value, target) == Some(Ordering::Greater)
            }
            Predicate::Ge { value: target } => {
                matches!(
                    compare_values(value, target),
                    Some(Ordering::Greater | Ordering::Equal)
                )
            }

            Predicate::Contains { substring } => {
                if let AlgorithmParamValue::Text(s) = value {
                    s.contains(substring)
                } else {
                    false
                }
            }

            Predicate::Range { min, max } => {
                let lower = compare_values(value, &AlgorithmParamValue::Float(*min));
                let upper = compare_values(value, &AlgorithmParamValue::Float(*max));
                matches!(lower, Some(Ordering::Greater | Ordering::Equal))
                    && matches!(upper, Some(Ordering::Less | Ordering::Equal))
            }
        }
    }
}

pub(super) fn compare_values(a: &AlgorithmParamValue, b: &AlgorithmParamValue) -> Option<Ordering> {
    use AlgorithmParamValue as APV;

    match (a, b) {
        (APV::Int(x), APV::Int(y)) => Some(x.cmp(y)),
        (APV::Float(x), APV::Float(y)) => x.partial_cmp(y),
        (APV::Int(x), APV::Float(y)) => compare_inputs(
            NumericInput::Integer(i128::from(*x)),
            NumericInput::Float(*y),
        ),
        (APV::Float(x), APV::Int(y)) => compare_inputs(
            NumericInput::Float(*x),
            NumericInput::Integer(i128::from(*y)),
        ),
        (APV::Text(x), APV::Text(y)) => Some(x.cmp(y)),
        (APV::Bool(x), APV::Bool(y)) => Some(x.cmp(y)),
        _ => Option::None,
    }
}

/// Validate one comparable scalar domain before sorting, then break ties by ID.
/// Missing/null values sort last in either direction; NaNs and mixed domains fail.
fn sort_checked(
    ctx: &Context,
    values: &mut [(NodeId, Option<AlgorithmParamValue>)],
    order: SortOrder,
) -> Result<()> {
    let mut domain = None;
    for (_, value) in values.iter() {
        if ctx.is_cancelled() {
            return Err(anyhow!("Ordering cancelled"));
        }
        let kind = match value {
            None | Some(AlgorithmParamValue::None) => continue,
            Some(AlgorithmParamValue::Int(_)) => 0,
            Some(AlgorithmParamValue::Float(value)) if !value.is_nan() => 0,
            Some(AlgorithmParamValue::Text(_)) => 1,
            Some(AlgorithmParamValue::Bool(_)) => 2,
            _ => return Err(anyhow!("Ordering requires comparable scalars without NaN")),
        };
        if domain.is_some_and(|old| old != kind) {
            return Err(anyhow!(
                "Ordering cannot mix numeric, text or Boolean domains"
            ));
        }
        domain = Some(kind);
    }
    values.sort_by(|(left_id, left), (right_id, right)| {
        let present = |value: &Option<AlgorithmParamValue>| {
            !matches!(value, None | Some(AlgorithmParamValue::None))
        };
        let compared = match (present(left), present(right)) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => Ordering::Equal,
            (true, true) => {
                // Domain/NaN validation above guarantees comparability.
                let compared = left
                    .as_ref()
                    .zip(right.as_ref())
                    .and_then(|(left, right)| compare_values(left, right))
                    .unwrap_or(Ordering::Equal);
                match order {
                    SortOrder::Ascending => compared,
                    SortOrder::Descending => compared.reverse(),
                }
            }
        };
        compared.then_with(|| left_id.cmp(right_id))
    });
    if ctx.is_cancelled() {
        return Err(anyhow!("Ordering cancelled"));
    }
    Ok(())
}

/// Sort order for ordering operations.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Sort nodes by attribute value.
pub struct SortNodesByAttrStep {
    attr: String,
    order: SortOrder,
    target: String,
}

impl SortNodesByAttrStep {
    pub fn new(attr: impl Into<String>, order: SortOrder, target: impl Into<String>) -> Self {
        Self {
            attr: attr.into(),
            order,
            target: target.into(),
        }
    }
}

impl Step for SortNodesByAttrStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.sort_nodes_by_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Sort nodes by attribute '{}' ({:?})", self.attr, self.order),
            cost_hint: CostHint::Linearithmic,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("sort_nodes_by_attr cancelled"));
        }

        let attr_name = AttrName::from(self.attr.clone());
        let mut node_values = Vec::with_capacity(scope.subgraph().node_count());
        for &node in scope.node_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("sort_nodes_by_attr cancelled"));
            }
            let value = match scope.node_attribute(ctx, node, &attr_name)? {
                None => None,
                Some(value) => {
                    Some(AlgorithmParamValue::from_attr_value(value).ok_or_else(|| {
                        anyhow!("Node {} property '{}' cannot be ordered", node, attr_name)
                    })?)
                }
            };
            node_values.push((node, value));
        }
        sort_checked(ctx, &mut node_values, self.order)?;

        // Create result map with sort index
        let mut result = HashMap::with_capacity(node_values.len());
        for (index, (node, _)) in node_values.iter().enumerate() {
            result.insert(
                *node,
                AlgorithmParamValue::Int(
                    i64::try_from(index).map_err(|_| anyhow!("Sort rank exceeds i64"))?,
                ),
            );
        }

        if ctx.is_cancelled() {
            return Err(anyhow!(
                "Filtering/ordering cancelled before installing results"
            ));
        }
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);

        Ok(())
    }
}

/// Filter nodes by attribute predicate.
pub struct FilterNodesByAttrStep {
    attr: String,
    predicate: Predicate,
    target: String,
}

impl FilterNodesByAttrStep {
    pub fn new(attr: impl Into<String>, predicate: Predicate, target: impl Into<String>) -> Self {
        Self {
            attr: attr.into(),
            predicate,
            target: target.into(),
        }
    }
}

impl Step for FilterNodesByAttrStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.filter_nodes_by_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Filter nodes by attribute '{}' with predicate", self.attr),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("filter_nodes_by_attr cancelled"));
        }

        let attr_name = AttrName::from(self.attr.clone());
        let mut result = HashMap::new();

        self.predicate.validate()?;
        for &node in scope.node_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("filter_nodes_by_attr cancelled"));
            }
            if let Some(attr_value) = scope.node_attribute(ctx, node, &attr_name)? {
                if let Some(param_value) = AlgorithmParamValue::from_attr_value(attr_value) {
                    if self.predicate.eval(&param_value) {
                        result.insert(node, AlgorithmParamValue::Int(1)); // Mark as passing
                    }
                }
            }
        }

        if ctx.is_cancelled() {
            return Err(anyhow!(
                "Filtering/ordering cancelled before installing results"
            ));
        }
        scope
            .variables_mut()
            .set_node_map(self.target.clone(), result);

        Ok(())
    }
}

/// Filter edges by attribute predicate.
pub struct FilterEdgesByAttrStep {
    attr: String,
    predicate: Predicate,
    target: String,
}

impl FilterEdgesByAttrStep {
    pub fn new(attr: impl Into<String>, predicate: Predicate, target: impl Into<String>) -> Self {
        Self {
            attr: attr.into(),
            predicate,
            target: target.into(),
        }
    }
}

impl Step for FilterEdgesByAttrStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.filter_edges_by_attr"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Filter edges by attribute '{}' with predicate", self.attr),
            cost_hint: CostHint::Linear,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("filter_edges_by_attr cancelled"));
        }

        let attr_name = AttrName::from(self.attr.clone());
        let mut result = HashMap::new();

        self.predicate.validate()?;
        for &edge in scope.edge_ids() {
            if ctx.is_cancelled() {
                return Err(anyhow!("filter_edges_by_attr cancelled"));
            }
            if let Some(attr_value) = scope.edge_attribute(ctx, edge, &attr_name)? {
                if let Some(param_value) = AlgorithmParamValue::from_attr_value(attr_value) {
                    if self.predicate.eval(&param_value) {
                        result.insert(edge, AlgorithmParamValue::Int(1)); // Mark as passing
                    }
                }
            }
        }

        if ctx.is_cancelled() {
            return Err(anyhow!(
                "Filtering/ordering cancelled before installing results"
            ));
        }
        scope
            .variables_mut()
            .set_edge_map(self.target.clone(), result);

        Ok(())
    }
}

/// Select top-k elements from a node map by value.
pub struct TopKStep {
    source: String,
    k: usize,
    target: String,
    order: SortOrder,
}

impl TopKStep {
    pub fn new(
        source: impl Into<String>,
        k: usize,
        target: impl Into<String>,
        order: SortOrder,
    ) -> Self {
        Self {
            source: source.into(),
            k,
            target: target.into(),
            order,
        }
    }
}

impl Step for TopKStep {
    fn supports_atomic_properties(&self) -> bool {
        true
    }

    fn id(&self) -> &'static str {
        "core.top_k"
    }

    fn metadata(&self) -> StepMetadata {
        StepMetadata {
            id: self.id().to_string(),
            description: format!("Select top-{} values ({:?})", self.k, self.order),
            cost_hint: CostHint::Linearithmic,
        }
    }

    fn apply(&self, ctx: &mut Context, scope: &mut StepScope<'_>) -> Result<()> {
        if ctx.is_cancelled() {
            return Err(anyhow!("top_k cancelled"));
        }

        let source_map = scope.variables().node_map(&self.source)?;

        let mut items = Vec::with_capacity(source_map.len());
        {
            let graph = scope.subgraph().graph();
            let graph = graph.borrow();
            for (&node, value) in source_map {
                if !scope.subgraph().has_node(node) || !graph.has_node(node) {
                    return Err(anyhow!(
                        "top_k source node {} is outside active selection",
                        node
                    ));
                }
                items.push((node, Some(value.clone())));
            }
        }
        sort_checked(ctx, &mut items, self.order)?;

        // Take top k
        let mut result = HashMap::new();
        for (rank, (node, _value)) in items.into_iter().take(self.k).enumerate() {
            result.insert(
                node,
                AlgorithmParamValue::Int(
                    i64::try_from(rank).map_err(|_| anyhow!("Top-k rank exceeds i64"))?,
                ),
            );
            // Could also preserve original value instead of rank
        }

        if ctx.is_cancelled() {
            return Err(anyhow!(
                "Filtering/ordering cancelled before installing results"
            ));
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

    #[test]
    fn exact_numeric_sort_and_ties_do_not_depend_on_input_order() {
        let base = 1i64 << 53;
        for reverse in [false, true] {
            let mut values = vec![
                (0, Some(AlgorithmParamValue::Int(base + 1))),
                (1, Some(AlgorithmParamValue::Float(base as f64))),
                (2, Some(AlgorithmParamValue::Int(base + 2))),
                (3, Some(AlgorithmParamValue::Float(base as f64))),
            ];
            if reverse {
                values.reverse();
            }
            sort_checked(&Context::new(), &mut values, SortOrder::Ascending).expect("sort");
            assert_eq!(
                values.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                [1, 3, 0, 2]
            );
        }
        assert!(!Predicate::Range {
            min: base as f64,
            max: base as f64
        }
        .eval(&AlgorithmParamValue::Int(base + 1)));
        assert!(!Predicate::Eq {
            value: AlgorithmParamValue::Float(1.0)
        }
        .eval(&AlgorithmParamValue::Int(1)));
    }

    #[test]
    fn invalid_sort_values_and_ranges_return_errors() {
        for values in [
            vec![(0, Some(AlgorithmParamValue::Float(f64::NAN)))],
            vec![
                (0, Some(AlgorithmParamValue::Int(1))),
                (1, Some(AlgorithmParamValue::Text("x".into()))),
            ],
            vec![(0, Some(AlgorithmParamValue::IntList(vec![1])))],
        ] {
            let mut values = values;
            assert!(sort_checked(&Context::new(), &mut values, SortOrder::Ascending).is_err());
        }
        for (min, max) in [(2.0, 1.0), (f64::NAN, 1.0)] {
            assert!(Predicate::Range { min, max }.validate().is_err());
        }
    }

    #[test]
    fn stale_property_reads_propagate_without_replacing_result() {
        use crate::{algorithms::steps::StepVariables, api::graph::Graph, subgraphs::Subgraph};
        use std::{cell::RefCell, rc::Rc};
        let mut graph = Graph::new();
        graph.add_node();
        let shared = Rc::new(RefCell::new(graph));
        let selection =
            Subgraph::from_nodes(shared.clone(), [0].into_iter().collect(), "stale".into())
                .expect("selection");
        shared.borrow_mut().remove_node(0).expect("remove");
        let mut vars = StepVariables::default();
        vars.set_node_map(
            "result",
            [(99, AlgorithmParamValue::Int(7))].into_iter().collect(),
        );
        let mut scope = StepScope::new(&selection, &mut vars);
        let filter = FilterNodesByAttrStep::new(
            "x",
            Predicate::Eq {
                value: AlgorithmParamValue::Int(1),
            },
            "result",
        );
        assert!(filter.apply(&mut Context::new(), &mut scope).is_err());
        assert!(
            SortNodesByAttrStep::new("x", SortOrder::Ascending, "result")
                .apply(&mut Context::new(), &mut scope)
                .is_err()
        );
        assert_eq!(
            scope
                .variables()
                .node_map("result")
                .expect("old result")
                .get(&99),
            Some(&AlgorithmParamValue::Int(7))
        );
    }

    #[test]
    fn test_predicate_eq() {
        let pred = Predicate::Eq {
            value: AlgorithmParamValue::Int(42),
        };
        assert!(pred.eval(&AlgorithmParamValue::Int(42)));
        assert!(!pred.eval(&AlgorithmParamValue::Int(43)));
    }

    #[test]
    fn test_predicate_gt() {
        let pred = Predicate::Gt {
            value: AlgorithmParamValue::Float(10.0),
        };
        assert!(pred.eval(&AlgorithmParamValue::Float(15.0)));
        assert!(!pred.eval(&AlgorithmParamValue::Float(5.0)));
    }

    #[test]
    fn test_predicate_contains() {
        let pred = Predicate::Contains {
            substring: "test".to_string(),
        };
        assert!(pred.eval(&AlgorithmParamValue::Text("testing123".to_string())));
        assert!(!pred.eval(&AlgorithmParamValue::Text("hello".to_string())));
    }

    #[test]
    fn test_predicate_range() {
        let pred = Predicate::Range {
            min: 0.0,
            max: 100.0,
        };
        assert!(pred.eval(&AlgorithmParamValue::Float(50.0)));
        assert!(pred.eval(&AlgorithmParamValue::Int(75)));
        assert!(!pred.eval(&AlgorithmParamValue::Float(150.0)));
    }
}
