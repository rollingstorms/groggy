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

use crate::traits::SubgraphOperations;
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
                let num = match value {
                    AlgorithmParamValue::Int(i) => *i as f64,
                    AlgorithmParamValue::Float(f) => *f,
                    _ => return false,
                };
                num >= *min && num <= *max
            }
        }
    }
}

fn compare_values(a: &AlgorithmParamValue, b: &AlgorithmParamValue) -> Option<Ordering> {
    use AlgorithmParamValue as APV;

    match (a, b) {
        (APV::Int(x), APV::Int(y)) => Some(x.cmp(y)),
        (APV::Float(x), APV::Float(y)) => x.partial_cmp(y),
        (APV::Int(x), APV::Float(y)) => (*x as f64).partial_cmp(y),
        (APV::Float(x), APV::Int(y)) => x.partial_cmp(&(*y as f64)),
        (APV::Text(x), APV::Text(y)) => Some(x.cmp(y)),
        (APV::Bool(x), APV::Bool(y)) => Some(x.cmp(y)),
        _ => Option::None,
    }
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
        let mut node_values: Vec<(NodeId, Option<AlgorithmParamValue>)> = scope
            .node_ids()
            .map(|&node| {
                let value = scope
                    .subgraph()
                    .get_node_attribute(node, &attr_name)
                    .ok()
                    .flatten()
                    .and_then(AlgorithmParamValue::from_attr_value);
                (node, value)
            })
            .collect();

        // Stable sort by value
        node_values.sort_by(|(_, a), (_, b)| {
            match (a, b) {
                (Some(val_a), Some(val_b)) => {
                    let cmp = compare_values(val_a, val_b).unwrap_or(Ordering::Equal);
                    match self.order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    }
                }
                (Some(_), None) => Ordering::Less, // Values before None
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            }
        });

        // Create result map with sort index
        let mut result = HashMap::with_capacity(node_values.len());
        for (index, (node, _)) in node_values.iter().enumerate() {
            result.insert(*node, AlgorithmParamValue::Int(index as i64));
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

        for &node in scope.node_ids() {
            if let Ok(Some(attr_value)) = scope.subgraph().get_node_attribute(node, &attr_name) {
                if let Some(param_value) = AlgorithmParamValue::from_attr_value(attr_value) {
                    if self.predicate.eval(&param_value) {
                        result.insert(node, AlgorithmParamValue::Int(1)); // Mark as passing
                    }
                }
            }
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

        for &edge in scope.edge_ids() {
            if let Ok(Some(attr_value)) = scope.subgraph().get_edge_attribute(edge, &attr_name) {
                if let Some(param_value) = AlgorithmParamValue::from_attr_value(attr_value) {
                    if self.predicate.eval(&param_value) {
                        result.insert(edge, AlgorithmParamValue::Int(1)); // Mark as passing
                    }
                }
            }
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

        // Collect and sort by value
        let mut items: Vec<(NodeId, AlgorithmParamValue)> = source_map
            .iter()
            .map(|(&node, value)| (node, value.clone()))
            .collect();

        items.sort_by(|(_, a), (_, b)| {
            let cmp = compare_values(a, b).unwrap_or(Ordering::Equal);
            match self.order {
                SortOrder::Ascending => cmp,
                SortOrder::Descending => cmp.reverse(),
            }
        });

        // Take top k
        let mut result = HashMap::new();
        for (rank, (node, _value)) in items.into_iter().take(self.k).enumerate() {
            result.insert(node, AlgorithmParamValue::Int(rank as i64));
            // Could also preserve original value instead of rank
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
