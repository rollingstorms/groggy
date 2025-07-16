// src_new/graph/managers/filter.rs
//! FilterManager: Composable, efficient filtering for node and edge collections in Groggy.
//! Supports vectorized, zero-copy filtering and agent/LLM-friendly APIs.

use crate::graph::types::{NodeId, EdgeId};
use super::attributes::AttributeManager;
use pyo3::prelude::*;
// use crate::graph::columnar::{NodeColumnarStore, EdgeColumnarStore}; // Uncomment when available

/// Manages filter expressions and evaluation context for nodes/edges.
#[pyclass]
pub struct FilterManager {
    pub filters: Vec<FilterExpr>,
    pub last_result: Option<Vec<usize>>,
    pub is_node: bool, // true for nodes, false for edges
    pub attr_manager: AttributeManager,
}

#[derive(Clone)]
pub enum FilterExpr {
    IntEquals { attr: String, value: i64 },
    BoolEquals { attr: String, value: bool },
    StrEquals { attr: String, value: String },
    // Extend with more as needed
}

#[pymethods]
impl FilterManager {
    /// Create a new FilterManager for nodes or edges.
    #[new]
    pub fn new(attr_manager: AttributeManager, is_node: bool) -> Self {
        Self { filters: Vec::new(), last_result: None, is_node, attr_manager }
    }

    /// Applies all filters to the target collection of IDs.
    /// Returns a filtered view or set of IDs. Supports vectorized execution.
    pub fn apply(&mut self, ids: Vec<usize>) -> Vec<usize> {
        let mut result = ids;
        for expr in &self.filters {
            result = match (self.is_node, expr) {
                (true, FilterExpr::IntEquals { attr, value }) =>
                    self.attr_manager.columnar.filter_nodes_int_simd(attr.clone(), *value)
                        .into_iter().filter(|i| result.contains(i)).collect(),
                (false, FilterExpr::IntEquals { attr, value }) =>
                    self.attr_manager.columnar.filter_edges_int_simd(attr.clone(), *value)
                        .into_iter().filter(|i| result.contains(i)).collect(),
                (true, FilterExpr::BoolEquals { attr, value }) =>
                    self.attr_manager.columnar.filter_nodes_by_bool(attr.clone(), *value)
                        .into_iter().filter(|i| result.contains(i)).collect(),
                (false, FilterExpr::BoolEquals { attr, value }) =>
                    self.attr_manager.columnar.filter_edges_by_bool(attr.clone(), *value)
                        .into_iter().filter(|i| result.contains(i)).collect(),
                (true, FilterExpr::StrEquals { attr, value }) =>
                    self.attr_manager.columnar.filter_nodes_by_value(attr.clone(), serde_json::Value::String(value.clone()))
                        .into_iter().filter(|i| result.contains(i)).collect(),
                (false, FilterExpr::StrEquals { attr, value }) =>
                    self.attr_manager.columnar.filter_edges_by_value(attr.clone(), serde_json::Value::String(value.clone()))
                        .into_iter().filter(|i| result.contains(i)).collect(),
                _ => result,
            };
        }
        self.last_result = Some(result.clone());
        result
    }

    /// Returns the current set of filtered IDs or objects.
    pub fn results(&self) -> Option<Vec<usize>> {
        self.last_result.clone()
    }
}

impl FilterManager {
    /// Adds a filter expression to the manager (internal method, not exposed to Python).
    /// Allows chaining multiple filters for complex queries.
    pub fn add_filter(&mut self, expr: FilterExpr) {
        self.filters.push(expr);
    }
}
