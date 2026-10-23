//! Query Engine - Core filtering operations for graph data.

use crate::errors::GraphResult;
use crate::state::space::GraphSpace;
use crate::storage::pool::GraphPool;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::collections::HashSet;

/// The main query engine for filtering operations
#[derive(Debug)]
pub struct QueryEngine {}

impl QueryEngine {
    /// Create a stateless filtering engine.
    pub fn new() -> Self {
        Self {}
    }
    /// Find current nodes matching the filter.
    pub fn find_nodes_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter,
    ) -> GraphResult<Vec<NodeId>> {
        self.filter_nodes(pool, space, filter)
    }
    /// Filter current nodes using column-oriented shared predicates.
    pub fn filter_nodes(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter,
    ) -> GraphResult<Vec<NodeId>> {
        self.filter_nodes_columnar(&space.node_ids(), pool, space, filter)
    }
    pub(crate) fn filter_nodes_columnar(
        &self,
        nodes: &[NodeId],
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter,
    ) -> GraphResult<Vec<NodeId>> {
        for &id in nodes {
            if !space.contains_node(id) {
                return Err(crate::GraphError::InvalidInput(format!(
                    "Inactive query node {id}"
                )));
            }
        }
        super::filtering::nodes(
            &super::filtering::Live::new(pool, space, None),
            nodes,
            filter,
        )
    }
    /// Filter current edge instances with the shared predicate executor.
    pub fn filter_edges(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter,
    ) -> GraphResult<Vec<EdgeId>> {
        super::filtering::edges(
            &super::filtering::Live::new(pool, space, None),
            &space.edge_ids(),
            filter,
        )
    }
    /// Find current edge instances matching the filter.
    pub fn find_edges_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter,
    ) -> GraphResult<Vec<EdgeId>> {
        self.filter_edges(pool, space, filter)
    }
}

/// Simple attribute filter for basic comparisons
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeFilter {
    Equals(AttrValue),
    NotEquals(AttrValue),
    GreaterThan(AttrValue),
    LessThan(AttrValue),
    GreaterThanOrEqual(AttrValue),
    LessThanOrEqual(AttrValue),
}

impl AttributeFilter {
    /// Match scalar equality or exact numeric ordering, without narrowing integers.
    pub fn matches(&self, value: &AttrValue) -> bool {
        use crate::storage::numeric::scalar::{compare_inputs, NumericInput};
        use std::cmp::Ordering;
        let numeric = |value: &AttrValue| match value {
            AttrValue::Int(v) => Some(NumericInput::Integer(i128::from(*v))),
            AttrValue::SmallInt(v) => Some(NumericInput::Integer(i128::from(*v))),
            AttrValue::Float(v) => Some(NumericInput::Float(f64::from(*v))),
            AttrValue::Float64(v) => Some(NumericInput::Float(*v)),
            _ => None,
        };
        let order = |target| {
            numeric(value)
                .zip(numeric(target))
                .and_then(|(a, b)| compare_inputs(a, b))
        };
        match self {
            Self::Equals(target) => value == target,
            Self::NotEquals(target) => value != target,
            Self::GreaterThan(target) => order(target) == Some(Ordering::Greater),
            Self::LessThan(target) => order(target) == Some(Ordering::Less),
            Self::GreaterThanOrEqual(target) => {
                matches!(order(target), Some(Ordering::Greater | Ordering::Equal))
            }
            Self::LessThanOrEqual(target) => {
                matches!(order(target), Some(Ordering::Less | Ordering::Equal))
            }
        }
    }
}

/// Node filter for graph queries
#[derive(Debug, Clone, PartialEq)]
pub enum NodeFilter {
    HasAttribute {
        name: AttrName,
    },
    AttributeEquals {
        name: AttrName,
        value: AttrValue,
    },
    AttributeFilter {
        name: AttrName,
        filter: AttributeFilter,
    },
    DegreeRange {
        min: usize,
        max: usize,
    },
    HasNeighbor {
        neighbor_id: NodeId,
    },
    And(Vec<NodeFilter>),
    Or(Vec<NodeFilter>),
    Not(Box<NodeFilter>),
    /// Filter to nodes in a specific set (for subgraph node restriction)
    NodeSet(HashSet<NodeId>),
}

/// Edge filter for graph queries
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeFilter {
    HasAttribute {
        name: AttrName,
    },
    AttributeEquals {
        name: AttrName,
        value: AttrValue,
    },
    AttributeFilter {
        name: AttrName,
        filter: AttributeFilter,
    },
    ConnectsNodes {
        source: NodeId,
        target: NodeId,
    },
    ConnectsAny(Vec<NodeId>),
    And(Vec<EdgeFilter>),
    Or(Vec<EdgeFilter>),
    Not(Box<EdgeFilter>),
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}
