//! Lazy ArrayIterator - validated scalar plans with statement-order execution
#![doc = include_str!("../../../docs/api/lazy-scalar-arrays.md")]

use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::array::{ArrayOps, EdgeLike, MetaNodeLike, NodeIdLike, SubgraphLike};
use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Operations that can be chained in a lazy fashion
#[derive(Clone, Debug)]
pub enum LazyOperation {
    /// Scalar predicate, compiled once during collect.
    Filter(String),
    /// Reserved transform; rejects during collect until implemented.
    Map(String),
    /// Take first N elements - enables early termination
    Take(usize),
    /// Skip first N elements
    Skip(usize),
    /// Filter nodes within subgraphs (SubgraphLike operations)
    FilterNodes(String),
    /// Filter edges within subgraphs (SubgraphLike operations)  
    FilterEdges(String),
    /// Collapse subgraphs with aggregation functions
    Collapse(HashMap<String, String>),
    /// Sample elements (useful for large datasets)
    Sample(usize),
    /// Local reproducible reservoir sample; does not consume global RNG state.
    SampleSeeded(usize, u64),
}

/// Lazy ArrayIterator that defers execution until collect() is called
/// Defers scalar operations until collection while preserving statement order
#[derive(Clone)]
pub struct LazyArrayIterator<T> {
    /// The source elements (not modified until collect())
    elements: Vec<T>,
    /// Optional reference to the parent graph for graph-aware operations
    graph_ref: Option<Rc<RefCell<Graph>>>,
    /// Chain of operations to validate and apply in statement order
    operations: Vec<LazyOperation>,
}

impl<T> LazyArrayIterator<T> {
    /// Create a new lazy iterator
    pub fn new(elements: Vec<T>) -> Self {
        Self {
            elements,
            graph_ref: None,
            operations: Vec::new(),
        }
    }

    /// Create a new lazy iterator with graph reference
    pub fn with_graph(elements: Vec<T>, graph: Rc<RefCell<Graph>>) -> Self {
        Self {
            elements,
            graph_ref: Some(graph),
            operations: Vec::new(),
        }
    }

    /// Get the number of source elements
    pub fn source_len(&self) -> usize {
        self.elements.len()
    }

    /// Upper bound on output length; filters may produce fewer rows.
    /// This is computed in statement order, without retaining stale size hints.
    pub fn estimated_len(&self) -> usize {
        self.operations
            .iter()
            .fold(self.elements.len(), |size, op| match op {
                LazyOperation::Take(n)
                | LazyOperation::Sample(n)
                | LazyOperation::SampleSeeded(n, _) => size.min(*n),
                LazyOperation::Skip(n) => size.saturating_sub(*n),
                _ => size,
            })
    }

    /// Whether supported operations guarantee an empty result without execution.
    /// A filter's exact output cardinality is known only after collection.
    pub fn is_empty(&self) -> bool {
        self.estimated_len() == 0
    }
}

/// Validated scalar execution plan. Unsupported graph/type-changing operations
/// reject during lowering, even on an empty source or after take(0).
enum PreparedOperation {
    Filter(crate::storage::array::QueryEvaluator),
    Take(usize),
    Skip(usize),
    Sample(usize, Option<u64>),
}

impl<T: Clone + 'static> LazyArrayIterator<T> {
    /// Compile a scalar query at collection; no random filtering or pass-through.
    pub fn filter(mut self, query: &str) -> Self {
        self.operations
            .push(LazyOperation::Filter(query.to_owned()));
        self
    }
    /// Keep the first n rows at this point in the chain.
    pub fn take(mut self, n: usize) -> Self {
        self.operations.push(LazyOperation::Take(n));
        self
    }
    /// Skip the first n rows at this point in the chain.
    pub fn skip(mut self, n: usize) -> Self {
        self.operations.push(LazyOperation::Skip(n));
        self
    }
    /// Reservoir sample, preserving the existing random default.
    pub fn sample(mut self, n: usize) -> Self {
        self.operations.push(LazyOperation::Sample(n));
        self
    }
    /// Reservoir sample using a private RNG for repeatable runs.
    pub fn sample_seeded(mut self, n: usize, seed: u64) -> Self {
        self.operations.push(LazyOperation::SampleSeeded(n, seed));
        self
    }
    /// Validate the whole chain and materialize in statement order.
    pub fn collect(self) -> GraphResult<Vec<T>> {
        let mut plan = Vec::with_capacity(self.operations.len());
        for operation in self.operations {
            plan.push(match operation {
                LazyOperation::Filter(query) => {
                    if !scalar_type::<T>() {
                        return Err(GraphError::InvalidInput("Lazy scalar filtering requires AttrValue, i32/i64, f32/f64, bool or usize elements".into()));
                    }
                    let evaluator = crate::storage::array::QueryEvaluator::new(&query);
                    evaluator.validate()?;
                    PreparedOperation::Filter(evaluator)
                }
                LazyOperation::Take(n) => PreparedOperation::Take(n),
                LazyOperation::Skip(n) => PreparedOperation::Skip(n),
                LazyOperation::Sample(n) => PreparedOperation::Sample(n, None),
                LazyOperation::SampleSeeded(n, seed) => PreparedOperation::Sample(n, Some(seed)),
                unsupported => return Err(GraphError::InvalidInput(format!(
                    "Unsupported lazy array operation {unsupported:?}; use the explicit graph/collection API"
                ))),
            });
        }
        let mut result = self.elements;
        for operation in plan {
            result = match operation {
                PreparedOperation::Take(n) => {
                    result.truncate(n);
                    result
                }
                PreparedOperation::Skip(n) => {
                    result.drain(..n.min(result.len()));
                    result
                }
                PreparedOperation::Filter(evaluator) => {
                    retain_scalar_rows(&mut result, &evaluator)?;
                    result
                }
                PreparedOperation::Sample(0, _) => Vec::new(),
                PreparedOperation::Sample(n, seed) => {
                    if result.len() <= n {
                        result
                    } else {
                        let mut rng = seed.map(fastrand::Rng::with_seed).unwrap_or_default();
                        let mut reservoir = Vec::with_capacity(n);
                        for (position, element) in result.into_iter().enumerate() {
                            if position < n {
                                reservoir.push(element);
                            } else {
                                let chosen = rng.usize(0..=position);
                                if chosen < n {
                                    reservoir[chosen] = element;
                                }
                            }
                        }
                        reservoir
                    }
                }
            };
        }
        Ok(result)
    }
    /// Materialize using the same checked execution path.
    pub fn collect_as_array(self) -> GraphResult<Box<dyn ArrayOps<T>>> {
        Ok(Box::new(LazyCollectedArray::new(self.collect()?)))
    }
}

fn retain_scalar_rows<T: 'static>(
    rows: &mut Vec<T>,
    evaluator: &crate::storage::array::QueryEvaluator,
) -> GraphResult<()> {
    use super::query::NumericColumnKind;
    use crate::storage::numeric::scalar::NumericInput;
    // Dispatch once per column, never once per row. These numeric types cannot
    // raise value-conversion errors; heterogeneous/usize paths retain checked reads.
    let values = rows as &mut dyn Any;
    macro_rules! numeric_column {
        ($type:ty, $kind:expr, $convert:expr) => {
            if let Some(values) = values.downcast_mut::<Vec<$type>>() {
                let predicate = evaluator.numeric_predicate($kind)?;
                values.retain(|&value| predicate.evaluate(($convert)(value)));
                return Ok(());
            }
        };
    }
    numeric_column!(i32, NumericColumnKind::Integer, |value| {
        NumericInput::Integer(i128::from(value))
    });
    numeric_column!(i64, NumericColumnKind::Integer, |value| {
        NumericInput::Integer(i128::from(value))
    });
    numeric_column!(f32, NumericColumnKind::Floating, |value| {
        NumericInput::Float(f64::from(value))
    });
    numeric_column!(f64, NumericColumnKind::Floating, NumericInput::Float);
    if let Some(values) = values.downcast_mut::<Vec<crate::AttrValue>>() {
        let integers = evaluator.numeric_predicate(NumericColumnKind::Integer)?;
        let floats = evaluator.numeric_predicate(NumericColumnKind::Floating)?;
        return retain_checked(values, |value| {
            match crate::storage::numeric::scalar::attribute_number(value) {
                Some(number @ NumericInput::Integer(_)) => Ok(integers.evaluate(number)),
                Some(number @ NumericInput::Float(_)) => Ok(floats.evaluate(number)),
                _ => evaluator.evaluate(value),
            }
        });
    }
    retain_checked(rows, |element| {
        scalar_value(element).and_then(|value| evaluator.evaluate(&value))
    })
}

fn retain_checked<T>(
    rows: &mut Vec<T>,
    mut predicate: impl FnMut(&T) -> GraphResult<bool>,
) -> GraphResult<()> {
    let mut failure = None;
    rows.retain(|element| {
        if failure.is_some() {
            return false;
        }
        match predicate(element) {
            Ok(keep) => keep,
            Err(error) => {
                failure = Some(error);
                false
            }
        }
    });
    failure.map_or(Ok(()), Err)
}

fn scalar_type<T: 'static>() -> bool {
    let id = TypeId::of::<T>();
    [
        TypeId::of::<crate::AttrValue>(),
        TypeId::of::<i32>(),
        TypeId::of::<i64>(),
        TypeId::of::<f32>(),
        TypeId::of::<f64>(),
        TypeId::of::<bool>(),
        TypeId::of::<usize>(),
    ]
    .contains(&id)
}
fn scalar_value<T: 'static>(value: &T) -> GraphResult<Cow<'_, crate::AttrValue>> {
    use crate::AttrValue as V;
    let value = value as &dyn Any;
    if let Some(value) = value.downcast_ref::<V>() {
        return Ok(Cow::Borrowed(value));
    }
    macro_rules! convert {
        ($ty:ty, $variant:ident) => {
            if let Some(value) = value.downcast_ref::<$ty>() {
                return Ok(Cow::Owned(V::$variant(*value)));
            }
        };
    }
    convert!(i32, SmallInt);
    convert!(i64, Int);
    convert!(f32, Float);
    convert!(f64, Float64);
    convert!(bool, Bool);
    if let Some(value) = value.downcast_ref::<usize>() {
        return i64::try_from(*value)
            .map(|value| Cow::Owned(V::Int(value)))
            .map_err(|_| {
                GraphError::InvalidInput("Lazy scalar ID exceeds i64 value range".into())
            });
    }
    Err(GraphError::InvalidInput(
        "Unsupported lazy scalar element type".into(),
    ))
}

// =============================================================================
// Trait-based lazy operations - only available for specific types
// =============================================================================

impl<T: SubgraphLike + Clone + 'static> LazyArrayIterator<T> {
    /// Lazily filter nodes within subgraphs
    pub fn filter_nodes(mut self, query: &str) -> Self {
        self.operations
            .push(LazyOperation::FilterNodes(query.to_string()));
        self
    }

    /// Lazily filter edges within subgraphs
    pub fn filter_edges(mut self, query: &str) -> Self {
        self.operations
            .push(LazyOperation::FilterEdges(query.to_string()));
        self
    }

    /// Lazily collapse subgraphs with aggregation
    pub fn collapse(self, aggs: HashMap<String, String>) -> LazyArrayIterator<()> {
        // Note: collapse changes the type to () (unit type) as it aggregates
        LazyArrayIterator {
            elements: Vec::new(), // Unsupported transforms reject during collect.
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Collapse(aggs));
                ops
            },
        }
    }
}

// =============================================================================
// NodeIdLike trait-based operations for LazyArrayIterator
// =============================================================================

impl<T: NodeIdLike + Clone + 'static> LazyArrayIterator<T> {
    /// Lazily filter nodes by their degree
    /// Only available when T implements NodeIdLike
    pub fn filter_by_degree(mut self, min_degree: usize) -> Self {
        self.operations
            .push(LazyOperation::Filter(format!("degree >= {}", min_degree)));
        self
    }

    /// Lazily get neighbors for each node  
    /// Only available when T implements NodeIdLike
    pub fn get_neighbors(self) -> LazyArrayIterator<Vec<crate::types::NodeId>> {
        // Transform to neighbor lists - this changes the element type
        LazyArrayIterator {
            elements: Vec::new(), // Unsupported transforms reject during collect.
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("get_neighbors".to_string()));
                ops
            },
        }
    }

    /// Lazily convert node IDs to subgraphs
    /// Only available when T implements NodeIdLike
    pub fn to_subgraph(self) -> LazyArrayIterator<crate::subgraphs::subgraph::Subgraph> {
        // Transform to subgraphs - this changes the element type
        LazyArrayIterator {
            elements: Vec::new(), // Unsupported transforms reject during collect.
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("to_subgraph".to_string()));
                ops
            },
        }
    }
}

// =============================================================================
// MetaNodeLike trait-based operations for LazyArrayIterator
// =============================================================================

impl<T: MetaNodeLike + Clone + 'static> LazyArrayIterator<T> {
    /// Lazily expand meta-nodes back into subgraphs
    /// Only available when T implements MetaNodeLike
    pub fn expand(self) -> LazyArrayIterator<crate::subgraphs::subgraph::Subgraph> {
        // Transform to subgraphs - this changes the element type
        LazyArrayIterator {
            elements: Vec::new(), // Unsupported transforms reject during collect.
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("expand".to_string()));
                ops
            },
        }
    }

    /// Lazily re-aggregate meta-nodes with new aggregation functions
    /// Only available when T implements MetaNodeLike
    pub fn re_aggregate(mut self, aggs: HashMap<String, String>) -> Self {
        // Re-aggregation is a form of collapse operation
        self.operations.push(LazyOperation::Collapse(aggs));
        self
    }
}

// =============================================================================
// EdgeLike trait-based operations for LazyArrayIterator
// =============================================================================

impl<T: EdgeLike + Clone + 'static> LazyArrayIterator<T> {
    /// Lazily filter edges by weight or other attributes
    /// Only available when T implements EdgeLike
    pub fn filter_by_weight(mut self, min_weight: f64) -> Self {
        self.operations
            .push(LazyOperation::Filter(format!("weight >= {}", min_weight)));
        self
    }

    /// Lazily filter edges by source and target node criteria
    /// Only available when T implements EdgeLike
    pub fn filter_by_endpoints(
        mut self,
        source_query: Option<String>,
        target_query: Option<String>,
    ) -> Self {
        let mut filter_parts = Vec::new();

        if let Some(sq) = source_query {
            filter_parts.push(format!("source: {}", sq));
        }
        if let Some(tq) = target_query {
            filter_parts.push(format!("target: {}", tq));
        }

        if !filter_parts.is_empty() {
            self.operations
                .push(LazyOperation::Filter(filter_parts.join(" AND ")));
        }

        self
    }

    /// Lazily group edges by source node
    /// Only available when T implements EdgeLike
    pub fn group_by_source(self) -> LazyArrayIterator<Vec<T>> {
        // Grouping changes the element type from T to Vec<T>
        LazyArrayIterator {
            elements: Vec::new(), // Unsupported transforms reject during collect.
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("group_by_source".to_string()));
                ops
            },
        }
    }
}

// =============================================================================
// Helper type for collected arrays
// =============================================================================

/// A simple collected array that implements ArrayOps
struct LazyCollectedArray<T> {
    elements: Vec<T>,
}

impl<T> LazyCollectedArray<T> {
    fn new(elements: Vec<T>) -> Self {
        Self { elements }
    }
}

impl<T: Clone + 'static> ArrayOps<T> for LazyCollectedArray<T> {
    fn len(&self) -> usize {
        self.elements.len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.elements.get(index)
    }

    fn iter(&self) -> crate::storage::array::ArrayIterator<T> {
        crate::storage::array::ArrayIterator::new(self.elements.clone())
    }
}
