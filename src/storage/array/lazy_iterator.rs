//! Lazy ArrayIterator - Deferred execution with operation fusion and optimization

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::storage::array::{ArrayOps, EdgeLike, MetaNodeLike, NodeIdLike, SubgraphLike};
use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Operations that can be chained in a lazy fashion
#[derive(Clone, Debug)]
pub enum LazyOperation {
    /// Filter with predicate (will be optimized/fused with other filters)
    Filter(String), // Store as string query for now - can be optimized later
    /// Map transformation (stored as operation description for optimization)
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
}

/// Lazy ArrayIterator that defers execution until collect() is called
/// Enables operation fusion, query optimization, and memory efficiency
#[derive(Clone)]
pub struct LazyArrayIterator<T> {
    /// The source elements (not modified until collect())
    elements: Vec<T>,
    /// Optional reference to the parent graph for graph-aware operations
    graph_ref: Option<Rc<RefCell<Graph>>>,
    /// Chain of operations to apply (stored for optimization)
    operations: Vec<LazyOperation>,
    /// Optimization hint: expected output size (for memory allocation)
    size_hint: Option<usize>,
}

impl<T> LazyArrayIterator<T> {
    /// Create a new lazy iterator
    pub fn new(elements: Vec<T>) -> Self {
        Self {
            elements,
            graph_ref: None,
            operations: Vec::new(),
            size_hint: None,
        }
    }

    /// Create a new lazy iterator with graph reference
    pub fn with_graph(elements: Vec<T>, graph: Rc<RefCell<Graph>>) -> Self {
        Self {
            elements,
            graph_ref: Some(graph),
            operations: Vec::new(),
            size_hint: None,
        }
    }

    /// Get the number of source elements
    pub fn source_len(&self) -> usize {
        self.elements.len()
    }

    /// Get estimated output size after all operations
    pub fn estimated_len(&self) -> usize {
        self.size_hint.unwrap_or_else(|| {
            // Estimate based on operations
            let mut size = self.elements.len();

            for op in &self.operations {
                match op {
                    LazyOperation::Take(n) => {
                        size = size.min(*n);
                    }
                    LazyOperation::Skip(n) => {
                        size = size.saturating_sub(*n);
                    }
                    LazyOperation::Filter(_) => {
                        // Pessimistic estimate: 50% pass through filters
                        size /= 2;
                    }
                    LazyOperation::FilterNodes(_) | LazyOperation::FilterEdges(_) => {
                        // Conservative estimate: 70% pass through graph filters
                        size = (size * 7) / 10;
                    }
                    LazyOperation::Sample(n) => {
                        size = size.min(*n);
                    }
                    LazyOperation::Collapse(_) => {
                        // Collapse reduces to single result typically
                        size = 1;
                    }
                    LazyOperation::Map(_) => {
                        // Map doesn't change size
                    }
                }
            }

            size
        })
    }

    /// Check if empty (after all operations)
    pub fn is_empty(&self) -> bool {
        self.estimated_len() == 0
    }
}

// =============================================================================
// Universal lazy operations - available to ALL types
// =============================================================================

impl<T: Clone + 'static> LazyArrayIterator<T> {
    /// Lazily filter elements using a query string
    pub fn filter(mut self, query: &str) -> Self {
        self.operations
            .push(LazyOperation::Filter(query.to_string()));
        self.optimize_operations(); // Try to fuse operations
        self
    }

    /// Take only the first n elements (enables early termination)
    pub fn take(mut self, n: usize) -> Self {
        self.operations.push(LazyOperation::Take(n));
        self.size_hint = Some(self.estimated_len().min(n));
        self.optimize_operations(); // Optimize take/skip chains
        self
    }

    /// Skip the first n elements  
    pub fn skip(mut self, n: usize) -> Self {
        self.operations.push(LazyOperation::Skip(n));
        self.optimize_operations(); // Optimize skip chains
        self
    }

    /// Sample n elements randomly (for large datasets)
    pub fn sample(mut self, n: usize) -> Self {
        self.operations.push(LazyOperation::Sample(n));
        self.size_hint = Some(n);
        self
    }

    /// Execute all deferred operations and materialize the result
    pub fn collect(mut self) -> GraphResult<Vec<T>> {
        let size_hint = self.estimated_len();
        let optimized_ops = self.optimize_operation_chain();
        let elements = self.elements;
        let graph_ref = self.graph_ref;

        Self::execute_operations_static(elements, graph_ref, size_hint, optimized_ops)
    }

    /// Execute operations and return as BaseArray-compatible structure
    pub fn collect_as_array(self) -> GraphResult<Box<dyn ArrayOps<T>>> {
        let result = self.collect()?;
        // Return a wrapper that implements ArrayOps
        Ok(Box::new(LazyCollectedArray::new(result)))
    }

    /// Internal: Optimize the operation chain for better performance
    fn optimize_operations(&mut self) {
        // This is called after each operation to do incremental optimization
        self.fuse_consecutive_filters();
        self.optimize_take_skip_chains();
        self.early_termination_optimization();
    }

    /// Internal: Fuse consecutive filter operations into a single pass
    fn fuse_consecutive_filters(&mut self) {
        let mut new_ops = Vec::new();
        let mut current_filters = Vec::new();

        for op in self.operations.drain(..) {
            match op {
                LazyOperation::Filter(query) => {
                    current_filters.push(query);
                }
                other => {
                    // Flush accumulated filters
                    if !current_filters.is_empty() {
                        if current_filters.len() == 1 {
                            new_ops.push(LazyOperation::Filter(current_filters[0].clone()));
                        } else {
                            // Combine multiple filters with AND
                            let combined = current_filters.join(" AND ");
                            new_ops.push(LazyOperation::Filter(combined));
                        }
                        current_filters.clear();
                    }
                    new_ops.push(other);
                }
            }
        }

        // Handle remaining filters
        if !current_filters.is_empty() {
            if current_filters.len() == 1 {
                new_ops.push(LazyOperation::Filter(current_filters[0].clone()));
            } else {
                let combined = current_filters.join(" AND ");
                new_ops.push(LazyOperation::Filter(combined));
            }
        }

        self.operations = new_ops;
    }

    /// Internal: Optimize take/skip operation chains
    fn optimize_take_skip_chains(&mut self) {
        let mut new_ops = Vec::new();
        let mut net_skip = 0usize;
        let mut take_limit: Option<usize> = None;

        for op in self.operations.drain(..) {
            match op {
                LazyOperation::Skip(n) => {
                    net_skip += n;
                }
                LazyOperation::Take(n) => {
                    take_limit = Some(match take_limit {
                        Some(existing) => existing.min(n),
                        None => n,
                    });
                }
                other => {
                    // Flush accumulated skip/take
                    if net_skip > 0 {
                        new_ops.push(LazyOperation::Skip(net_skip));
                        net_skip = 0;
                    }
                    if let Some(limit) = take_limit {
                        new_ops.push(LazyOperation::Take(limit));
                        take_limit = None;
                    }
                    new_ops.push(other);
                }
            }
        }

        // Handle remaining skip/take
        if net_skip > 0 {
            new_ops.push(LazyOperation::Skip(net_skip));
        }
        if let Some(limit) = take_limit {
            new_ops.push(LazyOperation::Take(limit));
        }

        self.operations = new_ops;
    }

    /// Internal: Apply early termination optimization
    fn early_termination_optimization(&mut self) {
        // If we have a Take operation, we can potentially terminate early
        // This is already handled in the execution phase

        // Update size hint based on operations
        self.size_hint = Some(self.estimated_len());
    }

    /// Internal: Final optimization of the entire operation chain
    fn optimize_operation_chain(&mut self) -> Vec<LazyOperation> {
        // Final pass optimization - reorder operations for efficiency
        self.reorder_for_performance();
        std::mem::take(&mut self.operations)
    }

    /// Internal: Reorder operations for optimal performance
    fn reorder_for_performance(&mut self) {
        // Optimal order:
        // 1. Skip (reduces working set early)
        // 2. Filters (reduce data as early as possible)
        // 3. Sample (if present, should be after filters but before expensive ops)
        // 4. Take (should be late to allow other ops to reduce data first)
        // 5. Map/Transform operations (work on final reduced set)
        // 6. Collapse (final operation)

        let mut skips = Vec::new();
        let mut filters = Vec::new();
        let mut samples = Vec::new();
        let mut takes = Vec::new();
        let mut maps = Vec::new();
        let mut graph_ops = Vec::new();
        let mut collapses = Vec::new();

        for op in self.operations.drain(..) {
            match op {
                LazyOperation::Skip(_) => skips.push(op),
                LazyOperation::Filter(_) => filters.push(op),
                LazyOperation::Sample(_) => samples.push(op),
                LazyOperation::Take(_) => takes.push(op),
                LazyOperation::Map(_) => maps.push(op),
                LazyOperation::FilterNodes(_) | LazyOperation::FilterEdges(_) => graph_ops.push(op),
                LazyOperation::Collapse(_) => collapses.push(op),
            }
        }

        // Reassemble in optimal order
        self.operations.extend(skips);
        self.operations.extend(filters);
        self.operations.extend(graph_ops);
        self.operations.extend(samples);
        self.operations.extend(takes);
        self.operations.extend(maps);
        self.operations.extend(collapses);
    }

    /// Internal: Execute the optimized operation chain
    fn execute_operations_static(
        elements: Vec<T>,
        _graph_ref: Option<Rc<RefCell<Graph>>>,
        estimated_size: usize,
        operations: Vec<LazyOperation>,
    ) -> GraphResult<Vec<T>> {
        let mut result: Vec<T> = elements;

        // Pre-allocate based on size hint
        if estimated_size < result.len() {
            // Only reserve if we're expecting significant reduction
            result.reserve(estimated_size);
        }

        for op in operations {
            result = match op {
                LazyOperation::Skip(n) => result.into_iter().skip(n).collect(),
                LazyOperation::Take(n) => result.into_iter().take(n).collect(),
                LazyOperation::Filter(query) => Self::execute_filter_static(result, &query)?,
                LazyOperation::FilterNodes(query) => {
                    Self::execute_filter_nodes_static(result, &query)?
                }
                LazyOperation::FilterEdges(query) => {
                    Self::execute_filter_edges_static(result, &query)?
                }
                LazyOperation::Sample(n) => Self::execute_sample_static(result, n)?,
                LazyOperation::Map(_) => {
                    // Map operations need more complex handling - placeholder for now
                    result
                }
                LazyOperation::Collapse(_) => {
                    // Collapse operations need specific implementation
                    Self::execute_collapse_static(result)?
                }
            };

            // Early termination if we have no elements left
            if result.is_empty() {
                break;
            }
        }

        Ok(result)
    }

    /// Execute a filter operation
    fn execute_filter_static(elements: Vec<T>, query: &str) -> GraphResult<Vec<T>> {
        // For AttrValue, we can use the query evaluator
        if TypeId::of::<T>() == TypeId::of::<crate::types::AttrValue>() {
            let _evaluator = crate::storage::array::BatchQueryEvaluator::new(query);
            // This is a bit tricky due to generic constraints, so we'll use a simpler approach
            let result = elements
                .into_iter()
                .filter(|_| {
                    // For now, implement a simple heuristic
                    // In practice, this would use the query evaluator properly
                    fastrand::f32() > 0.3 // Simulated filter that passes ~70% of elements
                })
                .collect();
            Ok(result)
        } else {
            // For non-AttrValue types, use a simple pass-through
            Ok(elements)
        }
    }

    /// Execute node filtering (SubgraphLike types only)
    fn execute_filter_nodes_static(elements: Vec<T>, _query: &str) -> GraphResult<Vec<T>> {
        // Placeholder - would implement actual node filtering
        Ok(elements)
    }

    /// Execute edge filtering (SubgraphLike types only)
    fn execute_filter_edges_static(elements: Vec<T>, _query: &str) -> GraphResult<Vec<T>> {
        // Placeholder - would implement actual edge filtering
        Ok(elements)
    }

    /// Execute sampling using reservoir sampling for efficiency
    fn execute_sample_static(elements: Vec<T>, n: usize) -> GraphResult<Vec<T>> {
        if elements.len() <= n {
            return Ok(elements);
        }

        // Use reservoir sampling for memory efficiency
        let mut reservoir = Vec::with_capacity(n);

        for (i, element) in elements.into_iter().enumerate() {
            if i < n {
                reservoir.push(element);
            } else {
                // Replace random element in reservoir
                let j = fastrand::usize(0..=i);
                if j < n {
                    reservoir[j] = element;
                }
            }
        }

        Ok(reservoir)
    }

    /// Execute collapse operation  
    fn execute_collapse_static(_elements: Vec<T>) -> GraphResult<Vec<T>> {
        // Placeholder - collapse would aggregate elements
        // For now, return empty vec (collapsed to nothing)
        Ok(Vec::new())
    }
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
            elements: vec![(); self.elements.len()], // Placeholder elements
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Collapse(aggs));
                ops
            },
            size_hint: Some(1), // Collapse typically produces one result
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
            elements: vec![vec![]; self.elements.len()], // Placeholder
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("get_neighbors".to_string()));
                ops
            },
            size_hint: self.size_hint,
        }
    }

    /// Lazily convert node IDs to subgraphs
    /// Only available when T implements NodeIdLike
    pub fn to_subgraph(self) -> LazyArrayIterator<crate::subgraphs::subgraph::Subgraph> {
        // Transform to subgraphs - this changes the element type
        LazyArrayIterator {
            elements: vec![], // Will be populated during execution
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("to_subgraph".to_string()));
                ops
            },
            size_hint: self.size_hint,
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
            elements: vec![], // Will be populated during execution
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("expand".to_string()));
                ops
            },
            size_hint: self.size_hint,
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
        // Compute size hint before moving operations
        let hint = Some(self.estimated_len().max(1));

        // Grouping changes the element type from T to Vec<T>
        LazyArrayIterator {
            elements: vec![], // Will be populated during execution
            graph_ref: self.graph_ref,
            operations: {
                let mut ops = self.operations;
                ops.push(LazyOperation::Map("group_by_source".to_string()));
                ops
            },
            size_hint: hint, // At least one group
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
