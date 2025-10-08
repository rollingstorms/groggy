//! ArrayIterator<T> - Universal iterator with trait-based method injection

use crate::api::graph::Graph;
use crate::storage::array::{ArrayOps, EdgeLike, MetaNodeLike, NodeIdLike, SubgraphLike};
use std::cell::RefCell;
use std::rc::Rc;

/// Universal iterator that supports chaining operations on any element type
/// Methods become available automatically based on what traits T implements
#[derive(Clone)]
pub struct ArrayIterator<T> {
    /// The elements being iterated over
    elements: Vec<T>,
    /// Optional reference to the parent graph for graph-aware operations
    graph_ref: Option<Rc<RefCell<Graph>>>,
}

impl<T> ArrayIterator<T> {
    /// Create a new ArrayIterator
    pub fn new(elements: Vec<T>) -> Self {
        Self {
            elements,
            graph_ref: None,
        }
    }

    /// Create a new ArrayIterator with graph reference
    pub fn with_graph(elements: Vec<T>, graph: Rc<RefCell<Graph>>) -> Self {
        Self {
            elements,
            graph_ref: Some(graph),
        }
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

// =============================================================================
// Universal methods available to ALL types
// =============================================================================

impl<T: 'static> ArrayIterator<T> {
    /// Filter elements using a predicate function
    /// Available for any element type T
    pub fn filter<F>(self, predicate: F) -> Self
    where
        F: Fn(&T) -> bool,
    {
        let filtered: Vec<T> = self
            .elements
            .into_iter()
            .filter(|item| predicate(item))
            .collect();

        Self {
            elements: filtered,
            graph_ref: self.graph_ref,
        }
    }

    /// Transform elements using a mapping function
    /// Available for any element type T, transforms to type U
    pub fn map<U, F>(self, func: F) -> ArrayIterator<U>
    where
        F: Fn(T) -> U,
    {
        let mapped: Vec<U> = self.elements.into_iter().map(func).collect();

        ArrayIterator {
            elements: mapped,
            graph_ref: self.graph_ref,
        }
    }

    /// Materialize the iterator into a concrete collection
    /// Returns a boxed array implementing ArrayOps<T>
    pub fn collect(self) -> Box<dyn ArrayOps<T>> {
        // For now, return a simple BaseArray-like implementation
        // This will be enhanced when we implement BaseArray
        Box::new(CollectedArray::new(self.elements))
    }

    /// Get the underlying elements (consumes iterator)
    pub fn into_vec(self) -> Vec<T> {
        self.elements
    }

    /// Take only the first n elements
    pub fn take(self, n: usize) -> Self {
        let taken: Vec<T> = self.elements.into_iter().take(n).collect();

        Self {
            elements: taken,
            graph_ref: self.graph_ref,
        }
    }

    /// Skip the first n elements  
    pub fn skip(self, n: usize) -> Self {
        let skipped: Vec<T> = self.elements.into_iter().skip(n).collect();

        Self {
            elements: skipped,
            graph_ref: self.graph_ref,
        }
    }
}

// =============================================================================
// Trait-based method injection - methods only available for specific types
// =============================================================================

impl<T: SubgraphLike> ArrayIterator<T> {
    /// Filter nodes within subgraphs using a query string
    /// Only available when T implements SubgraphLike
    pub fn filter_nodes(self, query: &str) -> Self {
        // For now, this is a passthrough that preserves all subgraphs
        // In a full implementation, this would filter nodes within each subgraph
        // based on the query string (e.g., "age > 25")

        println!("filter_nodes called with query: {}", query);

        // Return self unchanged for now - actual filtering would be implemented here
        self
    }

    /// Filter edges within subgraphs using a query string  
    /// Only available when T implements SubgraphLike
    pub fn filter_edges(self, query: &str) -> Self {
        // For now, this is a passthrough that preserves all subgraphs
        // In a full implementation, this would filter edges within each subgraph
        // based on the query string (e.g., "weight > 0.5")

        println!("filter_edges called with query: {}", query);

        // Return self unchanged for now - actual filtering would be implemented here
        self
    }

    /// Collapse subgraphs into meta-nodes with aggregations
    /// Only available when T implements SubgraphLike
    pub fn collapse(self, aggs: std::collections::HashMap<String, String>) -> ArrayIterator<()> {
        // For now, this creates empty meta-nodes as placeholders
        // In a full implementation, this would:
        // 1. Take each subgraph in self.elements
        // 2. Aggregate attributes according to aggs specification
        // 3. Create MetaNode objects containing the collapsed subgraphs

        println!("collapse called with aggregations: {:?}", aggs);

        // Create placeholder for demonstration - in real implementation this would create meta-nodes
        let placeholders: Vec<()> = vec![];

        ArrayIterator::new(placeholders)
    }
}

impl<T: NodeIdLike> ArrayIterator<T> {
    /// Filter nodes by their degree
    /// Only available when T implements NodeIdLike
    pub fn filter_by_degree(self, min_degree: usize) -> Self {
        // For demonstration: filter nodes based on minimum degree
        // In a full implementation, this would query the graph for each node's degree

        println!("filter_by_degree called with min_degree: {}", min_degree);

        if let Some(_graph_ref) = &self.graph_ref {
            // If we have a graph reference, we could actually check degrees
            let filtered: Vec<T> = self
                .elements
                .into_iter()
                .filter(|_node| {
                    // Placeholder: In real implementation, would check:
                    // graph_ref.borrow().degree(*node) >= min_degree
                    true // For now, keep all nodes
                })
                .collect();

            ArrayIterator {
                elements: filtered,
                graph_ref: self.graph_ref,
            }
        } else {
            // No graph reference - can't check degrees, return unchanged
            self
        }
    }

    /// Get neighbors for each node
    /// Only available when T implements NodeIdLike  
    pub fn get_neighbors(self) -> ArrayIterator<Vec<crate::types::NodeId>> {
        // For demonstration: return empty neighbor lists
        // In a full implementation, this would query the graph for neighbors

        println!("get_neighbors called for {} nodes", self.elements.len());

        // Create empty neighbor lists for each node
        let neighbor_lists: Vec<Vec<crate::types::NodeId>> = self
            .elements
            .iter()
            .map(|_node| {
                // Placeholder: In real implementation, would do:
                // graph_ref.borrow().neighbors(*node).collect()
                vec![] // Empty neighbors for now
            })
            .collect();

        ArrayIterator::new(neighbor_lists)
    }

    /// Convert node IDs to subgraphs
    /// Only available when T implements NodeIdLike
    pub fn to_subgraph(self) -> ArrayIterator<crate::subgraphs::subgraph::Subgraph> {
        // For demonstration: create empty subgraphs
        // In a full implementation, this would create subgraphs around each node

        println!("to_subgraph called for {} nodes", self.elements.len());

        let subgraphs: Vec<crate::subgraphs::subgraph::Subgraph> = vec![];
        ArrayIterator::new(subgraphs)
    }
}

impl<T: MetaNodeLike> ArrayIterator<T> {
    /// Expand meta-nodes back into subgraphs
    /// Only available when T implements MetaNodeLike
    pub fn expand(self) -> ArrayIterator<crate::subgraphs::subgraph::Subgraph> {
        // For demonstration: expand meta-nodes to their original subgraphs
        // In a full implementation, this would retrieve the original subgraph from each meta-node

        println!("expand called for {} meta-nodes", self.elements.len());

        let subgraphs: Vec<crate::subgraphs::subgraph::Subgraph> = self
            .elements
            .iter()
            .filter_map(|_meta_node| {
                // Placeholder: In real implementation, would do:
                // meta_node.expand() or meta_node.subgraph()
                None // No subgraphs for now
            })
            .collect();

        ArrayIterator::new(subgraphs)
    }

    /// Re-aggregate meta-nodes with new aggregation functions
    /// Only available when T implements MetaNodeLike
    pub fn re_aggregate(self, _aggs: std::collections::HashMap<String, String>) -> Self {
        // For demonstration: re-aggregate meta-nodes with different aggregation functions

        println!("re_aggregate called for {} meta-nodes", self.elements.len());

        // For now, return unchanged
        self
    }
}

impl<T: EdgeLike> ArrayIterator<T> {
    /// Filter edges by weight or other attributes
    /// Only available when T implements EdgeLike
    pub fn filter_by_weight(self, min_weight: f64) -> Self {
        // For demonstration: filter edges by minimum weight
        // In a full implementation, this would query edge attributes

        println!("filter_by_weight called with min_weight: {}", min_weight);

        if let Some(_graph_ref) = &self.graph_ref {
            let filtered: Vec<T> = self
                .elements
                .into_iter()
                .filter(|_edge| {
                    // Placeholder: In real implementation, would check:
                    // graph_ref.borrow().edge_attribute(*edge, "weight").as_float() >= Some(min_weight)
                    true // For now, keep all edges
                })
                .collect();

            ArrayIterator {
                elements: filtered,
                graph_ref: self.graph_ref,
            }
        } else {
            // No graph reference - can't check weights, return unchanged
            self
        }
    }

    /// Filter edges by source and target node criteria
    /// Only available when T implements EdgeLike
    pub fn filter_by_endpoints(
        self,
        source_predicate: Option<fn(crate::types::NodeId) -> bool>,
        target_predicate: Option<fn(crate::types::NodeId) -> bool>,
    ) -> Self {
        // For demonstration: filter edges by their endpoints

        println!("filter_by_endpoints called");

        // For now, return unchanged - real implementation would check source/target nodes
        let _ = (source_predicate, target_predicate); // Prevent unused warnings
        self
    }

    /// Group edges by source node
    /// Only available when T implements EdgeLike
    pub fn group_by_source(self) -> ArrayIterator<Vec<T>> {
        // For demonstration: group edges by their source nodes

        println!("group_by_source called for {} edges", self.elements.len());

        // Placeholder: In real implementation, would group by source node
        let edge_groups: Vec<Vec<T>> = vec![self.elements]; // Single group for now
        ArrayIterator::new(edge_groups)
    }
}

// =============================================================================
// Simple collection implementation for .collect()
// =============================================================================

/// Simple collected array implementation - temporary until BaseArray is ready
struct CollectedArray<T> {
    elements: Vec<T>,
}

impl<T> CollectedArray<T> {
    fn new(elements: Vec<T>) -> Self {
        Self { elements }
    }
}

impl<T> ArrayOps<T> for CollectedArray<T> {
    fn len(&self) -> usize {
        self.elements.len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.elements.get(index)
    }

    fn iter(&self) -> ArrayIterator<T>
    where
        T: Clone + 'static,
    {
        ArrayIterator::new(self.elements.clone())
    }
}
