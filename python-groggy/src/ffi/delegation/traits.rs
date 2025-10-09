//! Core delegation traits defining universal operations
//!
//! These traits separate algorithms from carriers, enabling any array type
//! or iterator to delegate operations to optimized implementations.

use groggy::types::{AttrValue, NodeId};
use pyo3::PyResult;

/// Operations available on subgraph objects
pub trait SubgraphOps {
    /// Expand subgraph to include neighbors within given radius
    fn neighborhood(
        &self,
        radius: Option<usize>,
    ) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph>;

    /// Convert subgraph to nodes table representation
    fn table(&self) -> PyResult<crate::ffi::storage::table::PyNodesTable>;

    /// Sample k random nodes/edges from the subgraph
    fn sample(&self, k: usize) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph>;

    /// Filter nodes in subgraph based on query
    fn filter_nodes(&self, query: &str) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph>;

    /// Get edges table for this subgraph
    fn edges_table(&self) -> PyResult<crate::ffi::storage::table::PyEdgesTable>;

    /// Calculate density of the subgraph
    fn density(&self) -> PyResult<f64>;

    /// Check if subgraph is connected
    fn is_connected(&self) -> PyResult<bool>;
}

/// Operations available on table objects (both nodes and edges)
pub trait TableOps {
    /// Aggregate table using specification
    fn agg(&self, spec: &str) -> PyResult<crate::ffi::storage::table::PyBaseTable>;

    /// Filter table rows based on expression
    fn filter(&self, expr: &str) -> PyResult<Self>
    where
        Self: Sized;

    /// Group table by specified columns
    fn group_by(&self, columns: &[&str]) -> PyResult<crate::ffi::storage::table::PyBaseTable>;

    /// Join this table with another
    fn join(&self, other: &Self, on: &str) -> PyResult<Self>
    where
        Self: Sized;

    /// Sort table by column
    fn sort_by(&self, column: &str, ascending: bool) -> PyResult<Self>
    where
        Self: Sized;

    /// Select specific columns
    fn select(&self, columns: &[&str]) -> PyResult<Self>
    where
        Self: Sized;

    /// Get unique values in specified column
    fn unique(&self, column: &str) -> PyResult<Vec<AttrValue>>;

    /// Count rows in table
    fn count(&self) -> PyResult<usize>;
}

/// Operations available on graph objects
pub trait GraphOps {
    /// Find connected components
    fn connected_components(
        &self,
    ) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray>;

    /// Find shortest path between nodes
    fn shortest_path(
        &self,
        from: NodeId,
        to: NodeId,
    ) -> PyResult<Option<crate::ffi::subgraphs::subgraph::PySubgraph>>;

    /// Breadth-first search from starting node
    fn bfs(&self, start: NodeId) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph>;

    /// Depth-first search from starting node
    fn dfs(&self, start: NodeId) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph>;

    /// Calculate pagerank for all nodes
    fn pagerank(&self, damping: Option<f64>) -> PyResult<crate::ffi::storage::table::PyNodesTable>;

    /// Find minimum spanning tree
    fn minimum_spanning_tree(&self) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph>;

    /// Calculate clustering coefficient
    fn clustering_coefficient(&self) -> PyResult<f64>;
}

/// Basic array operations (for all array types)
pub trait BaseArrayOps {
    type Item;

    /// Get array length
    fn len(&self) -> usize;

    /// Check if array is empty
    fn is_empty(&self) -> bool;

    /// Get item at index
    fn get(&self, index: usize) -> Option<&Self::Item>;

    /// Get iterator over items
    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Self::Item> + 'a>;

    /// Filter array by predicate
    fn filter<F>(&self, predicate: F) -> PyResult<Self>
    where
        F: Fn(&Self::Item) -> bool,
        Self: Sized,
        Self::Item: Clone;

    /// Map array to new type
    fn map<U, F>(&self, f: F) -> PyResult<DelegatingIterator<U>>
    where
        F: Fn(&Self::Item) -> U + 'static,
        U: 'static;

    /// Take first n elements
    fn take(&self, n: usize) -> PyResult<Self>
    where
        Self: Sized,
        Self::Item: Clone;

    /// Skip first n elements
    fn skip(&self, n: usize) -> PyResult<Self>
    where
        Self: Sized,
        Self::Item: Clone;
}

/// Statistical operations (for numerical array types)
pub trait NumArrayOps: BaseArrayOps {
    /// Calculate mean of numerical values
    fn mean(&self) -> PyResult<Option<f64>>;

    /// Calculate sum of numerical values  
    fn sum(&self) -> PyResult<f64>;

    /// Find minimum value
    fn min(&self) -> PyResult<Option<f64>>;

    /// Find maximum value
    fn max(&self) -> PyResult<Option<f64>>;

    /// Calculate standard deviation
    fn std_dev(&self) -> PyResult<Option<f64>>;

    /// Calculate median
    fn median(&self) -> PyResult<Option<f64>>;

    /// Calculate variance
    fn variance(&self) -> PyResult<Option<f64>>;

    /// Calculate percentile
    fn percentile(&self, p: f64) -> PyResult<Option<f64>>;

    /// Element-wise addition with another array
    fn add(&self, other: &Self) -> PyResult<Self>
    where
        Self: Sized;

    /// Scalar multiplication
    fn multiply(&self, scalar: f64) -> PyResult<Self>
    where
        Self: Sized;

    /// Calculate correlation with another array
    fn correlate(&self, other: &Self) -> PyResult<Option<f64>>;
}

/// Universal delegating iterator that forwards operations by mapping over elements
pub struct DelegatingIterator<T> {
    inner: Box<dyn Iterator<Item = T>>,
}

impl<T> DelegatingIterator<T> {
    /// Create new delegating iterator
    pub fn new<I: Iterator<Item = T> + 'static>(iter: I) -> Self {
        Self {
            inner: Box::new(iter),
        }
    }

    /// Map over elements to create new iterator
    pub fn map<U, F>(self, f: F) -> DelegatingIterator<U>
    where
        F: Fn(T) -> U + 'static,
        T: 'static,
        U: 'static,
    {
        DelegatingIterator::new(self.inner.map(f))
    }

    /// Flat map over elements
    pub fn flat_map<U, I, F>(self, f: F) -> DelegatingIterator<U>
    where
        F: Fn(T) -> I + 'static,
        I: IntoIterator<Item = U> + 'static,
        <I as IntoIterator>::IntoIter: 'static,
        T: 'static,
        U: 'static,
    {
        DelegatingIterator::new(self.inner.flat_map(f))
    }

    /// Filter elements by predicate
    pub fn filter<F>(self, predicate: F) -> DelegatingIterator<T>
    where
        F: Fn(&T) -> bool + 'static,
        T: 'static,
    {
        DelegatingIterator::new(self.inner.filter(predicate))
    }

    /// Take first n elements
    pub fn take(self, n: usize) -> DelegatingIterator<T>
    where
        T: 'static,
    {
        DelegatingIterator::new(self.inner.take(n))
    }

    /// Skip first n elements
    pub fn skip(self, n: usize) -> DelegatingIterator<T>
    where
        T: 'static,
    {
        DelegatingIterator::new(self.inner.skip(n))
    }

    /// Collect into vector
    pub fn collect_vec(self) -> Vec<T> {
        self.inner.collect()
    }
}

impl<T> Iterator for DelegatingIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

// Trait implementations for delegating operations across iterator elements
impl DelegatingIterator<crate::ffi::subgraphs::subgraph::PySubgraph> {
    /// Apply neighborhood operation to each subgraph in the iterator
    pub fn neighborhood(
        self,
        _radius: Option<usize>,
    ) -> DelegatingIterator<crate::ffi::subgraphs::subgraph::PySubgraph> {
        // For now, return a placeholder - full implementation would map the operation
        // This shows the pattern for trait-based delegation
        self
    }

    /// Convert each subgraph to a table
    pub fn table(self) -> DelegatingIterator<crate::ffi::storage::table::PyNodesTable> {
        DelegatingIterator::new(std::iter::empty()) // Placeholder
    }

    /// Sample from each subgraph
    pub fn sample(
        self,
        _k: usize,
    ) -> DelegatingIterator<crate::ffi::subgraphs::subgraph::PySubgraph> {
        self // Placeholder - would map sample operation
    }
}

impl DelegatingIterator<crate::ffi::storage::table::PyNodesTable> {
    /// Apply aggregation to each table in the iterator
    pub fn agg(self, _spec: String) -> DelegatingIterator<crate::ffi::storage::table::PyBaseTable> {
        DelegatingIterator::new(std::iter::empty()) // Placeholder
    }

    /// Filter each table in the iterator (specialized implementation)
    pub fn filter_table(
        self,
        _expr: String,
    ) -> DelegatingIterator<crate::ffi::storage::table::PyNodesTable> {
        self // Placeholder - would map filter operation
    }
}
