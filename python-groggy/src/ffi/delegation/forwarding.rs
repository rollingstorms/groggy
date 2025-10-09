//! Forwarding implementations for arrays and iterators
//!
//! This module implements trait forwarding for all array types,
//! enabling universal method availability across the delegation system.

use super::traits::{BaseArrayOps, DelegatingIterator, NumArrayOps};
use pyo3::PyResult;

/// Generic forwarding array that delegates to inner array implementations
pub struct ForwardingArray<T> {
    items: Vec<T>,
}

impl<T> ForwardingArray<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { items }
    }

    /// Create a delegating iterator that yields references to items
    pub fn iter_refs(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Create a delegating iterator that yields cloned items (if Clone is implemented)
    pub fn iter_cloned(&self) -> DelegatingIterator<T>
    where
        T: Clone + 'static,
    {
        let cloned_items: Vec<T> = self.items.to_vec();
        DelegatingIterator::new(cloned_items.into_iter())
    }

    /// Create an owning iterator from this array
    pub fn into_iter(self) -> DelegatingIterator<T>
    where
        T: 'static,
    {
        DelegatingIterator::new(self.items.into_iter())
    }
}

impl<T> BaseArrayOps for ForwardingArray<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.items.get(index)
    }

    fn iter<'a>(&'a self) -> Box<dyn Iterator<Item = &'a Self::Item> + 'a> {
        Box::new(self.items.iter())
    }

    fn filter<F>(&self, predicate: F) -> PyResult<Self>
    where
        F: Fn(&Self::Item) -> bool,
        Self: Sized,
        T: Clone,
    {
        let filtered_items: Vec<T> = self
            .items
            .iter()
            .filter(|item| predicate(item))
            .cloned()
            .collect();
        Ok(ForwardingArray::new(filtered_items))
    }

    fn map<U, F>(&self, f: F) -> PyResult<DelegatingIterator<U>>
    where
        F: Fn(&Self::Item) -> U + 'static,
        U: 'static,
    {
        let mapped_items: Vec<U> = self.items.iter().map(f).collect();
        Ok(DelegatingIterator::new(mapped_items.into_iter()))
    }

    fn take(&self, n: usize) -> PyResult<Self>
    where
        Self: Sized,
        T: Clone,
    {
        let taken_items: Vec<T> = self.items.iter().take(n).cloned().collect();
        Ok(ForwardingArray::new(taken_items))
    }

    fn skip(&self, n: usize) -> PyResult<Self>
    where
        Self: Sized,
        T: Clone,
    {
        let skipped_items: Vec<T> = self.items.iter().skip(n).cloned().collect();
        Ok(ForwardingArray::new(skipped_items))
    }
}

// Implement statistical operations for numerical array types
impl NumArrayOps for ForwardingArray<f64> {
    fn mean(&self) -> PyResult<Option<f64>> {
        if self.items.is_empty() {
            return Ok(None);
        }

        let sum: f64 = self.items.iter().sum();
        Ok(Some(sum / self.items.len() as f64))
    }

    fn sum(&self) -> PyResult<f64> {
        Ok(self.items.iter().sum())
    }

    fn min(&self) -> PyResult<Option<f64>> {
        Ok(self
            .items
            .iter()
            .cloned()
            .fold(None, |acc, x| Some(acc.map_or(x, |a| a.min(x)))))
    }

    fn max(&self) -> PyResult<Option<f64>> {
        Ok(self
            .items
            .iter()
            .cloned()
            .fold(None, |acc, x| Some(acc.map_or(x, |a| a.max(x)))))
    }

    fn std_dev(&self) -> PyResult<Option<f64>> {
        match self.variance()? {
            Some(var) => Ok(Some(var.sqrt())),
            None => Ok(None),
        }
    }

    fn median(&self) -> PyResult<Option<f64>> {
        if self.items.is_empty() {
            return Ok(None);
        }

        let mut sorted = self.items.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len();
        if len % 2 == 0 {
            Ok(Some((sorted[len / 2 - 1] + sorted[len / 2]) / 2.0))
        } else {
            Ok(Some(sorted[len / 2]))
        }
    }

    fn variance(&self) -> PyResult<Option<f64>> {
        if self.items.len() < 2 {
            return Ok(None);
        }

        let mean = match self.mean()? {
            Some(m) => m,
            None => return Ok(None),
        };

        let var: f64 = self.items.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.items.len() - 1) as f64;

        Ok(Some(var))
    }

    fn percentile(&self, p: f64) -> PyResult<Option<f64>> {
        if self.items.is_empty() || !(0.0..=100.0).contains(&p) {
            return Ok(None);
        }

        let mut sorted = self.items.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        Ok(Some(sorted[index.min(sorted.len() - 1)]))
    }

    fn add(&self, other: &Self) -> PyResult<Self> {
        if self.items.len() != other.items.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Arrays must have same length for element-wise addition",
            ));
        }

        let result: Vec<f64> = self
            .items
            .iter()
            .zip(other.items.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(ForwardingArray::new(result))
    }

    fn multiply(&self, scalar: f64) -> PyResult<Self> {
        let result: Vec<f64> = self.items.iter().map(|x| x * scalar).collect();
        Ok(ForwardingArray::new(result))
    }

    fn correlate(&self, other: &Self) -> PyResult<Option<f64>> {
        if self.items.len() != other.items.len() || self.items.len() < 2 {
            return Ok(None);
        }

        let mean_x = self.mean()?.unwrap_or(0.0);
        let mean_y = other.mean()?.unwrap_or(0.0);

        let numerator: f64 = self
            .items
            .iter()
            .zip(other.items.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();

        let sum_sq_x: f64 = self.items.iter().map(|x| (x - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = other.items.iter().map(|y| (y - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            Ok(None)
        } else {
            Ok(Some(numerator / denominator))
        }
    }
}

/// Generic forwarding iterator that provides delegation capabilities
pub struct ForwardingIterator<T> {
    inner: DelegatingIterator<T>,
}

impl<T> ForwardingIterator<T> {
    pub fn new<I: Iterator<Item = T> + 'static>(iter: I) -> Self {
        Self {
            inner: DelegatingIterator::new(iter),
        }
    }

    /// Forward to the inner delegating iterator
    pub fn into_delegating_iter(self) -> DelegatingIterator<T> {
        self.inner
    }
}

impl<T> Iterator for ForwardingIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

// Implement forwarding for specific array types used in the system
impl ForwardingArray<crate::ffi::subgraphs::subgraph::PySubgraph> {
    /// Forward neighborhood operation to each subgraph
    pub fn neighborhood(
        &self,
        _radius: Option<usize>,
    ) -> PyResult<ForwardingArray<crate::ffi::subgraphs::subgraph::PySubgraph>> {
        // Placeholder implementation - would apply neighborhood to each subgraph
        // For now, return self to demonstrate the pattern
        Ok(ForwardingArray::new(vec![]))
    }

    /// Forward table operation to each subgraph  
    pub fn table(&self) -> PyResult<ForwardingArray<crate::ffi::storage::table::PyNodesTable>> {
        // Placeholder implementation - would convert each subgraph to table
        Ok(ForwardingArray::new(vec![]))
    }
}

impl ForwardingArray<crate::ffi::storage::table::PyNodesTable> {
    /// Forward aggregation to each table
    pub fn agg(
        &self,
        _spec: &str,
    ) -> PyResult<ForwardingArray<crate::ffi::storage::table::PyBaseTable>> {
        // Placeholder implementation - would aggregate each table
        Ok(ForwardingArray::new(vec![]))
    }

    /// Forward filter operation to each table
    pub fn filter_tables(
        &self,
        _expr: &str,
    ) -> PyResult<ForwardingArray<crate::ffi::storage::table::PyNodesTable>> {
        // Placeholder implementation - would filter each table
        Ok(ForwardingArray::new(vec![]))
    }
}
