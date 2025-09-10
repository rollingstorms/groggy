//! Minimal StatsArray for testing

use pyo3::prelude::*;
use groggy::storage::array::StatsArray;

/// Minimal statistical array for testing
#[pyclass(name = "SimpleStatsArray", unsendable)]
#[derive(Clone)]
pub struct PySimpleStatsArray {
    inner: StatsArray<f64>,
}

impl PySimpleStatsArray {
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            inner: StatsArray::new(values),
        }
    }
}

#[pymethods]
impl PySimpleStatsArray {
    /// Get the number of elements
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("SimpleStatsArray({} elements)", self.inner.len())
    }
    
    /// Calculate the mean (average)
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
    
    /// Calculate the sum
    fn sum(&self) -> f64 {
        self.inner.sum()
    }
}