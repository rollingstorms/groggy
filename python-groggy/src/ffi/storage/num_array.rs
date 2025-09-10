//! PyNumArray - Numerical array with statistical methods (preferred over StatsArray)
//!
//! Wraps the Rust StatsArray<f64> to provide mean, sum, std_dev, and other operations.

use pyo3::prelude::*;
use groggy::storage::array::StatsArray;
use groggy::AttrValue;

/// Numerical array with statistical operations
#[pyclass(name = "NumArray", unsendable)]
#[derive(Clone)]
pub struct PyNumArray {
    pub(crate) inner: StatsArray<f64>,
}

impl PyNumArray {
    /// Create new PyNumArray from numerical values
    pub fn new(values: Vec<f64>) -> Self {
        Self { inner: StatsArray::new(values) }
    }

    /// Create from AttrValues, converting numerical ones to f64
    pub fn from_attr_values(attr_values: Vec<AttrValue>) -> PyResult<Self> {
        let mut numerical_values = Vec::new();
        for attr in attr_values {
            match attr {
                AttrValue::Int(i) => numerical_values.push(i as f64),
                AttrValue::SmallInt(i) => numerical_values.push(i as f64),
                AttrValue::Float(f) => numerical_values.push(f as f64),
                AttrValue::Bool(b) => numerical_values.push(if b { 1.0 } else { 0.0 }),
                _ => return Err(pyo3::exceptions::PyTypeError::new_err(
                    "NumArray can only contain numerical values (int, float, bool)"
                )),
            }
        }
        Ok(Self::new(numerical_values))
    }
}

#[pymethods]
impl PyNumArray {
    fn __len__(&self) -> usize { self.inner.len() }
    fn is_empty(&self) -> bool { self.inner.is_empty() }

    fn __getitem__(&self, index: isize) -> PyResult<f64> {
        let len = self.inner.len() as isize;
        let actual = if index < 0 { len + index } else { index } as usize;
        self.inner.get(actual)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(format!("Index {} out of range", index)))
    }

    fn to_list(&self) -> Vec<f64> { self.inner.iter().copied().collect() }

    fn __repr__(&self) -> String {
        let preview: Vec<f64> = self.inner.iter().take(5).copied().collect();
        if self.inner.len() <= 5 {
            format!("NumArray({:?})", preview)
        } else {
            format!("NumArray({:?}... {} total)", preview, self.inner.len())
        }
    }

    // Stats
    fn mean(&self) -> Option<f64> { self.inner.mean() }
    fn sum(&self) -> f64 { self.inner.sum() }
    fn min(&self) -> Option<f64> { self.inner.min() }
    fn max(&self) -> Option<f64> { self.inner.max() }
    fn std(&self) -> Option<f64> { self.inner.std_dev() }
    fn var(&self) -> Option<f64> { self.inner.variance() }
    fn median(&self) -> Option<f64> { self.inner.median() }
    fn percentile(&self, p: f64) -> Option<f64> { self.inner.percentile(p) }
    fn first(&self) -> Option<f64> { self.inner.first().copied() }
    fn last(&self) -> Option<f64> { self.inner.last().copied() }
    fn corr(&self, other: &PyNumArray) -> Option<f64> { self.inner.correlate(&other.inner) }
    fn add(&self, other: &PyNumArray) -> Option<PyNumArray> { self.inner.add(&other.inner).map(|inner| PyNumArray { inner }) }
    fn multiply(&self, scalar: f64) -> PyNumArray { PyNumArray { inner: self.inner.multiply(scalar) } }

    fn describe(&self) -> PyResult<PyObject> {
        let summary = self.inner.describe();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("count", summary.count)?;
            dict.set_item("mean", summary.mean)?;
            dict.set_item("std", summary.std_dev)?;
            dict.set_item("min", summary.min)?;
            dict.set_item("25%", summary.percentile_25)?;
            dict.set_item("50%", summary.median)?;
            dict.set_item("75%", summary.percentile_75)?;
            dict.set_item("max", summary.max)?;
            Ok(dict.to_object(py))
        })
    }

    fn __iter__(slf: PyRef<Self>) -> PyNumArrayIterator { PyNumArrayIterator { array: slf.into(), index: 0 } }
}

/// Iterator for PyNumArray
#[pyclass]
pub struct PyNumArrayIterator { array: Py<PyNumArray>, index: usize }

#[pymethods]
impl PyNumArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> { slf }
    fn __next__(&mut self, py: Python) -> PyResult<Option<f64>> {
        let array = self.array.borrow(py);
        if self.index < array.inner.len() {
            let v = array.inner.get(self.index).copied();
            self.index += 1;
            Ok(v)
        } else {
            Ok(None)
        }
    }
}

