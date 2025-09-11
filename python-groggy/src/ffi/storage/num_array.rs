//! PyNumArray - Statistical array operations with numerical methods
//!
//! Wraps the Rust NumArray to provide mean, sum, std_dev, and other statistical operations

use pyo3::prelude::*;
use groggy::storage::array::NumArray;
use groggy::AttrValue;

/// Statistical array with numerical operations
#[pyclass(name = "NumArray", unsendable)]
#[derive(Clone)]
pub struct PyNumArray {
    pub(crate) inner: NumArray<f64>,
}

impl PyNumArray {
    /// Create new PyNumArray from numerical values
    pub fn new(values: Vec<f64>) -> Self {
        Self {
            inner: NumArray::new(values),
        }
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
    /// Create new NumArray from a list of numbers
    #[new]
    fn __new__(values: Vec<f64>) -> Self {
        Self {
            inner: NumArray::new(values),
        }
    }
    
    /// Get the number of elements
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Get element at index
    fn __getitem__(&self, index: isize) -> PyResult<f64> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };
        
        self.inner.get(actual_index)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(
                format!("Index {} out of range", index)
            ))
    }
    
    /// Convert to Python list
    fn to_list(&self) -> Vec<f64> {
        self.inner.iter().copied().collect()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        let preview: Vec<f64> = self.inner.iter().take(5).copied().collect();
        if self.inner.len() <= 5 {
            format!("NumArray({:?})", preview)
        } else {
            format!("NumArray({:?}... {} total)", preview, self.inner.len())
        }
    }
    
    // Statistical Methods
    
    /// Calculate the mean (average)
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
    
    /// Calculate the sum
    fn sum(&self) -> f64 {
        self.inner.sum()
    }
    
    /// Find the minimum value
    fn min(&self) -> Option<f64> {
        self.inner.min()
    }
    
    /// Find the maximum value
    fn max(&self) -> Option<f64> {
        self.inner.max()
    }
    
    /// Calculate standard deviation
    fn std(&self) -> Option<f64> {
        self.inner.std_dev()
    }
    
    /// Calculate variance
    fn var(&self) -> Option<f64> {
        self.inner.variance()
    }
    
    /// Calculate median
    fn median(&self) -> Option<f64> {
        self.inner.median()
    }
    
    /// Calculate percentile (p between 0.0 and 1.0)
    fn percentile(&self, p: f64) -> Option<f64> {
        self.inner.percentile(p)
    }
    
    /// Get first element
    fn first(&self) -> Option<f64> {
        self.inner.first().copied()
    }
    
    /// Get last element
    fn last(&self) -> Option<f64> {
        self.inner.last().copied()
    }
    
    /// Calculate correlation with another NumArray
    fn corr(&self, other: &PyNumArray) -> Option<f64> {
        self.inner.correlate(&other.inner)
    }
    
    /// Element-wise addition with another NumArray
    fn add(&self, other: &PyNumArray) -> Option<PyNumArray> {
        self.inner.add(&other.inner).map(|result| PyNumArray { inner: result })
    }
    
    /// Multiply all elements by a scalar
    fn multiply(&self, scalar: f64) -> PyNumArray {
        PyNumArray {
            inner: self.inner.multiply(scalar)
        }
    }
    
    /// Get descriptive statistics summary as a dictionary
    // TODO: Fix linking issue with describe method
    // fn describe(&self) -> PyResult<PyObject> {
    //     let summary = self.inner.describe();
    //     
    //     Python::with_gil(|py| {
    //         let dict = pyo3::types::PyDict::new(py);
    //         dict.set_item("count", summary.count)?;
    //         dict.set_item("mean", summary.mean)?;
    //         dict.set_item("std", summary.std_dev)?;
    //         dict.set_item("min", summary.min)?;
    //         dict.set_item("25%", summary.percentile_25)?;
    //         dict.set_item("50%", summary.median)?;
    //         dict.set_item("75%", summary.percentile_75)?;
    //         dict.set_item("max", summary.max)?;
    //         Ok(dict.to_object(py))
    //     })
    // }
    
    /// Create iterator
    fn __iter__(slf: PyRef<Self>) -> PyNumArrayIterator {
        PyNumArrayIterator {
            array: slf.into(),
            index: 0,
        }
    }

    // Comparison operators for boolean masking
    
    /// Greater than comparison (>) - returns Vec<bool> for boolean masking
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x > scalar).collect();
            Ok(result.to_object(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Less than comparison (<) - returns Vec<bool> for boolean masking
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x < scalar).collect();
            Ok(result.to_object(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Greater than or equal comparison (>=) - returns Vec<bool> for boolean masking
    fn __ge__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x >= scalar).collect();
            Ok(result.to_object(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Less than or equal comparison (<=) - returns Vec<bool> for boolean masking
    fn __le__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x <= scalar).collect();
            Ok(result.to_object(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Equality comparison (==) - returns Vec<bool> for boolean masking
    fn __eq__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| (x - scalar).abs() < f64::EPSILON).collect();
            Ok(result.to_object(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Not equal comparison (!=) - returns Vec<bool> for boolean masking
    fn __ne__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| (x - scalar).abs() >= f64::EPSILON).collect();
            Ok(result.to_object(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }
}

/// Iterator for PyNumArray
#[pyclass]
pub struct PyNumArrayIterator {
    array: Py<PyNumArray>,
    index: usize,
}

// Back-compat aliases
pub type PyStatsArray = PyNumArray;
pub type PyStatsArrayIterator = PyNumArrayIterator;

#[pymethods]
impl PyNumArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<f64>> {
        let array = self.array.borrow(py);
        if self.index < array.inner.len() {
            let result = array.inner.get(self.index).copied().unwrap();
            self.index += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}