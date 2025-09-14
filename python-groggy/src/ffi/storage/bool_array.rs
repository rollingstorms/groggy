//! Python FFI bindings for BoolArray
//!
//! Provides Python access to efficient boolean array operations

use pyo3::prelude::*;
use pyo3::exceptions::{PyIndexError, PyValueError, PyTypeError};
use pyo3::types::{PyList, PyTuple};
use groggy::storage::array::BoolArray;
use std::collections::HashMap;

/// Python wrapper for BoolArray
#[pyclass(name = "BoolArray", unsendable)]
pub struct PyBoolArray {
    pub(crate) inner: BoolArray,
}

#[pymethods]
impl PyBoolArray {
    /// Create a new BoolArray from a Python list or other iterable
    #[new]
    pub fn new(data: &PyAny) -> PyResult<Self> {
        // Try to extract as a list of booleans
        if let Ok(py_list) = data.downcast::<PyList>() {
            let mut bool_data = Vec::new();
            for item in py_list {
                if let Ok(val) = item.extract::<bool>() {
                    bool_data.push(val);
                } else {
                    return Err(PyTypeError::new_err(
                        format!("Cannot convert {} to boolean", item)
                    ));
                }
            }
            Ok(Self {
                inner: BoolArray::new(bool_data),
            })
        } else {
            Err(PyTypeError::new_err("Expected a list or iterable of booleans"))
        }
    }
    
    /// Get the length of the boolean array
    #[getter]
    pub fn length(&self) -> usize {
        self.inner.len()
    }
    
    /// Python len() support
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Advanced index access supporting multiple indexing types
    pub fn __getitem__(&self, py: Python, index: &PyAny) -> PyResult<PyObject> {
        use groggy::storage::array::AdvancedIndexing;
        use crate::ffi::utils::indexing::python_index_to_slice_index;
        
        // Handle simple integer case directly for performance
        if let Ok(int_val) = index.extract::<isize>() {
            let len = self.inner.len() as isize;
            let actual_index = if int_val < 0 {
                (len + int_val) as usize
            } else {
                int_val as usize
            };
            
            return self.inner.get(actual_index)
                .map(|val| val.to_object(py))
                .ok_or_else(|| PyIndexError::new_err(format!("Index {} out of bounds", int_val)));
        }
        
        // Handle advanced indexing cases
        let slice_index = python_index_to_slice_index(py, index)?;
        
        match self.inner.get_slice(&slice_index) {
            Ok(sliced_array) => {
                let py_array = PyBoolArray { inner: sliced_array };
                Ok(py_array.into_py(py))
            }
            Err(e) => Err(PyIndexError::new_err(format!("{}", e)))
        }
    }
    
    /// Iterator support
    pub fn __iter__(slf: PyRef<Self>) -> PyResult<PyBoolArrayIterator> {
        Ok(PyBoolArrayIterator {
            array: slf.inner.to_vec(),
            index: 0,
        })
    }
    
    /// Count the number of True values
    pub fn count(&self) -> usize {
        self.inner.count()
    }
    
    /// Count the number of False values  
    pub fn count_false(&self) -> usize {
        self.inner.count_false()
    }
    
    /// Check if any value is True
    pub fn any(&self) -> bool {
        self.inner.any()
    }
    
    /// Check if all values are True
    pub fn all(&self) -> bool {
        self.inner.all()
    }
    
    /// Get percentage of True values
    pub fn percentage(&self) -> f64 {
        self.inner.percentage()
    }
    
    /// Get indices where value is True
    pub fn nonzero(&self) -> Vec<usize> {
        self.inner.nonzero()
    }
    
    /// Alias for nonzero()
    pub fn to_indices(&self) -> Vec<usize> {
        self.inner.to_indices()
    }
    
    /// Get indices where value is False
    pub fn false_indices(&self) -> Vec<usize> {
        self.inner.false_indices()
    }
    
    /// Convert to Python list
    pub fn to_list(&self) -> Vec<bool> {
        self.inner.to_vec()
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
    
    /// Debug representation  
    pub fn __repr__(&self) -> String {
        let len = self.inner.len();
        
        if len == 0 {
            return "BoolArray(dtype: bool)\n[]".to_string();
        }
        
        let true_count = self.inner.count();
        let false_count = self.inner.count_false();
        let percentage = self.inner.percentage();
        
        let max_display = 10;
        let show_all = len <= max_display;
        
        let mut output = format!("BoolArray(dtype: bool)\n");
        
        if show_all {
            // Show all elements with indices
            for (i, &value) in self.inner.iter().enumerate() {
                let symbol = if value { "T" } else { "F" };
                output.push_str(&format!("[{:>3}] {}\n", i, symbol));
            }
        } else {
            // Show first few and last few with ellipsis
            for (i, &value) in self.inner.iter().take(5).enumerate() {
                let symbol = if value { "T" } else { "F" };
                output.push_str(&format!("[{:>3}] {}\n", i, symbol));
            }
            output.push_str("      ...\n");
            let start_idx = len - 3;
            for (i, &value) in self.inner.iter().skip(start_idx).enumerate() {
                let symbol = if value { "T" } else { "F" };
                output.push_str(&format!("[{:>3}] {}\n", start_idx + i, symbol));
            }
        }
        
        // Add statistical summary
        output.push_str(&format!("\nStats: {} True ({:.1}%), {} False ({:.1}%)", 
            true_count, percentage, false_count, 100.0 - percentage));
        
        output
    }
    
    /// Boolean AND operation (& operator)
    pub fn __and__(&self, other: &PyBoolArray) -> PyResult<PyBoolArray> {
        Ok(PyBoolArray {
            inner: &self.inner & &other.inner,
        })
    }
    
    /// Boolean OR operation (| operator)
    pub fn __or__(&self, other: &PyBoolArray) -> PyResult<PyBoolArray> {
        Ok(PyBoolArray {
            inner: &self.inner | &other.inner,
        })
    }
    
    /// Boolean NOT operation (~ operator)
    pub fn __invert__(&self) -> PyResult<PyBoolArray> {
        Ok(PyBoolArray {
            inner: !&self.inner,
        })
    }
    
    /// Equality comparison
    pub fn __eq__(&self, other: &PyBoolArray) -> bool {
        if self.inner.len() != other.inner.len() {
            return false;
        }
        
        self.inner.iter()
            .zip(other.inner.iter())
            .all(|(&a, &b)| a == b)
    }
    
    /// Hash support (for use in sets/dicts)
    pub fn __hash__(&self) -> isize {
        // Simple hash based on content
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        
        self.inner.len().hash(&mut hasher);
        for &val in self.inner.iter() {
            val.hash(&mut hasher);
        }
        
        hasher.finish() as isize
    }
    
    /// Apply this boolean mask to a Python list
    pub fn apply_mask(&self, data: &PyList) -> PyResult<Vec<PyObject>> {
        if data.len() != self.inner.len() {
            return Err(PyValueError::new_err(
                format!("Length mismatch: mask has {} elements, data has {}", 
                    self.inner.len(), data.len())
            ));
        }
        
        let mut result = Vec::new();
        for (i, &mask_val) in self.inner.iter().enumerate() {
            if mask_val {
                result.push(data.get_item(i)?.to_object(data.py()));
            }
        }
        
        Ok(result)
    }
    
    /// Get summary statistics as a dictionary
    pub fn stats(&self) -> PyResult<HashMap<String, PyObject>> {
        let py = unsafe { Python::assume_gil_acquired() };
        let mut stats = HashMap::new();
        
        stats.insert("count".to_string(), self.inner.count().to_object(py));
        stats.insert("count_false".to_string(), self.inner.count_false().to_object(py));
        stats.insert("length".to_string(), self.inner.len().to_object(py));
        stats.insert("percentage".to_string(), self.inner.percentage().to_object(py));
        stats.insert("any".to_string(), self.inner.any().to_object(py));
        stats.insert("all".to_string(), self.inner.all().to_object(py));
        
        Ok(stats)
    }
}

/// Iterator for BoolArray
#[pyclass]
pub struct PyBoolArrayIterator {
    array: Vec<bool>,
    index: usize,
}

#[pymethods]
impl PyBoolArrayIterator {
    pub fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    pub fn __next__(mut slf: PyRefMut<Self>) -> Option<bool> {
        if slf.index < slf.array.len() {
            let val = slf.array[slf.index];
            slf.index += 1;
            Some(val)
        } else {
            None
        }
    }
}

/// Helper functions for creating BoolArrays
#[pyfunction]
pub fn bool_array(data: &PyAny) -> PyResult<PyBoolArray> {
    PyBoolArray::new(data)
}

/// Create a BoolArray of all True values
#[pyfunction] 
pub fn ones_bool(length: usize) -> PyResult<PyBoolArray> {
    Ok(PyBoolArray {
        inner: BoolArray::new(vec![true; length]),
    })
}

/// Create a BoolArray of all False values
#[pyfunction]
pub fn zeros_bool(length: usize) -> PyResult<PyBoolArray> {
    Ok(PyBoolArray {
        inner: BoolArray::new(vec![false; length]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;
    
    #[test]
    fn test_pybool_array_creation() {
        Python::with_gil(|py| {
            let data = PyList::new(py, &[true, false, true]);
            let bool_array = PyBoolArray::new(data).unwrap();
            
            assert_eq!(bool_array.length(), 3);
            assert_eq!(bool_array.__getitem__(0).unwrap(), true);
            assert_eq!(bool_array.__getitem__(1).unwrap(), false);
            assert_eq!(bool_array.__getitem__(2).unwrap(), true);
        });
    }
    
    #[test] 
    fn test_pybool_array_operations() {
        Python::with_gil(|py| {
            let data1 = PyList::new(py, &[true, false, true]);
            let data2 = PyList::new(py, &[false, true, true]);
            let array1 = PyBoolArray::new(data1).unwrap();
            let array2 = PyBoolArray::new(data2).unwrap();
            
            // Test AND
            let and_result = array1.__and__(&array2).unwrap();
            assert_eq!(and_result.to_list(), vec![false, false, true]);
            
            // Test OR
            let or_result = array1.__or__(&array2).unwrap();
            assert_eq!(or_result.to_list(), vec![true, true, true]);
            
            // Test NOT
            let not_result = array1.__invert__().unwrap();
            assert_eq!(not_result.to_list(), vec![false, true, false]);
        });
    }
}