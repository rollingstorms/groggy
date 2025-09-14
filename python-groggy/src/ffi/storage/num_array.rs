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

/// Integer array for node IDs and other integer data
#[pyclass(name = "IntArray", unsendable)]
#[derive(Clone)]
pub struct PyIntArray {
    pub(crate) inner: NumArray<i64>,
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

impl PyIntArray {
    /// Create new PyIntArray from integer values
    pub fn new(values: Vec<i64>) -> Self {
        Self {
            inner: NumArray::new(values),
        }
    }
    
    /// Create new PyIntArray from integer values (alias for compatibility)
    pub fn from_vec(values: Vec<i64>) -> Self {
        Self::new(values)
    }
    
    /// Create from node IDs (usize converted to i64)
    pub fn from_node_ids(node_ids: Vec<usize>) -> Self {
        let values: Vec<i64> = node_ids.into_iter().map(|id| id as i64).collect();
        Self::new(values)
    }
    
    /// Create from AttrValues, converting integer ones to i64
    pub fn from_attr_values(attr_values: Vec<AttrValue>) -> PyResult<Self> {
        let mut integer_values = Vec::new();
        
        for attr in attr_values {
            match attr {
                AttrValue::Int(i) => integer_values.push(i),
                AttrValue::SmallInt(i) => integer_values.push(i as i64),
                AttrValue::Bool(b) => integer_values.push(if b { 1 } else { 0 }),
                _ => return Err(pyo3::exceptions::PyTypeError::new_err(
                    "IntArray can only contain integer values (int, bool)"
                )),
            }
        }
        
        Ok(Self::new(integer_values))
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
    /// Advanced index access supporting multiple indexing types
    fn __getitem__(&self, py: Python, index: &PyAny) -> PyResult<PyObject> {
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
                .copied()
                .map(|val| val.to_object(py))
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(
                    format!("Index {} out of range", int_val)
                ));
        }
        
        // Handle advanced indexing cases
        let slice_index = python_index_to_slice_index(py, index)?;
        
        match self.inner.get_slice(&slice_index) {
            Ok(sliced_array) => {
                let py_array = PyNumArray { inner: sliced_array };
                Ok(py_array.into_py(py))
            }
            Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e)))
        }
    }
    
    /// Convert to Python list
    fn to_list(&self) -> Vec<f64> {
        self.inner.iter().copied().collect()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        
        if len == 0 {
            return "NumArray(dtype: f64)\n[]".to_string();
        }
        
        let max_display = 10;
        let show_all = len <= max_display;
        
        let mut output = format!("NumArray(dtype: f64)\n");
        
        if show_all {
            // Show all elements with indices
            for (i, &value) in self.inner.iter().enumerate() {
                output.push_str(&format!("[{:>3}] {:.6}\n", i, value));
            }
        } else {
            // Show first few and last few with ellipsis
            for (i, &value) in self.inner.iter().take(5).enumerate() {
                output.push_str(&format!("[{:>3}] {:.6}\n", i, value));
            }
            output.push_str("      ...\n");
            let start_idx = len - 3;
            for (i, &value) in self.inner.iter().skip(start_idx).enumerate() {
                output.push_str(&format!("[{:>3}] {:.6}\n", start_idx + i, value));
            }
        }
        
        output.trim_end().to_string()
    }
    
    /// Rich HTML representation for Jupyter notebooks
    /// 
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells, similar to pandas Series.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;
        
        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }
    
    /// Launch interactive streaming table view in browser
    /// 
    /// Converts the array into a table format and launches a streaming
    /// interactive view in the browser. The table will have two columns:
    /// 'index' and 'value' for easy exploration of the array data.
    /// 
    /// Returns:
    ///     str: URL of the interactive table interface
    pub fn interactive(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let table = self.to_table()?;
        
        // Use the table's interactive method
        table.interactive()
    }
    
    /// Generate embedded iframe HTML for Jupyter notebooks
    /// 
    /// Creates an interactive streaming table representation of the array
    /// that can be embedded directly in a Jupyter notebook cell.
    /// 
    /// Returns:
    ///     str: HTML iframe code for embedding in Jupyter
    pub fn interactive_embed(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let table = self.to_table()?;
        
        // Use the table's interactive_embed method
        table.interactive_embed()
    }
    
    /// Convert array to table format for streaming visualization
    /// 
    /// Creates a BaseTable with 'index' and 'value' columns from the array data.
    /// This enables the array to use the rich streaming table infrastructure.
    fn to_table(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;
        
        // Create columns for the table
        let mut columns = HashMap::new();
        
        // Index column (0, 1, 2, ...)
        let indices: Vec<AttrValue> = (0..self.inner.len())
            .map(|i| AttrValue::SmallInt(i as i32))
            .collect();
        columns.insert("index".to_string(), groggy::storage::array::BaseArray::new(indices));
        
        // Value column (array data)
        let values: Vec<AttrValue> = self.inner.iter()
            .map(|&val| AttrValue::Float(val as f32))
            .collect();
        columns.insert("value".to_string(), groggy::storage::array::BaseArray::new(values));
        
        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e)))?;
        
        Ok(crate::ffi::storage::table::PyBaseTable::from_table(base_table))
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
    
    /// Greater than comparison (>) - returns BoolArray for boolean masking
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x > scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Less than comparison (<) - returns BoolArray for boolean masking
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x < scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Greater than or equal comparison (>=) - returns BoolArray for boolean masking
    fn __ge__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x >= scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Less than or equal comparison (<=) - returns BoolArray for boolean masking
    fn __le__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x <= scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Equality comparison (==) - returns BoolArray for boolean masking
    fn __eq__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| (x - scalar).abs() < f64::EPSILON).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }

    /// Not equal comparison (!=) - returns BoolArray for boolean masking
    fn __ne__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<f64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| (x - scalar).abs() >= f64::EPSILON).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "NumArray comparison requires a numeric value"
            ))
        }
    }
}

#[pymethods]
impl PyIntArray {
    /// Create new IntArray from a list of integers
    #[new]
    fn __new__(values: Vec<i64>) -> Self {
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
    
    /// Get element at index or slice
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        use pyo3::exceptions::{PyIndexError, PyValueError};
        use pyo3::types::PySlice;
        
        // Integer indexing: arr[5]
        if let Ok(index) = key.extract::<isize>() {
            let len = self.inner.len() as isize;
            let actual_index = if index < 0 {
                (len + index) as usize
            } else {
                index as usize
            };
            
            match self.inner.get(actual_index) {
                Some(&value) => Ok(value.to_object(py)),
                None => Err(PyIndexError::new_err(format!("Index {} out of range", index))),
            }
        }
        // Slice indexing: arr[start:end], arr[start:end:step]
        else if let Ok(slice) = key.downcast::<PySlice>() {
            let len = self.inner.len();
            let indices = slice.indices(
                len.try_into()
                    .map_err(|_| PyValueError::new_err("Array too large for slice"))?,
            )?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step;

            let mut result_values = Vec::new();

            if step == 1 {
                // Simple slice [start:stop]
                for i in start..stop.min(len) {
                    if let Some(&value) = self.inner.get(i) {
                        result_values.push(value);
                    }
                }
            } else if step > 1 {
                // Step slice [start:stop:step]
                let mut i = start;
                while i < stop.min(len) {
                    if let Some(&value) = self.inner.get(i) {
                        result_values.push(value);
                    }
                    i += step as usize;
                }
            } else {
                return Err(PyValueError::new_err("Negative step not supported"));
            }

            // Return new PyIntArray with sliced data
            let result_array = PyIntArray::from_vec(result_values);
            Ok(result_array.into_py(py))
        }
        else {
            Err(PyIndexError::new_err(
                "Index must be int or slice"
            ))
        }
    }
    
    /// Convert to Python list
    fn to_list(&self) -> Vec<i64> {
        self.inner.iter().copied().collect()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        let preview: Vec<i64> = self.inner.iter().take(3).copied().collect();
        
        let preview_str = if self.inner.len() <= 3 {
            format!("{:?}", preview)
        } else {
            format!("{:?}...", preview)
        };
        
        format!(
            "IntArray[{}] {} (dtype: i64)\n💡 Use .interactive() for rich table view or .astype('float64') for numerical operations",
            len, preview_str
        )
    }
    
    /// Convert to float NumArray for numerical operations
    fn astype(&self, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "float64" | "f64" => {
                let float_values: Vec<f64> = self.inner.iter().map(|&x| x as f64).collect();
                let num_array = PyNumArray::new(float_values);
                Python::with_gil(|py| Ok(num_array.into_py(py)))
            }
            "int64" | "i64" => {
                Python::with_gil(|py| Ok(self.clone().into_py(py)))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported dtype: {}. Supported: 'int64', 'float64'", dtype)
            ))
        }
    }
    
    /// Calculate the sum
    fn sum(&self) -> i64 {
        self.inner.iter().sum()
    }
    
    /// Find the minimum value
    fn min(&self) -> Option<i64> {
        self.inner.iter().min().copied()
    }
    
    /// Find the maximum value
    fn max(&self) -> Option<i64> {
        self.inner.iter().max().copied()
    }
    
    /// Calculate the mean as float
    fn mean(&self) -> Option<f64> {
        if self.inner.is_empty() {
            None
        } else {
            let sum: i64 = self.inner.iter().sum();
            Some(sum as f64 / self.inner.len() as f64)
        }
    }
    
    /// Count unique values
    fn nunique(&self) -> usize {
        use std::collections::HashSet;
        let unique: HashSet<_> = self.inner.iter().collect();
        unique.len()
    }
    
    /// Equality comparison (==) - returns BoolArray for boolean masking
    fn __eq__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<i64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x == scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else if other.is_none() {
            // Node IDs are never None, so comparison with None is always False
            let result: Vec<bool> = vec![false; self.inner.len()];
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "IntArray comparison requires an integer value or None"
            ))
        }
    }
    
    /// Not equal comparison (!=) - returns BoolArray for boolean masking
    fn __ne__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<i64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x != scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else if other.is_none() {
            // Node IDs are never None, so != None is always True
            let result: Vec<bool> = vec![true; self.inner.len()];
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "IntArray comparison requires an integer value or None"
            ))
        }
    }
    
    /// Greater than comparison (>) - returns BoolArray for boolean masking
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<i64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x > scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else if other.is_none() {
            // Integers are never greater than None
            let result: Vec<bool> = vec![false; self.inner.len()];
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "IntArray comparison requires an integer value or None"
            ))
        }
    }
    
    /// Less than comparison (<) - returns BoolArray for boolean masking
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        if let Ok(scalar) = other.extract::<i64>() {
            let result: Vec<bool> = self.inner.iter().map(|&x| x < scalar).collect();
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else if other.is_none() {
            // Integers are never less than None
            let result: Vec<bool> = vec![false; self.inner.len()];
            // Convert Vec<bool> to BoolArray
            let bool_array = groggy::storage::BoolArray::new(result);
            let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
            Ok(py_bool_array.into_py(py))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "IntArray comparison requires an integer value or None"
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