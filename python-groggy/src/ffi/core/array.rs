//! Array FFI Bindings
//!
//! Python bindings for statistical arrays and matrices.

use groggy::core::array::{GraphArray, StatsSummary};
use groggy::AttrValue as RustAttrValue;
use pyo3::exceptions::{PyImportError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

// Use utility functions from utils module
use crate::ffi::utils::{attr_value_to_python_value, python_value_to_attr_value};

/// Native performance-oriented GraphArray for statistical operations
#[pyclass(name = "GraphArray")]
pub struct PyGraphArray {
    pub inner: GraphArray,
}

#[pymethods]
impl PyGraphArray {
    /// Create a new GraphArray from a list of values
    #[new]
    fn new(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            Ok(PyGraphArray {
                inner: GraphArray::from_vec(attr_values),
            })
        })
    }

    // === LIST COMPATIBILITY ===

    /// Get the number of elements (len())
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Advanced indexing support: arr[i], arr[start:end], arr[start:end:step], arr[mask]
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Single integer indexing: arr[5]
        if let Ok(index) = key.extract::<isize>() {
            let len = self.inner.len() as isize;

            // Handle negative indexing (Python-style)
            let actual_index = if index < 0 { len + index } else { index };

            // Check bounds
            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Index out of range"));
            }

            match self.inner.get(actual_index as usize) {
                Some(attr_value) => attr_value_to_python_value(py, attr_value),
                None => Err(PyIndexError::new_err("Index out of range")),
            }
        }
        // Slice indexing: arr[start:end], arr[start:end:step]
        else if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
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
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                }
            } else if step > 1 {
                // Step slice [start:stop:step]
                let mut i = start;
                while i < stop.min(len) {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                    i += step as usize;
                }
            } else {
                return Err(PyValueError::new_err("Negative step not supported"));
            }

            // Return new GraphArray with sliced data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        }
        // List of indices: arr[[1, 3, 5]]
        else if let Ok(indices) = key.extract::<Vec<isize>>() {
            let len = self.inner.len() as isize;
            let mut result_values = Vec::new();

            for &index in &indices {
                let actual_index = if index < 0 { len + index } else { index };

                if actual_index >= 0 && actual_index < len {
                    if let Some(attr_value) = self.inner.get(actual_index as usize) {
                        result_values.push(attr_value.clone());
                    }
                } else {
                    return Err(PyIndexError::new_err("Index out of range"));
                }
            }

            // Return new GraphArray with selected data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        }
        // Boolean mask indexing: arr[mask] where mask is a list of booleans
        else if let Ok(mask) = key.extract::<Vec<bool>>() {
            let len = self.inner.len();
            if mask.len() != len {
                return Err(PyValueError::new_err(
                    "Boolean mask length must match array length",
                ));
            }

            let mut result_values = Vec::new();
            for (i, &include) in mask.iter().enumerate() {
                if include {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                }
            }

            // Return new GraphArray with masked data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        } else {
            Err(PyTypeError::new_err(
                "Index must be int, slice, list of ints, or list of bools",
            ))
        }
    }

    /// String representation with rich display formatting
    fn __repr__(&self, py: Python) -> PyResult<String> {
        // Try rich display formatting first, with graceful fallback
        match self._try_rich_display(py) {
            Ok(formatted) => Ok(formatted),
            Err(_) => {
                // Fallback to simple representation
                let len = self.inner.len();
                let dtype = self._get_dtype();
                Ok(format!("GraphArray(len={}, dtype={})", len, dtype))
            }
        }
    }

    /// Try to use rich display formatting
    fn _try_rich_display(&self, py: Python) -> PyResult<String> {
        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_array function from Python
        let groggy_module = py.import("groggy")?;
        let format_array = groggy_module.getattr("format_array")?;

        // Call the Python formatter
        let result = format_array.call1((display_data,))?;
        let formatted_str: String = result.extract()?;

        Ok(formatted_str)
    }

    /// String representation (same as __repr__ for consistency)
    fn __str__(&self, py: Python) -> PyResult<String> {
        self.__repr__(py)
    }

    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Simple, effective HTML display for GraphArray
        let len = self.inner.len();
        let dtype = self._get_dtype();

        if len == 0 {
            return Ok(format!(
                "<div style='font-family: monospace; color: #666;'><strong>GraphArray:</strong> 0 elements, dtype: {}</div>",
                dtype
            ));
        }

        // Show first few elements
        let display_count = std::cmp::min(len, 10);
        let mut elements = Vec::new();

        for i in 0..display_count {
            if let Some(value) = self.inner.get(i) {
                let display_value = match value {
                    groggy::AttrValue::Int(i) => i.to_string(),
                    groggy::AttrValue::SmallInt(i) => i.to_string(),
                    groggy::AttrValue::Float(f) => format!("{:.3}", f),
                    groggy::AttrValue::Text(s) => format!("'{}'", s),
                    groggy::AttrValue::Bool(b) => b.to_string(),
                    groggy::AttrValue::CompactText(cs) => format!("'{}'", cs.as_str()),
                    _ => format!("{:?}", value),
                };
                elements.push(display_value);
            }
        }

        let elements_str = elements.join(", ");
        let ellipsis = if len > display_count { ", ..." } else { "" };

        Ok(format!(
            "<div style='font-family: monospace; padding: 8px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9;'>\
            <div style='font-weight: bold; margin-bottom: 4px;'>GraphArray: {} elements, dtype: {}</div>\
            <div style='color: #333;'>[{}{}]</div>\
            </div>",
            len, dtype, elements_str, ellipsis
        ))
    }

    /// Try to use rich HTML display formatting
    fn _try_rich_html_display(&self, py: Python) -> PyResult<String> {
        // Get display data for formatting
        let display_data = self._get_display_data(py)?;

        // Import the format_array_html function from Python
        let groggy_module = py.import("groggy")?;
        let display_module = groggy_module.getattr("display")?;
        let format_array_html = display_module.getattr("format_array_html")?;

        // Call the Python HTML formatter
        let result = format_array_html.call1((display_data,))?;
        let html_str: String = result.extract()?;

        Ok(html_str)
    }

    /// Iterator support (for value in array)
    fn __iter__(slf: PyRef<Self>) -> GraphArrayIterator {
        GraphArrayIterator {
            array: slf.inner.clone(),
            index: 0,
        }
    }

    /// Convert to plain Python list
    pub fn to_list(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut py_values = Vec::with_capacity(self.inner.len());

        for attr_value in self.inner.iter() {
            py_values.push(attr_value_to_python_value(py, attr_value)?);
        }

        Ok(py_values)
    }

    // === STATISTICAL OPERATIONS ===

    /// Calculate mean (average) of numeric values
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }

    /// Calculate standard deviation of numeric values
    fn std(&self) -> Option<f64> {
        self.inner.std()
    }

    /// Get minimum value
    fn min(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.inner.min() {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None),
        }
    }

    /// Get maximum value
    fn max(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.inner.max() {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None),
        }
    }

    /// Count non-null values
    fn count(&self) -> usize {
        let values = self.inner.materialize();
        values
            .iter()
            .filter(|value| !matches!(value, groggy::AttrValue::Null))
            .count()
    }

    /// Check if array contains any null values
    fn has_null(&self) -> bool {
        let values = self.inner.materialize();
        values
            .iter()
            .any(|value| matches!(value, groggy::AttrValue::Null))
    }

    /// Count null values
    fn null_count(&self) -> usize {
        let values = self.inner.materialize();
        values
            .iter()
            .filter(|value| matches!(value, groggy::AttrValue::Null))
            .count()
    }

    /// Drop null values, returning a new array
    fn drop_na(&self, py: Python) -> PyResult<PyObject> {
        let values = self.inner.materialize();
        let non_null_values: Vec<groggy::AttrValue> = values
            .iter()
            .filter(|value| !matches!(value, groggy::AttrValue::Null))
            .cloned()
            .collect();

        let new_array = groggy::core::array::GraphArray::from_vec(non_null_values);
        let py_array = PyGraphArray::from_graph_array(new_array);
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Fill null values with a specified value, returning a new array
    fn fill_na(&self, py: Python, fill_value: &PyAny) -> PyResult<PyObject> {
        let fill_attr_value = crate::ffi::utils::python_value_to_attr_value(fill_value)?;
        let values = self.inner.materialize();

        let filled_values: Vec<groggy::AttrValue> = values
            .iter()
            .map(|value| {
                if matches!(value, groggy::AttrValue::Null) {
                    fill_attr_value.clone()
                } else {
                    value.clone()
                }
            })
            .collect();

        let new_array = groggy::core::array::GraphArray::from_vec(filled_values);
        let py_array = PyGraphArray::from_graph_array(new_array);
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Calculate quantile (percentile)
    fn quantile(&self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }

    /// Calculate percentile (user-friendly wrapper for quantile)
    /// percentile: 0-100 (e.g., 25 for 25th percentile, 90 for 90th percentile)
    fn percentile(&self, p: f64) -> Option<f64> {
        if p < 0.0 || p > 100.0 {
            return None;
        }
        self.inner.quantile(p / 100.0)
    }

    /// Calculate median (50th percentile)
    fn median(&self) -> Option<f64> {
        self.inner.median()
    }

    /// Get unique values as a new GraphArray
    fn unique(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        use std::collections::HashSet;

        // Use HashSet to find unique values
        let mut unique_set = HashSet::new();
        let mut unique_values = Vec::new();

        for attr_value in self.inner.iter() {
            // Create a simple hash key based on the value
            let key = match attr_value {
                RustAttrValue::Int(i) => format!("i:{}", i),
                RustAttrValue::SmallInt(i) => format!("si:{}", i),
                RustAttrValue::Float(f) => format!("f:{}", f),
                RustAttrValue::Text(s) => format!("t:{}", s),
                RustAttrValue::CompactText(s) => format!("ct:{}", s.as_str()),
                RustAttrValue::Bool(b) => format!("b:{}", b),
                RustAttrValue::Bytes(b) => format!("bytes:{}", b.len()), // Simple hash for bytes
                _ => format!("other:{:?}", attr_value),                  // Fallback for other types
            };

            if unique_set.insert(key) {
                // This is a new unique value
                unique_values.push(attr_value.clone());
            }
        }

        // Create new GraphArray with unique values
        let unique_array = GraphArray::from_vec(unique_values);
        let py_unique = PyGraphArray {
            inner: unique_array,
        };

        Ok(Py::new(py, py_unique)?)
    }

    /// Get value counts (frequency of each unique value) as Python dict
    fn value_counts(&self, py: Python) -> PyResult<PyObject> {
        use std::collections::HashMap;

        let mut counts: HashMap<String, (i32, RustAttrValue)> = HashMap::new();

        for attr_value in self.inner.iter() {
            // Create a string representation for HashMap key
            let key_str = match attr_value {
                RustAttrValue::Int(i) => format!("i:{}", i),
                RustAttrValue::SmallInt(i) => format!("si:{}", i),
                RustAttrValue::Float(f) => format!("f:{}", f),
                RustAttrValue::Text(s) => format!("t:{}", s),
                RustAttrValue::CompactText(s) => format!("ct:{}", s.as_str()),
                RustAttrValue::Bool(b) => format!("b:{}", b),
                RustAttrValue::Bytes(b) => format!("bytes:{}", b.len()),
                _ => format!("other:{:?}", attr_value),
            };

            match counts.get_mut(&key_str) {
                Some((count, _)) => *count += 1,
                None => {
                    counts.insert(key_str, (1, attr_value.clone()));
                }
            }
        }

        // Convert to Python dict
        let dict = pyo3::types::PyDict::new(py);
        for (_, (count, attr_value)) in counts {
            let py_key = attr_value_to_python_value(py, &attr_value)?;
            dict.set_item(py_key, count)?;
        }

        Ok(dict.to_object(py))
    }

    /// Get comprehensive statistical summary
    fn describe(&self, _py: Python) -> PyResult<PyStatsSummary> {
        Ok(PyStatsSummary {
            inner: self.inner.describe(),
        })
    }

    // ========================================================================
    // LAZY EVALUATION & MATERIALIZATION
    // ========================================================================

    /// Get array values (materializes data to Python objects)
    /// This is the primary materialization method - use sparingly for large arrays
    #[getter]
    fn values(&self, py: Python) -> PyResult<PyObject> {
        let materialized = self.inner.materialize();
        let py_values: PyResult<Vec<PyObject>> = materialized
            .iter()
            .map(|val| attr_value_to_python_value(py, val))
            .collect();

        Ok(py_values?.to_object(py))
    }

    /// Get preview of array for display (first 10 elements by default)
    fn preview(&self, py: Python, limit: Option<usize>) -> PyResult<PyObject> {
        let limit = limit.unwrap_or(10);
        let preview_values = self.inner.preview(limit);
        let py_values: PyResult<Vec<PyObject>> = preview_values
            .iter()
            .map(|val| attr_value_to_python_value(py, val))
            .collect();

        Ok(py_values?.to_object(py))
    }

    /// Check if array is sparse (has many default values)
    #[getter]
    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    /// Get summary information without materializing data
    fn summary(&self) -> String {
        self.inner.summary_info()
    }

    // ========================================================================
    // SCIENTIFIC COMPUTING CONVERSIONS
    // ========================================================================

    /// Convert to NumPy array (when numpy available)
    /// Uses .values property to materialize data
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .values property
        let values = self.values(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (values,))?;
        Ok(array.to_object(py))
    }

    /// Convert to Pandas Series (when pandas available)
    fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Get data as Python list
        let values = self.values(py)?;

        // Create Series
        let series = pandas.call_method1("Series", (values,))?;
        Ok(series.to_object(py))
    }

    /// Convert to SciPy sparse array (for compatibility - GraphArray is dense by nature)
    fn to_scipy_sparse(&self, py: Python) -> PyResult<PyObject> {
        // Try to import scipy.sparse
        let scipy_sparse = py.import("scipy.sparse").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "scipy is required for to_scipy_sparse(). Install with: pip install scipy",
            )
        })?;

        // Get data as Python list
        let values = self.values(py)?;

        // Convert to numpy first, then to sparse
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (values,))?;

        // Create CSR matrix (compressed sparse row) from the dense array
        let sparse_matrix = scipy_sparse.call_method1("csr_matrix", (array,))?;
        Ok(sparse_matrix.to_object(py))
    }

    // ========================================================================
    // DISPLAY INTEGRATION METHODS
    // ========================================================================

    /// Extract display data for Python display formatters
    /// Returns a dictionary with the structure expected by array_display.py
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);

        // Extract array data - convert to Python list
        let data = self.to_list(py)?;
        dict.set_item("data", data)?;

        // Get array metadata
        dict.set_item("shape", (self.inner.len(),))?;
        dict.set_item("dtype", self._get_dtype())?;
        dict.set_item(
            "name",
            self._get_name().unwrap_or_else(|| "array".to_string()),
        )?;

        Ok(dict.to_object(py))
    }

    /// Get data type string for display
    fn _get_dtype(&self) -> String {
        // Sample first few elements to determine predominant type
        let sample_size = std::cmp::min(self.inner.len(), 5);
        if sample_size == 0 {
            return "object".to_string();
        }

        let mut type_counts = std::collections::HashMap::new();

        for i in 0..sample_size {
            let type_name = match &self.inner[i] {
                RustAttrValue::Int(_) | RustAttrValue::SmallInt(_) => "int64",
                RustAttrValue::Float(_) => "f32",
                RustAttrValue::Bool(_) => "bool",
                RustAttrValue::Text(_) | RustAttrValue::CompactText(_) => "str",
                _ => "object",
            };
            *type_counts.entry(type_name).or_insert(0) += 1;
        }

        // Return the most common type
        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(type_name, _)| type_name.to_string())
            .unwrap_or_else(|| "object".to_string())
    }

    /// Get name for display (optional)
    fn _get_name(&self) -> Option<String> {
        // For now, we don't store names in GraphArray
        // This can be enhanced later when we add named arrays
        None
    }

    // === COMPARISON OPERATORS FOR BOOLEAN INDEXING ===

    /// Greater than comparison - returns boolean array
    fn __gt__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a > b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a > b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a > b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) > (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) > *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) > (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() > b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() > b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() > b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() > b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a > b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Less than comparison - returns boolean array
    fn __lt__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a < b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a < b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a < b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) < (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) < *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) < (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() < b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() < b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() < b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() < b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a < b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Greater than or equal comparison - returns boolean array
    fn __ge__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a >= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a >= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a >= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) >= (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) >= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) >= (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() >= b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a >= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Comparison not supported for these types",
                    ))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Less than or equal comparison - returns boolean array
    fn __le__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a <= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a <= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a <= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as i64) <= (*b as i64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) <= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) <= (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() <= b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a <= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Equality comparison - returns boolean array
    fn __eq__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = value == &other_value;
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Not equal comparison - returns boolean array
    fn __ne__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = value != &other_value;
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }
}

/// Python wrapper for StatsSummary
#[pyclass(name = "StatsSummary")]
pub struct PyStatsSummary {
    pub inner: StatsSummary,
}

#[pymethods]
impl PyStatsSummary {
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }

    #[getter]
    fn mean(&self) -> Option<f64> {
        self.inner.mean
    }

    #[getter]
    fn std(&self) -> Option<f64> {
        self.inner.std
    }

    #[getter]
    fn min(&self, py: Python) -> PyResult<Option<PyObject>> {
        match &self.inner.min {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn max(&self, py: Python) -> PyResult<Option<PyObject>> {
        match &self.inner.max {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn median(&self) -> Option<f64> {
        self.inner.median
    }

    #[getter]
    fn q25(&self) -> Option<f64> {
        self.inner.q25
    }

    #[getter]
    fn q75(&self) -> Option<f64> {
        self.inner.q75
    }

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Iterator for GraphArray
#[pyclass]
pub struct GraphArrayIterator {
    array: GraphArray,
    index: usize,
}

#[pymethods]
impl GraphArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.array.len() {
            let attr_value = &self.array[self.index];
            self.index += 1;
            Ok(Some(attr_value_to_python_value(py, attr_value)?))
        } else {
            Ok(None)
        }
    }
}

// Helper function to create PyGraphArray from GraphArray
impl PyGraphArray {
    pub fn from_graph_array(array: GraphArray) -> Self {
        PyGraphArray { inner: array }
    }

    /// Public constructor for use by builder functions
    pub fn from_py_objects(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            Ok(PyGraphArray {
                inner: GraphArray::from_vec(attr_values),
            })
        })
    }
}
