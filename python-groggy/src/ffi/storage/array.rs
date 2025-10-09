//! Array FFI Bindings
//!
//! Python bindings for statistical arrays and matrices.

#![allow(unused_variables)]

use groggy::entities::meta_node::MetaNode;
use groggy::storage::array::{
    ArrayIterator, ArrayOps, BaseArray, EdgesArray, MetaNodeArray, NodesArray,
};
use groggy::types::{AttrValue as RustAttrValue, AttrValueType, EdgeId, NodeId};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Use utility functions from utils module
use crate::ffi::utils::{attr_value_to_python_value, python_value_to_attr_value};

/// New BaseArray-powered array with chaining support
#[derive(Clone)]
#[pyclass(name = "BaseArray")]
pub struct PyBaseArray {
    pub inner: BaseArray<RustAttrValue>,
}

#[pymethods]
impl PyBaseArray {
    /// Create a new BaseArray from a list of values
    #[new]
    fn new(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            // Infer the data type from the first non-null element
            let dtype = attr_values
                .iter()
                .find(|v| !matches!(v, RustAttrValue::Null))
                .map(|v| v.dtype())
                .unwrap_or(AttrValueType::Text);

            Ok(PyBaseArray {
                inner: BaseArray::new(attr_values),
            })
        })
    }

    /// Create a BaseArray from existing AttrValues (internal constructor)
    /// TODO: Fix PyO3 type inference issue
    // pub fn from_attr_values(attr_values: Vec<RustAttrValue>) -> PyResult<Self> {
    //     Ok(PyBaseArray {
    //         inner: BaseArray::new(attr_values),
    //     })
    // }
    /// Get the number of elements (len())
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check whether the array contains any elements
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check whether the array contains the provided value
    fn contains(&self, item: &PyAny) -> PyResult<bool> {
        let attr_value = python_value_to_attr_value(item)?;
        Ok(self.inner.contains(&attr_value))
    }

    /// Convert the array to a plain Python list
    fn to_list(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut items = Vec::with_capacity(self.inner.len());
        for value in self.inner.iter() {
            items.push(attr_value_to_python_value(py, value)?);
        }
        Ok(items)
    }

    /// Advanced index access supporting multiple indexing types
    /// Supports:
    /// - Single integer: array[5], array[-1]
    /// - Integer lists: array[[0, 2, 5]]
    /// - Boolean arrays: array[bool_mask]
    /// - Slices: array[:5], array[::2], array[1:10:2]
    fn __getitem__(&self, py: Python, index: &PyAny) -> PyResult<PyObject> {
        use crate::ffi::utils::indexing::python_index_to_slice_index;
        use groggy::storage::array::AdvancedIndexing;

        // Handle simple integer case directly for performance
        if let Ok(int_val) = index.extract::<isize>() {
            let len = self.inner.len() as isize;
            let actual_index = if int_val < 0 { len + int_val } else { int_val };

            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Index out of range"));
            }

            match self.inner.get(actual_index as usize) {
                Some(attr_value) => return attr_value_to_python_value(py, attr_value),
                None => return Err(PyIndexError::new_err("Index out of range")),
            }
        }

        // Handle advanced indexing cases
        let slice_index = python_index_to_slice_index(py, index)?;

        match self.inner.get_slice(&slice_index) {
            Ok(sliced_array) => {
                let py_array = PyBaseArray {
                    inner: sliced_array,
                };
                Ok(py_array.into_py(py))
            }
            Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
        }
    }

    /// String representation with rich display hints
    fn __repr__(&self) -> String {
        let dtype = self.dtype();
        let len = self.inner.len();

        // Show a preview of the data
        let preview = if len == 0 {
            "empty".to_string()
        } else {
            let preview_items: Vec<String> = self
                .inner
                .iter()
                .take(3)
                .map(|v| match v {
                    groggy::types::AttrValue::Text(s) => format!("'{}'", s),
                    groggy::types::AttrValue::SmallInt(i) => i.to_string(),
                    groggy::types::AttrValue::Int(i) => i.to_string(),
                    groggy::types::AttrValue::Float(f) => format!("{:.2}", f),
                    groggy::types::AttrValue::Bool(b) => b.to_string(),
                    groggy::types::AttrValue::Null => "None".to_string(),
                    _ => format!("{:?}", v), // Catch-all for other variants
                })
                .collect();

            let preview_str = preview_items.join(", ");
            if len > 3 {
                format!("[{}, ...]", preview_str)
            } else {
                format!("[{}]", preview_str)
            }
        };

        format!("BaseArray[{}] {} (dtype: {})", len, preview, dtype)
    }

    /// Get the data type of the array based on the first non-null element
    fn dtype(&self) -> String {
        let dtype = self
            .inner
            .first()
            .map(|v| v.dtype())
            .unwrap_or(AttrValueType::Null);
        format!("{:?}", dtype)
    }

    // Statistical operations
    fn head(&self, n: usize) -> Self {
        let data: Vec<RustAttrValue> = self.inner.iter().take(n).cloned().collect();
        PyBaseArray {
            inner: BaseArray::new(data),
        }
    }

    fn tail(&self, n: usize) -> Self {
        let len = self.inner.len();
        let start = len.saturating_sub(n);
        let data: Vec<RustAttrValue> = self.inner.iter().skip(start).cloned().collect();
        PyBaseArray {
            inner: BaseArray::new(data),
        }
    }

    fn unique(&self) -> Self {
        let mut unique_values: Vec<RustAttrValue> = Vec::new();
        let mut seen: std::collections::HashSet<RustAttrValue> = std::collections::HashSet::new();

        for value in self.inner.iter() {
            if seen.insert(value.clone()) {
                unique_values.push(value.clone());
            }
        }

        PyBaseArray {
            inner: BaseArray::new(unique_values),
        }
    }

    fn describe(&self, py: Python) -> PyResult<PyObject> {
        // Basic descriptive statistics
        let count = self.inner.len();
        let non_null_count = self
            .inner
            .iter()
            .filter(|v| !matches!(v, RustAttrValue::Null))
            .count();
        let null_count = count - non_null_count;
        let unique_count = self.unique().inner.len();

        let dict = PyDict::new(py);
        dict.set_item("count", count)?;
        dict.set_item("non_null", non_null_count)?;
        dict.set_item("null", null_count)?;
        dict.set_item("unique", unique_count)?;
        dict.set_item("dtype", self.dtype())?;

        Ok(dict.into())
    }

    /// Calculate sum of numeric values in the array
    fn sum(&self, py: Python) -> PyResult<PyObject> {
        match self.inner.sum() {
            Ok(result) => attr_value_to_python_value(py, &result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Sum calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate mean of numeric values in the array
    fn mean(&self) -> PyResult<f64> {
        match self.inner.mean() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Mean calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate median of numeric values in the array
    fn median(&self) -> PyResult<f64> {
        match self.inner.median() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Median calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate standard deviation of numeric values in the array
    fn std(&self) -> PyResult<f64> {
        match self.inner.std() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Standard deviation calculation failed: {}",
                e
            ))),
        }
    }

    /// Calculate variance of numeric values in the array
    fn var(&self) -> PyResult<f64> {
        match self.inner.var() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Variance calculation failed: {}",
                e
            ))),
        }
    }

    /// Find minimum value in the array
    fn min(&self, py: Python) -> PyResult<PyObject> {
        match self.inner.min() {
            Ok(result) => attr_value_to_python_value(py, &result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Min calculation failed: {}",
                e
            ))),
        }
    }

    /// Find maximum value in the array
    fn max(&self, py: Python) -> PyResult<PyObject> {
        match self.inner.max() {
            Ok(result) => attr_value_to_python_value(py, &result),
            Err(e) => Err(PyValueError::new_err(format!(
                "Max calculation failed: {}",
                e
            ))),
        }
    }

    /// Count non-null values in the array
    fn count(&self) -> usize {
        self.inner.count()
    }

    /// Count unique non-null values in the array
    fn nunique(&self) -> usize {
        self.inner.nunique()
    }

    /// Detect missing/null values in the array
    /// Returns a boolean BaseArray where True indicates a null value
    /// Similar to pandas Series.isna()
    fn isna(&self) -> PyBaseArray {
        let bool_array = self.inner.isna();
        let attr_values: Vec<RustAttrValue> =
            bool_array.into_iter().map(RustAttrValue::Bool).collect();
        PyBaseArray {
            inner: BaseArray::new(attr_values),
        }
    }

    /// Detect non-missing/non-null values in the array
    /// Returns a boolean BaseArray where True indicates a non-null value
    /// Similar to pandas Series.notna()
    fn notna(&self) -> PyBaseArray {
        let bool_array = self.inner.notna();
        let attr_values: Vec<RustAttrValue> =
            bool_array.into_iter().map(RustAttrValue::Bool).collect();
        PyBaseArray {
            inner: BaseArray::new(attr_values),
        }
    }

    /// Remove missing/null values from the array
    /// Returns a new array with null values filtered out
    /// Similar to pandas Series.dropna()
    fn dropna(&self) -> PyBaseArray {
        PyBaseArray {
            inner: self.inner.dropna(),
        }
    }

    /// Check if the array contains any null values
    /// Similar to pandas Series.hasnans
    fn has_nulls(&self) -> bool {
        self.inner.has_nulls()
    }

    /// Count the number of null values in the array
    fn null_count(&self) -> usize {
        self.inner.null_count()
    }

    /// Fill null values with a specified value
    /// Returns a new array with nulls replaced by the fill value
    /// Similar to pandas Series.fillna()
    fn fillna(&self, fill_value: PyObject, py: Python) -> PyResult<PyBaseArray> {
        let rust_fill_value = python_value_to_attr_value(fill_value.as_ref(py))?;
        Ok(PyBaseArray {
            inner: self.inner.fillna(rust_fill_value),
        })
    }

    /// NEW: Enable fluent chaining with .iter() method
    fn iter(slf: PyRef<Self>) -> PyResult<PyBaseArrayIterator> {
        // Use our ArrayOps implementation to create the iterator
        let array_iterator = ArrayOps::iter(&slf.inner);

        Ok(PyBaseArrayIterator {
            inner: array_iterator,
        })
    }

    /// Delegation-based method application to each element
    /// This demonstrates the concept: apply a method to each element and return new array
    fn apply_to_each(&self, py: Python, method_name: &str, args: &PyTuple) -> PyResult<PyObject> {
        let mut results = Vec::new();

        // Apply the method to each element in the array
        for value in self.inner.data() {
            // Convert AttrValue to a Python object that might have the method
            let py_value = attr_value_to_python_value(py, value)?;

            // Check if this value has the requested method
            if py_value.as_ref(py).hasattr(method_name)? {
                // Get the method and call it with the provided arguments
                let method = py_value.as_ref(py).getattr(method_name)?;
                let result = method.call1(args)?;

                // Convert result back to AttrValue
                let attr_result = python_value_to_attr_value(result)?;
                results.push(attr_result);
            } else {
                return Err(PyAttributeError::new_err(format!(
                    "Elements in array don't have method '{}'",
                    method_name
                )));
            }
        }

        // Create new BaseArray with results
        let result_array = BaseArray::from_attr_values(results);
        let py_base = PyBaseArray {
            inner: result_array,
        };
        Ok(Py::new(py, py_base)?.to_object(py))
    }

    /// BaseArray __getattr__ delegation: automatically apply methods to each element
    /// This enables: array.some_method() -> applies some_method to each element
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        // Use apply_to_each internally to delegate to each element
        let args = pyo3::types::PyTuple::empty(py);
        let result = self.apply_to_each(py, name, args)?;
        Ok(result.into_py(py))
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells, similar to pandas DataFrames.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;

        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
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
        columns.insert(
            "index".to_string(),
            groggy::storage::array::BaseArray::new(indices),
        );

        // Value column (array data)
        let values: Vec<AttrValue> = self.inner.clone_vec();
        columns.insert(
            "value".to_string(),
            groggy::storage::array::BaseArray::new(values),
        );

        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(
            base_table,
        ))
    }

    // === COMPARISON OPERATORS FOR BOOLEAN INDEXING ===

    /// Greater than comparison (>) - returns Vec<bool> for boolean masking
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();

        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a > b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a > b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a > b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a > (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) > *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) > *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a > (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) > *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a > (*b as f32),

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
            result.push(comparison_result);
        }

        // Convert Vec<bool> to unified NumArray with bool dtype
        let py_bool_array = crate::ffi::storage::num_array::PyNumArray::new_bool(result);
        Ok(py_bool_array.into_py(py))
    }

    /// Less than comparison (<) - returns BoolArray for boolean masking
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();

        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a < b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a < b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a < b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a < (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) < *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) < *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a < (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) < *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a < (*b as f32),

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
            result.push(comparison_result);
        }

        // Convert Vec<bool> to unified NumArray with bool dtype
        let py_bool_array = crate::ffi::storage::num_array::PyNumArray::new_bool(result);
        Ok(py_bool_array.into_py(py))
    }

    /// Greater than or equal comparison (>=) - returns BoolArray for boolean masking
    fn __ge__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();

        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a >= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a >= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a >= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a >= (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) >= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) >= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a >= (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) >= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a >= (*b as f32),

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
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(comparison_result);
        }

        // Convert Vec<bool> to unified NumArray with bool dtype
        let py_bool_array = crate::ffi::storage::num_array::PyNumArray::new_bool(result);
        Ok(py_bool_array.into_py(py))
    }

    /// Less than or equal comparison (<=) - returns BoolArray for boolean masking
    fn __le__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();

        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a <= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a <= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a <= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a <= (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) <= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) <= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a <= (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) <= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a <= (*b as f32),

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
            result.push(comparison_result);
        }

        // Convert Vec<bool> to unified NumArray with bool dtype
        let py_bool_array = crate::ffi::storage::num_array::PyNumArray::new_bool(result);
        Ok(py_bool_array.into_py(py))
    }

    /// Equality comparison (==) - returns BoolArray for boolean masking
    fn __eq__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();

        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Exact type matches
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a == b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a == b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => {
                    (a - b).abs() < f32::EPSILON
                }
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a == b,
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a == b,
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() == b.as_str()
                }
                (groggy::AttrValue::Null, groggy::AttrValue::Null) => true,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a == (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) == *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f32 - *b).abs() < f32::EPSILON
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a - *b as f32).abs() < f32::EPSILON
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f32 - *b).abs() < f32::EPSILON
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a - *b as f32).abs() < f32::EPSILON
                }

                // Mixed string comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() == b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() == b.as_str()
                }

                // Different types are not equal (including nulls with non-nulls)
                _ => false,
            };
            result.push(comparison_result);
        }

        // Convert Vec<bool> to unified NumArray with bool dtype
        let py_bool_array = crate::ffi::storage::num_array::PyNumArray::new_bool(result);
        Ok(py_bool_array.into_py(py))
    }

    /// Not equal comparison (!=) - returns BoolArray for boolean masking
    fn __ne__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        // Just negate the equality result
        let eq_result = self.__eq__(py, other)?;
        let bool_vec: Vec<bool> = eq_result.extract(py)?;
        let ne_result: Vec<bool> = bool_vec.into_iter().map(|x| !x).collect();
        // Convert Vec<bool> to unified NumArray with bool dtype
        let py_bool_array = crate::ffi::storage::num_array::PyNumArray::new_bool(ne_result);
        Ok(py_bool_array.into_py(py))
    }

    /// Convert BaseArray to different numeric types when possible
    fn to_type(&self, py: Python, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "int64" | "i64" => {
                // Try to convert all elements to i64
                let mut int_values = Vec::new();
                for value in self.inner.iter() {
                    match value {
                        groggy::AttrValue::Int(i) => int_values.push(*i),
                        groggy::AttrValue::SmallInt(i) => int_values.push(*i as i64),
                        groggy::AttrValue::Float(f) => int_values.push(f.round() as i64),
                        groggy::AttrValue::Bool(b) => int_values.push(if *b { 1 } else { 0 }),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Cannot convert {:?} to int64",
                                value
                            )))
                        }
                    }
                }
                let int_array = crate::ffi::storage::num_array::PyNumArray::new_int64(int_values);
                Ok(int_array.into_py(py))
            }
            "float64" | "f64" => {
                // Try to convert all elements to f64
                let mut float_values = Vec::new();
                for value in self.inner.iter() {
                    match value {
                        groggy::AttrValue::Int(i) => float_values.push(*i as f64),
                        groggy::AttrValue::SmallInt(i) => float_values.push(*i as f64),
                        groggy::AttrValue::Float(f) => float_values.push(*f as f64),
                        groggy::AttrValue::Bool(b) => float_values.push(if *b { 1.0 } else { 0.0 }),
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Cannot convert {:?} to float64",
                                value
                            )))
                        }
                    }
                }
                let num_array =
                    crate::ffi::storage::num_array::PyNumArray::new_float64(float_values);
                Ok(num_array.into_py(py))
            }
            "basearray" => {
                // Return a copy of self
                Ok(self.clone().into_py(py))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype: {}. Supported: 'int64', 'float64', 'basearray'",
                dtype
            ))),
        }
    }

    // === BASEARRAY INTELLIGENCE METHODS (Week 5-6: BaseArray Intelligence) ===

    /// Infer the optimal numeric type for this BaseArray
    /// Analyzes all elements to determine the most appropriate NumericType
    /// Returns None if the array contains non-numeric data
    fn infer_numeric_type(&self) -> Option<String> {
        use groggy::NumericType;

        if self.inner.is_empty() {
            return None;
        }

        let mut current_type: Option<NumericType> = None;

        for value in self.inner.iter() {
            let value_type = match value {
                groggy::AttrValue::Bool(_) => Some(NumericType::Bool),
                groggy::AttrValue::SmallInt(_) => Some(NumericType::Int32),
                groggy::AttrValue::Int(_) => Some(NumericType::Int64),
                groggy::AttrValue::Float(_) => Some(NumericType::Float32),
                groggy::AttrValue::Null => continue, // Skip nulls in type inference
                _ => return None,                    // Non-numeric data found
            };

            if let Some(vt) = value_type {
                current_type = Some(match current_type {
                    None => vt,
                    Some(ct) => ct.promote_with(vt),
                });
            }
        }

        current_type.map(|t| format!("{:?}", t))
    }

    /// Convert BaseArray to NumArray if all elements are numeric
    /// Returns appropriate numeric array type based on inferred type
    /// OPTIMIZED: Single-pass conversion with pre-allocated vectors
    fn to_num_array(&self, py: Python) -> PyResult<PyObject> {
        // OPTIMIZATION: Single-pass type inference and conversion
        if self.inner.is_empty() {
            // Return empty NumArray for empty input
            return Ok(
                crate::ffi::storage::num_array::PyNumArray::new_float64(Vec::new()).into_py(py),
            );
        }

        let len = self.inner.len();
        let mut current_type: Option<groggy::NumericType> = None;

        // Pre-allocate vectors with known capacity for better performance
        let mut bool_values = Vec::with_capacity(len);
        let mut int_values = Vec::with_capacity(len);
        let mut float_values = Vec::with_capacity(len);

        // Single pass: determine type and convert simultaneously
        for value in self.inner.iter() {
            let (value_type, bool_val, int_val, float_val) = match value {
                groggy::AttrValue::Bool(b) => (
                    Some(groggy::NumericType::Bool),
                    *b,
                    if *b { 1 } else { 0 },
                    if *b { 1.0 } else { 0.0 },
                ),
                groggy::AttrValue::SmallInt(i) => (
                    Some(groggy::NumericType::Int32),
                    *i != 0,
                    *i as i64,
                    *i as f64,
                ),
                groggy::AttrValue::Int(i) => {
                    (Some(groggy::NumericType::Int64), *i != 0, *i, *i as f64)
                }
                groggy::AttrValue::Float(f) => (
                    Some(groggy::NumericType::Float32),
                    *f != 0.0,
                    f.round() as i64,
                    *f as f64,
                ),
                groggy::AttrValue::Null => (None, false, 0, 0.0), // Skip nulls in type inference
                _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "BaseArray contains non-numeric data and cannot be converted to numeric array",
                )),
            };

            // Update promoted type
            if let Some(vt) = value_type {
                current_type = Some(match current_type {
                    None => vt,
                    Some(ct) => ct.promote_with(vt),
                });
            }

            // Store converted values (we'll use the appropriate one based on final type)
            bool_values.push(bool_val);
            int_values.push(int_val);
            float_values.push(float_val);
        }

        // Return appropriate array type based on promoted type
        match current_type {
            Some(groggy::NumericType::Bool) => {
                // Convert to unified NumArray with bool dtype
                let py_bool = crate::ffi::storage::num_array::PyNumArray::new_bool(bool_values);
                Ok(py_bool.into_py(py))
            }
            Some(groggy::NumericType::Int32) | Some(groggy::NumericType::Int64) => {
                let int_array = crate::ffi::storage::num_array::PyNumArray::new_int64(int_values);
                Ok(int_array.into_py(py))
            }
            Some(groggy::NumericType::Float32) | Some(groggy::NumericType::Float64) => {
                let num_array =
                    crate::ffi::storage::num_array::PyNumArray::new_float64(float_values);
                Ok(num_array.into_py(py))
            }
            None => {
                // All nulls - default to empty NumArray
                let num_array = crate::ffi::storage::num_array::PyNumArray::new_float64(Vec::new());
                Ok(num_array.into_py(py))
            }
        }
    }

    /// Check if this BaseArray can be converted to a numeric array
    fn is_numeric(&self) -> bool {
        self.infer_numeric_type().is_some()
    }

    /// Get statistics about the numeric compatibility of this BaseArray
    fn numeric_compatibility_info(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        let info = PyDict::new(py);

        if self.inner.is_empty() {
            info.set_item("is_numeric", false)?;
            info.set_item("reason", "Array is empty")?;
            return Ok(info.to_object(py));
        }

        let total_count = self.inner.len();
        let mut numeric_count = 0;
        let mut null_count = 0;
        let mut type_counts = std::collections::HashMap::new();

        for value in self.inner.iter() {
            match value {
                groggy::AttrValue::Bool(_)
                | groggy::AttrValue::SmallInt(_)
                | groggy::AttrValue::Int(_)
                | groggy::AttrValue::Float(_) => {
                    numeric_count += 1;
                    let type_name = match value {
                        groggy::AttrValue::Bool(_) => "bool",
                        groggy::AttrValue::SmallInt(_) => "int32",
                        groggy::AttrValue::Int(_) => "int64",
                        groggy::AttrValue::Float(_) => "float32",
                        _ => unreachable!(),
                    };
                    *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
                }
                groggy::AttrValue::Null => null_count += 1,
                _ => {
                    let type_name = match value {
                        groggy::AttrValue::Text(_) => "text",
                        groggy::AttrValue::CompactText(_) => "compact_text",
                        groggy::AttrValue::FloatVec(_) => "float_vec",
                        groggy::AttrValue::Bytes(_) => "bytes",
                        _ => "other",
                    };
                    *type_counts.entry(type_name.to_string()).or_insert(0) += 1;
                }
            }
        }

        let is_numeric = numeric_count + null_count == total_count;
        let numeric_percentage = (numeric_count as f64 / total_count as f64) * 100.0;

        info.set_item("is_numeric", is_numeric)?;
        info.set_item("total_count", total_count)?;
        info.set_item("numeric_count", numeric_count)?;
        info.set_item("null_count", null_count)?;
        info.set_item("numeric_percentage", numeric_percentage)?;
        info.set_item("type_counts", type_counts)?;

        if let Some(inferred_type) = self.infer_numeric_type() {
            info.set_item("inferred_numeric_type", inferred_type)?;
            info.set_item("recommended_conversion", "to_num_array()")?;
        } else if numeric_count > 0 {
            info.set_item("reason", "Contains mixed numeric and non-numeric data")?;
            info.set_item("suggestion", "Filter out non-numeric values first")?;
        } else {
            info.set_item("reason", "No numeric data found")?;
        }

        Ok(info.to_object(py))
    }

    // ==================================================================================
    // COLUMN MANAGEMENT OPERATIONS FOR TABLE CONVERSION
    // ==================================================================================

    /// Convert array to single-column table with specified column name
    ///
    /// # Arguments
    /// * `column_name` - Name for the column when converting to table
    ///
    /// # Examples
    /// ```python
    /// table = array.to_table_with_name("scores")
    /// ```
    pub fn to_table_with_name(
        &self,
        column_name: &str,
    ) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.to_table_with_name(column_name).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Table conversion failed: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Convert array to single-column table with prefix added to default column name
    ///
    /// # Arguments
    /// * `prefix` - Prefix to add to the default 'values' column name
    ///
    /// # Examples
    /// ```python
    /// table = array.to_table_with_prefix("old_")  # Creates column "old_values"
    /// ```
    pub fn to_table_with_prefix(
        &self,
        prefix: &str,
    ) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.to_table_with_prefix(prefix).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Table conversion failed: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Convert array to single-column table with suffix added to default column name
    ///
    /// # Arguments
    /// * `suffix` - Suffix to add to the default 'values' column name
    ///
    /// # Examples
    /// ```python
    /// table = array.to_table_with_suffix("_v1")  # Creates column "values_v1"
    /// ```
    pub fn to_table_with_suffix(
        &self,
        suffix: &str,
    ) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.to_table_with_suffix(suffix).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Table conversion failed: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    // ==================================================================================
    // PHASE 2.2: ELEMENT OPERATIONS (Array equivalent of row operations)
    // ==================================================================================

    /// Append a single element to the array
    ///
    /// # Arguments
    /// * `element` - Value to append to the array
    ///
    /// # Examples
    /// ```python
    /// new_array = array.append_element(42)
    /// ```
    pub fn append_element(&self, element: PyObject, py: Python) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(element.as_ref(py))?;
        let result = self.inner.append_element(attr_value);

        Ok(Self { inner: result })
    }

    /// Extend array with multiple elements
    ///
    /// # Arguments
    /// * `elements` - List of values to append to the array
    ///
    /// # Examples
    /// ```python
    /// new_array = array.extend_elements([42, "hello", 3.14])
    /// ```
    pub fn extend_elements(&self, elements: Vec<PyObject>, py: Python) -> PyResult<Self> {
        let mut attr_elements = Vec::new();
        for element in elements {
            let attr_value = crate::ffi::utils::python_value_to_attr_value(element.as_ref(py))?;
            attr_elements.push(attr_value);
        }

        let result = self.inner.extend_elements(attr_elements);
        Ok(Self { inner: result })
    }

    /// Drop elements by indices
    ///
    /// # Arguments
    /// * `indices` - List of indices to drop from the array
    ///
    /// # Examples
    /// ```python
    /// new_array = array.drop_elements([0, 2, 4])  # Drop elements at indices 0, 2, and 4
    /// ```
    pub fn drop_elements(&self, indices: Vec<usize>) -> PyResult<Self> {
        let result = self.inner.drop_elements(&indices).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Drop elements failed: {}", e))
        })?;

        Ok(Self { inner: result })
    }

    /// Drop duplicate elements
    ///
    /// # Examples
    /// ```python
    /// new_array = array.drop_duplicates_elements()
    /// ```
    pub fn drop_duplicates_elements(&self) -> Self {
        let result = self.inner.drop_duplicates_elements();
        Self { inner: result }
    }

    /// Get the length of the array (standard Python method)
    ///
    /// # Returns
    /// The number of elements in the array
    ///
    /// # Examples
    /// ```python
    /// size = array.len()
    /// ```
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get element at specific index
    ///
    /// # Arguments
    /// * `index` - Index to retrieve (0-based)
    ///
    /// # Returns
    /// The element at the specified index
    ///
    /// # Examples
    /// ```python
    /// element = array.get(0)  # Get first element
    /// ```
    pub fn get(&self, index: usize, py: Python) -> PyResult<PyObject> {
        use crate::ffi::utils::attr_value_to_python_value;

        if let Some(value) = self.inner.get(index) {
            attr_value_to_python_value(py, value)
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of bounds for array of length {}",
                index,
                self.inner.len()
            )))
        }
    }

    /// Append a single element (standard Python method)
    ///
    /// # Arguments
    /// * `element` - Value to append to the array
    ///
    /// # Returns
    /// New array with the element appended
    ///
    /// # Examples
    /// ```python
    /// new_array = array.append(42)
    /// ```
    pub fn append(&self, element: PyObject, py: Python) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(element.as_ref(py))?;
        let result = self.inner.append_element(attr_value);
        Ok(Self { inner: result })
    }

    /// Extend array with multiple elements (standard Python method)
    ///
    /// # Arguments
    /// * `elements` - List of values to append to the array
    ///
    /// # Returns
    /// New array with all elements appended
    ///
    /// # Examples
    /// ```python
    /// new_array = array.extend([42, "hello", 3.14])
    /// ```
    pub fn extend(&self, elements: Vec<PyObject>, py: Python) -> PyResult<Self> {
        let mut attr_elements = Vec::new();
        for element in elements {
            let attr_value = crate::ffi::utils::python_value_to_attr_value(element.as_ref(py))?;
            attr_elements.push(attr_value);
        }
        let result = self.inner.extend_elements(attr_elements);
        Ok(Self { inner: result })
    }

    /// Insert element at specific index
    ///
    /// # Arguments
    /// * `index` - Index to insert at
    /// * `element` - Value to insert
    ///
    /// # Returns
    /// New array with element inserted
    ///
    /// # Examples
    /// ```python
    /// new_array = array.insert(2, "new_value")
    /// ```
    pub fn insert(&self, index: usize, element: PyObject, py: Python) -> PyResult<Self> {
        use crate::ffi::utils::python_value_to_attr_value;

        let attr_value = python_value_to_attr_value(element.as_ref(py))?;

        // Insert by creating new array with element at specified position
        let mut new_values = Vec::new();

        // Add elements before insertion point
        for i in 0..index.min(self.inner.len()) {
            if let Some(val) = self.inner.get(i) {
                new_values.push(val.clone());
            }
        }

        // Add the new element
        new_values.push(attr_value);

        // Add remaining elements
        for i in index..self.inner.len() {
            if let Some(val) = self.inner.get(i) {
                new_values.push(val.clone());
            }
        }

        Ok(Self {
            inner: groggy::storage::array::BaseArray::new(new_values),
        })
    }

    /// Remove element at specific index
    ///
    /// # Arguments
    /// * `index` - Index to remove
    ///
    /// # Returns
    /// New array with element removed
    ///
    /// # Examples
    /// ```python
    /// new_array = array.remove(0)  # Remove first element
    /// ```
    pub fn remove(&self, index: usize) -> PyResult<Self> {
        if index >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of bounds for array of length {}",
                index,
                self.inner.len()
            )));
        }

        let result = self.inner.drop_elements(&[index]).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Remove failed: {}", e))
        })?;

        Ok(Self { inner: result })
    }

    /// Filter elements using a Python function
    ///
    /// # Arguments
    /// * `predicate` - Python function that returns True/False for each element
    ///
    /// # Returns
    /// New array with elements where predicate returns True
    ///
    /// # Examples
    /// ```python
    /// filtered = array.filter(lambda x: x > 5)
    /// ```
    pub fn filter(&self, predicate: PyObject, py: Python) -> PyResult<Self> {
        use crate::ffi::utils::attr_value_to_python_value;

        let mut filtered_values = Vec::new();

        for value in self.inner.data().iter() {
            // Convert AttrValue to Python object
            let py_value = attr_value_to_python_value(py, value)?;

            // Call Python predicate function
            let result = predicate.call1(py, (py_value,))?;

            // Check if result is truthy
            if result.is_true(py)? {
                filtered_values.push(value.clone());
            }
        }

        Ok(Self {
            inner: groggy::storage::array::BaseArray::new(filtered_values),
        })
    }

    /// Map elements using a Python function
    ///
    /// # Arguments
    /// * `func` - Python function to apply to each element
    ///
    /// # Returns
    /// New array with transformed elements
    ///
    /// # Examples
    /// ```python
    /// mapped = array.map(lambda x: x * 2)
    /// ```
    pub fn map(&self, func: PyObject, py: Python) -> PyResult<Self> {
        use crate::ffi::utils::{attr_value_to_python_value, python_value_to_attr_value};

        let mut mapped_values = Vec::new();

        for value in self.inner.data().iter() {
            // Convert AttrValue to Python object
            let py_value = attr_value_to_python_value(py, value)?;

            // Call Python function
            let result = func.call1(py, (py_value,))?;

            // Convert result back to AttrValue
            let attr_result = python_value_to_attr_value(result.as_ref(py))?;
            mapped_values.push(attr_result);
        }

        Ok(Self {
            inner: groggy::storage::array::BaseArray::new(mapped_values),
        })
    }

    /// Sort the array elements
    ///
    /// # Arguments
    /// * `ascending` - Whether to sort in ascending order (default: true)
    ///
    /// # Returns
    /// New array with sorted elements
    ///
    /// # Examples
    /// ```python
    /// sorted_asc = array.sort()
    /// sorted_desc = array.sort(ascending=False)
    /// ```
    pub fn sort(&self, ascending: Option<bool>) -> PyResult<Self> {
        let ascending = ascending.unwrap_or(true);

        let mut indexed_values: Vec<(usize, groggy::types::AttrValue)> = self
            .inner
            .data()
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect();

        // Sort by AttrValue comparison
        indexed_values.sort_by(|(_, a), (_, b)| {
            use std::cmp::Ordering;
            match (a, b) {
                // Numeric comparisons
                (groggy::types::AttrValue::Int(a), groggy::types::AttrValue::Int(b)) => {
                    if ascending {
                        a.cmp(b)
                    } else {
                        b.cmp(a)
                    }
                }
                (groggy::types::AttrValue::Float(a), groggy::types::AttrValue::Float(b)) => {
                    let ord = a.partial_cmp(b).unwrap_or(Ordering::Equal);
                    if ascending {
                        ord
                    } else {
                        ord.reverse()
                    }
                }
                (groggy::types::AttrValue::Int(a), groggy::types::AttrValue::Float(b)) => {
                    let ord = (*a as f32).partial_cmp(b).unwrap_or(Ordering::Equal);
                    if ascending {
                        ord
                    } else {
                        ord.reverse()
                    }
                }
                (groggy::types::AttrValue::Float(a), groggy::types::AttrValue::Int(b)) => {
                    let ord = a.partial_cmp(&(*b as f32)).unwrap_or(Ordering::Equal);
                    if ascending {
                        ord
                    } else {
                        ord.reverse()
                    }
                }
                // String comparisons
                (groggy::types::AttrValue::Text(a), groggy::types::AttrValue::Text(b)) => {
                    if ascending {
                        a.cmp(b)
                    } else {
                        b.cmp(a)
                    }
                }
                // Default case - maintain original order
                _ => Ordering::Equal,
            }
        });

        let sorted_values: Vec<groggy::types::AttrValue> =
            indexed_values.into_iter().map(|(_, v)| v).collect();

        Ok(Self {
            inner: groggy::storage::array::BaseArray::new(sorted_values),
        })
    }

    /// Reverse the order of elements
    ///
    /// # Returns
    /// New array with elements in reverse order
    ///
    /// # Examples
    /// ```python
    /// reversed_array = array.reverse()
    /// ```
    pub fn reverse(&self) -> Self {
        let mut reversed_values: Vec<groggy::types::AttrValue> = self.inner.data().to_vec();
        reversed_values.reverse();

        Self {
            inner: groggy::storage::array::BaseArray::new(reversed_values),
        }
    }

    /// Count the frequency of unique values (pandas-style value_counts)
    ///
    /// # Arguments
    /// * `sort` - Whether to sort results by count (default: true)
    /// * `ascending` - Sort order when sort=true (default: false, most frequent first)
    /// * `dropna` - Whether to exclude null values (default: true)
    ///
    /// # Returns
    /// Table with 'value' and 'count' columns showing frequency of each unique value
    ///
    /// # Examples
    /// ```python
    /// # Basic usage
    /// counts = array.value_counts()
    ///
    /// # Custom sorting
    /// counts = array.value_counts(sort=True, ascending=True)
    ///
    /// # Include null values
    /// counts = array.value_counts(dropna=False)
    /// ```
    pub fn value_counts(
        &self,
        sort: Option<bool>,
        ascending: Option<bool>,
        dropna: Option<bool>,
    ) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let sort = sort.unwrap_or(true);
        let ascending = ascending.unwrap_or(false);
        let dropna = dropna.unwrap_or(true);

        let result = self
            .inner
            .value_counts(sort, ascending, dropna)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Apply a Python function to each element in the array (pandas-style apply)
    ///
    /// # Arguments
    /// * `func` - Python function to apply to each element
    ///
    /// # Returns
    /// New BaseArray with transformed values
    ///
    /// # Examples
    /// ```python
    /// # Square each number
    /// def square(x):
    ///     if isinstance(x, (int, float)):
    ///         return x * x
    ///     return x
    ///
    /// squared = array.apply(square)
    ///
    /// # Using lambda
    /// doubled = array.apply(lambda x: x * 2 if isinstance(x, (int, float)) else x)
    /// ```
    pub fn apply(&self, func: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let transformed_data: PyResult<Vec<RustAttrValue>> = self
                .inner
                .iter()
                .map(|value| {
                    // Convert AttrValue to Python object
                    let py_value = attr_value_to_python_value(py, value)?;

                    // Call the Python function
                    let result = func.call1(py, (py_value,))?;

                    // Convert result back to AttrValue
                    python_value_to_attr_value(result.as_ref(py))
                })
                .collect();

            let transformed_data = transformed_data?;
            Ok(PyBaseArray {
                inner: BaseArray::from_attr_values(transformed_data),
            })
        })
    }

    /// Compute quantile for the array (pandas-style quantile)
    ///
    /// # Arguments
    /// * `q` - Quantile to compute (0.0 to 1.0)
    /// * `interpolation` - Method for interpolation ("linear", "lower", "higher", "midpoint", "nearest")
    ///
    /// # Returns
    /// AttrValue containing the computed quantile
    ///
    /// # Examples
    /// ```python
    /// # Get median (50th percentile)
    /// median = array.quantile(0.5, "linear")
    ///
    /// # Get 95th percentile with nearest interpolation
    /// p95 = array.quantile(0.95, "nearest")
    /// ```
    pub fn quantile(&self, q: f64, interpolation: Option<&str>) -> PyResult<PyObject> {
        use crate::ffi::utils::attr_value_to_python_value;

        let interpolation = interpolation.unwrap_or("linear");

        let result = self
            .inner
            .quantile(q, interpolation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Python::with_gil(|py| attr_value_to_python_value(py, &result))
    }

    /// Compute multiple quantiles for the array
    ///
    /// # Arguments
    /// * `quantiles` - List of quantiles to compute (each 0.0 to 1.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Returns
    /// BaseArray containing the computed quantiles
    pub fn quantiles(&self, quantiles: Vec<f64>, interpolation: Option<&str>) -> PyResult<Self> {
        let interpolation = interpolation.unwrap_or("linear");

        let result = self
            .inner
            .quantiles(&quantiles, interpolation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Compute percentile for the array (equivalent to quantile * 100)
    ///
    /// # Arguments
    /// * `percentile` - Percentile to compute (0.0 to 100.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Examples
    /// ```python
    /// # Get median (50th percentile)
    /// median = array.percentile(50.0, "linear")
    ///
    /// # Get quartiles
    /// q1 = array.percentile(25.0)
    /// q3 = array.percentile(75.0)
    /// ```
    pub fn percentile(&self, percentile: f64, interpolation: Option<&str>) -> PyResult<PyObject> {
        use crate::ffi::utils::attr_value_to_python_value;

        let interpolation = interpolation.unwrap_or("linear");

        let result = self
            .inner
            .get_percentile(percentile, interpolation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Python::with_gil(|py| attr_value_to_python_value(py, &result))
    }

    /// Compute specific percentile for the array (direct method for consistency)
    ///
    /// # Arguments
    /// * `percentile` - Percentile to compute (0.0 to 100.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Returns
    /// The computed percentile as PyObject
    ///
    /// # Examples
    /// ```python
    /// # Get median (50th percentile)
    /// median = array.get_percentile(50.0)
    ///
    /// # Get 95th percentile with nearest interpolation
    /// p95 = array.get_percentile(95.0, "nearest")
    /// ```
    pub fn get_percentile(
        &self,
        percentile: f64,
        interpolation: Option<&str>,
    ) -> PyResult<PyObject> {
        use crate::ffi::utils::attr_value_to_python_value;

        let interpolation = interpolation.unwrap_or("linear");

        let result = self
            .inner
            .get_percentile(percentile, interpolation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Python::with_gil(|py| attr_value_to_python_value(py, &result))
    }

    /// Compute multiple percentiles for the array
    ///
    /// # Arguments
    /// * `percentiles` - List of percentiles to compute (each 0.0 to 100.0)
    /// * `interpolation` - Method for interpolation
    ///
    /// # Returns
    /// BaseArray containing the computed percentiles
    pub fn percentiles(
        &self,
        percentiles: Vec<f64>,
        interpolation: Option<&str>,
    ) -> PyResult<Self> {
        let interpolation = interpolation.unwrap_or("linear");

        let result = self
            .inner
            .percentiles(&percentiles, interpolation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Compute correlation coefficient with another array
    ///
    /// # Arguments
    /// * `other` - The other array to compute correlation with
    /// * `method` - Correlation method: "pearson" (default), "spearman", or "kendall"
    ///
    /// # Returns
    /// Correlation coefficient as a Python float
    pub fn corr(&self, other: &PyBaseArray, method: Option<&str>) -> PyResult<PyObject> {
        let method = method.unwrap_or("pearson");

        let result = self
            .inner
            .corr(&other.inner, method)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Python::with_gil(|py| attr_value_to_python_value(py, &result))
    }

    /// Compute covariance with another array
    ///
    /// # Arguments
    /// * `other` - The other array to compute covariance with
    /// * `ddof` - Delta degrees of freedom (default: 1)
    ///
    /// # Returns
    /// Covariance as a Python float
    pub fn cov(&self, other: &PyBaseArray, ddof: Option<i32>) -> PyResult<PyObject> {
        let ddof = ddof.unwrap_or(1);

        let result = self
            .inner
            .cov(&other.inner, ddof)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Python::with_gil(|py| attr_value_to_python_value(py, &result))
    }

    /// Rolling window operation with specified window size
    ///
    /// # Arguments
    /// * `window` - Window size for rolling operations
    /// * `operation` - Function to apply to each window (e.g., "mean", "sum", "min", "max", "std")
    ///
    /// # Returns
    /// New BaseArray with rolling operation results
    pub fn rolling(&self, window: usize, operation: &str) -> PyResult<Self> {
        let result = self
            .inner
            .rolling(window, operation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Expanding window operation (cumulative from start)
    ///
    /// # Arguments
    /// * `operation` - Function to apply to expanding window (e.g., "mean", "sum", "min", "max", "std")
    ///
    /// # Returns
    /// New BaseArray with expanding operation results
    pub fn expanding(&self, operation: &str) -> PyResult<Self> {
        let result = self
            .inner
            .expanding(operation)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Cumulative sum operation
    ///
    /// # Returns
    /// New BaseArray with cumulative sum values
    pub fn cumsum(&self) -> PyResult<Self> {
        let result = self
            .inner
            .cumsum()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Cumulative minimum operation
    ///
    /// # Returns
    /// New BaseArray with cumulative minimum values
    pub fn cummin(&self) -> PyResult<Self> {
        let result = self
            .inner
            .cummin()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Cumulative maximum operation
    ///
    /// # Returns
    /// New BaseArray with cumulative maximum values
    pub fn cummax(&self) -> PyResult<Self> {
        let result = self
            .inner
            .cummax()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Shift operation - shift values by specified periods
    ///
    /// # Arguments
    /// * `periods` - Number of periods to shift (positive = shift right, negative = shift left)
    /// * `fill_value` - Value to use for filling gaps (default: Null)
    ///
    /// # Returns
    /// New BaseArray with shifted values
    pub fn shift(&self, periods: i32, fill_value: Option<PyObject>) -> PyResult<Self> {
        let rust_fill_value = if let Some(py_value) = fill_value {
            Python::with_gil(|py| python_value_to_attr_value(py_value.as_ref(py)))?
        } else {
            groggy::AttrValue::Null
        };

        let result = self
            .inner
            .shift(periods, Some(rust_fill_value))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }

    /// Percentage change operation
    ///
    /// # Arguments
    /// * `periods` - Number of periods to use for comparison (default: 1)
    ///
    /// # Returns
    /// New BaseArray with percentage change values
    pub fn pct_change(&self, periods: Option<usize>) -> PyResult<Self> {
        let result = self
            .inner
            .pct_change(periods)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(PyBaseArray { inner: result })
    }
}

/// Python wrapper for ArrayIterator<AttrValue> that supports method chaining
#[pyclass(name = "BaseArrayIterator", unsendable)]
pub struct PyBaseArrayIterator {
    inner: ArrayIterator<RustAttrValue>,
}

#[pymethods]
impl PyBaseArrayIterator {
    /// Filter elements using a Python predicate function
    fn filter(slf: PyRefMut<Self>, predicate: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = slf.inner.clone();
            let filtered = inner.filter(|attr_value| {
                // Convert AttrValue to Python and call predicate
                match attr_value_to_python_value(py, attr_value) {
                    Ok(py_value) => match predicate.call1(py, (py_value,)) {
                        Ok(result) => result.is_true(py).unwrap_or(false),
                        Err(_) => false,
                    },
                    Err(_) => false,
                }
            });

            Ok(Self { inner: filtered })
        })
    }

    /// Map elements using a Python function
    fn map(slf: PyRefMut<Self>, func: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = slf.inner.clone();
            let mapped = inner.map(|attr_value| {
                // Convert AttrValue to Python, call function, convert back
                match attr_value_to_python_value(py, &attr_value) {
                    Ok(py_value) => {
                        match func.call1(py, (py_value,)) {
                            Ok(result) => {
                                match python_value_to_attr_value(result.as_ref(py)) {
                                    Ok(new_attr_value) => new_attr_value,
                                    Err(_) => attr_value, // Keep original on conversion error
                                }
                            }
                            Err(_) => attr_value, // Keep original on call error
                        }
                    }
                    Err(_) => attr_value, // Keep original on conversion error
                }
            });

            Ok(Self { inner: mapped })
        })
    }

    /// Take first n elements
    fn take(slf: PyRefMut<Self>, n: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.take(n),
        })
    }

    /// Skip first n elements
    fn skip(slf: PyRefMut<Self>, n: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.skip(n),
        })
    }

    /// Collect back into a BaseArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyBaseArray> {
        let inner = slf.inner.clone();

        // Extract the data directly
        let elements = inner.into_vec();

        // Infer dtype from first non-null element
        let dtype = elements
            .iter()
            .find(|v| !matches!(v, RustAttrValue::Null))
            .map(|v| v.dtype())
            .unwrap_or(AttrValueType::Text);

        Ok(PyBaseArray {
            inner: BaseArray::new(elements),
        })
    }
}

// =============================================================================
// Specialized typed arrays with trait-based method injection
// =============================================================================

/// Python wrapper for NodesArray - specialized array for NodeId collections
#[pyclass(name = "NodesArray", unsendable)]
pub struct PyNodesArray {
    pub inner: NodesArray,
}

#[pymethods]
impl PyNodesArray {
    /// Create a new NodesArray from node IDs
    #[new]
    fn new(node_ids: Vec<usize>) -> PyResult<Self> {
        // Convert Python usize to NodeId
        let nodes: Vec<NodeId> = node_ids.into_iter().collect();

        Ok(PyNodesArray {
            inner: NodesArray::new(nodes),
        })
    }

    /// Get the number of nodes
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get node ID by index
    fn __getitem__(&self, index: isize) -> PyResult<usize> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 { len + index } else { index };

        if actual_index < 0 || actual_index >= len {
            return Err(PyIndexError::new_err("Index out of range"));
        }

        match self.inner.get(actual_index as usize) {
            Some(node_id) => Ok(*node_id),
            None => Err(PyIndexError::new_err("Index out of range")),
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        let preview = if len == 0 {
            "empty".to_string()
        } else {
            let preview_ids: Vec<String> = self
                .inner
                .node_ids()
                .iter()
                .take(3)
                .map(|&id| id.to_string())
                .collect();
            let preview_str = preview_ids.join(", ");
            if len > 3 {
                format!("[{}, ...]", preview_str)
            } else {
                format!("[{}]", preview_str)
            }
        };

        format!("NodesArray[{}] {} node_ids", len, preview)
    }

    /// Enable fluent chaining with .iter() method for node-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyNodesArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);

        Ok(PyNodesArrayIterator {
            inner: array_iterator,
        })
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells for NodesArray data.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;

        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }

    /// Convert nodes array to table format for streaming visualization
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
        columns.insert(
            "index".to_string(),
            groggy::storage::array::BaseArray::new(indices),
        );

        // Node ID column
        let node_ids: Vec<AttrValue> = self
            .inner
            .node_ids()
            .iter()
            .map(|&node_id| AttrValue::SmallInt(node_id as i32))
            .collect();
        columns.insert(
            "node_id".to_string(),
            groggy::storage::array::BaseArray::new(node_ids),
        );

        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(
            base_table,
        ))
    }
}

/// Python wrapper for ArrayIterator<NodeId> with node-specific chaining operations
#[pyclass(name = "NodesArrayIterator", unsendable)]
pub struct PyNodesArrayIterator {
    inner: ArrayIterator<NodeId>,
}

#[pymethods]
impl PyNodesArrayIterator {
    /// Filter nodes by minimum degree
    /// Enables: g.nodes.ids().iter().filter_by_degree(3)
    fn filter_by_degree(slf: PyRefMut<Self>, min_degree: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_by_degree(min_degree),
        })
    }

    /// Get neighbors for each node  
    /// Enables: node_ids.iter().get_neighbors()
    fn get_neighbors(slf: PyRefMut<Self>) -> PyResult<PyNeighborsArrayIterator> {
        let inner = slf.inner.clone();
        let neighbors_iterator = inner.get_neighbors();

        Ok(PyNeighborsArrayIterator {
            inner: neighbors_iterator,
        })
    }

    /// Convert nodes to subgraphs
    /// Enables: node_ids.iter().to_subgraph()
    fn to_subgraph(slf: PyRefMut<Self>) -> PyResult<PySubgraphArrayIterator> {
        let inner = slf.inner.clone();
        let subgraph_iterator = inner.to_subgraph();

        Ok(PySubgraphArrayIterator {
            inner: subgraph_iterator,
        })
    }

    /// Collect back into a NodesArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyNodesArray> {
        let inner = slf.inner.clone();
        let node_ids = inner.into_vec();

        Ok(PyNodesArray {
            inner: NodesArray::new(node_ids),
        })
    }
}

/// Python wrapper for EdgesArray - specialized array for EdgeId collections
#[pyclass(name = "EdgesArray", unsendable)]
pub struct PyEdgesArray {
    pub inner: EdgesArray,
}

#[pymethods]
impl PyEdgesArray {
    /// Create a new EdgesArray from edge IDs
    #[new]
    fn new(edge_ids: Vec<usize>) -> PyResult<Self> {
        // Convert Python usize to EdgeId
        let edges: Vec<EdgeId> = edge_ids.into_iter().collect();

        Ok(PyEdgesArray {
            inner: EdgesArray::new(edges),
        })
    }

    /// Get the number of edges
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        let preview = if len == 0 {
            "empty".to_string()
        } else {
            let preview_ids: Vec<String> = self
                .inner
                .edge_ids()
                .iter()
                .take(3)
                .map(|&id| id.to_string())
                .collect();
            let preview_str = preview_ids.join(", ");
            if len > 3 {
                format!("[{}, ...]", preview_str)
            } else {
                format!("[{}]", preview_str)
            }
        };

        format!("EdgesArray[{}] {} edge_ids", len, preview)
    }

    /// Enable fluent chaining with .iter() method for edge-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyEdgesArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);

        Ok(PyEdgesArrayIterator {
            inner: array_iterator,
        })
    }

    /// Rich HTML representation for Jupyter notebooks
    ///
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells for EdgesArray data.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;

        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }

    /// Convert edges array to table format for streaming visualization
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
        columns.insert(
            "index".to_string(),
            groggy::storage::array::BaseArray::new(indices),
        );

        // Edge ID column
        let edge_ids: Vec<AttrValue> = self
            .inner
            .edge_ids()
            .iter()
            .map(|&edge_id| AttrValue::SmallInt(edge_id as i32))
            .collect();
        columns.insert(
            "edge_id".to_string(),
            groggy::storage::array::BaseArray::new(edge_ids),
        );

        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e))
        })?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(
            base_table,
        ))
    }
}

/// Python wrapper for ArrayIterator<EdgeId> with edge-specific chaining operations
#[pyclass(name = "EdgesArrayIterator", unsendable)]
pub struct PyEdgesArrayIterator {
    inner: ArrayIterator<EdgeId>,
}

#[pymethods]
impl PyEdgesArrayIterator {
    /// Filter edges by minimum weight
    /// Enables: edges.iter().filter_by_weight(0.5)
    fn filter_by_weight(slf: PyRefMut<Self>, min_weight: f64) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_by_weight(min_weight),
        })
    }

    /// Group edges by source node
    /// Enables: edges.iter().group_by_source()
    fn group_by_source(slf: PyRefMut<Self>) -> PyResult<PyEdgeGroupsIterator> {
        let inner = slf.inner.clone();
        let groups_iterator = inner.group_by_source();

        Ok(PyEdgeGroupsIterator {
            inner: groups_iterator,
        })
    }

    /// Collect back into an EdgesArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyEdgesArray> {
        let inner = slf.inner.clone();
        let edge_ids = inner.into_vec();

        Ok(PyEdgesArray {
            inner: EdgesArray::new(edge_ids),
        })
    }
}

/// Python wrapper for MetaNodeArray - specialized array for MetaNode collections
#[pyclass(name = "MetaNodeArray", unsendable)]
pub struct PyMetaNodeArray {
    pub inner: MetaNodeArray,
}

#[pymethods]
impl PyMetaNodeArray {
    /// Get the number of meta-nodes
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("MetaNodeArray[{}]", self.inner.len())
    }

    /// Enable fluent chaining with .iter() method for meta-node-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyMetaNodeArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);

        Ok(PyMetaNodeArrayIterator {
            inner: array_iterator,
        })
    }
}

/// Python wrapper for ArrayIterator<MetaNode> with meta-node-specific chaining operations
#[pyclass(name = "MetaNodeArrayIterator", unsendable)]
pub struct PyMetaNodeArrayIterator {
    inner: ArrayIterator<MetaNode>,
}

#[pymethods]
impl PyMetaNodeArrayIterator {
    /// Expand meta-nodes back into subgraphs
    /// Enables: meta_nodes.iter().expand()
    fn expand(slf: PyRefMut<Self>) -> PyResult<PySubgraphArrayIterator> {
        let inner = slf.inner.clone();
        let subgraph_iterator = inner.expand();

        Ok(PySubgraphArrayIterator {
            inner: subgraph_iterator,
        })
    }

    /// Re-aggregate meta-nodes with new aggregation functions
    fn re_aggregate(
        slf: PyRefMut<Self>,
        aggs: std::collections::HashMap<String, String>,
    ) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.re_aggregate(aggs),
        })
    }

    /// Collect back into a MetaNodeArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyMetaNodeArray> {
        let inner = slf.inner.clone();
        let meta_nodes = inner.into_vec();

        Ok(PyMetaNodeArray {
            inner: MetaNodeArray::new(meta_nodes),
        })
    }
}

// =============================================================================
// Supporting iterator types for complex return values
// =============================================================================

#[pyclass(name = "NeighborsArrayIterator", unsendable)]
pub struct PyNeighborsArrayIterator {
    inner: ArrayIterator<Vec<NodeId>>,
}

#[pymethods]
impl PyNeighborsArrayIterator {
    /// Flatten neighbor lists into a single NodesArray
    fn flatten(slf: PyRefMut<Self>) -> PyResult<PyNodesArray> {
        let inner = slf.inner.clone();
        let neighbor_lists = inner.into_vec();
        let flattened: Vec<NodeId> = neighbor_lists.into_iter().flatten().collect();

        Ok(PyNodesArray {
            inner: NodesArray::new(flattened),
        })
    }

    /// Collect into a list of lists
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<Vec<usize>>> {
        let inner = slf.inner.clone();
        let neighbor_lists = inner.into_vec();
        let py_lists: Vec<Vec<usize>> = neighbor_lists
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect())
            .collect();

        Ok(py_lists)
    }
}

#[pyclass(name = "SubgraphArrayIterator", unsendable)]
pub struct PySubgraphArrayIterator {
    inner: ArrayIterator<groggy::subgraphs::subgraph::Subgraph>,
}

#[pymethods]
impl PySubgraphArrayIterator {
    /// Filter nodes within subgraphs (inherits from SubgraphLike)
    fn filter_nodes(slf: PyRefMut<Self>, query: &str) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_nodes(query),
        })
    }

    /// Collect into a list of subgraphs
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<String>> {
        let inner = slf.inner.clone();
        let _subgraphs = inner.into_vec();

        // Placeholder: return string representations
        Ok(vec!["placeholder_subgraph".to_string()])
    }
}

#[pyclass(name = "EdgeGroupsIterator", unsendable)]
pub struct PyEdgeGroupsIterator {
    inner: ArrayIterator<Vec<EdgeId>>,
}

#[pymethods]
impl PyEdgeGroupsIterator {
    /// Collect into a list of edge groups
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<Vec<usize>>> {
        let inner = slf.inner.clone();
        let edge_groups = inner.into_vec();
        let py_groups: Vec<Vec<usize>> = edge_groups
            .into_iter()
            .map(|group| group.into_iter().collect())
            .collect();

        Ok(py_groups)
    }
}
