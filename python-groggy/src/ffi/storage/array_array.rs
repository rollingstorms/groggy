//! PyArrayArray - FFI wrapper for ArrayArray
//!
//! Provides Python bindings for ArrayArray with aggregation operations.

use groggy::storage::array::{ArrayArray, BaseArray};
use groggy::types::AttrValue;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::array::PyBaseArray;
use super::table::PyBaseTable;

/// Python wrapper for ArrayArray
#[pyclass(name = "ArrayArray", module = "groggy")]
#[derive(Clone)]
pub struct PyArrayArray {
    /// Inner ArrayArray supporting all AttrValue types
    inner: ArrayArray<AttrValue>,
}

impl PyArrayArray {
    /// Create from Rust ArrayArray of AttrValues
    pub fn from_array_array(arr: ArrayArray<AttrValue>) -> Self {
        Self { inner: arr }
    }

    /// Create from vector of BaseArrays
    pub fn new(arrays: Vec<BaseArray<AttrValue>>) -> Self {
        Self {
            inner: ArrayArray::new(arrays),
        }
    }

    /// Create with keys
    pub fn with_keys(arrays: Vec<BaseArray<AttrValue>>, keys: Vec<String>) -> PyResult<Self> {
        ArrayArray::with_keys(arrays, keys)
            .map(|arr| Self { inner: arr })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Helper: Convert AttrValue arrays to numeric f64 arrays for aggregations
    /// Converts: Int, SmallInt, Float, Bool (1.0/0.0)
    /// Skips: Text, Null, and other non-numeric types
    fn to_numeric_array_array(&self) -> ArrayArray<f64> {
        // Access the inner arrays by iterating through the ArrayArray
        // We need to collect all arrays and convert each one
        let mut numeric_arrays = Vec::new();

        for i in 0..self.inner.len() {
            if let Some(arr) = self.inner.get(i) {
                let numeric_values: Vec<f64> = arr
                    .iter()
                    .filter_map(|val| match val {
                        AttrValue::Float(f) => Some(*f as f64),
                        AttrValue::Int(i) => Some(*i as f64),
                        AttrValue::SmallInt(i) => Some(*i as f64),
                        AttrValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                        _ => None,
                    })
                    .collect();
                numeric_arrays.push(BaseArray::from(numeric_values));
            }
        }

        if let Some(keys) = self.inner.keys() {
            ArrayArray::with_keys(numeric_arrays, keys.to_vec()).unwrap()
        } else {
            ArrayArray::new(numeric_arrays)
        }
    }
}

#[pymethods]
impl PyArrayArray {
    /// Create new ArrayArray from list of arrays
    #[new]
    fn py_new() -> Self {
        // For now, create an empty ArrayArray
        // Users will add arrays via other methods
        Self {
            inner: ArrayArray::new(vec![]),
        }
    }

    /// Get the number of arrays
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Retrieve an individual BaseArray by index (supports negative indices)
    fn __getitem__(&self, index: isize) -> PyResult<PyBaseArray> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 { len + index } else { index };

        if actual_index < 0 || actual_index >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Array index {} out of range for {} arrays",
                index, len
            )));
        }

        let base_array = self
            .inner
            .get(actual_index as usize)
            .cloned()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Array retrieval failed"))?;

        Ok(PyBaseArray { inner: base_array })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("ArrayArray({} arrays)", self.inner.len())
    }

    // --- Aggregation Methods ---

    /// Calculate mean of each array
    ///
    /// Returns a BaseTable with group keys and means if keys are present,
    /// otherwise returns a Python list of means.
    ///
    /// Example:
    /// ```python
    /// arr_arr = ArrayArray([
    ///     BaseArray([1.0, 2.0, 3.0]),
    ///     BaseArray([4.0, 5.0, 6.0])
    /// ])
    /// means = arr_arr.mean()  # [2.0, 5.0]
    /// ```
    fn mean(&self) -> PyResult<PyObject> {
        let numeric_arrays = self.to_numeric_array_array();
        let means = numeric_arrays.mean();

        Python::with_gil(|py| {
            if self.inner.keys().is_some() {
                // Package as table with keys
                numeric_arrays
                    .to_table_with_aggregation("mean", means)
                    .map(|table| PyBaseTable::from_table(table).into_py(py))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                // Return as Python list
                Ok(means.into_py(py))
            }
        })
    }

    /// Calculate sum of each array
    fn sum(&self) -> PyResult<PyObject> {
        let numeric_arrays = self.to_numeric_array_array();
        let sums = numeric_arrays.sum();

        Python::with_gil(|py| {
            if self.inner.keys().is_some() {
                numeric_arrays
                    .to_table_with_aggregation("sum", sums)
                    .map(|table| PyBaseTable::from_table(table).into_py(py))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                Ok(sums.into_py(py))
            }
        })
    }

    /// Calculate minimum of each array
    fn min(&self) -> PyResult<PyObject> {
        let numeric_arrays = self.to_numeric_array_array();
        let mins = numeric_arrays.min();

        Python::with_gil(|py| {
            if self.inner.keys().is_some() {
                numeric_arrays
                    .to_table_with_aggregation("min", mins)
                    .map(|table| PyBaseTable::from_table(table).into_py(py))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                Ok(mins.into_py(py))
            }
        })
    }

    /// Calculate maximum of each array
    fn max(&self) -> PyResult<PyObject> {
        let numeric_arrays = self.to_numeric_array_array();
        let maxs = numeric_arrays.max();

        Python::with_gil(|py| {
            if self.inner.keys().is_some() {
                numeric_arrays
                    .to_table_with_aggregation("max", maxs)
                    .map(|table| PyBaseTable::from_table(table).into_py(py))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                Ok(maxs.into_py(py))
            }
        })
    }

    /// Calculate standard deviation of each array
    fn std(&self) -> PyResult<PyObject> {
        let numeric_arrays = self.to_numeric_array_array();
        let stds = numeric_arrays.std();

        Python::with_gil(|py| {
            if self.inner.keys().is_some() {
                numeric_arrays
                    .to_table_with_aggregation("std", stds)
                    .map(|table| PyBaseTable::from_table(table).into_py(py))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                Ok(stds.into_py(py))
            }
        })
    }

    /// Count elements in each array
    fn count(&self) -> PyResult<PyObject> {
        let numeric_arrays = self.to_numeric_array_array();
        let counts = numeric_arrays.count();

        Python::with_gil(|py| {
            if self.inner.keys().is_some() {
                let counts_f64: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
                numeric_arrays
                    .to_table_with_aggregation("count", counts_f64)
                    .map(|table| PyBaseTable::from_table(table).into_py(py))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            } else {
                Ok(counts.into_py(py))
            }
        })
    }

    /// Apply multiple aggregations at once
    ///
    /// Args:
    ///     spec: Dictionary mapping aggregation names to functions
    ///           e.g., {"mean": "mean", "std": "std"}
    ///
    /// Returns:
    ///     BaseTable with columns for each aggregation
    ///
    /// Example:
    /// ```python
    /// result = arr_arr.agg({"mean": "mean", "std": "std", "count": "count"})
    /// # Returns table with columns: [group_key, mean, std, count]
    /// ```
    fn agg(&self, spec: &PyDict) -> PyResult<PyBaseTable> {
        use groggy::storage::array::BaseArray;
        use groggy::types::AttrValue;

        let mut columns = std::collections::HashMap::new();

        // Add group keys if present
        if let Some(keys) = self.inner.keys() {
            let key_col: Vec<AttrValue> = keys.iter().map(|s| AttrValue::Text(s.clone())).collect();
            columns.insert(
                "group_key".to_string(),
                BaseArray::from_attr_values(key_col),
            );
        }

        // Apply each aggregation
        let numeric_arrays = self.to_numeric_array_array();

        for (col_name, agg_func) in spec.iter() {
            let col_name_str: String = col_name.extract()?;
            let agg_func_str: String = agg_func.extract()?;

            let values: Vec<f64> = match agg_func_str.as_str() {
                "mean" => numeric_arrays.mean(),
                "sum" => numeric_arrays.sum(),
                "min" => numeric_arrays.min(),
                "max" => numeric_arrays.max(),
                "std" => numeric_arrays.std(),
                "count" => numeric_arrays.count().iter().map(|&c| c as f64).collect(),
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown aggregation function: {}",
                        agg_func_str
                    )))
                }
            };

            let value_col: Vec<AttrValue> =
                values.iter().map(|&v| AttrValue::Float(v as f32)).collect();
            columns.insert(col_name_str, BaseArray::from_attr_values(value_col));
        }

        groggy::storage::table::BaseTable::from_columns(columns)
            .map(PyBaseTable::from_table)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get keys if available
    fn keys(&self) -> Option<Vec<String>> {
        self.inner.keys().cloned()
    }

    /// Convert ArrayArray to numeric NumArray for operations
    ///
    /// Converts all arrays to f64 and returns as NumArray.
    /// Non-numeric values are filtered out during conversion.
    ///
    /// Example:
    /// ```python
    /// arr_arr = ArrayArray([
    ///     BaseArray([1, 2, 3]),
    ///     BaseArray([4.0, 5.0, 6.0])
    /// ])
    /// num_arr = arr_arr.to_type("float64")  # Returns NumArray with aggregations
    /// ```
    fn to_type(&self, py: Python, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "float64" | "f64" | "numeric" => {
                // Convert to numeric and flatten into a single NumArray
                let numeric_arrays = self.to_numeric_array_array();

                // Flatten all arrays into a single vector
                let mut all_values = Vec::new();
                for i in 0..numeric_arrays.len() {
                    if let Some(arr) = numeric_arrays.get(i) {
                        for val in arr.iter() {
                            all_values.push(*val);
                        }
                    }
                }

                let num_array = crate::ffi::storage::num_array::PyNumArray::new_float64(all_values);
                Ok(num_array.into_py(py))
            }
            "arrayarray" => {
                // Return self as-is
                Ok(self.clone().into_py(py))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported type conversion: {}. Use 'float64', 'numeric', or 'arrayarray'",
                dtype
            ))),
        }
    }

    // Temporarily comment out to_list and iteration until we properly implement PyBaseArray conversion
    // fn to_list(&self) -> Vec<PyBaseArray> { ... }
    // fn __iter__(slf: PyRef<Self>) -> PyArrayArrayIterator { ... }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_array_array_creation() {
        use groggy::types::AttrValue;

        let arrays = vec![
            BaseArray::from_attr_values(vec![
                AttrValue::Float(1.0),
                AttrValue::Float(2.0),
                AttrValue::Float(3.0),
            ]),
            BaseArray::from_attr_values(vec![
                AttrValue::Float(4.0),
                AttrValue::Float(5.0),
                AttrValue::Float(6.0),
            ]),
        ];
        let arr_arr = PyArrayArray::new(arrays);

        assert_eq!(arr_arr.__len__(), 2);
        assert!(!arr_arr.is_empty());
    }

    #[test]
    fn test_py_array_array_getitem() {
        use groggy::types::AttrValue;
        use pyo3::Python;

        Python::with_gil(|_| {
            let arrays = vec![
                BaseArray::from_attr_values(vec![
                    AttrValue::Int(1),
                    AttrValue::Int(2),
                    AttrValue::Int(3),
                ]),
                BaseArray::from_attr_values(vec![AttrValue::Int(4), AttrValue::Int(5)]),
            ];

            let wrapper = PyArrayArray::from_array_array(ArrayArray::new(arrays));

            let first = wrapper.__getitem__(0).expect("index 0 should succeed");
            assert_eq!(first.inner.len(), 3);

            let last = wrapper.__getitem__(-1).expect("negative index should work");
            assert_eq!(last.inner.len(), 2);

            assert!(wrapper.__getitem__(2).is_err());
        });
    }

    #[test]
    fn test_py_array_array_aggregations() {
        use groggy::types::AttrValue;

        let arrays = vec![
            BaseArray::from_attr_values(vec![
                AttrValue::Float(1.0),
                AttrValue::Float(2.0),
                AttrValue::Float(3.0),
            ]),
            BaseArray::from_attr_values(vec![
                AttrValue::Float(4.0),
                AttrValue::Float(5.0),
                AttrValue::Float(6.0),
            ]),
        ];
        let arr_arr = PyArrayArray::new(arrays);

        // Test aggregations work through numeric conversion
        let numeric = arr_arr.to_numeric_array_array();
        assert_eq!(numeric.mean(), vec![2.0, 5.0]);
        assert_eq!(numeric.sum(), vec![6.0, 15.0]);
    }
}
