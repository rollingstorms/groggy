//! PyTableArray - Specialized array for table objects
//!
//! Provides a typed container for collections of table objects with full ArrayOps support

use pyo3::prelude::*;
use std::sync::Arc;

/// Specialized array for table objects (PyObject wrapper for tables)
#[pyclass(name = "TableArray", unsendable)]
#[derive(Clone)]
pub struct PyTableArray {
    /// Internal storage of tables as PyObjects
    inner: Arc<Vec<PyObject>>,
}

impl PyTableArray {
    /// Create new PyTableArray from vector of table objects
    pub fn new(tables: Vec<PyObject>) -> Self {
        Self {
            inner: Arc::new(tables),
        }
    }

    /// Create from Arc<Vec<PyObject>> for zero-copy sharing
    pub fn from_arc(tables: Arc<Vec<PyObject>>) -> Self {
        Self { inner: tables }
    }
}

#[pymethods]
impl PyTableArray {
    /// Get the number of tables
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the array is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get table at index or extract column as ArrayArray
    ///
    /// Supports two modes:
    /// - Integer index: returns the table at that position
    /// - String key: extracts column from all tables, returns ArrayArray
    fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Try string key first (column extraction)
            if let Ok(column_name) = key.extract::<String>() {
                return self.extract_column(py, &column_name);
            }

            // Try integer index
            if let Ok(index) = key.extract::<isize>() {
                let len = self.inner.len() as isize;

                // Handle negative indexing
                let actual_index = if index < 0 {
                    (len + index) as usize
                } else {
                    index as usize
                };

                if actual_index >= self.inner.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Table index {} out of range (0-{})",
                        index,
                        self.inner.len() - 1
                    )));
                }

                return Ok(self.inner[actual_index].clone());
            }

            Err(pyo3::exceptions::PyTypeError::new_err(
                "TableArray indices must be integers or strings",
            ))
        })
    }

    /// Extract a column from all tables and return as ArrayArray
    fn extract_column(&self, py: Python, column_name: &str) -> PyResult<PyObject> {
        use crate::ffi::storage::array::PyBaseArray;
        use groggy::storage::array::ArrayArray;

        let mut arrays = Vec::new();
        let mut keys = Vec::new();

        for (idx, table_obj) in self.inner.iter().enumerate() {
            // Try to get the column from each table
            if let Ok(column) = table_obj.getattr(py, column_name) {
                // Try to extract as PyBaseArray
                if let Ok(py_array) = column.extract::<PyBaseArray>(py) {
                    arrays.push(py_array.inner.clone());
                    keys.push(format!("table_{}", idx));
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Column '{}' in table {} is not a BaseArray",
                        column_name, idx
                    )));
                }
            } else {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "Column '{}' not found in table {}",
                    column_name, idx
                )));
            }
        }

        // Create ArrayArray with keys
        let array_array = ArrayArray::with_keys(arrays, keys)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Wrap in PyArrayArray
        let py_array_array = crate::PyArrayArray::from_array_array(array_array);
        Ok(py_array_array.into_py(py))
    }

    /// Iterate over tables
    fn __iter__(slf: PyRef<Self>) -> PyTableArrayIterator {
        PyTableArrayIterator {
            array: slf.into(),
            index: 0,
        }
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<PyObject> {
        self.inner.as_ref().clone()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("TableArray({} tables)", self.inner.len())
    }

    /// Collect all tables into a Python list (for compatibility with iterator patterns)
    fn collect(&self) -> Vec<PyObject> {
        self.to_list()
    }

    /// Create iterator for method chaining
    fn iter(&self) -> PyTableArrayChainIterator {
        PyTableArrayChainIterator {
            inner: self.inner.as_ref().clone(),
        }
    }

    /// Apply aggregation to all tables - placeholder method
    fn agg(&self, py: Python, agg_spec: PyObject) -> PyResult<Vec<PyObject>> {
        let mut aggregated = Vec::new();

        for table in self.inner.iter() {
            // Try to call agg method on each table
            if table.as_ref(py).hasattr("agg")? {
                match table.call_method1(py, "agg", (agg_spec.clone(),)) {
                    Ok(result) => aggregated.push(result),
                    Err(_) => continue, // Skip failed aggregations
                }
            }
        }

        Ok(aggregated)
    }

    /// Filter all tables using a query - placeholder method
    fn filter(&self, py: Python, query: String) -> PyResult<PyTableArray> {
        let mut filtered = Vec::new();

        for table in self.inner.iter() {
            // Try to call filter method on each table
            if table.as_ref(py).hasattr("filter")? {
                match table.call_method1(py, "filter", (query.clone(),)) {
                    Ok(result) => filtered.push(result),
                    Err(_) => continue, // Skip failed filters
                }
            }
        }

        Ok(PyTableArray::new(filtered))
    }

    /// Map a function over all tables and return a BaseArray
    ///
    /// Args:
    ///     func: Python callable that takes a table and returns a numeric value
    ///
    /// Returns:
    ///     BaseArray containing the results
    ///
    /// Example:
    /// ```python
    /// # Get row count for each table
    /// row_counts = table_array.map(lambda t: len(t))
    ///
    /// # Get average of a column
    /// avgs = table_array.map(lambda t: t['value'].mean())
    /// ```
    fn map(&self, py: Python, func: PyObject) -> PyResult<crate::ffi::storage::array::PyBaseArray> {
        use groggy::storage::array::BaseArray;
        use groggy::types::AttrValue;

        let mut results = Vec::new();

        for table in self.inner.iter() {
            // Call function with table
            let result = func.call1(py, (table.clone(),))?;

            // Convert result to AttrValue
            let attr_value = if let Ok(f) = result.extract::<f64>(py) {
                AttrValue::Float(f as f32)
            } else if let Ok(i) = result.extract::<i64>(py) {
                AttrValue::Int(i)
            } else if let Ok(s) = result.extract::<String>(py) {
                AttrValue::Text(s)
            } else if let Ok(b) = result.extract::<bool>(py) {
                AttrValue::Bool(b)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Map function must return int, float, str, or bool",
                ));
            };

            results.push(attr_value);
        }

        // Create BaseArray from results and wrap in PyBaseArray
        let base_array = BaseArray::from_attr_values(results);
        Ok(crate::ffi::storage::array::PyBaseArray { inner: base_array })
    }
}

// From implementations for easy conversion
impl From<Vec<PyObject>> for PyTableArray {
    fn from(tables: Vec<PyObject>) -> Self {
        Self::new(tables)
    }
}

impl From<PyTableArray> for Vec<PyObject> {
    fn from(array: PyTableArray) -> Self {
        Arc::try_unwrap(array.inner).unwrap_or_else(|arc| arc.as_ref().clone())
    }
}

/// Python iterator for PyTableArray
#[pyclass]
pub struct PyTableArrayIterator {
    array: Py<PyTableArray>,
    index: usize,
}

#[pymethods]
impl PyTableArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        let array = self.array.borrow(py);
        if self.index < array.inner.len() {
            let result = array.inner[self.index].clone();
            self.index += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}

/// Chainable iterator for PyTableArray that supports method forwarding
#[pyclass(name = "TableArrayIterator", unsendable)]
pub struct PyTableArrayChainIterator {
    inner: Vec<PyObject>,
}

#[pymethods]
impl PyTableArrayChainIterator {
    /// Apply aggregation to each table and return list of results
    fn agg(&mut self, py: Python, agg_spec: PyObject) -> PyResult<Vec<PyObject>> {
        let mut aggregated = Vec::new();

        for table in &self.inner {
            // Try to call agg method on each table
            if table.as_ref(py).hasattr("agg")? {
                match table.call_method1(py, "agg", (agg_spec.clone(),)) {
                    Ok(result) => aggregated.push(result),
                    Err(_) => continue, // Skip failed aggregations
                }
            }
        }

        Ok(aggregated)
    }

    /// Apply filter to each table
    fn filter(&mut self, py: Python, query: String) -> PyResult<Self> {
        let mut filtered = Vec::new();

        for table in &self.inner {
            // Try to call filter method on each table
            if table.as_ref(py).hasattr("filter")? {
                match table.call_method1(py, "filter", (query.clone(),)) {
                    Ok(result) => filtered.push(result),
                    Err(_) => continue, // Skip failed filters
                }
            }
        }

        Ok(Self { inner: filtered })
    }

    /// Materialize iterator back into PyTableArray
    fn collect(&mut self) -> PyResult<PyTableArray> {
        Ok(PyTableArray::new(self.inner.clone()))
    }

    /// Apply group_by to each table - placeholder method
    fn group_by(&mut self, py: Python, columns: Vec<String>) -> PyResult<Self> {
        let mut grouped = Vec::new();

        for table in &self.inner {
            // Try to call group_by method on each table
            if table.as_ref(py).hasattr("group_by")? {
                match table.call_method1(py, "group_by", (columns.clone(),)) {
                    Ok(result) => grouped.push(result),
                    Err(_) => continue, // Skip failed group_by
                }
            }
        }

        Ok(Self { inner: grouped })
    }

    /// Join with another iterator of tables - simplified implementation
    fn join(&mut self, py: Python, other: &Self, on: String) -> PyResult<Self> {
        let mut joined = Vec::new();

        // Simple cartesian join - in production would be more sophisticated
        for (i, left_table) in self.inner.iter().enumerate() {
            if let Some(right_table) = other.inner.get(i) {
                // Try to call join method on left table with right table
                if left_table.as_ref(py).hasattr("join")? {
                    match left_table.call_method1(py, "join", (right_table.clone(), on.clone())) {
                        Ok(result) => joined.push(result),
                        Err(_) => continue, // Skip failed joins
                    }
                }
            }
        }

        Ok(Self { inner: joined })
    }

    /// Take first n tables
    fn take(&mut self, n: usize) -> PyResult<Self> {
        let taken: Vec<PyObject> = self.inner.iter().take(n).cloned().collect();
        Ok(Self { inner: taken })
    }

    /// Skip first n tables
    fn skip(&mut self, n: usize) -> PyResult<Self> {
        let skipped: Vec<PyObject> = self.inner.iter().skip(n).cloned().collect();
        Ok(Self { inner: skipped })
    }
}
