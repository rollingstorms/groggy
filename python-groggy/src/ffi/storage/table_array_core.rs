//! Core-delegating TableArray FFI implementation
//!
//! This replaces the old PyTableArray with one that delegates to the core TableArray

use groggy::storage::table::TableArray as CoreTableArray;
use groggy::storage::table::BaseTable;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Python wrapper for core TableArray
#[pyclass(name = "TableArray")]
pub struct PyTableArrayCore {
    pub inner: CoreTableArray,
}

#[pymethods]
impl PyTableArrayCore {
    /// Create new TableArray from list of tables
    #[new]
    fn new(tables: Vec<PyObject>, py: Python) -> PyResult<Self> {
        let mut core_tables = Vec::new();

        for table_obj in tables {
            // Convert PyObject to BaseTable
            if let Ok(py_table) = table_obj.extract::<crate::ffi::storage::table::PyBaseTable>(py) {
                core_tables.push(py_table.table.clone());
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "All elements must be BaseTable instances"
                ));
            }
        }

        Ok(Self {
            inner: CoreTableArray::from_tables(core_tables),
        })
    }

    /// Get the number of tables
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get table at index
    fn __getitem__(&self, index: isize) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };

        if let Some(table) = self.inner.get(actual_index) {
            Ok(crate::ffi::storage::table::PyBaseTable::from_table(table.clone()))
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Table index {} out of range", index
            )))
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("TableArray({} tables)", self.inner.len())
    }

    /// Sum aggregation across all tables
    fn sum(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.sum()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Sum failed: {}", e)))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Mean aggregation across all tables
    fn mean(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.mean()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Mean failed: {}", e)))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Count aggregation across all tables
    fn count(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.count()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Count failed: {}", e)))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Generic aggregation with function specifications
    fn agg(&self, agg_spec: &PyDict) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let mut agg_functions = HashMap::new();

        for (key, value) in agg_spec.iter() {
            let col_name: String = key.extract()?;
            let func_name: String = value.extract()?;
            agg_functions.insert(col_name, func_name);
        }

        let result = self.inner.agg(agg_functions)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Agg failed: {}", e)))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Take first n tables
    fn take(&self, n: usize) -> Self {
        Self {
            inner: self.inner.take(n),
        }
    }

    /// Skip first n tables
    fn skip(&self, n: usize) -> Self {
        Self {
            inner: self.inner.skip(n),
        }
    }

    /// Concatenate all tables into one
    fn concat(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let result = self.inner.concat()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Concat failed: {}", e)))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(result))
    }

    /// Get shape (num_tables, avg_rows_per_table)
    fn shape(&self) -> (usize, f64) {
        self.inner.shape()
    }

    /// Filter tables with a Python predicate function
    fn filter(&self, predicate: PyObject, py: Python) -> PyResult<Self> {
        let filtered = self.inner.filter(|table| {
            // Convert table to PyBaseTable and call predicate
            let py_table = crate::ffi::storage::table::PyBaseTable::from_table(table.clone());

            match predicate.call1(py, (py_table,)) {
                Ok(result) => result.is_true(py).unwrap_or(false),
                Err(_) => false,
            }
        });

        Ok(Self {
            inner: filtered,
        })
    }

    /// Convert to list of tables
    fn to_list(&self) -> PyResult<Vec<crate::ffi::storage::table::PyBaseTable>> {
        let mut tables = Vec::new();
        for table in self.inner.iter() {
            tables.push(crate::ffi::storage::table::PyBaseTable::from_table(table.clone()));
        }
        Ok(tables)
    }

    /// Apply a function to each table
    fn map(&self, func: PyObject, py: Python) -> PyResult<Self> {
        let mapped = self.inner.map(|table| {
            // Convert table to PyBaseTable, call function, convert back
            let py_table = crate::ffi::storage::table::PyBaseTable::from_table(table.clone());

            match func.call1(py, (py_table,)) {
                Ok(result) => {
                    if let Ok(result_table) = result.extract::<crate::ffi::storage::table::PyBaseTable>(py) {
                        Ok(result_table.table)
                    } else {
                        Err(groggy::errors::GraphError::InvalidInput(
                            "Function must return a BaseTable".to_string()
                        ))
                    }
                }
                Err(e) => Err(groggy::errors::GraphError::InvalidInput(
                    format!("Function call failed: {}", e)
                ))
            }
        }).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Map failed: {}", e)))?;

        Ok(Self {
            inner: mapped,
        })
    }

    /// Iterator support
    fn __iter__(slf: PyRef<Self>) -> PyTableArrayCoreIterator {
        PyTableArrayCoreIterator {
            inner: slf.inner.clone(),
            index: 0,
        }
    }

    /// Convenience method for fluent chaining
    fn iter(&self) -> PyTableArrayCoreIterator {
        PyTableArrayCoreIterator {
            inner: self.inner.clone(),
            index: 0,
        }
    }
}

/// Python iterator for core TableArray
#[pyclass]
pub struct PyTableArrayCoreIterator {
    inner: CoreTableArray,
    index: usize,
}

#[pymethods]
impl PyTableArrayCoreIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<crate::ffi::storage::table::PyBaseTable>> {
        if self.index < self.inner.len() {
            if let Some(table) = self.inner.get(self.index) {
                self.index += 1;
                Ok(Some(crate::ffi::storage::table::PyBaseTable::from_table(table.clone())))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Collect back to TableArray
    fn collect(&self) -> PyTableArrayCore {
        PyTableArrayCore {
            inner: self.inner.clone(),
        }
    }
}