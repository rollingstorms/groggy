//! Python FFI for BaseTable system

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use crate::ffi::storage::array::PyBaseArray;
use groggy::storage::{ArrayOps, array::BaseArray, table::{BaseTable, NodesTable, EdgesTable, Table, TableIterator}};
use groggy::GraphArray;  // Correct import path
use groggy::types::{NodeId, EdgeId, AttrValue, AttrValueType};
use std::collections::HashMap;
use serde_json::{Map, Value};

// =============================================================================
// PyBaseTable - Python wrapper for BaseTable
// =============================================================================

/// Python wrapper for BaseTable
#[pyclass(name = "BaseTable", module = "groggy")]
#[derive(Clone)]
pub struct PyBaseTable {
    pub(crate) table: BaseTable,
}

#[pymethods]
impl PyBaseTable {
    /// Create a new empty BaseTable
    #[new]
    pub fn new() -> Self {
        Self {
            table: BaseTable::new(),
        }
    }

    /// Create BaseTable from a Python dictionary
    /// 
    /// # Arguments
    /// * `data` - Dictionary mapping column names to lists of values
    /// 
    /// # Examples
    /// ```python
    /// data = {
    ///     'id': [1, 2, 3],
    ///     'name': ['Alice', 'Bob', 'Charlie'],
    ///     'age': [25, 30, 35]
    /// }
    /// table = BaseTable.from_dict(data)
    /// ```
    #[classmethod]
    pub fn from_dict(_cls: &PyType, py: Python, data: &PyDict) -> PyResult<Py<PyBaseTable>> {
        let mut columns = HashMap::new();
        
        for (col_name_py, values_py) in data.iter() {
            let col_name: String = col_name_py.extract()?;
            
            // Convert Python list to Vec<AttrValue>
            let values_list: &pyo3::types::PyList = values_py.extract()?;
            let mut attr_values = Vec::new();
            
            for value_py in values_list.iter() {
                let attr_value = crate::ffi::utils::python_value_to_attr_value(value_py)?;
                attr_values.push(attr_value);
            }
            
            // Infer dtype from the first non-null value, or default to Text
            let dtype = attr_values.iter()
                .find(|v| !matches!(v, AttrValue::Null))
                .map(|v| match v {
                    AttrValue::Int(_) => AttrValueType::Int,
                    AttrValue::SmallInt(_) => AttrValueType::SmallInt,
                    AttrValue::Float(_) => AttrValueType::Float,
                    AttrValue::Text(_) => AttrValueType::Text,
                    AttrValue::CompactText(_) => AttrValueType::CompactText,
                    AttrValue::Bool(_) => AttrValueType::Bool,
                    AttrValue::FloatVec(_) => AttrValueType::FloatVec,
                    AttrValue::Bytes(_) => AttrValueType::Bytes,
                    AttrValue::Null => AttrValueType::Null,
                    _ => AttrValueType::Text, // fallback for other types
                })
                .unwrap_or(AttrValueType::Text);
            
            columns.insert(col_name, BaseArray::new(attr_values));
        }
        
        let table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e)))?;
        
        Py::new(py, PyBaseTable { table })
    }
    
    /// Get number of rows
    #[getter]
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns
    #[getter]
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get column names
    #[getter]
    pub fn column_names(&self) -> Vec<String> {
        self.table.column_names().to_vec()
    }
    
    /// Get column names (alias for column_names)
    #[getter]
    pub fn columns(&self) -> Vec<String> {
        self.column_names()
    }
    
    /// Get a specific column as BaseArray for chaining operations
    /// This enables: table.column('age').iter().filter(...).collect()
    pub fn column(&self, column_name: &str) -> PyResult<crate::ffi::storage::array::PyBaseArray> {
        match self.table.column(column_name) {
            Some(base_array) => {
                Ok(crate::ffi::storage::array::PyBaseArray {
                    inner: base_array.clone()
                })
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Column '{}' not found", column_name)
            ))
        }
    }
    
    /// Get shape as (rows, cols)
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Check if column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.table.has_column(name)
    }
    
    /// Get first n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn head(&self, py: Python, n: usize) -> PyResult<Py<Self>> {
        let result_table = self.table.head(n);
        Py::new(py, Self {
            table: result_table,
        })
    }
    
    /// Get last n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn tail(&self, py: Python, n: usize) -> PyResult<Py<Self>> {
        let result_table = self.table.tail(n);
        Py::new(py, Self {
            table: result_table,
        })
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyBaseTableIterator {
        PyBaseTableIterator {
            iterator: self.table.iter(),
        }
    }
    
    /// Sort table by column
    /// 
    /// Args:
    ///     column: Name of the column to sort by
    ///     ascending: If True, sort in ascending order; if False, descending
    /// 
    /// Returns:
    ///     PyBaseTable: A new sorted table
    #[pyo3(signature = (column, ascending = true))]
    pub fn sort_by(&self, py: Python, column: &str, ascending: bool) -> PyResult<Py<Self>> {
        let sorted_table = self.table.sort_by(column, ascending)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Py::new(py, Self { table: sorted_table })
    }
    
    /// Select specific columns to create a new table
    ///
    /// Args:
    ///     columns: List of column names to select
    ///
    /// Returns:
    ///     PyBaseTable: A new table with only the selected columns
    pub fn select(&self, py: Python, columns: Vec<String>) -> PyResult<Py<Self>> {
        let selected_table = self.table.select(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Py::new(py, Self { table: selected_table })
    }
    
    /// Drop columns from the table
    ///
    /// Args:
    ///     columns: List of column names to drop
    ///
    /// Returns:
    ///     PyBaseTable: A new table without the specified columns
    pub fn drop_columns(&self, py: Python, columns: Vec<String>) -> PyResult<Py<Self>> {
        let new_table = self.table.drop_columns(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Py::new(py, Self { table: new_table })
    }
    
    // =============================================================================
    // Setting Methods - Comprehensive assignment and modification operations
    // =============================================================================
    
    /// Assign updates to multiple columns at once
    /// 
    /// Args:
    ///     updates: Dictionary mapping column names to values. Values can be:
    ///              - Lists: ['value1', 'value2', ...]  
    ///              - Dictionaries with integer keys: {0: 'value1', 1: 'value2', ...}
    ///     
    /// Examples:
    ///     # Using lists (updates entire columns)
    ///     updates = {"bonus": [1000, 1500], "status": ["active", "inactive"]}
    ///     table.assign(updates)
    ///     
    ///     # Using dictionaries with integer keys (sparse updates)
    ///     updates = {"bonus": {0: 1000, 3: 1500}, "status": {1: "active", 2: "inactive"}}
    ///     table.assign(updates)
    pub fn assign(&mut self, updates: &PyDict) -> PyResult<()> {
        for (col_name_py, values_py) in updates.iter() {
            let col_name: String = col_name_py.extract()?;
            
            if let Ok(values_list) = values_py.extract::<&pyo3::types::PyList>() {
                // Handle list format: [value1, value2, ...] - full column replacement
                let mut attr_values = Vec::new();
                for value_py in values_list.iter() {
                    let attr_value = crate::ffi::utils::python_value_to_attr_value(value_py)?;
                    attr_values.push(attr_value);
                }
                
                // Use set_column for full column updates
                self.table.set_column(&col_name, attr_values)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                
            } else if let Ok(values_dict) = values_py.extract::<&pyo3::types::PyDict>() {
                // Handle dictionary format: {0: value1, 1: value2, ...} - sparse updates
                
                // Check if column exists, create it if not
                if !self.table.has_column(&col_name) {
                    // Create a new column filled with nulls
                    let null_values = vec![groggy::AttrValue::Null; self.table.nrows()];
                    self.table.set_column(&col_name, null_values)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                }
                
                // Now update specific cells
                for (key_py, value_py) in values_dict.iter() {
                    let row_index: usize = key_py.extract()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Dictionary keys must be integers (row indices)"
                        ))?;
                    let attr_value = crate::ffi::utils::python_value_to_attr_value(value_py)?;
                    
                    // Use set_value for individual cell updates
                    self.table.set_value(row_index, &col_name, attr_value)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    format!("Values for column '{}' must be either a list or a dictionary with integer keys", col_name)
                ));
            };
        }
        
        Ok(())
    }
    
    /// Set an entire column with new values
    /// 
    /// Args:
    ///     column_name: Name of the column to set
    ///     values: List of new values for the column
    ///     
    /// Example:
    ///     table.set_column("score", [95, 87, 92, 88])
    pub fn set_column(&mut self, column_name: &str, values: &pyo3::types::PyList) -> PyResult<()> {
        let mut attr_values = Vec::new();
        for value_py in values.iter() {
            let attr_value = crate::ffi::utils::python_value_to_attr_value(value_py)?;
            attr_values.push(attr_value);
        }
        
        self.table.set_column(column_name, attr_values)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(())
    }
    
    /// Set a single value at a specific row and column
    /// 
    /// Args:
    ///     row: Row index (0-based)
    ///     column_name: Name of the column
    ///     value: New value to set
    ///     
    /// Example:
    ///     table.set_value(0, "name", "Alice Updated")
    pub fn set_value(&mut self, row: usize, column_name: &str, value: &PyAny) -> PyResult<()> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        
        self.table.set_value(row, column_name, attr_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(())
    }
    
    /// Set values for multiple rows in a column using a boolean mask
    /// 
    /// Args:
    ///     mask: List of booleans indicating which rows to update
    ///     column_name: Name of the column to update
    ///     value: Value to set for all masked rows
    ///     
    /// Example:
    ///     table.set_values_by_mask([True, False, True], "flag", "updated")
    pub fn set_values_by_mask(&mut self, mask: Vec<bool>, column_name: &str, value: &PyAny) -> PyResult<()> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        
        self.table.set_values_by_mask(&mask, column_name, attr_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(())
    }
    
    /// Set values for a range of rows in a column
    /// 
    /// Args:
    ///     start: Starting row index (inclusive)
    ///     end: Ending row index (exclusive)
    ///     step: Step size (default 1 for consecutive rows)
    ///     column_name: Name of the column to update
    ///     value: Value to set for all rows in the range
    ///     
    /// Example:
    ///     table.set_values_by_range(10, 20, 1, "score", 0.0)  # rows 10-19
    ///     table.set_values_by_range(0, 10, 2, "flag", True)   # rows 0,2,4,6,8
    #[pyo3(signature = (start, end, column_name, value, step = 1))]
    pub fn set_values_by_range(&mut self, start: usize, end: usize, column_name: &str, value: &PyAny, step: usize) -> PyResult<()> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        
        self.table.set_values_by_range(start, end, step, column_name, attr_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(())
    }
    
    /// Enable slice-based setting: table[rows, columns] = value
    /// 
    /// Supports multiple syntax forms:
    /// - t[10:20, "score"] = 0.0          # Set range of rows in one column
    /// - t[::2, ["a","b"]] = [1, 2]       # Set every 2nd row in multiple columns
    /// - t[mask, "note"] = "keep"         # Set rows matching boolean condition
    /// - t[5, "name"] = "Alice"           # Set single cell
    pub fn __setitem__(&mut self, key: &PyAny, value: &PyAny) -> PyResult<()> {
        use pyo3::types::{PySlice, PyTuple, PyList, PyString};
        
        // Handle different key types
        if let Ok(tuple) = key.downcast::<PyTuple>() {
            // Multi-dimensional indexing: table[rows, columns] = value
            if tuple.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Table indexing requires exactly 2 dimensions: [rows, columns]"
                ));
            }
            
            let row_key = tuple.get_item(0)?;
            let col_key = tuple.get_item(1)?;
            
            // Parse column specification
            let column_names = self.parse_column_key(col_key)?;
            
            // Parse row specification and apply updates
            self.apply_row_column_update(row_key, &column_names, value)?;
            
        } else {
            // Single-dimensional indexing (assume column-only): table["column"] = values
            let column_names = self.parse_column_key(key)?;
            if column_names.len() != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Single-dimensional setting requires exactly one column"
                ));
            }
            
            // Set entire column
            if let Ok(values_list) = value.downcast::<PyList>() {
                self.set_column(&column_names[0], values_list)?;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Setting entire column requires a list of values"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Filter rows using a query expression or Python function
    ///
    /// Args:
    ///     predicate: Either a string query expression (e.g. "age > 25") or a Python function
    ///
    /// Returns:
    ///     PyBaseTable: A new table with filtered rows
    pub fn filter(&self, predicate: &PyAny) -> PyResult<Self> {
        let filtered_table = if predicate.extract::<String>().is_ok() {
            // String predicate
            let pred_str = predicate.extract::<String>()?;
            self.table.filter(&pred_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else {
            // Python function predicate
            self.filter_by_python_function(predicate)?
        };
        
        Ok(Self { table: filtered_table })
    }
    
    /// Group by columns and return grouped tables
    ///
    /// Args:
    ///     columns: List of column names to group by
    ///
    /// Returns:
    ///     PyTableArray: Array-like container holding the grouped tables
    pub fn group_by(&self, columns: Vec<String>) -> PyResult<PyTableArray> {
        let grouped_tables = self.table.group_by(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert to PyBaseTable objects
        let py_tables: Vec<PyBaseTable> = grouped_tables.into_iter()
            .map(|table| PyBaseTable { table })
            .collect();
        
        Ok(PyTableArray { 
            tables: py_tables,
            group_columns: columns,
        })
    }
    
    /// Get a slice of rows [start, end)
    ///
    /// Args:
    ///     start: Starting row index (inclusive)
    ///     end: Ending row index (exclusive)
    ///
    /// Returns:
    ///     PyBaseTable: A new table with the specified row slice
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            table: self.table.slice(start, end),
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        // Simple fallback for now - return basic info
        format!("BaseTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("BaseTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
    
    
    /// Get length (number of rows) for len() function
    pub fn __len__(&self) -> usize {
        self.table.nrows()
    }
    
    /// Support iteration over rows: for row in table:
    pub fn __iter__(&self) -> PyBaseTableRowIterator {
        PyBaseTableRowIterator {
            table: self.table.clone(),
            current_row: 0,
        }
    }
    
    /// Convert to pandas DataFrame
    pub fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        
        // Try to import pandas
        let pandas = match py.import("pandas") {
            Ok(pd) => pd,
            Err(_) => return Err(PyErr::new::<pyo3::exceptions::PyImportError, _>(
                "pandas is required for to_pandas() but not installed"
            ))
        };
        
        // Convert table to dict format for pandas
        let mut data_dict = std::collections::HashMap::new();
        
        for col_name in self.table.column_names() {
            if let Some(column) = self.table.column(&col_name) {
                // Convert BaseArray to AttrValue list and then to Python list
                let attr_values = column.data();
                let py_objects: Vec<_> = attr_values.iter()
                    .map(|attr| {
                        let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                        py_attr.to_object(py)
                    })
                    .collect();
                data_dict.insert(col_name, py_objects);
            }
        }
        
        // Create DataFrame
        let df = pandas.call_method1("DataFrame", (data_dict,))?;
        Ok(df.to_object(py))
    }
    
    /// Enable subscripting: table[column_name], table[slice], or table[boolean_mask]
    pub fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        use pyo3::types::{PyString, PySlice};
        let py = key.py();
        
        if let Ok(column_name) = key.extract::<String>() {
            // Column access: table['column_name']
            if let Some(column) = self.table.column(&column_name) {
                let attr_values = column.data();
                // Prefer StatsArray for numeric columns; fallback to GraphArray
                if let Ok(stats) = crate::ffi::storage::num_array::PyNumArray::from_attr_values(attr_values.clone()) {
                    Ok(stats.into_py(py))
                } else {
                    // Fallback to BaseArray for non-numeric columns (GraphArray deprecated)
                    let base = groggy::storage::array::BaseArray::from_attr_values(attr_values.clone());
                    let py_base = crate::ffi::storage::array::PyBaseArray { inner: base };
                    Ok(py_base.into_py(py))
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Column '{}' not found", column_name)
                ))
            }
        } else if let Ok(column_names) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2', ...]]
            let selected_table = self.table.select(&column_names)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Failed to select columns: {}", e)
                ))?;
            Ok(PyBaseTable { table: selected_table }.into_py(py))
        } else if let Ok(row_index) = key.extract::<isize>() {
            // Row access by integer: table[5] or table[-1]
            let nrows = self.table.nrows() as isize;
            let actual_index = if row_index < 0 {
                (nrows + row_index) as usize
            } else {
                row_index as usize
            };
            
            if actual_index >= self.table.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Row index {} out of range (0-{})", row_index, self.table.nrows() - 1)
                ));
            }
            
            // Return single row as a BaseTable with one row
            let single_row_table = self.table.head(actual_index + 1).tail(1);
            Ok(PyBaseTable { table: single_row_table }.into_py(py))
        } else if let Ok(slice) = key.downcast::<PySlice>() {
            // Slice access: table[start:end]
            let indices = slice.indices(self.table.nrows() as i64)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step as usize;
            
            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Step slicing not yet implemented"
                ));
            }
            
            // Create a new BaseTable with the sliced rows
            let sliced_table = self.table.head(stop).tail(stop - start);
            Ok(PyBaseTable { table: sliced_table }.into_py(py))
        } else if let Ok(py_array) = key.extract::<crate::ffi::storage::array::PyGraphArray>() {
            // Boolean mask access: table[boolean_array]
            let mask_values = py_array.to_list(py)?;
            let mut boolean_mask = Vec::new();
            
            for value in mask_values {
                if let Ok(bool_val) = value.extract::<bool>(py) {
                    boolean_mask.push(bool_val);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Boolean mask must contain only boolean values"
                    ));
                }
            }
            
            if boolean_mask.len() != self.table.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Boolean mask length ({}) doesn't match table rows ({})", 
                           boolean_mask.len(), self.table.nrows())
                ));
            }
            
            // Apply boolean mask to filter rows
            let filtered_table = self.table.filter_by_mask(&boolean_mask)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyBaseTable { table: filtered_table }.into_py(py))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "BaseTable indices must be strings (column names), integers (row indices), lists of column names, slices, or boolean arrays"
            ))
        }
    }
    
    /// Get rich display representation using Rust formatter
    pub fn rich_display(&self, config: Option<&crate::ffi::display::PyDisplayConfig>) -> PyResult<String> {
        let display_data = self.to_display_data();
        let default_config = groggy::display::DisplayConfig::default();
        let rust_config = config.map(|c| c.get_config()).unwrap_or(&default_config);
        let mut formatted = groggy::display::format_table(display_data, rust_config);
        
        // Replace the footer with BaseTable-specific info
        let nrows = self.table.nrows();
        let ncols = self.table.ncols();
        let footer = format!("rows: {} • cols: {} • type: BaseTable", nrows, ncols);
        
        // Replace the last line (which contains the table stats) with our custom footer
        let lines: Vec<&str> = formatted.lines().collect();
        if let Some(last_line_idx) = lines.iter().rposition(|line| line.contains("rows:") || line.contains("•")) {
            let mut new_lines = lines[..last_line_idx].to_vec();
            new_lines.push(&footer);
            formatted = new_lines.join("\n");
        }
        
        Ok(formatted)
    }
    
    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        let display_data = self.to_display_data();
        let config = groggy::display::DisplayConfig::default();
        let formatted = groggy::display::format_table(display_data, &config);
        // Convert to HTML format for Jupyter
        Ok(format!("<pre>{}</pre>", html_escape::encode_text(&formatted)))
    }
    
    /// Export BaseTable to CSV file
    pub fn to_csv(&self, path: &str) -> PyResult<()> {
        self.table.to_csv(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import BaseTable from CSV file
    #[staticmethod]
    pub fn from_csv(path: &str) -> PyResult<PyBaseTable> {
        let table = groggy::storage::table::BaseTable::from_csv(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBaseTable { table })
    }
    
    /// Export BaseTable to Parquet file 
    pub fn to_parquet(&self, path: &str) -> PyResult<()> {
        self.table.to_parquet(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import BaseTable from Parquet file
    #[staticmethod]
    pub fn from_parquet(path: &str) -> PyResult<PyBaseTable> {
        let table = groggy::storage::table::BaseTable::from_parquet(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBaseTable { table })
    }
    
    /// Export BaseTable to JSON file
    pub fn to_json(&self, path: &str) -> PyResult<()> {
        self.table.to_json(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import BaseTable from JSON file
    #[staticmethod]
    pub fn from_json(path: &str) -> PyResult<PyBaseTable> {
        let table = groggy::storage::table::BaseTable::from_json(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBaseTable { table })
    }


    /// Aggregate entire table without grouping
    /// 
    /// # Arguments
    /// * `agg_specs` - Dictionary mapping column names to aggregation functions
    /// 
    /// # Examples
    /// ```python
    /// # Calculate summary statistics
    /// summary = table.aggregate({'sales': 'sum', 'price': 'avg', 'items': 'count'})
    /// ```
    pub fn aggregate(&self, agg_specs: HashMap<String, String>) -> PyResult<Self> {
        let result = self.table.aggregate(agg_specs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Aggregation failed: {}", e)))?;
        
        Ok(Self { table: result })
    }

    /// Alias for aggregate method (more concise)
    /// 
    /// # Arguments
    /// * `agg_specs` - Dictionary mapping column names to aggregation functions
    /// 
    /// # Returns
    /// PyBaseTable: A single table with aggregated results
    /// 
    /// # Example
    /// ```python
    /// # Aggregate the entire table
    /// summary = table.agg({'sales': 'sum', 'price': 'avg', 'items': 'count'})
    /// ```
    pub fn agg(&self, agg_specs: HashMap<String, String>) -> PyResult<Self> {
        self.aggregate(agg_specs)
    }

    // =============================================================================
    // Phase 2 Features: Multi-table Operations (Unified Join Interface)
    // =============================================================================

    /// Unified join method with pandas-style interface
    /// 
    /// # Arguments
    /// * `other` - The table to join with
    /// * `on` - Column name(s) to join on. Can be:
    ///   - String: single column name (same in both tables)
    ///   - List[str]: multiple column names (same in both tables)
    ///   - Dict: {"left": "col1", "right": "col2"} for different column names
    ///   - Dict: {"left": ["col1", "col2"], "right": ["col3", "col4"]} for multiple different columns
    /// * `how` - Join type: "inner", "left", "right", "outer"
    /// 
    /// # Examples
    /// ```python
    /// # Simple inner join on same column name
    /// result = table1.join(table2, on="id", how="inner")
    /// 
    /// # Left join on different column names  
    /// result = table1.join(table2, on={"left": "user_id", "right": "id"}, how="left")
    /// 
    /// # Multi-column join
    /// result = table1.join(table2, on=["key1", "key2"], how="outer")
    /// ```
    pub fn join(&self, other: &Self, on: &PyAny, how: &str) -> PyResult<Self> {
        // Parse the 'on' parameter
        let (left_cols, right_cols) = self.parse_join_on(on)?;
        
        // Validate join type
        let join_result = match how.to_lowercase().as_str() {
            "inner" => {
                if left_cols.len() == 1 {
                    self.table.inner_join(&other.table, &left_cols[0], &right_cols[0])
                } else {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "Multi-column joins not yet implemented"
                    ));
                }
            }
            "left" => {
                if left_cols.len() == 1 {
                    self.table.left_join(&other.table, &left_cols[0], &right_cols[0])
                } else {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "Multi-column joins not yet implemented"
                    ));
                }
            }
            "right" => {
                // Right join is just left join with tables swapped
                if left_cols.len() == 1 {
                    other.table.left_join(&self.table, &right_cols[0], &left_cols[0])
                } else {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        "Multi-column joins not yet implemented"
                    ));
                }
            }
            "outer" => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Outer join not yet implemented"
                ));
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Invalid join type '{}'. Must be one of: 'inner', 'left', 'right', 'outer'", how)
                ));
            }
        };

        let result = join_result
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Join failed: {}", e)))?;
        
        Ok(Self { table: result })
    }

    /// Union with another table (removes duplicates)
    /// 
    /// # Arguments
    /// * `other` - The table to union with
    /// 
    /// # Examples
    /// ```python
    /// # Combine two tables with same schema
    /// combined = table1.union(table2)
    /// ```
    pub fn union(&self, other: &Self) -> PyResult<Self> {
        let result = self.table.union(&other.table)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Union failed: {}", e)))?;
        
        Ok(Self { table: result })
    }

    /// Intersect with another table (returns common rows)
    /// 
    /// # Arguments
    /// * `other` - The table to intersect with
    /// 
    /// # Examples
    /// ```python
    /// # Find common rows between tables
    /// common = table1.intersect(table2)
    /// ```
    pub fn intersect(&self, other: &Self) -> PyResult<Self> {
        let result = self.table.intersect(&other.table)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Intersect failed: {}", e)))?;
        
        Ok(Self { table: result })
    }

}

// Implement display data conversion for PyBaseTable
impl PyBaseTable {
    /// Convert table to display data format expected by Rust formatter
    fn to_display_data(&self) -> HashMap<String, Value> {
        let mut data = HashMap::new();
        
        // Get table dimensions
        let nrows = self.table.nrows();
        let ncols = self.table.ncols();
        data.insert("shape".to_string(), Value::Array(vec![Value::from(nrows), Value::from(ncols)]));
        
        // Get column names
        let column_names = self.table.column_names();
        let columns_json: Vec<Value> = column_names.iter().map(|s| Value::String(s.clone())).collect();
        data.insert("columns".to_string(), Value::Array(columns_json));
        
        // Get data types for each column
        let mut dtypes_map = Map::new();
        for col_name in column_names {
            if let Some(column) = self.table.column(col_name) {
                let dtype = match column.data().first() {
                    Some(groggy::AttrValue::Int(_)) => "int64",
                    Some(groggy::AttrValue::SmallInt(_)) => "int32", 
                    Some(groggy::AttrValue::Float(_)) => "float64",
                    Some(groggy::AttrValue::Text(_)) => "string",
                    Some(groggy::AttrValue::CompactText(_)) => "string",
                    Some(groggy::AttrValue::Bool(_)) => "bool",
                    _ => "object",
                };
                dtypes_map.insert(col_name.clone(), Value::String(dtype.to_string()));
            }
        }
        data.insert("dtypes".to_string(), Value::Object(dtypes_map));
        
        // Get sample data (first 10 rows for display)
        let sample_size = std::cmp::min(10, nrows);
        let mut data_rows = Vec::new();
        
        for row_idx in 0..sample_size {
            let mut row_values = Vec::new();
            for col_name in column_names {
                if let Some(column) = self.table.column(col_name) {
                    let attr_values = column.data();
                    if let Some(value) = attr_values.get(row_idx) {
                        let json_value = match value {
                            groggy::AttrValue::Int(i) => Value::Number(serde_json::Number::from(*i)),
                            groggy::AttrValue::SmallInt(i) => Value::Number(serde_json::Number::from(*i as i64)),
                            groggy::AttrValue::Float(f) => Value::Number(serde_json::Number::from_f64((*f).into()).unwrap_or(serde_json::Number::from(0))),
                            groggy::AttrValue::Text(s) => Value::String(s.clone()),
                            groggy::AttrValue::CompactText(s) => Value::String(s.as_str().to_string()),
                            groggy::AttrValue::Bool(b) => Value::Bool(*b),
                            groggy::AttrValue::FloatVec(v) => Value::String(format!("{:?}", v)),
                            groggy::AttrValue::Bytes(b) => Value::String(format!("{:?}", b)),
                            groggy::AttrValue::CompressedText(cd) => {
                                match cd.decompress_text() {
                                    Ok(text) => Value::String(text),
                                    Err(_) => Value::String("[compressed text]".to_string()),
                                }
                            },
                            groggy::AttrValue::CompressedFloatVec(_) => Value::String("[compressed float vec]".to_string()),
                            groggy::AttrValue::Null => Value::Null,
                            groggy::AttrValue::SubgraphRef(id) => Value::String(format!("subgraph:{}", id)),
                            groggy::AttrValue::NodeArray(nodes) => Value::String(format!("{:?}", nodes)),
                            groggy::AttrValue::EdgeArray(edges) => Value::String(format!("{:?}", edges)),
                        };
                        row_values.push(json_value);
                    } else {
                        row_values.push(Value::Null);
                    }
                } else {
                    row_values.push(Value::Null);
                }
            }
            data_rows.push(Value::Array(row_values));
        }
        data.insert("data".to_string(), Value::Array(data_rows));
        
        // Set index type
        data.insert("index_type".to_string(), Value::String("int64".to_string()));
        
        data
    }

    // =============================================================================
    // Phase 2 Features: Group By Operations
    // =============================================================================

    /// Group by columns and apply aggregations
    /// 
    /// # Arguments
    /// * `group_cols` - List of column names to group by
    /// * `agg_specs` - Dictionary mapping column names to aggregation functions
    ///   Supported functions: "count", "sum", "avg", "mean", "min", "max"
    /// 
    /// # Examples
    /// ```python
    /// # Group by 'category' and aggregate 'value' column
    /// result = table.group_by_agg(['category'], {'value': 'sum', 'price': 'avg'})
    /// 
    /// # Multiple grouping columns
    /// result = table.group_by_agg(['region', 'category'], {'sales': 'sum', 'items': 'count'})
    /// ```
    pub fn group_by_agg(&self, group_cols: Vec<String>, agg_specs: HashMap<String, String>) -> PyResult<Self> {
        let result = self.table.group_by_agg(&group_cols, agg_specs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Group by aggregation failed: {}", e)))?;
        
        Ok(Self { table: result })
    }

    // =============================================================================
    // Helper Methods for Join Parameter Parsing
    // =============================================================================

    /// Parse the 'on' parameter for join operations
    fn parse_join_on(&self, on: &PyAny) -> PyResult<(Vec<String>, Vec<String>)> {
        // Case 1: String - single column name (same in both tables)
        if let Ok(col_name) = on.extract::<String>() {
            return Ok((vec![col_name.clone()], vec![col_name]));
        }
        
        // Case 2: List of strings - multiple column names (same in both tables)
        if let Ok(col_names) = on.extract::<Vec<String>>() {
            return Ok((col_names.clone(), col_names));
        }
        
        // Case 3: Dictionary with left/right keys
        if let Ok(dict) = on.extract::<&pyo3::types::PyDict>() {
            if let (Some(left_item), Some(right_item)) = (dict.get_item("left")?, dict.get_item("right")?) {
                // Try as single strings
                if let (Ok(left_col), Ok(right_col)) = (left_item.extract::<String>(), right_item.extract::<String>()) {
                    return Ok((vec![left_col], vec![right_col]));
                }
                
                // Try as lists of strings
                if let (Ok(left_cols), Ok(right_cols)) = (left_item.extract::<Vec<String>>(), right_item.extract::<Vec<String>>()) {
                    if left_cols.len() != right_cols.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Left and right column lists must have the same length"
                        ));
                    }
                    return Ok((left_cols, right_cols));
                }
                
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Dictionary values must be strings or lists of strings"
                ));
            } else {
                return Err(pyo3::exceptions::PyKeyError::new_err(
                    "Dictionary must have 'left' and 'right' keys"
                ));
            }
        }
        
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Join 'on' parameter must be a string, list of strings, or dict with 'left' and 'right' keys"
        ))
    }
}

// Internal helper methods (not exposed to Python)
impl PyBaseTable {
    /// Parse column key from various formats (string, list of strings, etc.)
    fn parse_column_key(&self, col_key: &PyAny) -> PyResult<Vec<String>> {
        use pyo3::types::{PyString, PyList};
        
        if let Ok(col_name) = col_key.extract::<String>() {
            // Single column name: "score"
            Ok(vec![col_name])
        } else if let Ok(col_list) = col_key.downcast::<PyList>() {
            // List of column names: ["a", "b"]
            let mut column_names = Vec::new();
            for item in col_list.iter() {
                let col_name: String = item.extract()
                    .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Column names must be strings"
                    ))?;
                column_names.push(col_name);
            }
            Ok(column_names)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Column key must be a string or list of strings"
            ))
        }
    }
    
    /// Apply updates to specified rows and columns
    fn apply_row_column_update(&mut self, row_key: &PyAny, column_names: &[String], value: &PyAny) -> PyResult<()> {
        use pyo3::types::{PySlice, PyList};
        
        if let Ok(slice) = row_key.downcast::<PySlice>() {
            // Slice-based row selection: table[10:20, "score"] = 0.0
            let (start, end, step) = self.parse_slice(slice)?;
            
            if column_names.len() == 1 {
                // Single column, single value
                let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
                self.table.set_values_by_range(start, end, step, &column_names[0], attr_value)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            } else {
                // Multiple columns - value should be a list
                if let Ok(values_list) = value.downcast::<PyList>() {
                    if values_list.len() != column_names.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Number of values ({}) must match number of columns ({})", values_list.len(), column_names.len())
                        ));
                    }
                    
                    for (col_name, val_py) in column_names.iter().zip(values_list.iter()) {
                        let attr_value = crate::ffi::utils::python_value_to_attr_value(val_py)?;
                        self.table.set_values_by_range(start, end, step, col_name, attr_value)
                            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "When setting multiple columns, value must be a list"
                    ));
                }
            }
        } else if let Ok(mask_list) = row_key.downcast::<PyList>() {
            // Boolean mask or index list: table[[True, False, True], "score"] = 0.0
            
            // Try to interpret as boolean mask first
            let mut is_boolean_mask = true;
            let mut mask = Vec::new();
            
            for item in mask_list.iter() {
                if let Ok(bool_val) = item.extract::<bool>() {
                    mask.push(bool_val);
                } else {
                    is_boolean_mask = false;
                    break;
                }
            }
            
            if is_boolean_mask {
                // Boolean mask
                if column_names.len() == 1 {
                    let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
                    self.table.set_values_by_mask(&mask, &column_names[0], attr_value)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                } else {
                    // Multiple columns - value should be a list
                    if let Ok(values_list) = value.downcast::<PyList>() {
                        if values_list.len() != column_names.len() {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                format!("Number of values ({}) must match number of columns ({})", values_list.len(), column_names.len())
                            ));
                        }
                        
                        for (col_name, val_py) in column_names.iter().zip(values_list.iter()) {
                            let attr_value = crate::ffi::utils::python_value_to_attr_value(val_py)?;
                            self.table.set_values_by_mask(&mask, col_name, attr_value)
                                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                        }
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "When setting multiple columns, value must be a list"
                        ));
                    }
                }
            } else {
                // Index list - convert to individual set_value calls
                let mut row_indices = Vec::new();
                for item in mask_list.iter() {
                    let idx: usize = item.extract()
                        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Row indices must be integers or booleans"
                        ))?;
                    row_indices.push(idx);
                }
                
                if column_names.len() == 1 {
                    let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
                    for &row_idx in &row_indices {
                        self.table.set_value(row_idx, &column_names[0], attr_value.clone())
                            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                        "Setting multiple columns with index lists not yet implemented"
                    ));
                }
            }
        } else if let Ok(row_idx) = row_key.extract::<usize>() {
            // Single row index: table[5, "name"] = "Alice"
            if column_names.len() == 1 {
                let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
                self.table.set_value(row_idx, &column_names[0], attr_value)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            } else {
                // Multiple columns - value should be a list
                if let Ok(values_list) = value.downcast::<PyList>() {
                    if values_list.len() != column_names.len() {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Number of values ({}) must match number of columns ({})", values_list.len(), column_names.len())
                        ));
                    }
                    
                    for (col_name, val_py) in column_names.iter().zip(values_list.iter()) {
                        let attr_value = crate::ffi::utils::python_value_to_attr_value(val_py)?;
                        self.table.set_value(row_idx, col_name, attr_value)
                            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "When setting multiple columns, value must be a list"
                    ));
                }
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Row key must be an integer, slice, or list of integers/booleans"
            ));
        }
        
        Ok(())
    }
    
    /// Parse Python slice object to (start, end, step) tuple
    fn parse_slice(&self, slice: &pyo3::types::PySlice) -> PyResult<(usize, usize, usize)> {
        let py = slice.py();
        let table_len = self.table.nrows();
        
        let indices = slice.indices(table_len as i64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid slice: {}", e)))?;
        
        let start = indices.start.max(0) as usize;
        let stop = indices.stop.max(0).min(table_len as isize) as usize;
        let step = indices.step.max(1) as usize;
        
        Ok((start, stop, step))
    }
    
    /// Helper method to filter using Python function
    fn filter_by_python_function(&self, func: &PyAny) -> PyResult<groggy::storage::table::BaseTable> {
        use pyo3::types::PyDict;
        let py = func.py();
        let mut mask = Vec::new();
        
        // Apply function to each row
        for i in 0..self.table.nrows() {
            // Create row dict
            let row_dict = PyDict::new(py);
            for col_name in self.table.column_names() {
                if let Some(column) = self.table.column(col_name) {
                    if let Some(value) = column.get(i) {
                        let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                        row_dict.set_item(col_name, py_attr.to_object(py))?;
                    } else {
                        row_dict.set_item(col_name, py.None())?;
                    }
                }
            }
            
            // Call the function with the row
            let result = func.call1((row_dict,))?;
            let keep_row = result.extract::<bool>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Filter function must return a boolean value"
                ))?;
            mask.push(keep_row);
        }
        
        // Apply mask to create filtered table
        self.table.filter_by_mask(&mask)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}

// =============================================================================
// PyBaseTableRowIterator - Iterator for table rows
// =============================================================================

/// Iterator for iterating over table rows as dictionaries
#[pyclass(name = "BaseTableRowIterator", module = "groggy")]
pub struct PyBaseTableRowIterator {
    pub(crate) table: BaseTable,
    pub(crate) current_row: usize,
}

#[pymethods]
impl PyBaseTableRowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.current_row >= self.table.nrows() {
            return Ok(None);
        }
        
        // Create a dictionary for the current row
        let dict = pyo3::types::PyDict::new(py);
        
        for col_name in self.table.column_names() {
            if let Some(column) = self.table.column(col_name) {
                let attr_values = column.data();
                if let Some(value) = attr_values.get(self.current_row) {
                    let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                    dict.set_item(col_name, py_attr.to_object(py))?;
                } else {
                    dict.set_item(col_name, py.None())?;
                }
            }
        }
        
        self.current_row += 1;
        Ok(Some(dict.to_object(py)))
    }
}

// =============================================================================
// PyBaseTableIterator - Python wrapper for TableIterator<BaseTable>
// =============================================================================

/// Python wrapper for BaseTable iterator
#[pyclass(name = "BaseTableIterator", module = "groggy")]
pub struct PyBaseTableIterator {
    pub(crate) iterator: TableIterator<BaseTable>,
}

#[pymethods]
impl PyBaseTableIterator {
    /// Execute all operations and return result
    pub fn collect(&self) -> PyResult<PyBaseTable> {
        let result = self.iterator.clone().collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyBaseTable { table: result })
    }
}

// =============================================================================
// PyNodesTable - Python wrapper for NodesTable
// =============================================================================

/// Python wrapper for NodesTable
#[pyclass(name = "NodesTable", module = "groggy")]
#[derive(Clone)]
pub struct PyNodesTable {
    pub(crate) table: NodesTable,
}

#[pymethods]
impl PyNodesTable {
    /// Create new NodesTable from node IDs
    #[new]
    pub fn new(node_ids: Vec<NodeId>) -> Self {
        Self {
            table: NodesTable::new(node_ids),
        }
    }
    
    /// Create NodesTable from a Python dictionary (must contain 'node_id' column)
    #[classmethod]
    pub fn from_dict(_cls: &PyType, py: Python, data: &PyDict) -> PyResult<Py<PyNodesTable>> {
        // First create a BaseTable from the dict
        let base_table_py = PyBaseTable::from_dict(_cls, py, data)?;
        let base_table = base_table_py.borrow(py).table.clone();
        
        // Convert BaseTable to NodesTable (requires 'node_id' column)
        let nodes_table = NodesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cannot create NodesTable: {}. Make sure your data contains a 'node_id' column.", e)
            ))?;
            
        Py::new(py, Self { table: nodes_table })
    }
    
    /// Get node IDs
    pub fn node_ids(&self) -> PyResult<crate::ffi::storage::array::PyGraphArray> {
        let node_ids = self.table.node_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Convert to AttrValue vector and then to PyGraphArray
        let attr_values: Vec<groggy::AttrValue> = node_ids.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        
        let graph_array = GraphArray::from_vec(attr_values);
        Ok(crate::ffi::storage::array::PyGraphArray::from_graph_array(graph_array))
    }
    
    /// Add node attributes - flexible input format
    ///
    /// Args:
    ///     attr_name: Name of the attribute column to add  
    ///     attributes: Can be:
    ///         - Dictionary mapping node_id to value: {0: "Alice", 1: "Bob"}
    ///         - List of {"id": node_id, "value": value} dicts: [{"id": 0, "value": "Alice"}]
    ///         - HashMap<NodeId, PyAttrValue> (advanced usage)
    ///
    /// Returns:
    ///     PyNodesTable: A new table with the attributes added
    pub fn with_attributes(&self, attr_name: &PyAny, attributes: &PyAny) -> PyResult<Self> {
        use pyo3::types::{PyDict, PyList};
        
        // Handle attr_name (can be string or other types)
        let attr_name_str = if let Ok(name) = attr_name.extract::<String>() {
            name
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Attribute name must be a string"
            ));
        };
        
        // TODO: Implement attribute conversion (temporarily disabled to fix compilation)
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "with_attributes method temporarily disabled - under development"
        ));
    }
    
    /// Filter nodes by attribute value
    pub fn filter_by_attr(&self, attr_name: &str, value: &PyAny) -> PyResult<Self> {
        // Convert PyAny to PyAttrValue for compatibility
        let py_attr_value = crate::ffi::types::PyAttrValue::from_py_value(value)?;
        let result = self.table.filter_by_attr(attr_name, &py_attr_value.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: result })
    }
    
    /// Get unique values for an attribute
    pub fn unique_attr_values(&self, attr_name: &str) -> PyResult<Vec<crate::ffi::types::PyAttrValue>> {
        let values = self.table.unique_attr_values(attr_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(values.into_iter()
            .map(|v| crate::ffi::types::PyAttrValue::new(v))
            .collect())
    }
    
    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Get reference to underlying BaseTable
    pub fn base_table(&self) -> PyBaseTable {
        PyBaseTable { table: self.table.base_table().clone() }
    }
    
    /// Convert to BaseTable (loses node-specific typing)
    pub fn into_base_table(&self) -> PyBaseTable {
        PyBaseTable { table: self.table.clone().into_base_table() }
    }
    
    /// Convert to pandas DataFrame
    pub fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Delegate to BaseTable implementation
        self.base_table().to_pandas(py)
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyNodesTableIterator {
        PyNodesTableIterator {
            iterator: self.table.iter(),
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("NodesTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
    
    
    /// Get length (number of rows) for len() function
    pub fn __len__(&self) -> usize {
        self.table.nrows()
    }
    
    /// Support iteration over rows: for row in table:
    pub fn __iter__(&self) -> PyNodesTableRowIterator {
        PyNodesTableRowIterator {
            table: self.table.clone(),
            current_row: 0,
        }
    }
    
    /// Get first n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn head(&self, n: usize) -> PyNodesTable {
        PyNodesTable { table: self.table.head(n) }
    }
    
    /// Get last n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn tail(&self, n: usize) -> PyNodesTable {
        PyNodesTable { table: self.table.tail(n) }
    }
    
    /// Sort table by column
    /// 
    /// Args:
    ///     column: Name of the column to sort by
    ///     ascending: If True, sort in ascending order; if False, descending
    /// 
    /// Returns:
    ///     PyNodesTable: A new sorted table
    #[pyo3(signature = (column, ascending = true))]
    pub fn sort_by(&self, column: &str, ascending: bool) -> PyResult<Self> {
        let sorted_table = self.table.sort_by(column, ascending)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: sorted_table })
    }
    
    /// Select specific columns to create a new table
    ///
    /// Args:
    ///     columns: List of column names to select
    ///
    /// Returns:
    ///     PyNodesTable: A new table with only the selected columns
    pub fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        // For NodesTable, ensure node_id column is always included
        let mut all_columns = vec!["node_id".to_string()];
        for col_name in columns {
            if col_name != "node_id" {
                all_columns.push(col_name);
            }
        }
        
        let selected_base = self.table.base_table().select(&all_columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let selected_nodes = groggy::storage::table::NodesTable::from_base_table(selected_base)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: selected_nodes })
    }
    
    /// Drop columns from the table (cannot drop node_id)
    ///
    /// Args:
    ///     columns: List of column names to drop
    ///
    /// Returns:
    ///     PyNodesTable: A new table without the specified columns
    pub fn drop_columns(&self, columns: Vec<String>) -> PyResult<Self> {
        // Prevent dropping node_id column
        if columns.contains(&"node_id".to_string()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot drop 'node_id' column from NodesTable"
            ));
        }
        
        let new_base = self.table.base_table().drop_columns(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let new_nodes = groggy::storage::table::NodesTable::from_base_table(new_base)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: new_nodes })
    }
    
    /// Filter rows using a query expression or Python function
    ///
    /// Args:
    ///     predicate: Either a string query expression (e.g. "age > 25") or a Python function
    ///
    /// Returns:
    ///     PyNodesTable: A new table with filtered rows
    pub fn filter(&self, predicate: &PyAny) -> PyResult<Self> {
        // Use base table's filter method (which handles both string and function predicates)
        let base_table_py = PyBaseTable { table: self.table.base_table().clone() };
        let filtered_base_py = base_table_py.filter(predicate)?;
        
        let filtered_nodes = groggy::storage::table::NodesTable::from_base_table(filtered_base_py.table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: filtered_nodes })
    }
    
    /// Group by columns and return grouped tables
    ///
    /// Args:
    ///     columns: List of column names to group by
    ///
    /// Returns:
    ///     PyNodesTableArray: Array-like container holding the grouped node tables
    pub fn group_by(&self, columns: Vec<String>) -> PyResult<PyNodesTableArray> {
        let grouped_bases = self.table.base_table().group_by(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert each grouped table to PyNodesTable
        let py_tables: Vec<PyNodesTable> = grouped_bases.into_iter()
            .map(|base_table| -> PyResult<PyNodesTable> {
                let nodes_table = groggy::storage::table::NodesTable::from_base_table(base_table)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(PyNodesTable { table: nodes_table })
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyNodesTableArray { 
            tables: py_tables,
            group_columns: columns,
        })
    }
    
    /// Get a slice of rows [start, end)
    ///
    /// Args:
    ///     start: Starting row index (inclusive)
    ///     end: Ending row index (exclusive)
    ///
    /// Returns:
    ///     PyNodesTable: A new table with the specified row slice
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            table: self.table.slice(start, end),
        }
    }
    
    /// Enable subscripting: table[column_name] or table[slice]  
    pub fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        use pyo3::types::{PyString, PySlice};
        let py = key.py();
        
        if let Ok(column_name) = key.extract::<String>() {
            // Column access: table['column_name'] 
            if let Some(column) = self.table.base_table().column(&column_name) {
                let attr_values = column.data();
                // Prefer StatsArray for numeric columns; fallback to GraphArray
                if let Ok(stats) = crate::ffi::storage::num_array::PyNumArray::from_attr_values(attr_values.clone()) {
                    Ok(stats.into_py(py))
                } else {
                    let base = groggy::storage::array::BaseArray::from_attr_values(attr_values.clone());
                    let py_base = crate::ffi::storage::array::PyBaseArray { inner: base };
                    Ok(py_base.into_py(py))
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Column '{}' not found", column_name)
                ))
            }
        } else if let Ok(column_names) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2', ...]]
            // For NodesTable, we need to ensure required columns are included
            let mut required_columns = vec!["node_id".to_string()];
            let mut all_columns = required_columns.clone();
            
            // Add requested columns that aren't already required
            for col_name in column_names {
                if !required_columns.contains(&col_name) {
                    all_columns.push(col_name);
                }
            }
            
            let selected_base = self.table.base_table().select(&all_columns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Failed to select columns: {}", e)
                ))?;
            let nodes_selected = groggy::storage::table::NodesTable::from_base_table(selected_base)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyNodesTable { table: nodes_selected }.into_py(py))
        } else if let Ok(row_index) = key.extract::<isize>() {
            // Row access by integer: table[5] or table[-1]
            let nrows = self.table.nrows() as isize;
            let actual_index = if row_index < 0 {
                (nrows + row_index) as usize
            } else {
                row_index as usize
            };
            
            if actual_index >= self.table.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Row index {} out of range (0-{})", row_index, self.table.nrows() - 1)
                ));
            }
            
            // Return single row as a NodesTable with one row
            let single_row_base = self.table.base_table().head(actual_index + 1).tail(1);
            let single_row_nodes = groggy::storage::table::NodesTable::from_base_table(single_row_base)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyNodesTable { table: single_row_nodes }.into_py(py))
        } else if let Ok(slice) = key.downcast::<PySlice>() {
            // Slice access: table[start:end]
            let indices = slice.indices(self.table.nrows() as i64)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step as usize;
            
            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Step slicing not yet implemented"
                ));
            }
            
            // Create a new NodesTable with sliced rows
            let base_sliced = self.table.base_table().head(stop).tail(stop - start);
            let nodes_sliced = groggy::storage::table::NodesTable::from_base_table(base_sliced)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyNodesTable { table: nodes_sliced }.into_py(py))
        } else if let Ok(py_array) = key.extract::<crate::ffi::storage::array::PyGraphArray>() {
            // Boolean mask access: table[boolean_array]
            let mask_values = py_array.to_list(py)?;
            let mut mask_booleans = Vec::new();
            
            for value in mask_values.iter() {
                // Try to extract as plain Python bool first
                if let Ok(b) = value.extract::<bool>(py) {
                    mask_booleans.push(b);
                } else if let Ok(py_attr) = value.extract::<crate::ffi::types::PyAttrValue>(py) {
                    if let groggy::AttrValue::Bool(b) = py_attr.inner {
                        mask_booleans.push(b);
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Boolean mask must contain only boolean values"
                        ));
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Boolean mask must contain boolean values"
                    ));
                }
            }
            
            if mask_booleans.len() != self.table.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Boolean mask length ({}) does not match table length ({})", 
                            mask_booleans.len(), self.table.nrows())
                ));
            }
            
            // Filter the base table using the boolean mask
            let filtered_base = self.table.base_table().filter_by_mask(&mask_booleans)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let nodes_filtered = groggy::storage::table::NodesTable::from_base_table(filtered_base)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyNodesTable { table: nodes_filtered }.into_py(py))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "NodesTable indices must be strings (column names), integers (row indices), lists of column names, slices, or boolean arrays"
            ))
        }
    }

    /// Get display data structure for formatters
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::{PyDict, PyList};
        
        let dict = PyDict::new(py);
        
        // Get table dimensions
        let nrows = self.table.nrows();
        let ncols = self.table.ncols();
        dict.set_item("shape", (nrows, ncols))?;
        
        // Get column names
        let column_names = self.table.column_names();
        let py_columns = PyList::new(py, column_names);
        dict.set_item("columns", py_columns)?;
        
        // Get data types for each column
        let dtypes_dict = PyDict::new(py);
        for col_name in column_names {
            if let Some(column) = self.table.column(col_name) {
                let dtype = match column.data().first() {
                    Some(groggy::AttrValue::Int(_)) => "int64",
                    Some(groggy::AttrValue::SmallInt(_)) => "int32",
                    Some(groggy::AttrValue::Float(_)) => "float64",
                    Some(groggy::AttrValue::Text(_)) => "string",
                    Some(groggy::AttrValue::CompactText(_)) => "string",
                    Some(groggy::AttrValue::Bool(_)) => "bool",
                    _ => "object",
                };
                dtypes_dict.set_item(col_name, dtype)?;
            }
        }
        dict.set_item("dtypes", dtypes_dict)?;
        
        // Get sample data (first 10 rows for display)
        let sample_size = std::cmp::min(10, nrows);
        let data_list = PyList::empty(py);
        
        for row_idx in 0..sample_size {
            let row_list = PyList::empty(py);
            for col_name in column_names {
                if let Some(column) = self.table.column(col_name) {
                    let attr_values = column.data();
                    if let Some(value) = attr_values.get(row_idx) {
                        let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                        row_list.append(py_attr.to_object(py))?;
                    } else {
                        row_list.append(py.None())?;
                    }
                } else {
                    row_list.append(py.None())?;
                }
            }
            data_list.append(row_list)?;
        }
        dict.set_item("data", data_list)?;
        
        // Set index type
        dict.set_item("index_type", "int64")?;
        
        Ok(dict.to_object(py))
    }
    
    /// Get rich display representation with NodesTable type
    pub fn rich_display(&self, config: Option<&crate::ffi::display::PyDisplayConfig>) -> PyResult<String> {
        let base_table = self.base_table();
        let display_data = base_table.to_display_data();
        let default_config = groggy::display::DisplayConfig::default();
        let rust_config = config.map(|c| c.get_config()).unwrap_or(&default_config);
        let mut formatted = groggy::display::format_table(display_data, rust_config);
        
        // Replace the footer with NodesTable-specific info
        let nrows = self.table.nrows();
        let ncols = self.table.ncols();
        let footer = format!("rows: {} • cols: {} • type: NodesTable", nrows, ncols);
        
        // Replace the last line with our custom footer
        let lines: Vec<&str> = formatted.lines().collect();
        if let Some(last_line_idx) = lines.iter().rposition(|line| line.contains("rows:") || line.contains("•")) {
            let mut new_lines = lines[..last_line_idx].to_vec();
            new_lines.push(&footer);
            formatted = new_lines.join("\n");
        }
        
        Ok(formatted)
    }
    
    /// Delegate to BaseTable for missing methods (enables rich display)
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let base_table = self.base_table();
        let base_obj = base_table.into_py(py);
        base_obj.getattr(py, name)
    }
    
    /// Export NodesTable to CSV file
    pub fn to_csv(&self, path: &str) -> PyResult<()> {
        self.table.base_table().to_csv(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import NodesTable from CSV file (must contain node_id column)
    #[staticmethod]
    pub fn from_csv(path: &str) -> PyResult<PyNodesTable> {
        let base_table = groggy::storage::table::BaseTable::from_csv(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let nodes_table = groggy::storage::table::NodesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyNodesTable { table: nodes_table })
    }
    
    /// Export NodesTable to Parquet file 
    pub fn to_parquet(&self, path: &str) -> PyResult<()> {
        self.table.base_table().to_parquet(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import NodesTable from Parquet file (must contain node_id column)
    #[staticmethod]
    pub fn from_parquet(path: &str) -> PyResult<PyNodesTable> {
        let base_table = groggy::storage::table::BaseTable::from_parquet(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let nodes_table = groggy::storage::table::NodesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyNodesTable { table: nodes_table })
    }
    
    /// Export NodesTable to JSON file
    pub fn to_json(&self, path: &str) -> PyResult<()> {
        self.table.base_table().to_json(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import NodesTable from JSON file (must contain node_id column)
    #[staticmethod]
    pub fn from_json(path: &str) -> PyResult<PyNodesTable> {
        let base_table = groggy::storage::table::BaseTable::from_json(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let nodes_table = groggy::storage::table::NodesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyNodesTable { table: nodes_table })
    }
}

// =============================================================================
// PyNodesTableRowIterator - Iterator for nodes table rows
// =============================================================================

/// Iterator for iterating over nodes table rows as dictionaries
#[pyclass(name = "NodesTableRowIterator", module = "groggy")]
pub struct PyNodesTableRowIterator {
    pub(crate) table: NodesTable,
    pub(crate) current_row: usize,
}

#[pymethods]
impl PyNodesTableRowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.current_row >= self.table.nrows() {
            return Ok(None);
        }
        
        // Create a dictionary for the current row
        let dict = pyo3::types::PyDict::new(py);
        
        for col_name in self.table.column_names() {
            if let Some(column) = self.table.column(col_name) {
                let attr_values = column.data();
                if let Some(value) = attr_values.get(self.current_row) {
                    let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                    dict.set_item(col_name, py_attr.to_object(py))?;
                } else {
                    dict.set_item(col_name, py.None())?;
                }
            }
        }
        
        self.current_row += 1;
        Ok(Some(dict.to_object(py)))
    }
}

// =============================================================================
// PyNodesTableIterator - Python wrapper for TableIterator<NodesTable>
// =============================================================================

/// Python wrapper for NodesTable iterator
#[pyclass(name = "NodesTableIterator", module = "groggy")]
pub struct PyNodesTableIterator {
    pub(crate) iterator: TableIterator<NodesTable>,
}

#[pymethods]
impl PyNodesTableIterator {
    /// Execute all operations and return result
    pub fn collect(&self) -> PyResult<PyNodesTable> {
        let result = self.iterator.clone().collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyNodesTable { table: result })
    }
}

// =============================================================================
// PyEdgesTable - Python wrapper for EdgesTable
// =============================================================================

/// Python wrapper for EdgesTable
#[pyclass(name = "EdgesTable", module = "groggy")]
#[derive(Clone)]
pub struct PyEdgesTable {
    pub(crate) table: EdgesTable,
}

#[pymethods]
impl PyEdgesTable {
    /// Create new EdgesTable from edge tuples
    #[new]
    pub fn new(edges: Vec<(EdgeId, NodeId, NodeId)>) -> Self {
        Self {
            table: EdgesTable::new(edges),
        }
    }
    
    /// Create EdgesTable from a Python dictionary (must contain 'edge_id', 'source', 'target' columns)
    #[classmethod]
    pub fn from_dict(_cls: &PyType, py: Python, data: &PyDict) -> PyResult<Py<PyEdgesTable>> {
        // First create a BaseTable from the dict
        let base_table_py = PyBaseTable::from_dict(_cls, py, data)?;
        let base_table = base_table_py.borrow(py).table.clone();
        
        // Convert BaseTable to EdgesTable (requires 'edge_id', 'source', 'target' columns)
        let edges_table = EdgesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Cannot create EdgesTable: {}. Make sure your data contains 'edge_id', 'source', and 'target' columns.", e)
            ))?;
            
        Py::new(py, Self { table: edges_table })
    }
    
    /// Get edge IDs
    pub fn edge_ids(&self) -> PyResult<crate::ffi::storage::array::PyGraphArray> {
        let edge_ids = self.table.edge_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Convert to AttrValue vector and then to PyGraphArray
        let attr_values: Vec<groggy::AttrValue> = edge_ids.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        
        let graph_array = GraphArray::from_vec(attr_values);
        Ok(crate::ffi::storage::array::PyGraphArray::from_graph_array(graph_array))
    }
    
    /// Get source node IDs
    pub fn sources(&self) -> PyResult<crate::ffi::storage::array::PyGraphArray> {
        let sources = self.table.sources()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Convert to AttrValue vector and then to PyGraphArray
        let attr_values: Vec<groggy::AttrValue> = sources.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        
        let graph_array = GraphArray::from_vec(attr_values);
        Ok(crate::ffi::storage::array::PyGraphArray::from_graph_array(graph_array))
    }
    
    /// Get target node IDs  
    pub fn targets(&self) -> PyResult<crate::ffi::storage::array::PyGraphArray> {
        let targets = self.table.targets()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        
        // Convert to AttrValue vector and then to PyGraphArray
        let attr_values: Vec<groggy::AttrValue> = targets.into_iter()
            .map(|id| groggy::AttrValue::Int(id as i64))
            .collect();
        
        let graph_array = GraphArray::from_vec(attr_values);
        Ok(crate::ffi::storage::array::PyGraphArray::from_graph_array(graph_array))
    }
    
    /// Get edges as tuples (edge_id, source, target)
    pub fn as_tuples(&self) -> PyResult<Vec<(EdgeId, NodeId, NodeId)>> {
        self.table.as_tuples()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    /// Filter edges by source nodes
    pub fn filter_by_sources(&self, source_nodes: Vec<NodeId>) -> PyResult<Self> {
        let result = self.table.filter_by_sources(&source_nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: result })
    }
    
    /// Filter edges by target nodes
    pub fn filter_by_targets(&self, target_nodes: Vec<NodeId>) -> PyResult<Self> {
        let result = self.table.filter_by_targets(&target_nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: result })
    }
    
    /// Filter edges by attribute value
    pub fn filter_by_attr(&self, attr_name: &str, value: &PyAny) -> PyResult<Self> {
        // Convert PyAny to PyAttrValue for compatibility
        let py_attr_value = crate::ffi::types::PyAttrValue::from_py_value(value)?;
        let result = self.table.filter_by_attr(attr_name, &py_attr_value.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: result })
    }
    
    /// Get unique values for an attribute
    pub fn unique_attr_values(&self, attr_name: &str) -> PyResult<Vec<crate::ffi::types::PyAttrValue>> {
        let values = self.table.unique_attr_values(attr_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(values.into_iter()
            .map(|v| crate::ffi::types::PyAttrValue::new(v))
            .collect())
    }
    
    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Get reference to underlying BaseTable
    pub fn base_table(&self) -> PyBaseTable {
        PyBaseTable { table: self.table.base_table().clone() }
    }
    
    /// Convert to BaseTable (loses edge-specific typing)
    pub fn into_base_table(&self) -> PyBaseTable {
        PyBaseTable { table: self.table.clone().into_base_table() }
    }
    
    /// Auto-assign edge IDs for null values (useful for meta nodes)
    pub fn auto_assign_edge_ids(&self) -> PyResult<Self> {
        let fixed_table = self.table.clone().auto_assign_edge_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { table: fixed_table })
    }
    
    /// Convert to pandas DataFrame
    pub fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Delegate to BaseTable implementation
        self.base_table().to_pandas(py)
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyEdgesTableIterator {
        PyEdgesTableIterator {
            iterator: self.table.iter(),
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("EdgesTable[{} x {}]", self.table.nrows(), self.table.ncols())
    }
    

    
    /// Get length (number of rows) for len() function
    pub fn __len__(&self) -> usize {
        self.table.nrows()
    }
    
    /// Support iteration over rows: for row in table:
    pub fn __iter__(&self) -> PyEdgesTableRowIterator {
        PyEdgesTableRowIterator {
            table: self.table.clone(),
            current_row: 0,
        }
    }
    
    /// Get first n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn head(&self, n: usize) -> PyEdgesTable {
        PyEdgesTable { table: self.table.head(n) }
    }
    
    /// Get last n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn tail(&self, n: usize) -> PyEdgesTable {
        PyEdgesTable { table: self.table.tail(n) }
    }
    
    /// Sort table by column
    /// 
    /// Args:
    ///     column: Name of the column to sort by
    ///     ascending: If True, sort in ascending order; if False, descending
    /// 
    /// Returns:
    ///     PyEdgesTable: A new sorted table
    #[pyo3(signature = (column, ascending = true))]
    pub fn sort_by(&self, column: &str, ascending: bool) -> PyResult<Self> {
        let sorted_table = self.table.sort_by(column, ascending)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: sorted_table })
    }
    
    /// Select specific columns to create a new table
    ///
    /// Args:
    ///     columns: List of column names to select
    ///
    /// Returns:
    ///     PyEdgesTable: A new table with only the selected columns
    pub fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        // For EdgesTable, ensure required columns are always included
        let mut all_columns = vec!["edge_id".to_string(), "source".to_string(), "target".to_string()];
        for col_name in columns {
            if !all_columns.contains(&col_name) {
                all_columns.push(col_name);
            }
        }
        
        let selected_base = self.table.base_table().select(&all_columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let selected_edges = groggy::storage::table::EdgesTable::from_base_table(selected_base)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: selected_edges })
    }
    
    /// Drop columns from the table (cannot drop edge_id, source, or target)
    ///
    /// Args:
    ///     columns: List of column names to drop
    ///
    /// Returns:
    ///     PyEdgesTable: A new table without the specified columns
    pub fn drop_columns(&self, columns: Vec<String>) -> PyResult<Self> {
        // Prevent dropping required columns
        let required = ["edge_id", "source", "target"];
        for req_col in &required {
            if columns.contains(&req_col.to_string()) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Cannot drop '{}' column from EdgesTable", req_col)
                ));
            }
        }
        
        let new_base = self.table.base_table().drop_columns(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let new_edges = groggy::storage::table::EdgesTable::from_base_table(new_base)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: new_edges })
    }
    
    /// Filter rows using a query expression or Python function
    ///
    /// Args:
    ///     predicate: Either a string query expression (e.g. "weight > 0.5") or a Python function
    ///
    /// Returns:
    ///     PyEdgesTable: A new table with filtered rows
    pub fn filter(&self, predicate: &PyAny) -> PyResult<Self> {
        // Use base table's filter method (which handles both string and function predicates)
        let base_table_py = PyBaseTable { table: self.table.base_table().clone() };
        let filtered_base_py = base_table_py.filter(predicate)?;
        
        let filtered_edges = groggy::storage::table::EdgesTable::from_base_table(filtered_base_py.table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: filtered_edges })
    }
    
    /// Group by columns and return grouped tables
    ///
    /// Args:
    ///     columns: List of column names to group by
    ///
    /// Returns:
    ///     PyEdgesTableArray: Array-like container holding the grouped edge tables
    pub fn group_by(&self, columns: Vec<String>) -> PyResult<PyEdgesTableArray> {
        let grouped_bases = self.table.base_table().group_by(&columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert each grouped table to PyEdgesTable
        let py_tables: Vec<PyEdgesTable> = grouped_bases.into_iter()
            .map(|base_table| -> PyResult<PyEdgesTable> {
                let edges_table = groggy::storage::table::EdgesTable::from_base_table(base_table)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(PyEdgesTable { table: edges_table })
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyEdgesTableArray { tables: py_tables })
    }
    
    /// Get a slice of rows [start, end)
    ///
    /// Args:
    ///     start: Starting row index (inclusive)
    ///     end: Ending row index (exclusive)
    ///
    /// Returns:
    ///     PyEdgesTable: A new table with the specified row slice
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self {
            table: self.table.slice(start, end),
        }
    }
    
    /// Enable subscripting: table[column_name] or table[slice]
    pub fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        use pyo3::types::{PyString, PySlice};
        let py = key.py();
        
        if let Ok(column_name) = key.extract::<String>() {
            // Column access: table['column_name']
            if let Some(column) = self.table.base_table().column(&column_name) {
                let attr_values = column.data();
                // Prefer StatsArray for numeric columns; fallback to GraphArray
                if let Ok(stats) = crate::ffi::storage::num_array::PyNumArray::from_attr_values(attr_values.clone()) {
                    Ok(stats.into_py(py))
                } else {
                    let base = groggy::storage::array::BaseArray::from_attr_values(attr_values.clone());
                    let py_base = crate::ffi::storage::array::PyBaseArray { inner: base };
                    Ok(py_base.into_py(py))
                }
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Column '{}' not found", column_name)
                ))
            }
        } else if let Ok(column_names) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2', ...]]
            // For EdgesTable, we need to ensure required columns are included
            let mut required_columns = vec!["edge_id".to_string(), "source".to_string(), "target".to_string()];
            let mut all_columns = required_columns.clone();
            
            // Add requested columns that aren't already required
            for col_name in column_names {
                if !required_columns.contains(&col_name) {
                    all_columns.push(col_name);
                }
            }
            
            let selected_base = self.table.base_table().select(&all_columns)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Failed to select columns: {}", e)
                ))?;
            let edges_selected = groggy::storage::table::EdgesTable::from_base_table(selected_base)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyEdgesTable { table: edges_selected }.into_py(py))
        } else if let Ok(slice) = key.downcast::<PySlice>() {
            // Slice access: table[start:end]
            let indices = slice.indices(self.table.nrows() as i64)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step as usize;
            
            if step != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Step slicing not yet implemented"
                ));
            }
            
            // Create a new EdgesTable with sliced rows
            let base_sliced = self.table.base_table().head(stop).tail(stop - start);
            let edges_sliced = groggy::storage::table::EdgesTable::from_base_table(base_sliced)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyEdgesTable { table: edges_sliced }.into_py(py))
        } else if let Ok(py_array) = key.extract::<crate::ffi::storage::array::PyGraphArray>() {
            // Boolean mask access: table[boolean_array]
            let mask_values = py_array.to_list(py)?;
            let mut mask_booleans = Vec::new();
            
            for value in mask_values.iter() {
                // Try to extract as plain Python bool first
                if let Ok(b) = value.extract::<bool>(py) {
                    mask_booleans.push(b);
                } else if let Ok(py_attr) = value.extract::<crate::ffi::types::PyAttrValue>(py) {
                    if let groggy::AttrValue::Bool(b) = py_attr.inner {
                        mask_booleans.push(b);
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "Boolean mask must contain only boolean values"
                        ));
                    }
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Boolean mask must contain boolean values"
                    ));
                }
            }
            
            if mask_booleans.len() != self.table.nrows() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Boolean mask length ({}) does not match table length ({})", 
                            mask_booleans.len(), self.table.nrows())
                ));
            }
            
            // Filter the base table using the boolean mask
            let filtered_base = self.table.base_table().filter_by_mask(&mask_booleans)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let edges_filtered = groggy::storage::table::EdgesTable::from_base_table(filtered_base)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyEdgesTable { table: edges_filtered }.into_py(py))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Table indices must be strings (column names), lists of column names, slices, or boolean arrays"
            ))
        }
    }

    /// Get display data structure for formatters
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::{PyDict, PyList};
        
        let dict = PyDict::new(py);
        
        // Get table dimensions
        let nrows = self.table.nrows();
        let ncols = self.table.ncols();
        dict.set_item("shape", (nrows, ncols))?;
        
        // Get column names
        let column_names = self.table.column_names();
        let py_columns = PyList::new(py, column_names);
        dict.set_item("columns", py_columns)?;
        
        // Get data types for each column
        let dtypes_dict = PyDict::new(py);
        for col_name in column_names {
            if let Some(column) = self.table.column(col_name) {
                let dtype = match column.data().first() {
                    Some(groggy::AttrValue::Int(_)) => "int64",
                    Some(groggy::AttrValue::SmallInt(_)) => "int32",
                    Some(groggy::AttrValue::Float(_)) => "float64",
                    Some(groggy::AttrValue::Text(_)) => "string",
                    Some(groggy::AttrValue::CompactText(_)) => "string",
                    Some(groggy::AttrValue::Bool(_)) => "bool",
                    _ => "object",
                };
                dtypes_dict.set_item(col_name, dtype)?;
            }
        }
        dict.set_item("dtypes", dtypes_dict)?;
        
        // Get sample data (first 10 rows for display)
        let sample_size = std::cmp::min(10, nrows);
        let data_list = PyList::empty(py);
        
        for row_idx in 0..sample_size {
            let row_list = PyList::empty(py);
            for col_name in column_names {
                if let Some(column) = self.table.column(col_name) {
                    let attr_values = column.data();
                    if let Some(value) = attr_values.get(row_idx) {
                        let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                        row_list.append(py_attr.to_object(py))?;
                    } else {
                        row_list.append(py.None())?;
                    }
                } else {
                    row_list.append(py.None())?;
                }
            }
            data_list.append(row_list)?;
        }
        dict.set_item("data", data_list)?;
        
        // Set index type
        dict.set_item("index_type", "int64")?;
        
        Ok(dict.to_object(py))
    }
    
    /// Get rich display representation with EdgesTable type
    pub fn rich_display(&self, config: Option<&crate::ffi::display::PyDisplayConfig>) -> PyResult<String> {
        let base_table = self.base_table();
        let display_data = base_table.to_display_data();
        let default_config = groggy::display::DisplayConfig::default();
        let rust_config = config.map(|c| c.get_config()).unwrap_or(&default_config);
        let mut formatted = groggy::display::format_table(display_data, rust_config);
        
        // Replace the footer with EdgesTable-specific info
        let nrows = self.table.nrows();
        let ncols = self.table.ncols();
        let footer = format!("rows: {} • cols: {} • type: EdgesTable", nrows, ncols);
        
        // Replace the last line with our custom footer
        let lines: Vec<&str> = formatted.lines().collect();
        if let Some(last_line_idx) = lines.iter().rposition(|line| line.contains("rows:") || line.contains("•")) {
            let mut new_lines = lines[..last_line_idx].to_vec();
            new_lines.push(&footer);
            formatted = new_lines.join("\n");
        }
        
        Ok(formatted)
    }
    
    /// Delegate to BaseTable for missing methods (enables rich display)
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let base_table = self.base_table();
        let base_obj = base_table.into_py(py);
        base_obj.getattr(py, name)
    }
    
    /// Export EdgesTable to CSV file
    pub fn to_csv(&self, path: &str) -> PyResult<()> {
        self.table.base_table().to_csv(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import EdgesTable from CSV file (must contain edge_id, source, target columns)
    #[staticmethod]
    pub fn from_csv(path: &str) -> PyResult<PyEdgesTable> {
        let base_table = groggy::storage::table::BaseTable::from_csv(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let edges_table = groggy::storage::table::EdgesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyEdgesTable { table: edges_table })
    }
    
    /// Export EdgesTable to Parquet file 
    pub fn to_parquet(&self, path: &str) -> PyResult<()> {
        self.table.base_table().to_parquet(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import EdgesTable from Parquet file (must contain edge_id, source, target columns)
    #[staticmethod]
    pub fn from_parquet(path: &str) -> PyResult<PyEdgesTable> {
        let base_table = groggy::storage::table::BaseTable::from_parquet(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let edges_table = groggy::storage::table::EdgesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyEdgesTable { table: edges_table })
    }
    
    /// Export EdgesTable to JSON file
    pub fn to_json(&self, path: &str) -> PyResult<()> {
        self.table.base_table().to_json(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Import EdgesTable from JSON file (must contain edge_id, source, target columns)
    #[staticmethod]
    pub fn from_json(path: &str) -> PyResult<PyEdgesTable> {
        let base_table = groggy::storage::table::BaseTable::from_json(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let edges_table = groggy::storage::table::EdgesTable::from_base_table(base_table)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyEdgesTable { table: edges_table })
    }
}

// =============================================================================
// PyEdgesTableRowIterator - Iterator for edges table rows
// =============================================================================

/// Iterator for iterating over edges table rows as dictionaries
#[pyclass(name = "EdgesTableRowIterator", module = "groggy")]
pub struct PyEdgesTableRowIterator {
    pub(crate) table: EdgesTable,
    pub(crate) current_row: usize,
}

#[pymethods]
impl PyEdgesTableRowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.current_row >= self.table.nrows() {
            return Ok(None);
        }
        
        // Create a dictionary for the current row
        let dict = pyo3::types::PyDict::new(py);
        
        for col_name in self.table.column_names() {
            if let Some(column) = self.table.column(col_name) {
                let attr_values = column.data();
                if let Some(value) = attr_values.get(self.current_row) {
                    let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                    dict.set_item(col_name, py_attr.to_object(py))?;
                } else {
                    dict.set_item(col_name, py.None())?;
                }
            }
        }
        
        self.current_row += 1;
        Ok(Some(dict.to_object(py)))
    }
}

// =============================================================================
// PyEdgesTableIterator - Python wrapper for TableIterator<EdgesTable>
// =============================================================================

/// Python wrapper for EdgesTable iterator
#[pyclass(name = "EdgesTableIterator", module = "groggy")]
pub struct PyEdgesTableIterator {
    pub(crate) iterator: TableIterator<EdgesTable>,
}

#[pymethods]
impl PyEdgesTableIterator {
    /// Execute all operations and return result
    pub fn collect(&self) -> PyResult<PyEdgesTable> {
        let result = self.iterator.clone().collect()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(PyEdgesTable { table: result })
    }
}

// =============================================================================
// PyGraphTable - Python wrapper for GraphTable (composite)
// =============================================================================

/// Python wrapper for GraphTable
#[pyclass(name = "GraphTable", module = "groggy")]
#[derive(Clone)]
pub struct PyGraphTable {
    pub(crate) table: groggy::storage::table::GraphTable,
}

#[pymethods]
impl PyGraphTable {
    /// Create a new GraphTable from NodesTable and EdgesTable
    #[new]
    pub fn new(nodes: PyNodesTable, edges: PyEdgesTable) -> Self {
        let graph_table = groggy::storage::table::GraphTable::new(nodes.table, edges.table);
        Self { table: graph_table }
    }
    
    /// Get number of total rows (nodes + edges)
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns (max of nodes and edges)
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Get NodesTable component
    #[getter]
    pub fn nodes(&self) -> PyNodesTable {
        PyNodesTable { table: self.table.nodes().clone() }
    }
    
    /// Get EdgesTable component  
    #[getter]
    pub fn edges(&self) -> PyEdgesTable {
        PyEdgesTable { table: self.table.edges().clone() }
    }
    
    /// Validate the GraphTable and return report
    pub fn validate(&self) -> PyResult<String> {
        let report = self.table.validate();
        Ok(format!("{:?}", report))
    }
    
    /// Auto-assign edge IDs for null values (useful for meta nodes and imported data)
    pub fn auto_assign_edge_ids(&self) -> PyResult<Self> {
        let fixed_table = self.table.clone().auto_assign_edge_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { table: fixed_table })
    }
    
    /// Convert back to Graph
    pub fn to_graph(&self) -> PyResult<crate::ffi::api::graph::PyGraph> {
        let graph = self.table.clone().to_graph()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(crate::ffi::api::graph::PyGraph {
            inner: std::rc::Rc::new(std::cell::RefCell::new(graph)),
            cached_view: std::cell::RefCell::new(None),
        })
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("{}", self.table)
    }
    
    /// String representation  
    pub fn __repr__(&self) -> String {
        let nodes_rows = self.table.nodes().nrows();
        let nodes_cols = self.table.nodes().ncols();
        let edges_rows = self.table.edges().nrows();
        let edges_cols = self.table.edges().ncols();
        
        format!(
            "GraphTable[\n  NodesTable: {} rows × {} cols\n  EdgesTable: {} rows × {} cols\n]",
            nodes_rows, nodes_cols, edges_rows, edges_cols
        )
    }

    
    /// Get length (total number of entities: nodes + edges) for len() function
    pub fn __len__(&self) -> usize {
        self.table.nodes().nrows() + self.table.edges().nrows()
    }
    
    /// Get first n rows (primarily from nodes table, default 5)
    #[pyo3(signature = (n = 5))]
    pub fn head(&self, n: usize) -> PyGraphTable {
        PyGraphTable { table: self.table.head(n) }
    }
    
    /// Get last n rows (primarily from nodes table, default 5) 
    #[pyo3(signature = (n = 5))]
    pub fn tail(&self, n: usize) -> PyGraphTable {
        PyGraphTable { table: self.table.tail(n) }
    }
    
    /// Merge multiple GraphTables into one
    #[staticmethod]
    pub fn merge(tables: Vec<PyGraphTable>) -> PyResult<PyGraphTable> {
        let rust_tables: Vec<groggy::storage::table::GraphTable> = 
            tables.into_iter().map(|t| t.table).collect();
            
        let merged = groggy::storage::table::GraphTable::merge(rust_tables)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table: merged })
    }
    
    /// Merge with conflict resolution strategy
    #[staticmethod] 
    pub fn merge_with_strategy(tables: Vec<PyGraphTable>, strategy: &str) -> PyResult<PyGraphTable> {
        use groggy::storage::table::ConflictResolution;
        
        let conflict_strategy = match strategy.to_lowercase().as_str() {
            "fail" => ConflictResolution::Fail,
            "keep_first" => ConflictResolution::KeepFirst,
            "keep_second" => ConflictResolution::KeepSecond,
            "merge_attributes" => ConflictResolution::MergeAttributes,
            "domain_prefix" => ConflictResolution::DomainPrefix,
            "auto_remap" => ConflictResolution::AutoRemap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown conflict resolution strategy: {}", strategy)
            ))
        };
        
        let rust_tables: Vec<groggy::storage::table::GraphTable> = 
            tables.into_iter().map(|t| t.table).collect();
            
        let merged = groggy::storage::table::GraphTable::merge_with_strategy(rust_tables, conflict_strategy)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table: merged })
    }
    
    /// Merge with another GraphTable
    pub fn merge_with(&mut self, other: PyGraphTable, strategy: &str) -> PyResult<()> {
        use groggy::storage::table::ConflictResolution;
        
        let conflict_strategy = match strategy.to_lowercase().as_str() {
            "fail" => ConflictResolution::Fail,
            "keep_first" => ConflictResolution::KeepFirst, 
            "keep_second" => ConflictResolution::KeepSecond,
            "merge_attributes" => ConflictResolution::MergeAttributes,
            "domain_prefix" => ConflictResolution::DomainPrefix,
            "auto_remap" => ConflictResolution::AutoRemap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown conflict resolution strategy: {}", strategy)
            ))
        };
        
        self.table.merge_with(other.table, conflict_strategy)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(())
    }
    
    /// Create federated GraphTable from multiple bundle paths
    #[staticmethod]
    pub fn from_federated_bundles(bundle_paths: Vec<&str>, domain_names: Option<Vec<String>>) -> PyResult<PyGraphTable> {
        use std::path::Path;
        
        let paths: Vec<&Path> = bundle_paths.iter().map(|s| Path::new(s)).collect();
        
        let table = groggy::storage::table::GraphTable::from_federated_bundles(paths, domain_names)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(PyGraphTable { table })
    }
    
    /// Get graph statistics  
    pub fn stats(&self) -> std::collections::HashMap<String, usize> {
        self.table.stats()
    }
    
    /// Enable subscripting: table[column_name] for unified access
    pub fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        let py = key.py();
        
        if let Ok(column_name) = key.extract::<String>() {
            // Try nodes first, then edges
            let nodes_table = self.table.nodes();
            if let Some(column) = nodes_table.base_table().column(&column_name) {
                let attr_values = column.data();
                if let Ok(stats) = crate::ffi::storage::num_array::PyNumArray::from_attr_values(attr_values.clone()) {
                    Ok(stats.into_py(py))
                } else {
                    let py_objects: Vec<_> = attr_values.iter()
                        .map(|attr| {
                            let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                            py_attr.to_object(py)
                        })
                        .collect();
                    let py_array = crate::ffi::storage::array::PyGraphArray::from_py_objects(py_objects)?;
                    Ok(py_array.into_py(py))
                }
            } else {
                let edges_table = self.table.edges();
                if let Some(column) = edges_table.base_table().column(&column_name) {
                    let attr_values = column.data();
                    if let Ok(stats) = crate::ffi::storage::num_array::PyNumArray::from_attr_values(attr_values.clone()) {
                        Ok(stats.into_py(py))
                    } else {
                        let base = groggy::storage::array::BaseArray::from_attr_values(attr_values.clone());
                        let py_base = crate::ffi::storage::array::PyBaseArray { inner: base };
                        Ok(py_base.into_py(py))
                    }
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        format!("Column '{}' not found in nodes or edges", column_name)
                    ))
                }
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "GraphTable indices must be strings (column names)"
            ))
        }
    }

    /// Get display data structure for formatters
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::{PyDict, PyList};
        
        let dict = PyDict::new(py);
        
        // GraphTable shows nodes by default, with metadata about edges
        let nodes_table = self.table.nodes();
        let edges_table = self.table.edges();
        
        // Get nodes table dimensions
        let nodes_nrows = nodes_table.nrows();
        let nodes_ncols = nodes_table.ncols();
        dict.set_item("shape", (nodes_nrows, nodes_ncols))?;
        
        // Add metadata about the composite nature
        dict.set_item("nodes_shape", (nodes_nrows, nodes_ncols))?;
        dict.set_item("edges_shape", (edges_table.nrows(), edges_table.ncols()))?;
        
        // Get nodes column names for display
        let column_names = nodes_table.column_names();
        let py_columns = PyList::new(py, column_names);
        dict.set_item("columns", py_columns)?;
        
        // Get data types for nodes columns
        let dtypes_dict = PyDict::new(py);
        for col_name in column_names {
            if let Some(column) = nodes_table.column(col_name) {
                let dtype = match column.data().first() {
                    Some(groggy::AttrValue::Int(_)) => "int64",
                    Some(groggy::AttrValue::SmallInt(_)) => "int32",
                    Some(groggy::AttrValue::Float(_)) => "float64",
                    Some(groggy::AttrValue::Text(_)) => "string",
                    Some(groggy::AttrValue::CompactText(_)) => "string",
                    Some(groggy::AttrValue::Bool(_)) => "bool",
                    _ => "object",
                };
                dtypes_dict.set_item(col_name, dtype)?;
            }
        }
        dict.set_item("dtypes", dtypes_dict)?;
        
        // Get sample data from nodes (first 10 rows for display)
        let sample_size = std::cmp::min(10, nodes_nrows);
        let data_list = PyList::empty(py);
        
        for row_idx in 0..sample_size {
            let row_list = PyList::empty(py);
            for col_name in column_names {
                if let Some(column) = nodes_table.column(col_name) {
                    let attr_values = column.data();
                    if let Some(value) = attr_values.get(row_idx) {
                        let py_attr = crate::ffi::types::PyAttrValue::new(value.clone());
                        row_list.append(py_attr.to_object(py))?;
                    } else {
                        row_list.append(py.None())?;
                    }
                } else {
                    row_list.append(py.None())?;
                }
            }
            data_list.append(row_list)?;
        }
        dict.set_item("data", data_list)?;
        
        // Set index type
        dict.set_item("index_type", "int64")?;
        
        // Add composite table info for special formatting
        dict.set_item("table_type", "GraphTable")?;
        
        Ok(dict.to_object(py))
    }

    // =============================================================================
    // Phase 2 Features: Enhanced Bundle System
    // =============================================================================

    /// Save GraphTable as a v2.0 bundle with comprehensive metadata and checksums
    /// 
    /// # Arguments
    /// * `bundle_path` - Directory path to save the bundle
    /// 
    /// # Examples
    /// ```python
    /// # Save with comprehensive metadata and validation
    /// graph_table.save_bundle("./graph_data_bundle")
    /// 
    /// # Bundle will contain:
    /// # - metadata.json: Comprehensive metadata with checksums
    /// # - MANIFEST.json: File integrity manifest
    /// # - validation_report.json: Structured validation results
    /// # - nodes.csv: Node data
    /// # - edges.csv: Edge data
    /// ```
    pub fn save_bundle(&self, bundle_path: &str) -> PyResult<()> {
        self.table.save_bundle(bundle_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to save bundle: {}", e)))
    }

    /// Load GraphTable from a bundle directory (supports both v1.0 and v2.0 formats)
    /// 
    /// # Arguments
    /// * `bundle_path` - Directory path containing the bundle
    /// 
    /// # Returns
    /// * `PyGraphTable` - Loaded graph table with validation policy restored
    /// 
    /// # Examples
    /// ```python
    /// # Load from v2.0 bundle (with integrity verification)
    /// graph_table = GraphTable.load_bundle("./graph_data_bundle")
    /// 
    /// # Also supports legacy v1.0 bundles
    /// graph_table = GraphTable.load_bundle("./old_bundle")
    /// ```
    #[staticmethod]
    pub fn load_bundle(bundle_path: &str) -> PyResult<Self> {
        let table = groggy::storage::table::GraphTable::load_bundle(bundle_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load bundle: {}", e)))?;
        
        Ok(Self { table })
    }

    /// Get bundle metadata information without loading the full bundle
    /// 
    /// # Arguments
    /// * `bundle_path` - Directory path containing the bundle
    /// 
    /// # Returns
    /// * `dict` - Bundle metadata information
    /// 
    /// # Examples
    /// ```python
    /// # Inspect bundle metadata
    /// metadata = GraphTable.get_bundle_info("./graph_data_bundle")
    /// print(f"Bundle version: {metadata['version']}")
    /// print(f"Nodes: {metadata['node_count']}, Edges: {metadata['edge_count']}")
    /// print(f"Created: {metadata['created_at']}")
    /// ```
    #[staticmethod]
    pub fn get_bundle_info(py: Python, bundle_path: &str) -> PyResult<PyObject> {
        use std::path::Path;
        
        let bundle_path = Path::new(bundle_path);
        
        // Try v2.0 format first (JSON metadata)
        let metadata_json_path = bundle_path.join("metadata.json");
        if metadata_json_path.exists() {
            let metadata_json = std::fs::read_to_string(&metadata_json_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read metadata: {}", e)))?;
            
            // Parse JSON to Python dict
            let metadata: serde_json::Value = serde_json::from_str(&metadata_json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to parse metadata JSON: {}", e)))?;
            
            // Convert to Python object
            let py_dict = pyo3::types::PyDict::new(py);
            json_value_to_py_dict(py, &metadata, py_dict)?;
            return Ok(py_dict.to_object(py));
        }
        
        // Fall back to v1.0 format (text metadata)
        let metadata_txt_path = bundle_path.join("metadata.txt");
        if metadata_txt_path.exists() {
            let metadata_text = std::fs::read_to_string(&metadata_txt_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read metadata: {}", e)))?;
            
            // Parse simple key-value format
            let py_dict = pyo3::types::PyDict::new(py);
            py_dict.set_item("format_version", "1.0")?;
            
            for line in metadata_text.lines() {
                if let Some((key, value)) = line.split_once(": ") {
                    py_dict.set_item(key.trim(), value.trim())?;
                }
            }
            
            return Ok(py_dict.to_object(py));
        }
        
        Err(pyo3::exceptions::PyFileNotFoundError::new_err(
            format!("Bundle metadata not found at: {}", bundle_path.display())
        ))
    }

    /// Verify bundle integrity without loading the full data
    /// 
    /// # Arguments
    /// * `bundle_path` - Directory path containing the bundle
    /// 
    /// # Returns
    /// * `dict` - Verification results with checksums and validation status
    /// 
    /// # Examples
    /// ```python
    /// # Verify bundle integrity
    /// verification = GraphTable.verify_bundle("./graph_data_bundle")
    /// if verification['is_valid']:
    ///     print("Bundle integrity verified!")
    /// else:
    ///     print(f"Issues found: {verification['errors']}")
    /// ```
    #[staticmethod]
    pub fn verify_bundle(py: Python, bundle_path: &str) -> PyResult<PyObject> {
        use std::path::Path;
        
        let bundle_path = Path::new(bundle_path);
        let py_dict = pyo3::types::PyDict::new(py);
        
        // Check if bundle directory exists
        if !bundle_path.exists() {
            py_dict.set_item("is_valid", false)?;
            py_dict.set_item("errors", vec!["Bundle directory does not exist"])?;
            return Ok(py_dict.to_object(py));
        }
        
        // For v2.0 bundles, verify checksums
        let metadata_json_path = bundle_path.join("metadata.json");
        let manifest_path = bundle_path.join("MANIFEST.json");
        
        if metadata_json_path.exists() && manifest_path.exists() {
            // This would implement full checksum verification
            // For now, just check file existence
            let required_files = vec!["metadata.json", "MANIFEST.json", "nodes.csv", "edges.csv"];
            let mut missing_files = Vec::new();
            
            for file_name in &required_files {
                if !bundle_path.join(file_name).exists() {
                    missing_files.push(file_name.to_string());
                }
            }
            
            py_dict.set_item("format_version", "2.0")?;
            py_dict.set_item("is_valid", missing_files.is_empty())?;
            py_dict.set_item("missing_files", missing_files)?;
            py_dict.set_item("checksum_verified", false)?; // TODO: Implement full verification
            
        } else {
            // v1.0 bundle format
            let required_files = vec!["metadata.txt", "nodes.csv", "edges.csv"];
            let mut missing_files = Vec::new();
            
            for file_name in &required_files {
                if !bundle_path.join(file_name).exists() {
                    missing_files.push(file_name.to_string());
                }
            }
            
            py_dict.set_item("format_version", "1.0")?;
            py_dict.set_item("is_valid", missing_files.is_empty())?;
            py_dict.set_item("missing_files", missing_files)?;
        }
        
        Ok(py_dict.to_object(py))
    }

    // =============================================================================
    // Helper Methods for Bundle System
    // =============================================================================

    // =============================================================================
    // Cross-Type Conversions for Phase 3 Delegation Architecture
    // =============================================================================

    /// Convert table to NodesAccessor by extracting node IDs from the table
    pub fn to_nodes(&self) -> PyResult<crate::ffi::storage::accessors::PyNodesAccessor> {
        // TODO: Implement proper conversion from GraphTable to NodesAccessor
        // This requires understanding the graph context and creating appropriate accessors
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GraphTable to NodesAccessor conversion not yet implemented. Use .nodes() property instead."
        ))
    }

    /// Convert table to EdgesAccessor by extracting edge IDs from the table
    pub fn to_edges(&self) -> PyResult<crate::ffi::storage::accessors::PyEdgesAccessor> {
        // TODO: Implement proper conversion from GraphTable to EdgesAccessor
        // This requires understanding the graph context and creating appropriate accessors
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GraphTable to EdgesAccessor conversion not yet implemented. Use .edges() property instead."
        ))
    }

    /// Convert table to SubgraphArray by creating subgraphs from table rows
    pub fn to_subgraphs(&self) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        // TODO: Implement proper conversion from GraphTable to SubgraphArray
        // This requires understanding the graph context and creating appropriate subgraphs
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "GraphTable to SubgraphArray conversion not yet implemented."
        ))
    }
}

/// Convert JSON value to Python dictionary recursively (helper function)
fn json_value_to_py_dict(py: Python, value: &serde_json::Value, py_dict: &pyo3::types::PyDict) -> PyResult<()> {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                match val {
                    serde_json::Value::Object(_) => {
                        let nested_dict = pyo3::types::PyDict::new(py);
                        json_value_to_py_dict(py, val, nested_dict)?;
                        py_dict.set_item(key, nested_dict)?;
                    }
                    serde_json::Value::Array(arr) => {
                        let py_list = pyo3::types::PyList::new(py, arr.iter().map(|v| match v {
                            serde_json::Value::String(s) => s.to_object(py),
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    i.to_object(py)
                                } else if let Some(f) = n.as_f64() {
                                    f.to_object(py)
                                } else {
                                    n.to_string().to_object(py)
                                }
                            }
                            serde_json::Value::Bool(b) => b.to_object(py),
                            _ => py.None(),
                        }));
                        py_dict.set_item(key, py_list)?;
                    }
                    serde_json::Value::String(s) => {
                        py_dict.set_item(key, s)?;
                    }
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            py_dict.set_item(key, i)?;
                        } else if let Some(f) = n.as_f64() {
                            py_dict.set_item(key, f)?;
                        } else {
                            py_dict.set_item(key, n.to_string())?;
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        py_dict.set_item(key, b)?;
                    }
                    serde_json::Value::Null => {
                        py_dict.set_item(key, py.None())?;
                    }
                }
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyTypeError::new_err("Expected JSON object"));
        }
    }
    Ok(())
}

// Internal methods (not exposed to Python)
impl PyGraphTable {
    /// Create PyGraphTable from GraphTable (used by accessors - not exposed to Python)
    pub(crate) fn from_graph_table(table: groggy::storage::table::GraphTable) -> Self {
        Self { table }
    }
}

// =============================================================================
// PyTableArray - Array-like container for grouped tables
// =============================================================================

/// Array-like container that holds multiple tables (used for group_by results)
#[pyclass(name = "TableArray", module = "groggy")]
#[derive(Clone)]
pub struct PyTableArray {
    tables: Vec<PyBaseTable>,
    /// Column names that were used for grouping (used for aggregation column selection)
    group_columns: Vec<String>,
}

#[pymethods]
impl PyTableArray {
    /// Create a new TableArray from a list of tables
    /// 
    /// Args:
    ///     tables: List of BaseTable objects
    ///     columns: Optional list of column names to slice/select from each table
    /// 
    /// Examples:
    /// ```python
    /// # Create TableArray without column slicing
    /// table_array = TableArray(tables)
    /// 
    /// # Create TableArray with column slicing
    /// table_array = TableArray(tables, columns=['col1', 'col2'])
    /// ```
    #[new]
    #[pyo3(signature = (tables, columns = None))]
    pub fn new(tables: Vec<PyBaseTable>, columns: Option<Vec<String>>) -> PyResult<Self> {
        let processed_tables = if let Some(column_names) = columns.as_ref() {
            // If columns are provided, slice each table to only include those columns
            let mut sliced_tables = Vec::new();
            for table in tables {
                let sliced_table = table.table.select(column_names)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Failed to slice columns: {}", e)
                    ))?;
                sliced_tables.push(PyBaseTable { table: sliced_table });
            }
            sliced_tables
        } else {
            // If no columns specified, use tables as-is
            tables
        };
        
        Ok(Self {
            tables: processed_tables,
            group_columns: columns.unwrap_or_default(),
        })
    }

    /// Get the number of grouped tables
    pub fn __len__(&self) -> usize {
        self.tables.len()
    }
    
    /// Get a table by index
    pub fn __getitem__(&self, index: isize) -> PyResult<PyBaseTable> {
        let len = self.tables.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        
        if idx < 0 || idx >= len {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Table index {} out of range for {} groups", index, len)
            ));
        }
        
        Ok(self.tables[idx as usize].clone())
    }
    
    /// Iterate over the tables
    pub fn __iter__(&self) -> PyTableArrayIterator {
        PyTableArrayIterator {
            tables: self.tables.clone(),
            index: 0,
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("TableArray[{} groups]", self.tables.len())
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("TableArray[{} groups]", self.tables.len())
    }
    
    /// Get all tables as a list
    pub fn to_list(&self) -> Vec<PyBaseTable> {
        self.tables.clone()
    }
    
    /// Aggregate across all tables in the array
    /// 
    /// Args:
    ///     agg_specs: Dictionary mapping column names to aggregation functions
    ///                Supported functions: "count", "sum", "avg", "mean", "min", "max"
    /// 
    /// Returns:
    ///     PyBaseTable: A single table with aggregated results
    /// 
    /// Examples:
    /// ```python
    /// # Group by department, then aggregate salary and count users
    /// grouped = table.group_by(['department'])
    /// result = grouped.agg({'salary': 'avg', 'user_id': 'count'})
    /// ```
    pub fn agg(&self, agg_specs: HashMap<String, String>) -> PyResult<PyBaseTable> {
        if self.tables.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot aggregate empty table array"
            ));
        }
        
        use std::collections::HashMap;
        use groggy::types::AttrValue;
        
        let group_columns = &self.group_columns;
        let agg_column_names: Vec<String> = agg_specs.keys().cloned().collect();
        
        // Prepare result columns: group columns + aggregated columns
        let mut result_columns: HashMap<String, Vec<AttrValue>> = HashMap::new();
        
        // Initialize columns
        for col_name in group_columns {
            result_columns.insert(col_name.clone(), Vec::new());
        }
        for col_name in &agg_column_names {
            result_columns.insert(col_name.clone(), Vec::new());
        }
        
        // Process each group (table)
        for group_table in &self.tables {
            if group_table.table.nrows() == 0 {
                continue; // Skip empty groups
            }
            
            // Get group key values (first row of the group for group-by columns)
            for col_name in group_columns {
                if let Some(column) = group_table.table.column(col_name) {
                    let group_value = column.get(0).cloned().unwrap_or(AttrValue::Null);
                    if let Some(col_vec) = result_columns.get_mut(col_name) {
                        col_vec.push(group_value);
                    }
                }
            }
            
            // Calculate aggregated values for this group
            for (col_name, agg_func) in &agg_specs {
                let agg_value = if let Some(column) = group_table.table.column(col_name) {
                    self.calculate_aggregation(column.data(), agg_func)?
                } else {
                    AttrValue::Null
                };
                
                if let Some(col_vec) = result_columns.get_mut(col_name) {
                    col_vec.push(agg_value);
                }
            }
        }
        
        // Convert to BaseArray columns and create table
        let mut final_columns = HashMap::new();
        for (col_name, values) in result_columns {
            final_columns.insert(col_name, groggy::storage::array::BaseArray::from_attr_values(values));
        }
        
        let result_table = groggy::storage::table::BaseTable::from_columns(final_columns)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create aggregated table: {}", e)
            ))?;
            
        Ok(PyBaseTable { table: result_table })
    }
    
    /// Iterator-style aggregation - applies aggregation functions as if iterating
    /// 
    /// This is equivalent to calling .iter().agg(), providing a more functional approach
    pub fn iter_agg(&self, agg_specs: HashMap<String, String>) -> PyResult<PyBaseTable> {
        // For now, delegate to the regular agg method
        // In the future, this could have different semantics (e.g., streaming)
        self.agg(agg_specs)
    }
}

// Implementation block for non-Python methods
impl PyTableArray {
    /// Helper method to calculate aggregation on a column's data
    /// Note: This method is not exposed to Python (no #[pyo3] annotation)
    fn calculate_aggregation(&self, data: &[AttrValue], agg_func: &str) -> PyResult<AttrValue> {
        match agg_func.to_lowercase().as_str() {
            "count" => {
                // Count non-null values
                let count = data.iter().filter(|v| !matches!(v, AttrValue::Null)).count() as i64;
                Ok(AttrValue::Int(count))
            }
            "sum" => {
                let mut sum = 0.0;
                let mut found_numeric = false;
                for value in data {
                    match value {
                        AttrValue::Int(i) => { sum += *i as f64; found_numeric = true; }
                        AttrValue::SmallInt(i) => { sum += *i as f64; found_numeric = true; }
                        AttrValue::Float(f) => { sum += *f as f64; found_numeric = true; }
                        AttrValue::Null => {} // Skip nulls
                        _ => {} // Skip non-numeric
                    }
                }
                if found_numeric {
                    Ok(AttrValue::Float(sum as f32))
                } else {
                    Ok(AttrValue::Null)
                }
            }
            "avg" | "mean" => {
                let mut sum = 0.0;
                let mut count = 0;
                for value in data {
                    match value {
                        AttrValue::Int(i) => { sum += *i as f64; count += 1; }
                        AttrValue::SmallInt(i) => { sum += *i as f64; count += 1; }
                        AttrValue::Float(f) => { sum += *f as f64; count += 1; }
                        AttrValue::Null => {} // Skip nulls
                        _ => {} // Skip non-numeric
                    }
                }
                if count > 0 {
                    Ok(AttrValue::Float((sum / count as f64) as f32))
                } else {
                    Ok(AttrValue::Null)
                }
            }
            "min" => {
                let mut min_val: Option<f64> = None;
                for value in data {
                    let numeric_val = match value {
                        AttrValue::Int(i) => Some(*i as f64),
                        AttrValue::SmallInt(i) => Some(*i as f64),
                        AttrValue::Float(f) => Some(*f as f64),
                        _ => None,
                    };
                    if let Some(val) = numeric_val {
                        min_val = Some(min_val.map_or(val, |current| current.min(val)));
                    }
                }
                Ok(min_val.map(|v| AttrValue::Float(v as f32)).unwrap_or(AttrValue::Null))
            }
            "max" => {
                let mut max_val: Option<f64> = None;
                for value in data {
                    let numeric_val = match value {
                        AttrValue::Int(i) => Some(*i as f64),
                        AttrValue::SmallInt(i) => Some(*i as f64),
                        AttrValue::Float(f) => Some(*f as f64),
                        _ => None,
                    };
                    if let Some(val) = numeric_val {
                        max_val = Some(max_val.map_or(val, |current| current.max(val)));
                    }
                }
                Ok(max_val.map(|v| AttrValue::Float(v as f32)).unwrap_or(AttrValue::Null))
            }
            _ => {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported aggregation function: {}", agg_func)
                ))
            }
        }
    }
}

/// Iterator for PyTableArray
#[pyclass(name = "TableArrayIterator", module = "groggy")]
pub struct PyTableArrayIterator {
    tables: Vec<PyBaseTable>,
    index: usize,
}

#[pymethods]
impl PyTableArrayIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self) -> Option<PyBaseTable> {
        if self.index < self.tables.len() {
            let table = self.tables[self.index].clone();
            self.index += 1;
            Some(table)
        } else {
            None
        }
    }
}

// =============================================================================
// PyNodesTableArray - Array-like container for grouped NodesTable
// =============================================================================

/// Array-like container that holds multiple nodes tables (used for group_by results)
#[pyclass(name = "NodesTableArray", module = "groggy")]
#[derive(Clone)]
pub struct PyNodesTableArray {
    tables: Vec<PyNodesTable>,
    group_columns: Vec<String>,
}

#[pymethods]
impl PyNodesTableArray {
    /// Get the number of grouped tables
    pub fn __len__(&self) -> usize {
        self.tables.len()
    }
    
    /// Get a table by index
    pub fn __getitem__(&self, index: isize) -> PyResult<PyNodesTable> {
        let len = self.tables.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        
        if idx < 0 || idx >= len {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Table index {} out of range for {} groups", index, len)
            ));
        }
        
        Ok(self.tables[idx as usize].clone())
    }
    
    /// Iterate over the tables
    pub fn __iter__(&self) -> PyNodesTableArrayIterator {
        PyNodesTableArrayIterator {
            tables: self.tables.clone(),
            index: 0,
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("NodesTableArray[{} groups]", self.tables.len())
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("NodesTableArray[{} groups]", self.tables.len())
    }
    
    /// Get all tables as a list
    pub fn to_list(&self) -> Vec<PyNodesTable> {
        self.tables.clone()
    }
    
    /// Delegate to PyTableArray methods (like .agg())
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        // Convert NodesTable vector to BaseTable vector for delegation
        let base_tables: Vec<PyBaseTable> = self.tables.iter()
            .map(|nodes_table| nodes_table.base_table())
            .collect();
        
        let table_array = PyTableArray { 
            tables: base_tables,
            group_columns: self.group_columns.clone(), // Pass group columns through delegation
        };
        let array_obj = table_array.into_py(py);
        array_obj.getattr(py, name)
    }
}

/// Iterator for PyNodesTableArray
#[pyclass(name = "NodesTableArrayIterator", module = "groggy")]
pub struct PyNodesTableArrayIterator {
    tables: Vec<PyNodesTable>,
    index: usize,
}

#[pymethods]
impl PyNodesTableArrayIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self) -> Option<PyNodesTable> {
        if self.index < self.tables.len() {
            let table = self.tables[self.index].clone();
            self.index += 1;
            Some(table)
        } else {
            None
        }
    }
}

// =============================================================================
// PyEdgesTableArray - Array-like container for grouped EdgesTable
// =============================================================================

/// Array-like container that holds multiple edges tables (used for group_by results)
#[pyclass(name = "EdgesTableArray", module = "groggy")]
#[derive(Clone)]
pub struct PyEdgesTableArray {
    tables: Vec<PyEdgesTable>,
}

#[pymethods]
impl PyEdgesTableArray {
    /// Get the number of grouped tables
    pub fn __len__(&self) -> usize {
        self.tables.len()
    }
    
    /// Get a table by index
    pub fn __getitem__(&self, index: isize) -> PyResult<PyEdgesTable> {
        let len = self.tables.len() as isize;
        let idx = if index < 0 { len + index } else { index };
        
        if idx < 0 || idx >= len {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Table index {} out of range for {} groups", index, len)
            ));
        }
        
        Ok(self.tables[idx as usize].clone())
    }
    
    /// Iterate over the tables
    pub fn __iter__(&self) -> PyEdgesTableArrayIterator {
        PyEdgesTableArrayIterator {
            tables: self.tables.clone(),
            index: 0,
        }
    }
    
    /// String representation
    pub fn __str__(&self) -> String {
        format!("EdgesTableArray[{} groups]", self.tables.len())
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("EdgesTableArray[{} groups]", self.tables.len())
    }
    
    /// Get all tables as a list
    pub fn to_list(&self) -> Vec<PyEdgesTable> {
        self.tables.clone()
    }
    
    /// Delegate to PyTableArray methods (like .agg())
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        // Convert EdgesTable vector to BaseTable vector for delegation
        let base_tables: Vec<PyBaseTable> = self.tables.iter()
            .map(|edges_table| edges_table.base_table())
            .collect();
        
        let table_array = PyTableArray { 
            tables: base_tables,
            group_columns: Vec::new(), // Empty group columns for delegation
        };
        let array_obj = table_array.into_py(py);
        array_obj.getattr(py, name)
    }
}

/// Iterator for PyEdgesTableArray
#[pyclass(name = "EdgesTableArrayIterator", module = "groggy")]
pub struct PyEdgesTableArrayIterator {
    tables: Vec<PyEdgesTable>,
    index: usize,
}

#[pymethods]
impl PyEdgesTableArrayIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __next__(&mut self) -> Option<PyEdgesTable> {
        if self.index < self.tables.len() {
            let table = self.tables[self.index].clone();
            self.index += 1;
            Some(table)
        } else {
            None
        }
    }
}
