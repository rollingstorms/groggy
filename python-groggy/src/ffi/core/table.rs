//! Table FFI Bindings
//! 
//! Python bindings for GraphTable - DataFrame-like views for graph data.
//! This is a thin wrapper around the core Rust GraphTable implementation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError, PyImportError, PyNotImplementedError};
use groggy::{NodeId, EdgeId, AttrValue as RustAttrValue, GraphTable, GraphArray, TableMetadata};
use std::collections::HashMap;
use std::rc::Rc;

// Import utilities
use crate::ffi::utils::{python_value_to_attr_value, attr_value_to_python_value, graph_error_to_py_err};
use crate::ffi::api::graph::PyGraph;
use crate::ffi::core::array::PyGraphArray;

/// Python wrapper around core GraphTable implementation
#[pyclass(name = "GraphTable", unsendable)]
pub struct PyGraphTable {
    /// Core GraphTable implementation
    pub inner: GraphTable,
}

impl PyGraphTable {
    /// Create from a core GraphTable
    pub fn from_graph_table(table: GraphTable) -> Self {
        Self {
            inner: table,
        }
    }
}

#[pymethods]
impl PyGraphTable {
    /// Create a new GraphTable from arrays
    #[new]
    #[pyo3(signature = (arrays, column_names = None))]
    pub fn new(
        py: Python,
        arrays: Vec<Py<PyGraphArray>>,
        column_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // Convert PyGraphArrays to core GraphArrays
        let core_arrays: Vec<GraphArray> = arrays.iter()
            .map(|py_array| py_array.borrow(py).inner.clone())
            .collect();
        
        // Create core GraphTable
        let table = GraphTable::from_arrays_standalone(core_arrays, column_names)
            .map_err(graph_error_to_py_err)?;
        
        Ok(Self::from_graph_table(table))
    }

    /// Create GraphTable from graph nodes
    #[classmethod]
    pub fn from_graph_nodes(
        _cls: &PyType,
        _py: Python,
        _graph: Py<PyGraph>,
        _nodes: Vec<u64>,
        _attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // TODO: Implement graph nodes integration in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "GraphTable.from_graph_nodes temporarily disabled during Phase 2 unification"
        ))
    }

    /// Create GraphTable from graph edges
    #[classmethod]
    pub fn from_graph_edges(
        _cls: &PyType,
        _py: Python,
        _graph: Py<PyGraph>,
        _edges: Vec<u64>,
        _attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        // TODO: Implement graph edges integration in Phase 2
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "GraphTable.from_graph_edges temporarily disabled during Phase 2 unification"
        ))
    }

    /// Number of rows in the table
    fn __len__(&self) -> usize {
        self.inner.shape().0
    }

    /// String representation with proper display formatting
    fn __repr__(&self, py: Python) -> PyResult<String> {
        // Try rich display formatting first
        match self._try_rich_display(py) {
            Ok(formatted) => Ok(formatted),
            Err(_) => {
                // Fallback to simple representation
                let (rows, cols) = self.inner.shape();
                Ok(format!("GraphTable({} rows, {} columns)", rows, cols))
            }
        }
    }

    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, py: Python) -> PyResult<String> {
        match self._try_rich_html_display(py) {
            Ok(html) => Ok(html),
            Err(_) => {
                // Fallback to basic HTML
                let (rows, cols) = self.inner.shape();
                Ok(format!(
                    r#"<div style="font-family: monospace; padding: 10px; border: 1px solid #ddd;">
                    <strong>GraphTable</strong><br>
                    Shape: ({}, {})
                    </div>"#,
                    rows, cols
                ))
            }
        }
    }

    /// Try to use rich display formatting
    fn _try_rich_display(&self, py: Python) -> PyResult<String> {
        // Get display data for formatting
        let display_data = self._get_display_data(py)?;
        
        // Import the format_table function from Python
        let groggy_module = py.import("groggy")?;
        let format_table = groggy_module.getattr("format_table")?;
        
        // Call the Python formatter
        let result = format_table.call1((display_data,))?;
        let formatted_str: String = result.extract()?;
        
        Ok(formatted_str)
    }

    /// Try to use rich HTML display formatting
    fn _try_rich_html_display(&self, py: Python) -> PyResult<String> {
        // Get display data for formatting
        let display_data = self._get_display_data(py)?;
        
        // Import the format_table_html function from Python
        let groggy_module = py.import("groggy")?;
        let display_module = groggy_module.getattr("display")?;
        let format_table_html = display_module.getattr("format_table_html")?;
        
        // Call the Python HTML formatter
        let result = format_table_html.call1((display_data,))?;
        let html_str: String = result.extract()?;
        
        Ok(html_str)
    }

    /// Get display data structure for formatters
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let (rows, cols) = self.inner.shape();
        
        // Build table data - convert first few rows for display
        let display_rows = std::cmp::min(rows, 100); // Limit for performance
        let mut data_rows = Vec::new();
        
        for row_idx in 0..display_rows {
            if let Some(row_data) = self.inner.iloc(row_idx) {
                let mut row_values = Vec::new();
                for col_name in self.inner.columns() {
                    let value = row_data.get(col_name).cloned().unwrap_or(RustAttrValue::Int(0));
                    row_values.push(attr_value_to_python_value(py, &value)?);
                }
                data_rows.push(row_values);
            }
        }
        
        // Set display data
        dict.set_item("data", data_rows)?;
        dict.set_item("columns", self.inner.columns().to_vec())?;
        dict.set_item("shape", (rows, cols))?;
        dict.set_item("source_type", &self.inner.metadata().source_type)?;
        
        // Add basic dtype detection
        let dtypes: HashMap<String, String> = self.inner.dtypes().into_iter()
            .map(|(k, v)| (k, format!("{:?}", v).to_lowercase()))
            .collect();
        dict.set_item("dtypes", dtypes)?;
        
        Ok(dict.to_object(py))
    }

    /// Get table shape
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get column names
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    /// Support indexing operations: table[key]
    /// - table[2] -> dict (single row access)
    /// - table[:2] -> table (slice access) 
    /// - table['column'] -> array (single column access)
    /// - table[['col1', 'col2']] -> table (multi-column access)
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Handle different key types
        if let Ok(row_index) = key.extract::<isize>() {
            // Single row access: table[2] -> dict
            let (rows, _) = self.inner.shape();
            let len = rows as isize;
            let actual_index = if row_index < 0 {
                len + row_index
            } else {
                row_index
            };
            
            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Row index out of range"));
            }
            
            if let Some(row_data) = self.inner.iloc(actual_index as usize) {
                let dict = pyo3::types::PyDict::new(py);
                for (key, value) in row_data {
                    dict.set_item(key, attr_value_to_python_value(py, &value)?)?;
                }
                Ok(dict.to_object(py))
            } else {
                Err(PyIndexError::new_err("Row index out of range"))
            }

        } else if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            // Slice access: table[:2] -> table
            let (rows, _) = self.inner.shape();
            let indices = slice.indices(rows as i64)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step;
            
            if step != 1 {
                return Err(PyNotImplementedError::new_err("Step slicing not supported"));
            }
            
            let sliced_table = if start < rows && start <= stop {
                let n = (stop.min(rows)).saturating_sub(start);
                self.inner.head(n) // This is a simplified approach - ideally we'd have a slice method
            } else {
                // Return empty table with same structure
                let empty_arrays = self.inner.columns().iter()
                    .map(|_| GraphArray::from_vec(vec![]))
                    .collect();
                GraphTable::from_arrays_standalone(empty_arrays, Some(self.inner.columns().to_vec()))
                    .map_err(graph_error_to_py_err)?
            };
            
            Ok(Py::new(py, PyGraphTable::from_graph_table(sliced_table))?.to_object(py))

        } else if let Ok(column_name) = key.extract::<String>() {
            // Single column access: table['column'] -> array
            if let Some(column) = self.inner.get_column_by_name(&column_name) {
                let py_array = PyGraphArray::from_graph_array(column.clone());
                Ok(Py::new(py, py_array)?.to_object(py))
            } else {
                Err(PyKeyError::new_err(format!("Column '{}' not found", column_name)))
            }

        } else if let Ok(column_list) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2']] -> table
            let selected_table = self.inner.select(&column_list.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                .map_err(graph_error_to_py_err)?;
            
            Ok(Py::new(py, PyGraphTable::from_graph_table(selected_table))?.to_object(py))

        } else {
            Err(PyTypeError::new_err(
                "Key must be: int (row), slice (:), string (column), or list of strings (columns)"
            ))
        }
    }

    /// Get first n rows
    pub fn head(&self, py: Python, n: usize) -> PyResult<PyObject> {
        let head_table = self.inner.head(n);
        let py_table = PyGraphTable::from_graph_table(head_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Get last n rows
    pub fn tail(&self, py: Python, n: usize) -> PyResult<PyObject> {
        let tail_table = self.inner.tail(n);
        let py_table = PyGraphTable::from_graph_table(tail_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Sort table by column
    pub fn sort_by(&self, py: Python, column: String, ascending: bool) -> PyResult<PyObject> {
        let sorted_table = self.inner.sort_by(&column, ascending)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(sorted_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Get summary statistics
    pub fn describe(&self, py: Python) -> PyResult<PyObject> {
        let desc_table = self.inner.describe();
        let py_table = PyGraphTable::from_graph_table(desc_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Convert to dictionary format
    pub fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict_data = self.inner.to_dict();
        let py_dict = pyo3::types::PyDict::new(py);
        
        for (column, values) in dict_data {
            let py_values: Vec<PyObject> = values.iter()
                .map(|v| attr_value_to_python_value(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            py_dict.set_item(column, py_values)?;
        }
        
        Ok(py_dict.to_object(py))
    }
}

