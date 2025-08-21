//! Table FFI Bindings
//! 
//! Python bindings for GraphTable - DataFrame-like views for graph data.
//! This is a thin wrapper around the core Rust GraphTable implementation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError, PyImportError, PyNotImplementedError};
use groggy::{NodeId, EdgeId, AttrValue as RustAttrValue, GraphTable, GraphArray, TableMetadata, GroupBy, AggregateOp};
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
        py: Python,
        graph: Py<PyGraph>,
        nodes: Vec<u64>,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let graph_ref = graph.borrow(py);
        
        // Default to just node_id if no attributes specified
        let attr_names = attrs.unwrap_or_else(|| vec!["node_id".to_string()]);
        let mut columns = Vec::new();
        
        for attr_name in &attr_names {
            let mut attr_values = Vec::new();
            
            if attr_name == "node_id" {
                // Special case: node IDs
                for &node_id in &nodes {
                    attr_values.push(RustAttrValue::Int(node_id as i64));
                }
            } else {
                // Regular node attributes
                for &node_id in &nodes {
                    if let Ok(Some(attr_value)) = graph_ref.inner.get_node_attr(node_id as usize, attr_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Handle missing attributes with default value
                        attr_values.push(RustAttrValue::Int(0));
                    }
                }
            }
            
            // Create GraphArray from attribute values
            let graph_array = GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }
        
        // Create GraphTable from arrays
        let table = GraphTable::from_arrays_standalone(columns, Some(attr_names))
            .map_err(graph_error_to_py_err)?;
        
        Ok(Self::from_graph_table(table))
    }

    /// Create GraphTable from graph edges
    #[classmethod]
    pub fn from_graph_edges(
        _cls: &PyType,
        py: Python,
        graph: Py<PyGraph>,
        edges: Vec<u64>,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let graph_ref = graph.borrow(py);
        
        // Default to edge_id, source, target if no attributes specified
        let attr_names = attrs.unwrap_or_else(|| vec![
            "edge_id".to_string(),
            "source".to_string(), 
            "target".to_string()
        ]);
        let mut columns = Vec::new();
        
        for attr_name in &attr_names {
            let mut attr_values = Vec::new();
            
            if attr_name == "edge_id" {
                // Special case: edge IDs
                for &edge_id in &edges {
                    attr_values.push(RustAttrValue::Int(edge_id as i64));
                }
            } else if attr_name == "source" || attr_name == "target" {
                // Special case: edge endpoints
                for &edge_id in &edges {
                    if let Ok((source, target)) = graph_ref.inner.edge_endpoints(edge_id as usize) {
                        let endpoint_id = if attr_name == "source" { source } else { target };
                        attr_values.push(RustAttrValue::Int(endpoint_id as i64));
                    } else {
                        // Handle missing edges with default value
                        attr_values.push(RustAttrValue::Int(0));
                    }
                }
            } else {
                // Regular edge attributes
                for &edge_id in &edges {
                    if let Ok(Some(attr_value)) = graph_ref.inner.get_edge_attr(edge_id as usize, attr_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Handle missing attributes with default value
                        attr_values.push(RustAttrValue::Int(0));
                    }
                }
            }
            
            // Create GraphArray from attribute values
            let graph_array = GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }
        
        // Create GraphTable from arrays
        let table = GraphTable::from_arrays_standalone(columns, Some(attr_names))
            .map_err(graph_error_to_py_err)?;
        
        Ok(Self::from_graph_table(table))
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
        let (rows, cols) = self.inner.shape();
        let columns = self.inner.columns();
        
        // Create HTML table with data
        let mut html = String::new();
        html.push_str(r#"<div style="max-height: 400px; overflow: auto;">"#);
        html.push_str(r#"<table border="1" class="dataframe" style="border-collapse: collapse; margin: 0;">"#);
        
        // Table header
        html.push_str("<thead><tr style=\"text-align: right;\">");
        html.push_str("<th style=\"padding: 8px; background-color: #f0f0f0;\"></th>"); // Index column
        for column in columns {
            html.push_str(&format!("<th style=\"padding: 8px; background-color: #f0f0f0;\">{}</th>", column));
        }
        html.push_str("</tr></thead>");
        
        // Table body - show first 5 rows for performance  
        html.push_str("<tbody>");
        let display_rows = std::cmp::min(rows, 5);
        
        for row_idx in 0..display_rows {
            html.push_str("<tr>");
            // Index column
            html.push_str(&format!("<th style=\"padding: 8px; background-color: #f9f9f9;\">{}</th>", row_idx));
            
            // Data columns
            if let Some(row_data) = self.inner.iloc(row_idx) {
                for column in columns {
                    let value = row_data.get(column).cloned().unwrap_or(groggy::AttrValue::Int(0));
                    let display_value = match &value {
                        groggy::AttrValue::Int(i) => i.to_string(),
                        groggy::AttrValue::SmallInt(i) => i.to_string(),
                        groggy::AttrValue::Float(f) => {
                            if f.fract() == 0.0 {
                                format!("{:.0}", f)
                            } else {
                                format!("{:.6}", f).trim_end_matches('0').trim_end_matches('.').to_string()
                            }
                        },
                        groggy::AttrValue::Text(s) => s.clone(),
                        groggy::AttrValue::CompactText(compact_str) => compact_str.as_str().to_string(),
                        groggy::AttrValue::Bool(b) => if *b { "True".to_string() } else { "False".to_string() },
                        groggy::AttrValue::Bytes(b) => format!("bytes[{}]", b.len()),
                        groggy::AttrValue::FloatVec(items) => {
                            if items.len() <= 3 {
                                format!("[{}]", items.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", "))
                            } else {
                                format!("[{}, ... {} items]", items.iter().take(2).map(|f| f.to_string()).collect::<Vec<_>>().join(", "), items.len())
                            }
                        },
                        groggy::AttrValue::CompressedText(_compressed) => {
                            // Compressed text - show placeholder for now
                            "compressed_text".to_string()
                        },
                        groggy::AttrValue::CompressedFloatVec(_compressed) => {
                            // Compressed float vector - show placeholder for now
                            "compressed_vec".to_string()
                        },
                        _ => format!("{:?}", value)
                    };
                    html.push_str(&format!("<td style=\"padding: 8px;\">{}</td>", display_value));
                }
            }
            html.push_str("</tr>");
        }
        
        // Show truncation message if needed
        if rows > display_rows {
            html.push_str(&format!(
                "<tr><td colspan=\"{}\" style=\"padding: 8px; text-align: center; font-style: italic;\">... {} more rows</td></tr>",
                cols + 1, rows - display_rows
            ));
        }
        
        html.push_str("</tbody></table></div>");
        
        // Add summary info
        html.push_str(&format!(
            r#"<div style="margin-top: 8px; font-size: 12px; color: #666;">
            <strong>GraphTable:</strong> {} rows × {} columns
            </div>"#,
            rows, cols
        ));
        
        Ok(html)
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

    /// Convert to pandas DataFrame
    pub fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Import pandas
        let pandas = py.import("pandas")?;
        
        // Get table data as dictionary
        let dict_data = self.inner.to_dict();
        let py_dict = pyo3::types::PyDict::new(py);
        
        // Convert each column to Python list
        for (column, values) in dict_data {
            let py_values: Vec<PyObject> = values.iter()
                .map(|v| attr_value_to_python_value(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            py_dict.set_item(column, py_values)?;
        }
        
        // Create DataFrame from dictionary
        let dataframe_class = pandas.getattr("DataFrame")?;
        let df = dataframe_class.call1((py_dict,))?;
        
        Ok(df.to_object(py))
    }

    /// Group by a column for aggregation operations
    pub fn group_by(&self, py: Python, column: String) -> PyResult<PyObject> {
        let group_by = self.inner.group_by(&column)
            .map_err(graph_error_to_py_err)?;
        let py_group_by = PyGroupBy::from_group_by(group_by);
        Ok(Py::new(py, py_group_by)?.to_object(py))
    }

    /// Extract neighborhood table for a given node
    #[classmethod]
    pub fn neighborhood_table(
        _cls: &PyType,
        py: Python,
        graph: Py<PyGraph>,
        node_id: u64,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let graph_ref = graph.borrow(py);
        
        // Convert attribute names to &str slice
        let attr_refs: Option<Vec<&str>> = attrs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let attr_slice = attr_refs.as_ref().map(|v| v.as_slice());
        
        let table = GraphTable::neighborhood_table(&graph_ref.inner, node_id as usize, attr_slice)
            .map_err(graph_error_to_py_err)?;
        
        Ok(Self::from_graph_table(table))
    }

    /// Extract neighborhood tables for multiple nodes
    #[classmethod]
    pub fn multi_neighborhood_table(
        _cls: &PyType,
        py: Python,
        graph: Py<PyGraph>,
        node_ids: Vec<u64>,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let graph_ref = graph.borrow(py);
        
        // Convert node IDs to usize
        let node_usize: Vec<usize> = node_ids.iter().map(|&id| id as usize).collect();
        
        // Convert attribute names to &str slice
        let attr_refs: Option<Vec<&str>> = attrs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let attr_slice = attr_refs.as_ref().map(|v| v.as_slice());
        
        let table = GraphTable::multi_neighborhood_table(&graph_ref.inner, &node_usize, attr_slice)
            .map_err(graph_error_to_py_err)?;
        
        Ok(Self::from_graph_table(table))
    }

    /// Extract k-hop neighborhood table for a given node
    #[classmethod]
    pub fn k_hop_neighborhood_table(
        _cls: &PyType,
        py: Python,
        graph: Py<PyGraph>,
        node_id: u64,
        k: usize,
        attrs: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let graph_ref = graph.borrow(py);
        
        // Convert attribute names to &str slice
        let attr_refs: Option<Vec<&str>> = attrs.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
        let attr_slice = attr_refs.as_ref().map(|v| v.as_slice());
        
        let table = GraphTable::k_hop_neighborhood_table(&graph_ref.inner, node_id as usize, k, attr_slice)
            .map_err(graph_error_to_py_err)?;
        
        Ok(Self::from_graph_table(table))
    }

    /// Inner join with another table
    pub fn inner_join(&self, py: Python, other: &PyGraphTable, left_on: String, right_on: String) -> PyResult<PyObject> {
        let result_table = self.inner.inner_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Left join with another table
    pub fn left_join(&self, py: Python, other: &PyGraphTable, left_on: String, right_on: String) -> PyResult<PyObject> {
        let result_table = self.inner.left_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Right join with another table
    pub fn right_join(&self, py: Python, other: &PyGraphTable, left_on: String, right_on: String) -> PyResult<PyObject> {
        let result_table = self.inner.right_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Outer join with another table
    pub fn outer_join(&self, py: Python, other: &PyGraphTable, left_on: String, right_on: String) -> PyResult<PyObject> {
        let result_table = self.inner.outer_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Union with another table (combine rows)
    pub fn union(&self, py: Python, other: &PyGraphTable) -> PyResult<PyObject> {
        let result_table = self.inner.union(&other.inner)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Intersect with another table (rows present in both)
    pub fn intersect(&self, py: Python, other: &PyGraphTable) -> PyResult<PyObject> {
        let result_table = self.inner.intersect(&other.inner)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Iterator support - iterates over rows as dictionaries (temporarily disabled)
    fn __iter__(slf: PyRef<Self>) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Table iteration temporarily disabled during Phase 3 - use table[i] for row access"
        ))
    }
}

/// Python wrapper around GroupBy for aggregation operations
#[pyclass(name = "GroupBy", unsendable)]
pub struct PyGroupBy {
    /// Core GroupBy implementation
    pub inner: GroupBy,
}

impl PyGroupBy {
    /// Create from a core GroupBy
    pub fn from_group_by(group_by: GroupBy) -> Self {
        Self {
            inner: group_by,
        }
    }
}

#[pymethods]
impl PyGroupBy {
    /// Apply aggregation operations to grouped data
    /// 
    /// Args:
    ///     operations: Dictionary mapping column names to aggregation operations
    ///                 Valid operations: "sum", "mean", "count", "min", "max", "std", "var"
    /// 
    /// Returns:
    ///     PyGraphTable: Aggregated results with group keys and aggregated values
    /// 
    /// Example:
    ///     grouped = table.group_by("category")
    ///     result = grouped.agg({"price": "mean", "quantity": "sum"})
    pub fn agg(&self, py: Python, operations: HashMap<String, String>) -> PyResult<PyObject> {
        // Convert string operations to AggregateOp enum
        let mut ops = HashMap::new();
        for (column, op_str) in operations {
            let op = match op_str.as_str() {
                "sum" => AggregateOp::Sum,
                "mean" => AggregateOp::Mean,
                "count" => AggregateOp::Count,
                "min" => AggregateOp::Min,
                "max" => AggregateOp::Max,
                "std" => AggregateOp::Std,
                "var" => AggregateOp::Var,
                "first" => AggregateOp::First,
                "last" => AggregateOp::Last,
                "unique" => AggregateOp::Unique,
                _ => return Err(PyValueError::new_err(format!("Unknown aggregation operation: {}", op_str))),
            };
            ops.insert(column, op);
        }
        
        // Apply aggregation
        let result_table = self.inner.agg(ops)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }
    
    /// Convenience method for summing grouped values
    pub fn sum(&self, py: Python, column: String) -> PyResult<PyObject> {
        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Sum);
        
        let result_table = self.inner.agg(ops)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }
    
    /// Convenience method for averaging grouped values
    pub fn mean(&self, py: Python, column: String) -> PyResult<PyObject> {
        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Mean);
        
        let result_table = self.inner.agg(ops)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }
    
    /// Convenience method for counting grouped values
    pub fn count(&self, py: Python, column: String) -> PyResult<PyObject> {
        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Count);
        
        let result_table = self.inner.agg(ops)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }
}


