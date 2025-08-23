//! Table FFI Bindings
//!
//! Python bindings for GraphTable - DataFrame-like views for graph data.
//! This is a thin wrapper around the core Rust GraphTable implementation.

use groggy::{
    AggregateOp, AttrValue as RustAttrValue, ConnectivityType, GraphArray, GraphTable, GroupBy,
};
use pyo3::exceptions::{
    PyImportError, PyIndexError, PyKeyError, PyNotImplementedError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;

// Import utilities
use crate::ffi::api::graph::PyGraph;
use crate::ffi::core::array::PyGraphArray;
use crate::ffi::core::matrix::PyGraphMatrix;
use crate::ffi::utils::{attr_value_to_python_value, graph_error_to_py_err};

/// Python wrapper around core GraphTable implementation
#[pyclass(name = "GraphTable", unsendable)]
pub struct PyGraphTable {
    /// Core GraphTable implementation
    pub inner: GraphTable,
}

/// Iterator for table rows that yields dictionaries
#[pyclass(unsendable)]
pub struct TableIterator {
    table: GraphTable,
    index: usize,
}

#[pymethods]
impl TableIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        let (num_rows, _) = self.table.shape();
        if self.index < num_rows {
            let row_dict = pyo3::types::PyDict::new(py);

            // Get column names and data
            let column_names = self.table.columns();

            // Access each column by name
            for (col_idx, col_name) in column_names.iter().enumerate() {
                // Get the column data by name
                if let Some(array) = self.table.get_column_by_name(col_name) {
                    if self.index < array.len() {
                        let value = &array[self.index];
                        let py_value = attr_value_to_python_value(py, value)?;
                        row_dict.set_item(col_name, py_value)?;
                    }
                }
            }

            self.index += 1;
            Ok(Some(row_dict.to_object(py)))
        } else {
            Ok(None)
        }
    }
}

impl PyGraphTable {
    /// Create from a core GraphTable
    pub fn from_graph_table(table: GraphTable) -> Self {
        Self { inner: table }
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
        let core_arrays: Vec<GraphArray> = arrays
            .iter()
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

        // If no attributes specified, discover all available attributes
        let attr_names = attrs.unwrap_or_else(|| {
            // Discover all available node attributes
            let mut all_attrs = std::collections::HashSet::new();
            for &node_id in &nodes {
                if let Ok(attrs) = graph_ref.inner.get_node_attrs(node_id as usize) {
                    for attr_name in attrs.keys() {
                        all_attrs.insert(attr_name.clone());
                    }
                }
            }

            // Always include node_id as first column
            let mut column_names = vec!["node_id".to_string()];
            column_names.extend(all_attrs.into_iter());
            column_names
        });
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
                    if let Ok(Some(attr_value)) =
                        graph_ref.inner.get_node_attr(node_id as usize, attr_name)
                    {
                        attr_values.push(attr_value);
                    } else {
                        // Handle missing attributes as Null instead of imputing to 0
                        attr_values.push(RustAttrValue::Null);
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

        // If no attributes specified, discover all available attributes
        let attr_names = attrs.unwrap_or_else(|| {
            // Discover all available edge attributes
            let mut all_attrs = std::collections::HashSet::new();
            for &edge_id in &edges {
                if let Ok(attrs) = graph_ref.inner.get_edge_attrs(edge_id as usize) {
                    for attr_name in attrs.keys() {
                        all_attrs.insert(attr_name.clone());
                    }
                }
            }

            // Always include edge_id, source, target as first columns
            let mut column_names = vec![
                "edge_id".to_string(),
                "source".to_string(),
                "target".to_string(),
            ];
            column_names.extend(all_attrs.into_iter());
            column_names
        });
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
                        let endpoint_id = if attr_name == "source" {
                            source
                        } else {
                            target
                        };
                        attr_values.push(RustAttrValue::Int(endpoint_id as i64));
                    } else {
                        // Handle missing edges with default value
                        attr_values.push(RustAttrValue::Int(0));
                    }
                }
            } else {
                // Regular edge attributes
                for &edge_id in &edges {
                    if let Ok(Some(attr_value)) =
                        graph_ref.inner.get_edge_attr(edge_id as usize, attr_name)
                    {
                        attr_values.push(attr_value);
                    } else {
                        // Handle missing attributes as Null instead of imputing to 0
                        attr_values.push(RustAttrValue::Null);
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
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();
        let columns = self.inner.columns();

        // Create HTML table with data
        let mut html = String::new();
        html.push_str(r#"<div style="max-height: 400px; overflow: auto;">"#);
        html.push_str(
            r#"<table border="1" class="dataframe" style="border-collapse: collapse; margin: 0;">"#,
        );

        // Table header
        html.push_str("<thead><tr style=\"text-align: right;\">");
        html.push_str("<th style=\"padding: 8px; background-color: #f0f0f0;\"></th>"); // Index column
        for column in columns {
            html.push_str(&format!(
                "<th style=\"padding: 8px; background-color: #f0f0f0;\">{}</th>",
                column
            ));
        }
        html.push_str("</tr></thead>");

        // Table body - show first 5 rows for performance
        html.push_str("<tbody>");
        let display_rows = std::cmp::min(rows, 5);

        for row_idx in 0..display_rows {
            html.push_str("<tr>");
            // Index column
            html.push_str(&format!(
                "<th style=\"padding: 8px; background-color: #f9f9f9;\">{}</th>",
                row_idx
            ));

            // Data columns
            if let Some(row_data) = self.inner.iloc(row_idx) {
                for column in columns {
                    let value = row_data
                        .get(column)
                        .cloned()
                        .unwrap_or(groggy::AttrValue::Null);
                    let display_value = match &value {
                        groggy::AttrValue::Int(i) => i.to_string(),
                        groggy::AttrValue::SmallInt(i) => i.to_string(),
                        groggy::AttrValue::Float(f) => {
                            if f.fract() == 0.0 {
                                format!("{:.0}", f)
                            } else {
                                format!("{:.6}", f)
                                    .trim_end_matches('0')
                                    .trim_end_matches('.')
                                    .to_string()
                            }
                        }
                        groggy::AttrValue::Text(s) => s.clone(),
                        groggy::AttrValue::CompactText(compact_str) => {
                            compact_str.as_str().to_string()
                        }
                        groggy::AttrValue::Bool(b) => {
                            if *b {
                                "True".to_string()
                            } else {
                                "False".to_string()
                            }
                        }
                        groggy::AttrValue::Bytes(b) => format!("bytes[{}]", b.len()),
                        groggy::AttrValue::FloatVec(items) => {
                            if items.len() <= 3 {
                                format!(
                                    "[{}]",
                                    items
                                        .iter()
                                        .map(|f| f.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )
                            } else {
                                format!(
                                    "[{}, ... {} items]",
                                    items
                                        .iter()
                                        .take(2)
                                        .map(|f| f.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", "),
                                    items.len()
                                )
                            }
                        }
                        groggy::AttrValue::CompressedText(_compressed) => {
                            // Compressed text - show placeholder for now
                            "compressed_text".to_string()
                        }
                        groggy::AttrValue::CompressedFloatVec(_compressed) => {
                            // Compressed float vector - show placeholder for now
                            "compressed_vec".to_string()
                        }
                        groggy::AttrValue::Null => "NaN".to_string(),
                        _ => format!("{:?}", value),
                    };
                    html.push_str(&format!(
                        "<td style=\"padding: 8px;\">{}</td>",
                        display_value
                    ));
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
                    let value = row_data
                        .get(col_name)
                        .cloned()
                        .unwrap_or(RustAttrValue::Null);
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
        let dtypes: HashMap<String, String> = self
            .inner
            .dtypes()
            .into_iter()
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
            let indices = slice.indices(
                rows.try_into()
                    .map_err(|_| PyValueError::new_err("Table too large for slice"))?,
            )?;
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
                let empty_arrays = self
                    .inner
                    .columns()
                    .iter()
                    .map(|_| GraphArray::from_vec(vec![]))
                    .collect();
                GraphTable::from_arrays_standalone(
                    empty_arrays,
                    Some(self.inner.columns().to_vec()),
                )
                .map_err(graph_error_to_py_err)?
            };

            Ok(Py::new(py, PyGraphTable::from_graph_table(sliced_table))?.to_object(py))
        } else if let Ok(column_name) = key.extract::<String>() {
            // Single column access: table['column'] -> array
            if let Some(column) = self.inner.get_column_by_name(&column_name) {
                let py_array = PyGraphArray::from_graph_array(column.clone());
                Ok(Py::new(py, py_array)?.to_object(py))
            } else {
                Err(PyKeyError::new_err(format!(
                    "Column '{}' not found",
                    column_name
                )))
            }
        } else if let Ok(column_list) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2']] -> table
            let selected_table = self
                .inner
                .select(&column_list.iter().map(|s| s.as_str()).collect::<Vec<_>>())
                .map_err(graph_error_to_py_err)?;

            Ok(Py::new(py, PyGraphTable::from_graph_table(selected_table))?.to_object(py))
        } else if let Ok(mask_array_ref) = key.extract::<PyRef<PyGraphArray>>() {
            // Boolean indexing: table[boolean_mask] -> table
            let mask_values = mask_array_ref.inner.to_list();
            let mut row_indices = Vec::new();

            // Check that mask is boolean and collect true indices
            for (i, value) in mask_values.iter().enumerate() {
                match value {
                    groggy::AttrValue::Bool(true) => row_indices.push(i),
                    groggy::AttrValue::Bool(false) => {} // Skip false values
                    _ => {
                        return Err(PyTypeError::new_err(
                            "Boolean mask must contain only boolean values",
                        ))
                    }
                }
            }

            // Check mask length matches table rows
            let (table_rows, _) = self.inner.shape();
            if mask_values.len() != table_rows {
                return Err(PyValueError::new_err(format!(
                    "Boolean mask length ({}) doesn't match table rows ({})",
                    mask_values.len(),
                    table_rows
                )));
            }

            // Create filtered table by selecting rows
            if row_indices.is_empty() {
                // Return empty table with same structure
                let empty_arrays = self
                    .inner
                    .columns()
                    .iter()
                    .map(|_| GraphArray::from_vec(vec![]))
                    .collect();
                let filtered_table = GraphTable::from_arrays_standalone(
                    empty_arrays,
                    Some(self.inner.columns().to_vec()),
                )
                .map_err(graph_error_to_py_err)?;
                Ok(Py::new(py, PyGraphTable::from_graph_table(filtered_table))?.to_object(py))
            } else {
                // Filter by row indices
                let mut filtered_columns = Vec::new();
                for col_name in self.inner.columns() {
                    if let Some(column) = self.inner.get_column_by_name(col_name) {
                        let mut filtered_values = Vec::new();
                        let col_values = column.to_list();
                        for &row_idx in &row_indices {
                            if row_idx < col_values.len() {
                                filtered_values.push(col_values[row_idx].clone());
                            }
                        }
                        filtered_columns.push(GraphArray::from_vec(filtered_values));
                    }
                }

                let filtered_table = GraphTable::from_arrays_standalone(
                    filtered_columns,
                    Some(self.inner.columns().to_vec()),
                )
                .map_err(graph_error_to_py_err)?;
                Ok(Py::new(py, PyGraphTable::from_graph_table(filtered_table))?.to_object(py))
            }
        } else {
            Err(PyTypeError::new_err(
                "Key must be: int (row), slice (:), string (column), list of strings (columns), or boolean mask (GraphArray)"
            ))
        }
    }

    /// Get first n rows
    #[pyo3(signature = (n = 5))]
    pub fn head(&self, py: Python, n: usize) -> PyResult<PyObject> {
        let head_table = self.inner.head(n);
        let py_table = PyGraphTable::from_graph_table(head_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Get last n rows
    #[pyo3(signature = (n = 5))]
    pub fn tail(&self, py: Python, n: usize) -> PyResult<PyObject> {
        let tail_table = self.inner.tail(n);
        let py_table = PyGraphTable::from_graph_table(tail_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Sort table by column
    #[pyo3(signature = (column, ascending = true))]
    pub fn sort_by(&self, py: Python, column: String, ascending: bool) -> PyResult<PyObject> {
        let sorted_table = self
            .inner
            .sort_by(&column, ascending)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(sorted_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Calculate mean for a specific column
    pub fn mean(&self, column_name: String) -> PyResult<f64> {
        self.inner.mean(&column_name).map_err(graph_error_to_py_err)
    }

    /// Calculate sum for a specific column
    pub fn sum(&self, column_name: String) -> PyResult<f64> {
        self.inner.sum(&column_name).map_err(graph_error_to_py_err)
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
            let py_values: Vec<PyObject> = values
                .iter()
                .map(|v| attr_value_to_python_value(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            py_dict.set_item(column, py_values)?;
        }

        Ok(py_dict.to_object(py))
    }

    /// Group by a column for aggregation operations
    pub fn group_by(&self, py: Python, column: String) -> PyResult<PyObject> {
        let group_by = self
            .inner
            .group_by(&column)
            .map_err(graph_error_to_py_err)?;
        let py_group_by = PyGroupBy::from_group_by(group_by);
        Ok(Py::new(py, py_group_by)?.to_object(py))
    }

    /// Convert this table to a GraphMatrix if all columns are compatible numeric types
    ///
    /// Returns a PyGraphMatrix that can be used for matrix operations.
    ///
    /// Examples:
    ///   # This works - both columns are numeric
    ///   matrix = table[['age', 'height']].matrix()
    ///   
    ///   # This fails - mixed numeric and text types
    ///   matrix = table[['age', 'name']].matrix()  # Raises ValueError
    ///
    /// Raises:
    ///   ValueError: If columns have incompatible types (e.g., mixing numeric and text)
    ///   ValueError: If the table is empty
    ///
    /// Returns:
    ///   PyGraphMatrix: A matrix representation of the numeric table data
    pub fn matrix(&self, py: Python) -> PyResult<PyObject> {
        let matrix = self.inner.matrix().map_err(graph_error_to_py_err)?;

        let py_matrix = PyGraphMatrix::from_graph_matrix(matrix);
        Ok(Py::new(py, py_matrix)?.to_object(py))
    }

    /// Filter table by node degree (number of connections)
    pub fn filter_by_degree(
        &self,
        py: Python,
        graph: Py<PyGraph>,
        node_id_column: String,
        min_degree: Option<usize>,
        max_degree: Option<usize>,
    ) -> PyResult<PyObject> {
        let graph_ref = graph.borrow(py);

        let filtered_table = self
            .inner
            .filter_by_degree(&graph_ref.inner, &node_id_column, min_degree, max_degree)
            .map_err(graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Filter table by connectivity to target nodes
    pub fn filter_by_connectivity(
        &self,
        py: Python,
        graph: Py<PyGraph>,
        node_id_column: String,
        target_nodes: Vec<u64>,
        connection_type: String,
    ) -> PyResult<PyObject> {
        let graph_ref = graph.borrow(py);

        // Convert target nodes to usize
        let target_usize: Vec<usize> = target_nodes.iter().map(|&id| id as usize).collect();

        // Parse connection type
        let connectivity = match connection_type.as_str() {
            "any" => ConnectivityType::ConnectedToAny,
            "all" => ConnectivityType::ConnectedToAll,
            "none" => ConnectivityType::NotConnectedToAny,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid connection type: {}. Use 'any', 'all', or 'none'",
                    connection_type
                )))
            }
        };

        let filtered_table = self
            .inner
            .filter_by_connectivity(
                &graph_ref.inner,
                &node_id_column,
                &target_usize,
                connectivity,
            )
            .map_err(graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Filter table by distance from target nodes
    pub fn filter_by_distance(
        &self,
        py: Python,
        graph: Py<PyGraph>,
        node_id_column: String,
        target_nodes: Vec<u64>,
        max_distance: usize,
    ) -> PyResult<PyObject> {
        let graph_ref = graph.borrow(py);

        // Convert target nodes to usize
        let target_usize: Vec<usize> = target_nodes.iter().map(|&id| id as usize).collect();

        let filtered_table = self
            .inner
            .filter_by_distance(
                &graph_ref.inner,
                &node_id_column,
                &target_usize,
                max_distance,
            )
            .map_err(graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Inner join with another table
    pub fn inner_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject> {
        let result_table = self
            .inner
            .inner_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Left join with another table
    pub fn left_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject> {
        let result_table = self
            .inner
            .left_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Right join with another table
    pub fn right_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject> {
        let result_table = self
            .inner
            .right_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Outer join with another table
    pub fn outer_join(
        &self,
        py: Python,
        other: &PyGraphTable,
        left_on: String,
        right_on: String,
    ) -> PyResult<PyObject> {
        let result_table = self
            .inner
            .outer_join(&other.inner, &left_on, &right_on)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Union with another table (combine rows)
    pub fn union(&self, py: Python, other: &PyGraphTable) -> PyResult<PyObject> {
        let result_table = self
            .inner
            .union(&other.inner)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Intersect with another table (rows present in both)
    pub fn intersect(&self, py: Python, other: &PyGraphTable) -> PyResult<PyObject> {
        let result_table = self
            .inner
            .intersect(&other.inner)
            .map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Iterator support - iterates over rows as dictionaries
    fn __iter__(&self) -> TableIterator {
        TableIterator {
            table: self.inner.clone(),
            index: 0,
        }
    }

    /// Explicit iterator method - same as __iter__ but more discoverable
    fn iter(&self) -> TableIterator {
        TableIterator {
            table: self.inner.clone(),
            index: 0,
        }
    }

    // ========================================================================
    // LAZY EVALUATION & MATERIALIZATION
    // ========================================================================

    /// Get table data (materializes data to Python objects)
    /// This is the primary materialization method - use sparingly for large tables
    #[getter]
    fn data(&self, py: Python) -> PyResult<PyObject> {
        let materialized = self.inner.materialize();
        let py_rows: PyResult<Vec<Vec<PyObject>>> = materialized
            .iter()
            .map(|row| {
                row.iter()
                    .map(|val| attr_value_to_python_value(py, val))
                    .collect()
            })
            .collect();

        Ok(py_rows?.to_object(py))
    }

    /// Get preview of table for display (first N rows by default)
    fn preview(
        &self,
        py: Python,
        row_limit: Option<usize>,
        col_limit: Option<usize>,
    ) -> PyResult<PyObject> {
        let row_limit = row_limit.unwrap_or(10);
        let (preview_data, _col_names) = self.inner.preview(row_limit, col_limit);

        Ok(preview_data.to_object(py))
    }

    /// Check if table is sparse (has many default values)
    #[getter]
    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    /// Get summary information without materializing data
    fn summary(&self) -> String {
        self.inner.summary_info()
    }

    /// Convert to NumPy array (when numpy available)
    /// Uses .data property to materialize data
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .data property
        let data = self.data(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (data,))?;
        Ok(array.to_object(py))
    }

    /// Fill null/missing values with a specified value
    #[pyo3(signature = (fill_value, inplace = false))]
    fn fill_na(&self, py: Python, fill_value: &PyAny, inplace: bool) -> PyResult<PyObject> {
        // Convert Python fill_value to AttrValue
        let attr_fill_value = crate::ffi::utils::python_value_to_attr_value(fill_value)?;

        if inplace {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "fill_na(inplace=True) is not yet implemented. Use fill_na(inplace=False) which returns a new table."
            ));
        }

        // Create new table with filled values
        let filled_table = self
            .inner
            .fill_na(attr_fill_value)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filled_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Drop rows containing any null/missing values
    fn drop_na(&self, py: Python) -> PyResult<PyObject> {
        let filtered_table = self
            .inner
            .drop_na()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;

        let py_table = PyGraphTable::from_graph_table(filtered_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Check if table contains any null values
    fn has_null(&self) -> bool {
        let (rows, _) = self.inner.shape();
        let columns = self.inner.columns();
        for row_idx in 0..rows {
            if let Some(row_data) = self.inner.iloc(row_idx) {
                for value in row_data.values() {
                    if matches!(value, groggy::AttrValue::Null) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Count null values in each column
    fn null_count(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let (rows, _) = self.inner.shape();
        let columns = self.inner.columns();

        for column_name in columns {
            let mut null_count = 0;

            for row_idx in 0..rows {
                if let Some(row_data) = self.inner.iloc(row_idx) {
                    if let Some(value) = row_data.get(column_name) {
                        if matches!(value, groggy::AttrValue::Null) {
                            null_count += 1;
                        }
                    } else {
                        // Missing value counts as null
                        null_count += 1;
                    }
                }
            }

            dict.set_item(column_name, null_count)?;
        }

        Ok(dict.to_object(py))
    }

    /// Convert to Pandas DataFrame (when pandas available)
    /// Uses .data property to materialize data
    fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Get materialized data and column names
        let data = self.data(py)?;
        let columns = self.inner.columns();

        // Create DataFrame
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("columns", columns)?;
        let df = pandas.call_method("DataFrame", (data,), Some(kwargs))?;
        Ok(df.to_object(py))
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
        Self { inner: group_by }
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
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown aggregation operation: {}",
                        op_str
                    )))
                }
            };
            ops.insert(column, op);
        }

        // Apply aggregation
        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Convenience method for summing grouped values
    pub fn sum(&self, py: Python, column: String) -> PyResult<PyObject> {
        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Sum);

        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Convenience method for averaging grouped values
    pub fn mean(&self, py: Python, column: String) -> PyResult<PyObject> {
        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Mean);

        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }

    /// Convenience method for counting grouped values
    pub fn count(&self, py: Python, column: String) -> PyResult<PyObject> {
        let mut ops = HashMap::new();
        ops.insert(column, AggregateOp::Count);

        let result_table = self.inner.agg(ops).map_err(graph_error_to_py_err)?;
        let py_table = PyGraphTable::from_graph_table(result_table);
        Ok(Py::new(py, py_table)?.to_object(py))
    }
}
