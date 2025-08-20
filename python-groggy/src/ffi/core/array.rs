//! Array FFI Bindings
//! 
//! Python bindings for statistical arrays and matrices.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError, PyImportError, PyNotImplementedError};
use groggy::core::array::{GraphArray, StatsSummary};
use groggy::AttrValue as RustAttrValue;

// Use utility functions from utils module
use crate::ffi::utils::{python_value_to_attr_value, attr_value_to_python_value, graph_error_to_py_err};

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
    
    /// Get element by index - supports arr[i] with negative indexing
    fn __getitem__(&self, py: Python, index: isize) -> PyResult<PyObject> {
        let len = self.inner.len() as isize;
        
        // Handle negative indexing (Python-style)
        let actual_index = if index < 0 {
            len + index
        } else {
            index
        };
        
        // Check bounds
        if actual_index < 0 || actual_index >= len {
            return Err(PyIndexError::new_err("Index out of range"));
        }
        
        match self.inner.get(actual_index as usize) {
            Some(attr_value) => attr_value_to_python_value(py, attr_value),
            None => Err(PyIndexError::new_err("Index out of range")),
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
    
    /// Iterator support (for value in array)
    fn __iter__(slf: PyRef<Self>) -> GraphArrayIterator {
        GraphArrayIterator {
            array: slf.inner.clone(),
            index: 0,
        }
    }
    
    /// Convert to plain Python list
    fn to_list(&self, py: Python) -> PyResult<Vec<PyObject>> {
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
    
    /// Get count of elements
    fn count(&self) -> usize {
        self.inner.count()
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
                _ => format!("other:{:?}", attr_value), // Fallback for other types
            };
            
            if unique_set.insert(key) {
                // This is a new unique value
                unique_values.push(attr_value.clone());
            }
        }
        
        // Create new GraphArray with unique values
        let unique_array = GraphArray::from_vec(unique_values);
        let py_unique = PyGraphArray { inner: unique_array };
        
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
    
    /// Get raw data as Python list (like pandas .values property) 
    #[getter]
    fn values(&self, py: Python) -> PyResult<Vec<PyObject>> {
        self.to_list(py)
    }
    
    /// Get comprehensive statistical summary
    fn describe(&self, _py: Python) -> PyResult<PyStatsSummary> {
        Ok(PyStatsSummary {
            inner: self.inner.describe(),
        })
    }
    
    // ========================================================================
    // SCIENTIFIC COMPUTING CONVERSIONS
    // ========================================================================
    
    /// Convert to NumPy array (when numpy available)
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>("numpy is required for to_numpy(). Install with: pip install numpy")
        })?;
        
        // Get data as Python list
        let values = self.values(py)?;
        
        // Convert to numpy array
        let array = numpy.call_method1("array", (values,))?;
        Ok(array.to_object(py))
    }
    
    /// Convert to Pandas Series (when pandas available)
    fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>("pandas is required for to_pandas(). Install with: pip install pandas")
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
            PyErr::new::<PyImportError, _>("scipy is required for to_scipy_sparse(). Install with: pip install scipy")
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
        dict.set_item("name", self._get_name().unwrap_or_else(|| "array".to_string()))?;
        
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
        type_counts.into_iter()
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

/// Python wrapper for GraphMatrix - wraps core GraphMatrix
/// This is the return type for adjacency matrices and other matrix operations
#[pyclass(name = "GraphMatrix")]
pub struct PyGraphMatrix {
    /// Core GraphMatrix
    pub inner: groggy::core::adjacency::GraphMatrix,
}

#[pymethods]
impl PyGraphMatrix {
    /// Get matrix dimensions as (rows, columns) tuple
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.size, self.inner.size)
    }
    
    /// Get number of rows (same as columns for square matrices)
    fn row_count(&self) -> usize {
        self.inner.size
    }
    
    /// Get number of columns (same as rows for square matrices)
    fn column_count(&self) -> usize {
        self.inner.size
    }
    
    /// Multi-index access for matrix elements: matrix[row, col] -> value
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Multi-index access (row, col) -> single cell value
        if let Ok(indices) = key.extract::<(usize, usize)>() {
            let (row, col) = indices;
            return self.get_cell(py, row, col);
        }
        
        // Single integer -> row access
        if let Ok(row_index) = key.extract::<usize>() {
            return self.get_row(py, row_index);
        }
        
        Err(PyTypeError::new_err(
            "Key must be: int (row index) or (row, col) tuple for multi-index access"
        ))
    }
    
    /// Get single cell value at (row, col) (FFI wrapper around core GraphMatrix)
    fn get_cell(&self, py: Python, row: usize, col: usize) -> PyResult<PyObject> {
        match self.inner.get(row, col) {
            Some(attr_value) => attr_value_to_python_value(py, attr_value),
            None => Err(PyIndexError::new_err(format!(
                "Index ({}, {}) out of range for {}x{} matrix", 
                row, col, self.inner.size, self.inner.size
            )))
        }
    }
    
    /// Get entire row as list (FFI wrapper around core GraphMatrix)
    fn get_row(&self, py: Python, row: usize) -> PyResult<PyObject> {
        match self.inner.get_row(row) {
            Some(row_values) => {
                let py_list = pyo3::types::PyList::empty(py);
                for value in row_values {
                    let py_value = attr_value_to_python_value(py, &value)?;
                    py_list.append(py_value)?;
                }
                Ok(py_list.to_object(py))
            },
            None => Err(PyIndexError::new_err(format!(
                "Row index {} out of range for {}x{} matrix", 
                row, self.inner.size, self.inner.size
            )))
        }
    }
    
    /// Get entire column as list (FFI wrapper around core GraphMatrix)
    fn get_column(&self, py: Python, col: usize) -> PyResult<PyObject> {
        match self.inner.get_column(col) {
            Some(col_values) => {
                let py_list = pyo3::types::PyList::empty(py);
                for value in col_values {
                    let py_value = attr_value_to_python_value(py, &value)?;
                    py_list.append(py_value)?;
                }
                Ok(py_list.to_object(py))
            },
            None => Err(PyIndexError::new_err(format!(
                "Column index {} out of range for {}x{} matrix", 
                col, self.inner.size, self.inner.size
            )))
        }
    }
    
    fn __repr__(&self) -> String {
        let (rows, cols) = (self.inner.size, self.inner.size);
        
        // For small matrices, show some data
        if rows <= 8 && cols <= 8 {
            let mut result = format!("GraphMatrix({}x{}):\n", rows, cols);
            for i in 0..rows {
                result.push_str("  [");
                for j in 0..cols {
                    if let Some(value) = self.inner.get(i, j) {
                        match value {
                            groggy::AttrValue::Int(v) => result.push_str(&format!("{:3}", v)),
                            groggy::AttrValue::Float(v) => result.push_str(&format!("{:6.2}", v)),
                            groggy::AttrValue::Bool(true) => result.push_str("  1"),
                            groggy::AttrValue::Bool(false) => result.push_str("  0"),
                            _ => result.push_str("  ?"),
                        }
                    } else {
                        result.push_str("  0");
                    }
                    if j < cols - 1 {
                        result.push_str(" ");
                    }
                }
                result.push_str("]\n");
            }
            result.trim_end().to_string()
        } else {
            // For large matrices, just show dimensions and type
            format!("GraphMatrix({}x{} adjacency matrix)", rows, cols)
        }
    }
    
    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self) -> String {
        let (rows, cols) = (self.inner.size, self.inner.size);
        
        if rows <= 50 && cols <= 50 {
            // For reasonably sized matrices, show interactive HTML table
            self._generate_html_table()
        } else {
            // For very large matrices, show summary with sample
            self._generate_html_summary()
        }
    }
    
    /// Generate interactive HTML table for small-medium matrices
    fn _generate_html_table(&self) -> String {
        let (rows, cols) = (self.inner.size, self.inner.size);
        let mut html = String::new();
        
        // Modern rounded CSS styling like the example
        html.push_str(r#"
<style>
.df-rounded {
  --bg: #ffffff; --fg: #222; --muted: #6b7280;
  --border: #e5e7eb; --border-strong: #d1d5db;
  --row-odd: #fafafa;
  font-family: 'SFMono-Regular','Menlo','Consolas','Liberation Mono',monospace;
  font-size: 12.5px; line-height: 1.5;
  color: var(--fg); background: var(--bg);
}
.df-rounded table {
  border-collapse: separate; border-spacing: 0; margin: .25rem 0 .125rem 0;
  border: 2px solid var(--border); border-radius: 10px; overflow: hidden;
}
.df-rounded th, .df-rounded td {
  padding: 2px 8px; border-right: 1px solid var(--border);
  white-space: nowrap; max-width: 420px; overflow: hidden; text-overflow: ellipsis;
}
.df-rounded tr:nth-child(odd) td.data { background: var(--row-odd); }
.df-rounded thead th {
  font-weight: 600; background: rgba(127,127,127,0.06);
  border-bottom: 1px solid var(--border-strong);
}
.df-rounded thead tr.dtype th {
  font-weight: 400; color: var(--muted);
}
.df-rounded td.index, .df-rounded th.index {
  color: var(--muted); text-align: right; background: rgba(127,127,127,0.04);
}
.df-rounded td:last-child, .df-rounded th:last-child { border-right: none; }
.df-rounded tfoot {
  color: var(--muted); font-size: 12px; padding-top: 2px;
}
.df-rounded td.ellipsis, .df-rounded th.ellipsis { text-align:center; color: var(--muted); }
.df-rounded .footer {
  margin-top: 2px; color: var(--muted); font-family: 'SFMono-Regular','Menlo','Consolas','Liberation Mono',monospace; font-size: 12px;
}
</style>
"#);
        
        html.push_str(r#"<div class="df-rounded">"#);
        html.push_str(r#"<table>"#);
        
        // Header row with column indices (if matrix is reasonably sized)
        if cols <= 20 && rows <= 20 {
            html.push_str("<thead><tr><th class=\"index\">#</th>");
            for j in 0..cols.min(20) {
                html.push_str(&format!("<th class=\"\">{}</th>", j));
            }
            if cols > 20 {
                html.push_str("<th class=\"ellipsis\">⋯</th>");
            }
            html.push_str("</tr>");
            
            // Type row (simplified for adjacency matrices)
            html.push_str("<tr class=\"dtype\"><th class=\"index\"></th>");
            for _ in 0..cols.min(20) {
                html.push_str("<th class=\"dtype\">adj</th>");
            }
            if cols > 20 {
                html.push_str("<th class=\"ellipsis\"></th>");
            }
            html.push_str("</tr></thead>");
        }
        
        // Data rows
        html.push_str("<tbody>");
        let display_rows = rows.min(20);
        for i in 0..display_rows {
            html.push_str("<tr>");
            
            // Row index (if we have headers)
            if cols <= 20 && rows <= 20 {
                html.push_str(&format!("<td class=\"index\">{}</td>", i));
            }
            
            let display_cols = cols.min(20);
            for j in 0..display_cols {
                if let Some(value) = self.inner.get(i, j) {
                    let val_str = match value {
                        groggy::AttrValue::Int(v) => {
                            if *v == 0 { 
                                "0".to_string()
                            } else { 
                                v.to_string()
                            }
                        },
                        groggy::AttrValue::Float(v) => {
                            if *v == 0.0 { 
                                "0.00".to_string()
                            } else { 
                                format!("{:.2}", v)
                            }
                        },
                        groggy::AttrValue::Bool(true) => "1".to_string(),
                        groggy::AttrValue::Bool(false) => "0".to_string(),
                        _ => "?".to_string(),
                    };
                    html.push_str(&format!(r#"<td class="data" style="text-align:right">{}</td>"#, val_str));
                } else {
                    html.push_str(r#"<td class="data" style="text-align:right">0</td>"#);
                }
            }
            
            if cols > 20 {
                html.push_str("<td class=\"ellipsis\">⋯</td>");
            }
            html.push_str("</tr>");
        }
        
        if rows > 20 {
            html.push_str("<tr>");
            if cols <= 20 && rows <= 20 {
                html.push_str("<td class=\"ellipsis\">⋮</td>");
            }
            for _ in 0..cols.min(20) {
                html.push_str("<td class=\"ellipsis\">⋮</td>");
            }
            if cols > 20 {
                html.push_str("<td class=\"ellipsis\">⋱</td>");
            }
            html.push_str("</tr>");
        }
        
        html.push_str("</tbody></table>");
        
        // Footer with shape and type info
        html.push_str(&format!(r#"<div class="footer">shape: ({}, {}) • dtype: adjacency • type: GraphMatrix</div>"#, rows, cols));
        html.push_str("</div>");
        
        html
    }
    
    /// Generate HTML summary for very large matrices
    fn _generate_html_summary(&self) -> String {
        let (rows, cols) = (self.inner.size, self.inner.size);
        
        // Count non-zero elements in a sample
        let mut nonzero_count = 0;
        let sample_size = 100.min(rows);
        
        for i in 0..sample_size {
            for j in 0..cols.min(100) {
                if let Some(value) = self.inner.get(i, j) {
                    match value {
                        groggy::AttrValue::Int(v) if *v != 0 => nonzero_count += 1,
                        groggy::AttrValue::Float(v) if *v != 0.0 => nonzero_count += 1,
                        groggy::AttrValue::Bool(true) => nonzero_count += 1,
                        _ => {}
                    }
                }
            }
        }
        
        let sample_total = sample_size * cols.min(100);
        let density = if sample_total > 0 {
            (nonzero_count as f64 / sample_total as f64) * 100.0
        } else {
            0.0
        };
        
        format!(r#"
<div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; background-color: #f9f9f9;">
    <strong>GraphMatrix</strong><br>
    <strong>Shape:</strong> {} × {}<br>
    <strong>Type:</strong> Adjacency Matrix<br>
    <strong>Estimated Density:</strong> {:.2}% (sample)<br>
    <strong>Total Elements:</strong> {}<br>
    <em>Matrix too large for full display. Use .to_dense() for smaller subsets.</em>
</div>
"#, rows, cols, density, rows * cols)
    }
}

/// Python wrapper for SparseGraphMatrix - wraps core sparse matrix
#[pyclass(name = "GraphSparseMatrix")]
pub struct PyGraphSparseMatrix {
    /// Core SparseGraphMatrix
    pub inner: groggy::core::adjacency::SparseGraphMatrix,
}

#[pymethods]
impl PyGraphSparseMatrix {
    /// Get matrix dimensions as (rows, columns) tuple
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape
    }
    
    /// Get number of non-zero entries
    fn nnz(&self) -> usize {
        self.inner.values.len()
    }
    
    /// Get row indices (for COO format access)
    #[getter]
    fn rows(&self) -> Vec<usize> {
        self.inner.rows.clone()
    }
    
    /// Get column indices (for COO format access)  
    #[getter]
    fn cols(&self) -> Vec<usize> {
        self.inner.cols.clone()
    }
    
    /// Convert to dense matrix (when needed)
    pub fn to_dense(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let (rows, cols) = self.inner.shape;
        let size = rows; // Square matrix
        
        // Build dense data from sparse representation
        let mut dense_data = Vec::with_capacity(size * size);
        let default_value = groggy::AttrValue::Int(0);
        
        // Initialize with zeros
        for _ in 0..(size * size) {
            dense_data.push(default_value.clone());
        }
        
        // Fill in non-zero values from sparse matrix
        for i in 0..self.inner.rows.len() {
            let row = self.inner.rows[i];
            let col = self.inner.cols[i];
            if let Some(value) = self.inner.values.get(i) {
                let index = row * size + col;
                if index < dense_data.len() {
                    dense_data[index] = value.clone();
                }
            }
        }
        
        // Create GraphMatrix from dense data
        let graph_array = groggy::GraphArray::from_vec(dense_data);
        let dense_matrix = groggy::core::adjacency::GraphMatrix {
            data: graph_array,
            size,
            row_major: true,
            labels: self.inner.labels.clone(),
        };
        
        let py_graph_matrix = PyGraphMatrix { inner: dense_matrix };
        Ok(Py::new(py, py_graph_matrix)?)
    }
    
    fn __repr__(&self) -> String {
        let (rows, cols) = self.inner.shape;
        let nnz = self.inner.values.len();
        let density = if rows * cols > 0 {
            (nnz as f64) / ((rows * cols) as f64) * 100.0
        } else {
            0.0
        };
        
        format!("GraphSparseMatrix({}x{}, {} non-zeros, {:.1}% dense)", 
                rows, cols, nnz, density)
    }
    
    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self) -> String {
        let (rows, cols) = self.inner.shape;
        let nnz = self.inner.values.len();
        let density = if rows * cols > 0 {
            (nnz as f64) / ((rows * cols) as f64) * 100.0
        } else {
            0.0
        };
        
        let mut html = format!(r#"
<div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; background-color: #f8f9fa; border-radius: 5px;">
    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">GraphSparseMatrix</h4>
    <table style="border-collapse: collapse; width: 100%;">
        <tr>
            <td style="padding: 5px; font-weight: bold; color: #34495e;">Shape:</td>
            <td style="padding: 5px;">{} × {}</td>
        </tr>
        <tr>
            <td style="padding: 5px; font-weight: bold; color: #34495e;">Non-zeros:</td>
            <td style="padding: 5px;">{}</td>
        </tr>
        <tr>
            <td style="padding: 5px; font-weight: bold; color: #34495e;">Density:</td>
            <td style="padding: 5px;">{:.2}%</td>
        </tr>
        <tr>
            <td style="padding: 5px; font-weight: bold; color: #34495e;">Total Elements:</td>
            <td style="padding: 5px;">{}</td>
        </tr>
    </table>
"#, rows, cols, nnz, density, rows * cols);
        
        // Show some sample non-zero values if available
        if nnz > 0 {
            html.push_str(r#"<h5 style="margin: 15px 0 5px 0; color: #2c3e50;">Sample Non-zero Values:</h5>"#);
            html.push_str(r#"<table style="border-collapse: collapse; font-family: monospace; font-size: 12px;">"#);
            html.push_str(r#"<tr style="background-color: #ecf0f1;"><th style="padding: 3px 8px; border: 1px solid #bdc3c7;">Row</th><th style="padding: 3px 8px; border: 1px solid #bdc3c7;">Col</th><th style="padding: 3px 8px; border: 1px solid #bdc3c7;">Value</th></tr>"#);
            
            let sample_count = nnz.min(10);
            for i in 0..sample_count {
                if i < self.inner.rows.len() && i < self.inner.cols.len() && i < self.inner.values.len() {
                    let row = self.inner.rows[i];
                    let col = self.inner.cols[i];
                    let value = &self.inner.values[i];
                    
                    let value_str = match value {
                        groggy::AttrValue::Int(v) => v.to_string(),
                        groggy::AttrValue::Float(v) => format!("{:.3}", v),
                        groggy::AttrValue::Bool(true) => "1".to_string(),
                        groggy::AttrValue::Bool(false) => "0".to_string(),
                        _ => "?".to_string(),
                    };
                    
                    html.push_str(&format!(
                        r#"<tr><td style="padding: 3px 8px; border: 1px solid #bdc3c7; text-align: center;">{}</td><td style="padding: 3px 8px; border: 1px solid #bdc3c7; text-align: center;">{}</td><td style="padding: 3px 8px; border: 1px solid #bdc3c7; text-align: center; font-weight: bold; color: #e74c3c;">{}</td></tr>"#,
                        row, col, value_str
                    ));
                }
            }
            
            html.push_str("</table>");
            
            if nnz > 10 {
                html.push_str(&format!(r#"<p style="margin: 5px 0 0 0; font-style: italic; color: #7f8c8d;">... and {} more non-zero values</p>"#, nnz - 10));
            }
        }
        
        html.push_str(r#"<p style="margin: 10px 0 0 0; font-style: italic; color: #7f8c8d;">Use .to_dense() to convert to dense matrix for full display</p>"#);
        html.push_str("</div>");
        
        html
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
}
