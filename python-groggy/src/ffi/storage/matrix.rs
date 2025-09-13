//! Matrix FFI Bindings
//!
//! Python bindings for GraphMatrix - general-purpose matrix operations.

use groggy::storage::array::BaseArray;
use groggy::storage::GraphMatrix;
use groggy::storage::array::NumArray;
use groggy::types::AttrValue as RustAttrValue;
use pyo3::exceptions::{
    PyImportError, PyIndexError, PyKeyError, PyNotImplementedError, PyRuntimeError, PyTypeError,
    PyValueError,
};
use pyo3::prelude::*;
use std::collections::HashMap;
use pyo3::types::PyType;

// Use utility functions from utils module
use crate::ffi::storage::array::PyBaseArray;
use crate::ffi::storage::num_array::PyNumArray;
use crate::ffi::utils::attr_value_to_python_value;

/// Python wrapper for GraphMatrix - general-purpose matrix for collections of GraphArrays
#[derive(Clone)]
#[pyclass(name = "GraphMatrix", unsendable)]
pub struct PyGraphMatrix {
    /// Core GraphMatrix
    pub inner: GraphMatrix,
}

#[pymethods]
impl PyGraphMatrix {
    /// Create a new GraphMatrix from BaseArrays (mixed types) or NumArrays (numerical)
    #[new]
    pub fn new(py: Python, arrays: Vec<PyObject>) -> PyResult<Self> {
        if arrays.is_empty() {
            return Err(PyValueError::new_err("Cannot create matrix from empty array list"));
        }

        let mut core_arrays: Vec<NumArray<f64>> = Vec::new();
        
        for (i, array_obj) in arrays.iter().enumerate() {
            // Try NumArray first (preferred for matrices)
            if let Ok(num_array) = array_obj.extract::<PyRef<PyNumArray>>(py) {
                // Use NumArray directly - no conversion needed
                core_arrays.push(num_array.inner.clone());
            }
            // Convert BaseArray to NumArray<f64>
            else if let Ok(base_array) = array_obj.extract::<PyRef<PyBaseArray>>(py) {
                let f64_values: Vec<f64> = base_array.inner.iter()
                    .map(|attr_value| match attr_value {
                        RustAttrValue::Float(f) => *f as f64,
                        RustAttrValue::Int(i) => *i as f64,
                        RustAttrValue::Bool(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0, // Default for non-numeric types
                    })
                    .collect();
                core_arrays.push(NumArray::new(f64_values));
            }
            // Convert legacy PyGraphArray to NumArray<f64>
            else if let Ok(base_array) = array_obj.extract::<PyRef<PyBaseArray>>(py) {
                let f64_values: Vec<f64> = base_array.inner.data()
                    .iter()
                    .map(|attr_value| match attr_value {
                        RustAttrValue::Float(f) => *f as f64,
                        RustAttrValue::Int(i) => *i as f64,
                        RustAttrValue::Bool(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0, // Default for non-numeric types
                    })
                    .collect();
                core_arrays.push(NumArray::new(f64_values));
            }
            else {
                return Err(PyTypeError::new_err(format!(
                    "Array {} must be BaseArray, NumArray, or GraphArray", i
                )));
            }
        }

        // Create core GraphMatrix
        let matrix = GraphMatrix::from_arrays(core_arrays)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create matrix: {:?}", e)))?;

        Ok(Self { inner: matrix })
    }

    /// Create a zero matrix with specified dimensions and type
    #[classmethod]
    fn zeros(
        _cls: &PyType,
        py: Python,
        rows: usize,
        cols: usize,
        dtype: Option<&str>,
    ) -> PyResult<Py<Self>> {
        // Parse dtype string to AttrValueType
        let attr_type = match dtype.unwrap_or("float") {
            "int" | "int64" => groggy::AttrValueType::Int,
            "float" | "float64" | "f64" => groggy::AttrValueType::Float,
            "bool" => groggy::AttrValueType::Bool,
            "str" | "string" | "text" => groggy::AttrValueType::Text,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported dtype: {}",
                    dtype.unwrap_or("unknown")
                )))
            }
        };

        let matrix = GraphMatrix::zeros(rows, cols, attr_type);
        Py::new(py, Self { inner: matrix })
    }

    /// Create an identity matrix with specified size
    #[classmethod]
    fn identity(_cls: &PyType, py: Python, size: usize) -> PyResult<Py<Self>> {
        let matrix = GraphMatrix::identity(size);
        Py::new(py, Self { inner: matrix })
    }

    /// Create matrix from graph attributes
    #[classmethod]
    fn from_graph_attributes(
        _cls: &PyType,
        _py: Python,
        _graph: PyObject,
        _attrs: Vec<String>,
        _entities: Vec<u64>,
    ) -> PyResult<Py<Self>> {
        // TODO: Implement graph integration in Phase 2
        // For now, return a placeholder error
        Err(PyNotImplementedError::new_err(
            "Graph integration not yet implemented in Phase 2",
        ))
    }

    // === PROPERTIES ===

    /// Get matrix dimensions as (rows, columns) tuple
    #[getter]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get matrix data type
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    /// Get column names
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }


    /// Check if matrix is square
    #[getter]
    fn is_square(&self) -> bool {
        self.inner.is_square()
    }

    /// Check if matrix is symmetric (for square numeric matrices)
    #[getter]
    fn is_symmetric(&self) -> bool {
        // TODO: Implement is_symmetric in core GraphMatrix
        false
    }

    /// Check if matrix contains only numeric data
    #[getter]
    fn is_numeric(&self) -> bool {
        self.inner.is_numeric()
    }

    // === ACCESS & INDEXING ===

    /// Multi-index access for matrix elements: matrix[row, col] -> value, matrix[row] -> row, matrix["col"] -> column
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

        // String -> column access
        if let Ok(col_name) = key.extract::<String>() {
            return self.get_column_by_name(py, col_name);
        }

        Err(PyTypeError::new_err(
            "Key must be: int (row index), string (column name), or (row, col) tuple for multi-index access"
        ))
    }

    /// Get single cell value at (row, col)
    fn get_cell(&self, py: Python, row: usize, col: usize) -> PyResult<PyObject> {
        match self.inner.get(row, col) {
            Some(f64_value) => Ok(f64_value.to_object(py)),
            None => {
                let (rows, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Index ({}, {}) out of range for {}x{} matrix",
                    row, col, rows, cols
                )))
            }
        }
    }

    /// Get entire row as BaseArray (mixed types) or NumArray (if all numerical)
    fn get_row(&self, py: Python, row: usize) -> PyResult<PyObject> {
        match self.inner.get_row(row) {
            Some(row_array) => {
                // GraphMatrix is always numerical, so wrap in PyNumArray
                let py_num_array = PyNumArray { inner: row_array };
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
            None => {
                let (rows, _) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Row index {} out of range for {} rows",
                    row, rows
                )))
            }
        }
    }

    /// Get column by name as BaseArray (mixed types) or NumArray (if all numerical)
    fn get_column_by_name(&self, py: Python, name: String) -> PyResult<PyObject> {
        match self.inner.get_column_by_name(&name) {
            Some(column) => {
                // GraphMatrix columns are always numerical NumArrays
                let py_num_array = PyNumArray { inner: column.clone() };
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
            None => Err(PyKeyError::new_err(format!("Column '{}' not found", name))),
        }
    }

    /// Get column by index as BaseArray (mixed types) or NumArray (if all numerical)
    fn get_column(&self, py: Python, col: usize) -> PyResult<PyObject> {
        match self.inner.get_column(col) {
            Some(column) => {
                // GraphMatrix columns are always numerical NumArrays
                let py_num_array = PyNumArray { inner: column.clone() };
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
            None => {
                let (_, cols) = self.inner.shape();
                Err(PyIndexError::new_err(format!(
                    "Column index {} out of range for {} columns",
                    col, cols
                )))
            }
        }
    }

    // === ITERATION ===

    /// Iterate over rows - returns iterator of BaseArrays or NumArrays
    fn iter_rows(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let (rows, _) = self.inner.shape();
        let mut row_arrays = Vec::with_capacity(rows);

        for i in 0..rows {
            match self.inner.get_row(i) {
                Some(row_array) => {
                    // GraphMatrix rows are always numerical NumArrays
                    let py_num_array = PyNumArray { inner: row_array };
                    row_arrays.push(Py::new(py, py_num_array)?.to_object(py));
                }
                None => return Err(PyIndexError::new_err(format!("Row {} not found", i))),
            }
        }

        Ok(row_arrays)
    }

    /// Iterate over columns - returns iterator of BaseArrays or NumArrays
    fn iter_columns(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let (_, cols) = self.inner.shape();
        let mut col_arrays = Vec::with_capacity(cols);

        for i in 0..cols {
            match self.inner.get_column(i) {
                Some(col_array) => {
                    // GraphMatrix columns are always numerical NumArrays
                    let py_num_array = PyNumArray { inner: col_array.clone() };
                    col_arrays.push(Py::new(py, py_num_array)?.to_object(py));
                }
                None => return Err(PyIndexError::new_err(format!("Column {} not found", i))),
            }
        }

        Ok(col_arrays)
    }

    // === LINEAR ALGEBRA OPERATIONS ===

    /// Transpose the matrix
    fn transpose(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let transposed = self.inner.transpose();
        Py::new(py, PyGraphMatrix { inner: transposed })
    }

    /// Matrix multiplication - multiply this matrix with another
    /// Returns: new GraphMatrix that is the product of self * other
    fn multiply(&self, py: Python, other: &PyGraphMatrix) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self.inner.multiply(&other.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Matrix multiplication failed: {:?}", e))
        })?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Matrix inverse (Phase 5 - placeholder for now)
    fn inverse(&self) -> PyResult<Py<PyGraphMatrix>> {
        Err(PyNotImplementedError::new_err(
            "Matrix inverse will be implemented in Phase 5",
        ))
    }

    /// Matrix power - raise matrix to integer power
    /// Returns: new GraphMatrix that is self^n
    fn power(&self, py: Python, n: u32) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self
            .inner
            .power(n)
            .map_err(|e| PyRuntimeError::new_err(format!("Matrix power failed: {:?}", e)))?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Elementwise multiplication (Hadamard product)
    /// Returns: new GraphMatrix with elementwise product
    fn elementwise_multiply(
        &self,
        py: Python,
        other: &PyGraphMatrix,
    ) -> PyResult<Py<PyGraphMatrix>> {
        let result_matrix = self.inner.elementwise_multiply(&other.inner).map_err(|e| {
            PyRuntimeError::new_err(format!("Elementwise multiplication failed: {:?}", e))
        })?;

        let py_result = PyGraphMatrix::from_graph_matrix(result_matrix);
        Py::new(py, py_result)
    }

    /// Determinant calculation (Phase 5 - placeholder for now)
    fn determinant(&self) -> PyResult<Option<f64>> {
        Err(PyNotImplementedError::new_err(
            "Determinant calculation will be implemented in Phase 5",
        ))
    }

    // === STATISTICAL OPERATIONS ===

    /// Sum along specified axis (0=rows, 1=columns)
    fn sum_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.sum_axis(axis_enum);
        
        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.iter().cloned().collect();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Mean along specified axis (0=rows, 1=columns)
    fn mean_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.mean_axis(axis_enum);
        
        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.iter().cloned().collect();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    /// Standard deviation along specified axis (0=rows, 1=columns)
    fn std_axis(&self, py: Python, axis: usize) -> PyResult<PyObject> {
        let axis_enum = match axis {
            0 => groggy::storage::Axis::Rows,
            1 => groggy::storage::Axis::Columns,
            _ => {
                return Err(PyValueError::new_err(
                    "Axis must be 0 (rows) or 1 (columns)",
                ))
            }
        };

        let result_array = self.inner.std_axis(axis_enum);
        
        // Statistical results are numerical NumArray<f64>, convert directly to PyNumArray
        let f64_values: Vec<f64> = result_array.iter().cloned().collect();
        let num_array = PyNumArray::new(f64_values);
        Ok(Py::new(py, num_array)?.to_object(py))
    }

    // === SCIENTIFIC COMPUTING CONVERSIONS ===

    /// Convert to Pandas DataFrame (when pandas available)
    fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Convert to dictionary of column name -> values
        let dict = pyo3::types::PyDict::new(py);
        let column_names = self.inner.column_names();
        let (_, cols) = self.inner.shape();

        for (col_idx, col_name) in column_names.iter().enumerate() {
            if col_idx < cols {
                match self.inner.get_column(col_idx) {
                    Some(column) => {
                        // Convert column values directly to Python list
                        let py_list = pyo3::types::PyList::empty(py);
                        for val in column.iter() {
                            py_list.append(val.to_object(py))?;
                        }
                        dict.set_item(col_name, py_list)?;
                    }
                    None => {
                        return Err(PyIndexError::new_err(format!(
                            "Column {} not found",
                            col_idx
                        )))
                    }
                }
            }
        }

        // Create DataFrame
        let dataframe = pandas.call_method1("DataFrame", (dict,))?;
        Ok(dataframe.to_object(py))
    }

    // === DISPLAY & REPRESENTATION ===

    /// String representation with rich display hints
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let (rows, cols) = self.inner.shape();
        let dtype = format!("{:?}", self.inner.dtype());
        
        // Show a preview of the matrix data
        let preview = if rows == 0 || cols == 0 {
            "empty".to_string()
        } else {
            // Show first few values as a preview
            let mut preview_values = Vec::new();
            let preview_rows = std::cmp::min(2, rows);
            let preview_cols = std::cmp::min(3, cols);
            
            for i in 0..preview_rows {
                let mut row_values = Vec::new();
                for j in 0..preview_cols {
                    match self.inner.get(i, j) {
                        Some(val) => {
                            let val_str = format!("{:.2}", val);
                            row_values.push(val_str);
                        }
                        None => row_values.push("0".to_string()),
                    }
                }
                if cols > preview_cols {
                    row_values.push("...".to_string());
                }
                preview_values.push(format!("[{}]", row_values.join(" ")));
            }
            
            if rows > preview_rows {
                preview_values.push("...".to_string());
            }
            
            format!("\n{}", preview_values.join("\n"))
        };
        
        Ok(format!(
            "GraphMatrix({} x {}, dtype={}){}\n💡 Use .interactive() for rich table view or .interactive_embed() for Jupyter",
            rows, cols, dtype, preview
        ))
    }

    /// String representation (same as __repr__ for consistency)
    fn __str__(&self, py: Python) -> PyResult<String> {
        self.__repr__(py)
    }

    /// Get rich display representation using Rust formatter
    pub fn rich_display(&self, config: Option<&crate::ffi::display::PyDisplayConfig>) -> PyResult<String> {
        let (total_rows, total_cols) = self.inner.shape();
        let dtype = format!("{:?}", self.inner.dtype());
        let sample_rows = std::cmp::min(5, total_rows);
        let sample_cols = std::cmp::min(5, total_cols);
        
        // Create custom matrix display without index column
        let mut lines = Vec::new();
        
        // Calculate column widths based on content
        let mut col_widths = vec![5; sample_cols]; // minimum width of 5
        
        // Check header widths
        for j in 0..sample_cols {
            let header = format!("col_{}", j);
            col_widths[j] = std::cmp::max(col_widths[j], header.len());
        }
        
        // Check data widths
        for i in 0..sample_rows {
            for j in 0..sample_cols {
                if let Some(value) = self.inner.get(i, j) {
                    let value_str = format!("{}", value);
                    col_widths[j] = std::cmp::max(col_widths[j], value_str.len());
                }
            }
        }
        
        // Build top border
        let mut top_border = "╭".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            top_border.push_str(&"─".repeat(width + 2));
            if j < col_widths.len() - 1 {
                top_border.push('┬');
            }
        }
        top_border.push('╮');
        lines.push(top_border);
        
        // Build header line
        let mut header_line = "│".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            let header = format!("col_{}", j);
            header_line.push_str(&format!(" {:<width$} │", header, width = width));
        }
        lines.push(header_line);
        
        // Build type header line
        let mut type_line = "│".to_string();
        for &width in &col_widths {
            type_line.push_str(&format!(" {:<width$} │", "obj", width = width));
        }
        lines.push(type_line);
        
        // Build separator
        let mut sep_border = "├".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            sep_border.push_str(&"─".repeat(width + 2));
            if j < col_widths.len() - 1 {
                sep_border.push('┼');
            }
        }
        sep_border.push('┤');
        lines.push(sep_border);
        
        // Build data rows
        for i in 0..sample_rows {
            let mut row_line = "│".to_string();
            for (j, &width) in col_widths.iter().enumerate() {
                let value_str = if let Some(value) = self.inner.get(i, j) {
                    format!("{}", value)
                } else {
                    "null".to_string()
                };
                row_line.push_str(&format!(" {:<width$} │", value_str, width = width));
            }
            lines.push(row_line);
        }
        
        // Build bottom border
        let mut bottom_border = "╰".to_string();
        for (j, &width) in col_widths.iter().enumerate() {
            bottom_border.push_str(&"─".repeat(width + 2));
            if j < col_widths.len() - 1 {
                bottom_border.push('┴');
            }
        }
        bottom_border.push('╯');
        lines.push(bottom_border);
        
        // Footer with matrix info
        let footer = format!("rows: {} • cols: {} • type: GraphMatrix • dtype: {}", total_rows, total_cols, dtype);
        lines.push(footer);
        
        Ok(lines.join("\n"))
    }
    
    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert matrix to table and get its HTML representation
        let table = self.to_table_for_streaming()?;
        
        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }


    /// Extract display data for Python display formatters
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        let (rows, cols) = self.inner.shape();

        // Convert matrix to nested list for display
        let mut matrix_data = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row_data = Vec::with_capacity(cols);
            for j in 0..cols {
                match self.inner.get(i, j) {
                    Some(attr_value) => {
                        row_data.push(attr_value.to_object(py));
                    }
                    None => row_data.push(py.None()),
                }
            }
            matrix_data.push(row_data);
        }

        dict.set_item("data", matrix_data)?;
        dict.set_item("shape", (rows, cols))?;
        dict.set_item("dtype", self.dtype())?;
        dict.set_item("columns", self.columns())?;
        dict.set_item("is_square", self.is_square())?;
        dict.set_item("is_symmetric", self.is_symmetric())?;

        Ok(dict.to_object(py))
    }

    /// Iterator support - iterates over rows as lists (temporarily disabled)
    fn __iter__(_slf: PyRef<Self>) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Matrix iteration temporarily disabled during Phase 3 - use matrix[i] for row access",
        ))
    }

    // ========================================================================
    // LAZY EVALUATION & MATERIALIZATION
    // ========================================================================

    /// Get matrix data (materializes data to Python objects)
    /// This is the primary materialization method - use sparingly for large matrices
    #[getter]
    fn data(&self, py: Python) -> PyResult<PyObject> {
        let materialized = self.inner.materialize();
        let py_matrix: PyResult<Vec<Vec<PyObject>>> = materialized
            .iter()
            .map(|row| {
                row.iter()
                    .map(|val| Ok(val.to_object(py)))
                    .collect()
            })
            .collect();

        Ok(py_matrix?.to_object(py))
    }

    /// Get preview of matrix for display (first N rows/cols by default)
    fn preview(
        &self,
        py: Python,
        row_limit: Option<usize>,
        col_limit: Option<usize>,
    ) -> PyResult<PyObject> {
        let row_limit = row_limit.unwrap_or(10);
        let col_limit = col_limit.unwrap_or(10);
        let (preview_data, _col_names) = self.inner.preview(row_limit, col_limit);

        Ok(preview_data.to_object(py))
    }

    /// Check if matrix is sparse (has many default values)
    #[getter]
    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    /// Get summary information without materializing data
    fn summary(&self) -> String {
        self.inner.summary_info()
    }

    /// Create a dense materialized version of the matrix
    fn dense(&self, py: Python) -> PyResult<Py<PyGraphMatrix>> {
        let dense_matrix = self.inner.dense();
        let py_result = PyGraphMatrix::from_graph_matrix(dense_matrix);
        Py::new(py, py_result)
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

    /// Launch interactive streaming table view in browser
    /// 
    /// Converts the matrix into a table format and launches a streaming
    /// interactive view in the browser. The table will show the matrix data
    /// with row and column indices for easy exploration.
    /// 
    /// Returns:
    ///     str: URL of the interactive table interface
    pub fn interactive(&self) -> PyResult<String> {
        // Convert matrix to table format for streaming
        let table = self.to_table_for_streaming()?;
        
        // Use the table's interactive method
        table.interactive()
    }
    
    /// Generate embedded iframe HTML for Jupyter notebooks
    /// 
    /// Creates an interactive streaming table representation of the matrix
    /// that can be embedded directly in a Jupyter notebook cell.
    /// 
    /// Returns:
    ///     str: HTML iframe code for embedding in Jupyter
    pub fn interactive_embed(&self) -> PyResult<String> {
        // Convert matrix to table format for streaming
        let mut table = self.to_table_for_streaming()?;
        
        // Use the table's interactive_embed method
        table.interactive_embed()
    }
    
    /// Convert matrix to table format for streaming visualization
    /// 
    /// Creates a BaseTable with columns representing matrix data:
    /// - 'row': row index
    /// - 'column': column index  
    /// - 'value': the matrix value at that position
    fn to_table_for_streaming(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;
        
        let (rows, cols) = self.shape();
        let mut columns = HashMap::new();
        
        // Prepare data vectors
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        
        // Flatten the matrix into table format (row, column, value)
        for row in 0..rows {
            for col in 0..cols {
                row_indices.push(AttrValue::SmallInt(row as i32));
                col_indices.push(AttrValue::SmallInt(col as i32));
                
                // Get the value from the matrix
                match self.inner.get(row, col) {
                    Some(f64_value) => values.push(AttrValue::Float(*f64_value as f32)),
                    None => values.push(AttrValue::Null),
                }
            }
        }
        
        // Create columns for the table
        columns.insert("row".to_string(), groggy::storage::array::BaseArray::new(row_indices));
        columns.insert("column".to_string(), groggy::storage::array::BaseArray::new(col_indices));
        columns.insert("value".to_string(), groggy::storage::array::BaseArray::new(values));
        
        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e)))?;
        
        Ok(crate::ffi::storage::table::PyBaseTable::from_table(base_table))
    }
}

// Implement display data conversion for PyGraphMatrix
impl PyGraphMatrix {
    /// Convert matrix to display data format expected by Rust formatter
    fn to_display_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        
        // Get matrix dimensions
        let (total_rows, total_cols) = self.inner.shape();
        let sample_rows = std::cmp::min(5, total_rows);
        let sample_cols = std::cmp::min(5, total_cols);
        
        // Table shape (show total dimensions, not just sample)
        data.insert("shape".to_string(), serde_json::Value::Array(vec![
            serde_json::Value::Number(serde_json::Number::from(total_rows)), 
            serde_json::Value::Number(serde_json::Number::from(total_cols))
        ]));
        
        // Generate column names (col_0, col_1, etc.)
        let mut column_names = Vec::new();
        for j in 0..sample_cols {
            column_names.push(serde_json::Value::String(format!("col_{}", j)));
        }
        data.insert("columns".to_string(), serde_json::Value::Array(column_names));
        
        // Create rows data  
        let mut table_rows = Vec::new();
        for i in 0..sample_rows {
            let mut row_data = Vec::new();
            for j in 0..sample_cols {
                if let Some(value) = self.inner.get(i, j) {
                    let json_value = serde_json::Value::Number(
                        serde_json::Number::from_f64(*value).unwrap_or(serde_json::Number::from(0))
                    );
                    row_data.push(json_value);
                } else {
                    row_data.push(serde_json::Value::Null);
                }
            }
            table_rows.push(serde_json::Value::Array(row_data));
        }
        data.insert("data".to_string(), serde_json::Value::Array(table_rows));
        
        data
    }
}

impl PyGraphMatrix {
    /// Create PyGraphMatrix from core GraphMatrix
    pub fn from_graph_matrix(matrix: GraphMatrix) -> Self {
        Self { inner: matrix }
    }

    /// Convert matrix to table representation (placeholder implementation)
    pub fn to_table(&self, _py: Python) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        // TODO: Implement proper matrix-to-table conversion
        // This would typically convert a matrix to a tabular format
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Matrix to table conversion not yet implemented."
        ))
    }
}
