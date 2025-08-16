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

/// Python wrapper for GraphMatrix - structured collection of GraphArray columns
/// This is the return type for multi-column operations like g.nodes[:][['age', 'dept']]
#[pyclass(name = "GraphMatrix")]
pub struct PyGraphMatrix {
    /// Column data as GraphArrays
    pub columns: Vec<Py<PyGraphArray>>,
    /// Column names
    pub column_names: Vec<String>,
    /// Number of rows
    pub num_rows: usize,
}

#[pymethods]
impl PyGraphMatrix {
    /// Get matrix dimensions as (rows, columns) tuple
    /// Get matrix shape as (rows, cols) tuple 
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.column_names.len())
    }
    
    /// Get column names as property
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.column_names.clone()
    }
    
    /// Get number of columns
    fn column_count(&self) -> usize {
        self.column_names.len()
    }
    
    /// Get number of rows
    fn row_count(&self) -> usize {
        self.num_rows
    }
    
    /// Enhanced access supporting multiple patterns:
    /// - Column by name: matrix['age'] -> GraphArray
    /// - Row by index: matrix[0] -> list/dict of row values  
    /// - Multi-index: matrix[0, 1] -> single cell value
    /// - Positional column: matrix[:, 0] -> GraphArray (first column)
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Pattern 1: Multi-index access (row, col) -> single cell value
        if let Ok(indices) = key.extract::<(usize, usize)>() {
            let (row, col) = indices;
            return self.get_cell(py, row, col);
        }
        
        // Pattern 2: String key -> column by name
        if let Ok(column_name) = key.extract::<String>() {
            if let Some(index) = self.column_names.iter().position(|name| name == &column_name) {
                return Ok(self.columns[index].clone_ref(py).to_object(py));
            } else {
                return Err(PyKeyError::new_err(format!("Column '{}' not found. Available columns: {:?}", column_name, self.column_names)));
            }
        }
        
        // Pattern 3: Single integer -> row access
        if let Ok(row_index) = key.extract::<usize>() {
            return self.get_row(py, row_index);
        }
        
        // Pattern 4: Slice notation (future enhancement)
        // if let Ok(slice) = key.extract::<PySlice>() { ... }
        
        Err(PyTypeError::new_err(
            "Key must be: string (column name), int (row index), or (row, col) tuple for multi-index access"
        ))
    }
    
    /// Get single cell value at (row, col)
    fn get_cell(&self, py: Python, row: usize, col: usize) -> PyResult<PyObject> {
        if col >= self.columns.len() {
            return Err(PyIndexError::new_err(format!("Column index {} out of range (0-{})", col, self.columns.len() - 1)));
        }
        
        let column = &self.columns[col];
        let graph_array = column.borrow(py);
        
        // Get the value at the specified row from this column
        graph_array.__getitem__(py, row as isize)
    }
    
    /// Get entire row as dict (if named columns) or list (if positional)
    fn get_row(&self, py: Python, row: usize) -> PyResult<PyObject> {
        if row >= self.num_rows {
            return Err(PyIndexError::new_err(format!("Row index {} out of range (0-{})", row, self.num_rows - 1)));
        }
        
        // Return as dict with column names as keys
        let dict = pyo3::types::PyDict::new(py);
        
        for (i, column_name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];
            let graph_array = column.borrow(py);
            let value = graph_array.__getitem__(py, row as isize)?;
            dict.set_item(column_name, value)?;
        }
        
        Ok(dict.to_object(py))
    }
    
    fn __repr__(&self) -> String {
        format!("GraphMatrix(shape=({}, {}), columns={:?})", 
                self.num_rows, self.column_names.len(), self.column_names)
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
