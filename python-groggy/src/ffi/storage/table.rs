//! Python FFI for BaseTable system

use pyo3::prelude::*;
use crate::ffi::storage::array::PyBaseArray;
use groggy::storage::table::{BaseTable, NodesTable, EdgesTable, Table, TableIterator};
use groggy::types::{NodeId, EdgeId};
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
    
    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.table.nrows()
    }
    
    /// Get number of columns
    pub fn ncols(&self) -> usize {
        self.table.ncols()
    }
    
    /// Get column names
    pub fn column_names(&self) -> Vec<String> {
        self.table.column_names().to_vec()
    }
    
    /// Get shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.table.shape()
    }
    
    /// Check if column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.table.has_column(name)
    }
    
    /// Get first n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn head(&self, n: usize) -> Self {
        Self {
            table: self.table.head(n),
        }
    }
    
    /// Get last n rows (default 5)
    #[pyo3(signature = (n = 5))]
    pub fn tail(&self, n: usize) -> Self {
        Self {
            table: self.table.tail(n),
        }
    }
    
    /// Get table iterator for chaining
    pub fn iter(&self) -> PyBaseTableIterator {
        PyBaseTableIterator {
            iterator: self.table.iter(),
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
                // Convert BaseArray to PyGraphArray via AttrValues
                let attr_values = column.data();
                let py_objects: Vec<_> = attr_values.iter()
                    .map(|attr| {
                        let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                        py_attr.to_object(py)
                    })
                    .collect();
                let py_array = crate::ffi::storage::array::PyGraphArray::from_py_objects(py_objects)?;
                Ok(py_array.into_py(py))
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
                "Table indices must be strings (column names), lists of column names, slices, or boolean arrays"
            ))
        }
    }
    
    /// Get rich display representation using Rust formatter
    pub fn rich_display(&self, config: Option<&crate::ffi::display::PyDisplayConfig>) -> PyResult<String> {
        let display_data = self.to_display_data();
        let default_config = groggy::display::DisplayConfig::default();
        let rust_config = config.map(|c| c.get_config()).unwrap_or(&default_config);
        Ok(groggy::display::format_table(display_data, rust_config))
    }
    
    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        let display_data = self.to_display_data();
        let config = groggy::display::DisplayConfig::default();
        let formatted = groggy::display::format_table(display_data, &config);
        // Convert to HTML format for Jupyter
        Ok(format!("<pre>{}</pre>", html_escape::encode_text(&formatted)))
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
    
    /// Get node IDs
    pub fn node_ids(&self) -> PyResult<Vec<NodeId>> {
        self.table.node_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    /// Add node attributes from a HashMap
    pub fn with_attributes(&self, attr_name: String, attributes: std::collections::HashMap<NodeId, crate::ffi::types::PyAttrValue>) -> PyResult<Self> {
        // Convert PyAttrValue HashMap to AttrValue HashMap
        let rust_attributes: std::collections::HashMap<NodeId, groggy::AttrValue> = attributes
            .into_iter()
            .map(|(k, v)| (k, v.inner))
            .collect();
            
        let result = self.table.clone().with_attributes(attr_name, rust_attributes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(Self { table: result })
    }
    
    /// Filter nodes by attribute value
    pub fn filter_by_attr(&self, attr_name: &str, value: &crate::ffi::types::PyAttrValue) -> PyResult<Self> {
        let result = self.table.filter_by_attr(attr_name, &value.inner)
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
    
    /// Enable subscripting: table[column_name] or table[slice]  
    pub fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        use pyo3::types::{PyString, PySlice};
        let py = key.py();
        
        if let Ok(column_name) = key.extract::<String>() {
            // Column access: table['column_name'] 
            if let Some(column) = self.table.base_table().column(&column_name) {
                // Convert BaseArray to GraphArray
                // Convert BaseArray to PyGraphArray via AttrValues
                let attr_values = column.data();
                let py_objects: Vec<_> = attr_values.iter()
                    .map(|attr| {
                        let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                        py_attr.to_object(py)
                    })
                    .collect();
                let py_array = crate::ffi::storage::array::PyGraphArray::from_py_objects(py_objects)?;
                Ok(py_array.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Column '{}' not found", column_name)
                ))
            }
        } else if let Ok(column_names) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2', ...]]
            let selected_base = self.table.base_table().select(&column_names)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Failed to select columns: {}", e)
                ))?;
            let nodes_selected = groggy::storage::table::NodesTable::from_base_table(selected_base)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyNodesTable { table: nodes_selected }.into_py(py))
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
    
    /// Delegate to BaseTable for missing methods (enables rich display)
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let base_table = self.base_table();
        let base_obj = base_table.into_py(py);
        base_obj.getattr(py, name)
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
    
    /// Get edge IDs
    pub fn edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        self.table.edge_ids()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    /// Get source node IDs
    pub fn sources(&self) -> PyResult<Vec<NodeId>> {
        self.table.sources()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
    
    /// Get target node IDs  
    pub fn targets(&self) -> PyResult<Vec<NodeId>> {
        self.table.targets()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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
    
    /// Enable subscripting: table[column_name] or table[slice]
    pub fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        use pyo3::types::{PyString, PySlice};
        let py = key.py();
        
        if let Ok(column_name) = key.extract::<String>() {
            // Column access: table['column_name']
            if let Some(column) = self.table.base_table().column(&column_name) {
                // Convert BaseArray to GraphArray  
                // Convert BaseArray to PyGraphArray via AttrValues
                let attr_values = column.data();
                let py_objects: Vec<_> = attr_values.iter()
                    .map(|attr| {
                        let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                        py_attr.to_object(py)
                    })
                    .collect();
                let py_array = crate::ffi::storage::array::PyGraphArray::from_py_objects(py_objects)?;
                Ok(py_array.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Column '{}' not found", column_name)
                ))
            }
        } else if let Ok(column_names) = key.extract::<Vec<String>>() {
            // Multi-column access: table[['col1', 'col2', ...]]
            let selected_base = self.table.base_table().select(&column_names)
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
    
    /// Delegate to BaseTable for missing methods (enables rich display)
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let base_table = self.base_table();
        let base_obj = base_table.into_py(py);
        base_obj.getattr(py, name)
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
    
    /// Save GraphTable as bundle to disk
    pub fn save_bundle(&self, path: &str) -> PyResult<()> {
        use std::path::Path;
        
        self.table.save_bundle(Path::new(path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(())
    }
    
    /// Load GraphTable from bundle on disk
    #[staticmethod]
    pub fn load_bundle(path: &str) -> PyResult<PyGraphTable> {
        use std::path::Path;
        
        let table = groggy::storage::table::GraphTable::load_bundle(Path::new(path))
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
                // Convert BaseArray to PyGraphArray via AttrValues
                let attr_values = column.data();
                let py_objects: Vec<_> = attr_values.iter()
                    .map(|attr| {
                        let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                        py_attr.to_object(py)
                    })
                    .collect();
                let py_array = crate::ffi::storage::array::PyGraphArray::from_py_objects(py_objects)?;
                Ok(py_array.into_py(py))
            } else {
                let edges_table = self.table.edges();
                if let Some(column) = edges_table.base_table().column(&column_name) {
                    // Convert BaseArray to PyGraphArray via AttrValues
                let attr_values = column.data();
                let py_objects: Vec<_> = attr_values.iter()
                    .map(|attr| {
                        let py_attr = crate::ffi::types::PyAttrValue::new(attr.clone());
                        py_attr.to_object(py)
                    })
                    .collect();
                let py_array = crate::ffi::storage::array::PyGraphArray::from_py_objects(py_objects)?;
                    Ok(py_array.into_py(py))
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
}

// Internal methods (not exposed to Python)
impl PyGraphTable {
    /// Create PyGraphTable from GraphTable (used by accessors - not exposed to Python)
    pub(crate) fn from_graph_table(table: groggy::storage::table::GraphTable) -> Self {
        Self { table }
    }
}