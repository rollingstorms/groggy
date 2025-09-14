//! Array FFI Bindings
//!
//! Python bindings for statistical arrays and matrices.

use groggy::storage::array::{BaseArray, ArrayOps, ArrayIterator, NodesArray, EdgesArray, MetaNodeArray, NodeIdLike, EdgeLike, MetaNodeLike};
use groggy::types::{AttrValue as RustAttrValue, AttrValueType, NodeId, EdgeId};
use groggy::entities::meta_node::MetaNode;
use pyo3::exceptions::{PyAttributeError, PyImportError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

// Use utility functions from utils module
use crate::ffi::utils::{attr_value_to_python_value, python_value_to_attr_value};

/// New BaseArray-powered array with chaining support
#[derive(Clone)]
#[pyclass(name = "BaseArray")]
pub struct PyBaseArray {
    pub inner: BaseArray<RustAttrValue>,
}

#[pymethods]
impl PyBaseArray {
    /// Create a new BaseArray from a list of values
    #[new]
    fn new(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            // Infer the data type from the first non-null element
            let dtype = attr_values
                .iter()
                .find(|v| !matches!(v, RustAttrValue::Null))
                .map(|v| v.dtype())
                .unwrap_or(AttrValueType::Text);

            Ok(PyBaseArray {
                inner: BaseArray::new(attr_values),
            })
        })
    }

    /// Get the number of elements (len())
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Advanced index access supporting multiple indexing types
    /// Supports: 
    /// - Single integer: array[5], array[-1]
    /// - Integer lists: array[[0, 2, 5]]
    /// - Boolean arrays: array[bool_mask]
    /// - Slices: array[:5], array[::2], array[1:10:2]
    fn __getitem__(&self, py: Python, index: &PyAny) -> PyResult<PyObject> {
        use groggy::storage::array::AdvancedIndexing;
        use crate::ffi::utils::indexing::python_index_to_slice_index;
        
        // Handle simple integer case directly for performance
        if let Ok(int_val) = index.extract::<isize>() {
            let len = self.inner.len() as isize;
            let actual_index = if int_val < 0 { len + int_val } else { int_val };
            
            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Index out of range"));
            }
            
            match self.inner.get(actual_index as usize) {
                Some(attr_value) => return attr_value_to_python_value(py, attr_value),
                None => return Err(PyIndexError::new_err("Index out of range")),
            }
        }
        
        // Handle advanced indexing cases
        let slice_index = python_index_to_slice_index(py, index)?;
        
        match self.inner.get_slice(&slice_index) {
            Ok(sliced_array) => {
                let py_array = PyBaseArray { inner: sliced_array };
                Ok(py_array.into_py(py))
            }
            Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e)))
        }
    }

    /// String representation with rich display hints
    fn __repr__(&self) -> String {
        let dtype = self.dtype();
        let len = self.inner.len();
        
        // Show a preview of the data
        let preview = if len == 0 {
            "empty".to_string()
        } else {
            let preview_items: Vec<String> = self.inner.iter()
                .take(3)
                .map(|v| match v {
                    groggy::types::AttrValue::Text(s) => format!("'{}'", s),
                    groggy::types::AttrValue::SmallInt(i) => i.to_string(),
                    groggy::types::AttrValue::Int(i) => i.to_string(),
                    groggy::types::AttrValue::Float(f) => format!("{:.2}", f),
                    groggy::types::AttrValue::Bool(b) => b.to_string(),
                    groggy::types::AttrValue::Null => "None".to_string(),
                    _ => format!("{:?}", v), // Catch-all for other variants
                })
                .collect();
            
            let preview_str = preview_items.join(", ");
            if len > 3 {
                format!("[{}, ...]", preview_str)
            } else {
                format!("[{}]", preview_str)
            }
        };
        
        format!(
            "BaseArray[{}] {} (dtype: {})\n💡 Use .interactive() for rich table view or .interactive_embed() for Jupyter",
            len, preview, dtype
        )
    }

    /// Get the data type of the array based on the first non-null element
    fn dtype(&self) -> String {
        let dtype = self.inner.first()
            .map(|v| v.dtype())
            .unwrap_or(AttrValueType::Null);
        format!("{:?}", dtype)
    }

    // Statistical operations
    fn head(&self, n: usize) -> Self {
        let data: Vec<RustAttrValue> = self.inner.iter().take(n).cloned().collect();
        PyBaseArray {
            inner: BaseArray::new(data),
        }
    }
    
    fn tail(&self, n: usize) -> Self {
        let len = self.inner.len();
        let start = if len > n { len - n } else { 0 };
        let data: Vec<RustAttrValue> = self.inner.iter().skip(start).cloned().collect();
        PyBaseArray {
            inner: BaseArray::new(data),
        }
    }
    
    fn unique(&self) -> Self {
        let mut unique_values: Vec<RustAttrValue> = Vec::new();
        let mut seen: std::collections::HashSet<RustAttrValue> = std::collections::HashSet::new();
        
        for value in self.inner.iter() {
            if seen.insert(value.clone()) {
                unique_values.push(value.clone());
            }
        }
        
        PyBaseArray {
            inner: BaseArray::new(unique_values),
        }
    }

    fn describe(&self, py: Python) -> PyResult<PyObject> {
        // Basic descriptive statistics
        let count = self.inner.len();
        let non_null_count = self.inner.iter().filter(|v| !matches!(v, RustAttrValue::Null)).count();
        let null_count = count - non_null_count;
        let unique_count = self.unique().inner.len();
        
        let dict = PyDict::new(py);
        dict.set_item("count", count)?;
        dict.set_item("non_null", non_null_count)?;
        dict.set_item("null", null_count)?;
        dict.set_item("unique", unique_count)?;
        dict.set_item("dtype", self.dtype())?;
        
        Ok(dict.into())
    }

    /// NEW: Enable fluent chaining with .iter() method
    fn iter(slf: PyRef<Self>) -> PyResult<PyBaseArrayIterator> {
        // Use our ArrayOps implementation to create the iterator
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyBaseArrayIterator {
            inner: array_iterator,
        })
    }
    
    /// Delegation-based method application to each element 
    /// This demonstrates the concept: apply a method to each element and return new array
    fn apply_to_each(&self, py: Python, method_name: &str, args: &PyTuple) -> PyResult<PyObject> {
        let mut results = Vec::new();
        
        // Apply the method to each element in the array
        for value in self.inner.data() {
            // Convert AttrValue to a Python object that might have the method
            let py_value = attr_value_to_python_value(py, value)?;
            
            // Check if this value has the requested method
            if py_value.as_ref(py).hasattr(method_name)? {
                // Get the method and call it with the provided arguments
                let method = py_value.as_ref(py).getattr(method_name)?;
                let result = method.call1(args)?;
                
                // Convert result back to AttrValue
                let attr_result = python_value_to_attr_value(result)?;
                results.push(attr_result);
            } else {
                return Err(PyAttributeError::new_err(
                    format!("Elements in array don't have method '{}'", method_name)
                ));
            }
        }
        
        // Create new BaseArray with results
        let result_array = BaseArray::from_attr_values(results);
        let py_base = PyBaseArray { inner: result_array };
        Ok(Py::new(py, py_base)?.to_object(py))
    }
    
    /// BaseArray __getattr__ delegation: automatically apply methods to each element
    /// This enables: array.some_method() -> applies some_method to each element
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        // Use apply_to_each internally to delegate to each element
        let args = pyo3::types::PyTuple::empty(py);
        let result = self.apply_to_each(py, name, args)?;
        Ok(result.into_py(py))
    }

    /// Launch interactive streaming table view in browser
    /// 
    /// Converts the array into a table format and launches a streaming
    /// interactive view in the browser. The table will have two columns:
    /// 'index' and 'value' for easy exploration of the array data.
    /// 
    /// Returns:
    ///     str: URL of the interactive table interface
    pub fn interactive(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let table = self.to_table()?;
        
        // Use the table's interactive method
        table.interactive()
    }
    
    /// Generate embedded iframe HTML for Jupyter notebooks
    /// 
    /// Creates an interactive streaming table representation of the array
    /// that can be embedded directly in a Jupyter notebook cell.
    /// 
    /// Returns:
    ///     str: HTML iframe code for embedding in Jupyter
    pub fn interactive_embed(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let mut table = self.to_table()?;
        
        // Use the table's interactive_embed method
        table.interactive_embed()
    }
    
    /// Rich HTML representation for Jupyter notebooks
    /// 
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells, similar to pandas DataFrames.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;
        
        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }
    
    /// Convert array to table format for streaming visualization
    /// 
    /// Creates a BaseTable with 'index' and 'value' columns from the array data.
    /// This enables the array to use the rich streaming table infrastructure.
    fn to_table(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;
        
        // Create columns for the table
        let mut columns = HashMap::new();
        
        // Index column (0, 1, 2, ...)
        let indices: Vec<AttrValue> = (0..self.inner.len())
            .map(|i| AttrValue::SmallInt(i as i32))
            .collect();
        columns.insert("index".to_string(), groggy::storage::array::BaseArray::new(indices));
        
        // Value column (array data)
        let values: Vec<AttrValue> = self.inner.clone_vec();
        columns.insert("value".to_string(), groggy::storage::array::BaseArray::new(values));
        
        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e)))?;
        
        Ok(crate::ffi::storage::table::PyBaseTable::from_table(base_table))
    }

    // === COMPARISON OPERATORS FOR BOOLEAN INDEXING ===

    /// Greater than comparison (>) - returns Vec<bool> for boolean masking
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();
        
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a > b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a > b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a > b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a > (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) > *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) > *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a > (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) > *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a > (*b as f32),

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() > b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => a.as_str() > b.as_str(),
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => a.as_str() > b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => a.as_str() > b.as_str(),

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a > b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(comparison_result);
        }
        
        // Convert Vec<bool> to BoolArray
        let bool_array = groggy::storage::BoolArray::new(result);
        let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
        Ok(py_bool_array.into_py(py))
    }

    /// Less than comparison (<) - returns BoolArray for boolean masking
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();
        
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a < b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a < b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a < b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a < (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) < *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) < *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a < (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) < *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a < (*b as f32),

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() < b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => a.as_str() < b.as_str(),
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => a.as_str() < b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => a.as_str() < b.as_str(),

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a < b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(comparison_result);
        }
        
        // Convert Vec<bool> to BoolArray
        let bool_array = groggy::storage::BoolArray::new(result);
        let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
        Ok(py_bool_array.into_py(py))
    }

    /// Greater than or equal comparison (>=) - returns BoolArray for boolean masking
    fn __ge__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();
        
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a >= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a >= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a >= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a >= (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) >= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) >= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a >= (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) >= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a >= (*b as f32),

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() >= b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => a.as_str() >= b.as_str(),
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => a.as_str() >= b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => a.as_str() >= b.as_str(),

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a >= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(comparison_result);
        }
        
        // Convert Vec<bool> to BoolArray
        let bool_array = groggy::storage::BoolArray::new(result);
        let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
        Ok(py_bool_array.into_py(py))
    }

    /// Less than or equal comparison (<=) - returns BoolArray for boolean masking
    fn __le__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();
        
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a <= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a <= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a <= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a <= (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) <= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32) <= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => *a <= (*b as f32),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32) <= *b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => *a <= (*b as f32),

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() <= b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => a.as_str() <= b.as_str(),
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => a.as_str() <= b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => a.as_str() <= b.as_str(),

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a <= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(comparison_result);
        }
        
        // Convert Vec<bool> to BoolArray
        let bool_array = groggy::storage::BoolArray::new(result);
        let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
        Ok(py_bool_array.into_py(py))
    }

    /// Equality comparison (==) - returns BoolArray for boolean masking
    fn __eq__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;
        let mut result = Vec::new();
        
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Exact type matches
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a == b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a == b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => (a - b).abs() < f32::EPSILON,
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a == b,
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a == b,
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => a.as_str() == b.as_str(),
                (groggy::AttrValue::Null, groggy::AttrValue::Null) => true,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a == (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) == *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => (*a as f32 - *b).abs() < f32::EPSILON,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => (*a - *b as f32).abs() < f32::EPSILON,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => (*a as f32 - *b).abs() < f32::EPSILON,
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => (*a - *b as f32).abs() < f32::EPSILON,

                // Mixed string comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => a.as_str() == b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => a.as_str() == b.as_str(),

                // Different types are not equal (including nulls with non-nulls)
                _ => false,
            };
            result.push(comparison_result);
        }
        
        // Convert Vec<bool> to BoolArray
        let bool_array = groggy::storage::BoolArray::new(result);
        let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
        Ok(py_bool_array.into_py(py))
    }

    /// Not equal comparison (!=) - returns BoolArray for boolean masking
    fn __ne__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        // Just negate the equality result
        let eq_result = self.__eq__(py, other)?;
        let bool_vec: Vec<bool> = eq_result.extract(py)?;
        let ne_result: Vec<bool> = bool_vec.into_iter().map(|x| !x).collect();
        // Convert Vec<bool> to BoolArray
        let bool_array = groggy::storage::BoolArray::new(ne_result);
        let py_bool_array = crate::ffi::storage::bool_array::PyBoolArray { inner: bool_array };
        Ok(py_bool_array.into_py(py))
    }
}

/// Python wrapper for ArrayIterator<AttrValue> that supports method chaining
#[pyclass(name = "BaseArrayIterator", unsendable)]
pub struct PyBaseArrayIterator {
    inner: ArrayIterator<RustAttrValue>,
}

#[pymethods]
impl PyBaseArrayIterator {
    /// Filter elements using a Python predicate function
    fn filter(slf: PyRefMut<Self>, predicate: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = slf.inner.clone();
            let filtered = inner.filter(|attr_value| {
                // Convert AttrValue to Python and call predicate
                match attr_value_to_python_value(py, attr_value) {
                    Ok(py_value) => {
                        match predicate.call1(py, (py_value,)) {
                            Ok(result) => result.is_true(py).unwrap_or(false),
                            Err(_) => false,
                        }
                    }
                    Err(_) => false,
                }
            });
            
            Ok(Self { inner: filtered })
        })
    }
    
    /// Map elements using a Python function
    fn map(slf: PyRefMut<Self>, func: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = slf.inner.clone();
            let mapped = inner.map(|attr_value| {
                // Convert AttrValue to Python, call function, convert back
                match attr_value_to_python_value(py, &attr_value) {
                    Ok(py_value) => {
                        match func.call1(py, (py_value,)) {
                            Ok(result) => {
                                match python_value_to_attr_value(result.as_ref(py)) {
                                    Ok(new_attr_value) => new_attr_value,
                                    Err(_) => attr_value, // Keep original on conversion error
                                }
                            }
                            Err(_) => attr_value, // Keep original on call error
                        }
                    }
                    Err(_) => attr_value, // Keep original on conversion error
                }
            });
            
            Ok(Self { inner: mapped })
        })
    }
    
    /// Take first n elements
    fn take(slf: PyRefMut<Self>, n: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self { inner: inner.take(n) })
    }
    
    /// Skip first n elements
    fn skip(slf: PyRefMut<Self>, n: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self { inner: inner.skip(n) })
    }
    
    /// Collect back into a BaseArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyBaseArray> {
        let inner = slf.inner.clone();
        
        // Extract the data directly
        let elements = inner.into_vec();
        
        // Infer dtype from first non-null element
        let dtype = elements
            .iter()
            .find(|v| !matches!(v, RustAttrValue::Null))
            .map(|v| v.dtype())
            .unwrap_or(AttrValueType::Text);
        
        Ok(PyBaseArray {
            inner: BaseArray::new(elements),
        })
    }
}

// =============================================================================
// Specialized typed arrays with trait-based method injection
// =============================================================================

/// Python wrapper for NodesArray - specialized array for NodeId collections
#[pyclass(name = "NodesArray", unsendable)]
pub struct PyNodesArray {
    pub inner: NodesArray,
}

#[pymethods]
impl PyNodesArray {
    /// Create a new NodesArray from node IDs
    #[new]
    fn new(node_ids: Vec<usize>) -> PyResult<Self> {
        // Convert Python usize to NodeId
        let nodes: Vec<NodeId> = node_ids.into_iter().collect();
        
        Ok(PyNodesArray {
            inner: NodesArray::new(nodes),
        })
    }
    
    /// Get the number of nodes
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Get node ID by index
    fn __getitem__(&self, index: isize) -> PyResult<usize> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 { len + index } else { index };
        
        if actual_index < 0 || actual_index >= len {
            return Err(PyIndexError::new_err("Index out of range"));
        }
        
        match self.inner.get(actual_index as usize) {
            Some(node_id) => Ok(*node_id),
            None => Err(PyIndexError::new_err("Index out of range")),
        }
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        let preview = if len == 0 {
            "empty".to_string()
        } else {
            let preview_ids: Vec<String> = self.inner.node_ids()
                .iter()
                .take(3)
                .map(|&id| id.to_string())
                .collect();
            let preview_str = preview_ids.join(", ");
            if len > 3 {
                format!("[{}, ...]", preview_str)
            } else {
                format!("[{}]", preview_str)
            }
        };
        
        format!(
            "NodesArray[{}] {} node_ids\n💡 Use .interactive() for rich table view or .interactive_embed() for Jupyter",
            len, preview
        )
    }
    
    /// Enable fluent chaining with .iter() method for node-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyNodesArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyNodesArrayIterator {
            inner: array_iterator,
        })
    }

    /// Launch interactive streaming table view in browser
    /// 
    /// Converts the nodes array into a table format and launches a streaming
    /// interactive view in the browser. The table will have an 'index' column
    /// and a 'node_id' column for exploring the node data.
    /// 
    /// Returns:
    ///     str: URL of the interactive table interface
    pub fn interactive(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let table = self.to_table()?;
        
        // Use the table's interactive method
        table.interactive()
    }
    
    /// Generate embedded iframe HTML for Jupyter notebooks
    /// 
    /// Creates an interactive streaming table representation of the nodes array.
    /// 
    /// Returns:
    ///     str: HTML iframe code for embedding in Jupyter
    pub fn interactive_embed(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let mut table = self.to_table()?;
        
        // Use the table's interactive_embed method
        table.interactive_embed()
    }
    
    /// Rich HTML representation for Jupyter notebooks
    /// 
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells for NodesArray data.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;
        
        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }
    
    /// Convert nodes array to table format for streaming visualization
    fn to_table(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;
        
        // Create columns for the table
        let mut columns = HashMap::new();
        
        // Index column (0, 1, 2, ...)
        let indices: Vec<AttrValue> = (0..self.inner.len())
            .map(|i| AttrValue::SmallInt(i as i32))
            .collect();
        columns.insert("index".to_string(), groggy::storage::array::BaseArray::new(indices));
        
        // Node ID column
        let node_ids: Vec<AttrValue> = self.inner.node_ids()
            .iter()
            .map(|&node_id| AttrValue::SmallInt(node_id as i32))
            .collect();
        columns.insert("node_id".to_string(), groggy::storage::array::BaseArray::new(node_ids));
        
        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e)))?;
        
        Ok(crate::ffi::storage::table::PyBaseTable::from_table(base_table))
    }
}

/// Python wrapper for ArrayIterator<NodeId> with node-specific chaining operations
#[pyclass(name = "NodesArrayIterator", unsendable)]
pub struct PyNodesArrayIterator {
    inner: ArrayIterator<NodeId>,
}

#[pymethods]
impl PyNodesArrayIterator {
    /// Filter nodes by minimum degree
    /// Enables: g.nodes.ids().iter().filter_by_degree(3)
    fn filter_by_degree(slf: PyRefMut<Self>, min_degree: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_by_degree(min_degree),
        })
    }
    
    /// Get neighbors for each node  
    /// Enables: node_ids.iter().get_neighbors()
    fn get_neighbors(slf: PyRefMut<Self>) -> PyResult<PyNeighborsArrayIterator> {
        let inner = slf.inner.clone();
        let neighbors_iterator = inner.get_neighbors();
        
        Ok(PyNeighborsArrayIterator {
            inner: neighbors_iterator,
        })
    }
    
    /// Convert nodes to subgraphs
    /// Enables: node_ids.iter().to_subgraph()
    fn to_subgraph(slf: PyRefMut<Self>) -> PyResult<PySubgraphArrayIterator> {
        let inner = slf.inner.clone();
        let subgraph_iterator = inner.to_subgraph();
        
        Ok(PySubgraphArrayIterator {
            inner: subgraph_iterator,
        })
    }
    
    /// Collect back into a NodesArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyNodesArray> {
        let inner = slf.inner.clone();
        let node_ids = inner.into_vec();
        
        Ok(PyNodesArray {
            inner: NodesArray::new(node_ids),
        })
    }
}

/// Python wrapper for EdgesArray - specialized array for EdgeId collections
#[pyclass(name = "EdgesArray", unsendable)]
pub struct PyEdgesArray {
    pub inner: EdgesArray,
}

#[pymethods]
impl PyEdgesArray {
    /// Create a new EdgesArray from edge IDs
    #[new]
    fn new(edge_ids: Vec<usize>) -> PyResult<Self> {
        // Convert Python usize to EdgeId
        let edges: Vec<EdgeId> = edge_ids.into_iter().collect();
        
        Ok(PyEdgesArray {
            inner: EdgesArray::new(edges),
        })
    }
    
    /// Get the number of edges
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        let preview = if len == 0 {
            "empty".to_string()
        } else {
            let preview_ids: Vec<String> = self.inner.edge_ids()
                .iter()
                .take(3)
                .map(|&id| id.to_string())
                .collect();
            let preview_str = preview_ids.join(", ");
            if len > 3 {
                format!("[{}, ...]", preview_str)
            } else {
                format!("[{}]", preview_str)
            }
        };
        
        format!(
            "EdgesArray[{}] {} edge_ids\n💡 Use .interactive() for rich table view or .interactive_embed() for Jupyter",
            len, preview
        )
    }
    
    /// Enable fluent chaining with .iter() method for edge-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyEdgesArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyEdgesArrayIterator {
            inner: array_iterator,
        })
    }

    /// Launch interactive streaming table view in browser
    /// 
    /// Converts the edges array into a table format and launches a streaming
    /// interactive view in the browser. The table will have an 'index' column
    /// and an 'edge_id' column for exploring the edge data.
    /// 
    /// Returns:
    ///     str: URL of the interactive table interface
    pub fn interactive(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let table = self.to_table()?;
        
        // Use the table's interactive method
        table.interactive()
    }
    
    /// Generate embedded iframe HTML for Jupyter notebooks
    /// 
    /// Creates an interactive streaming table representation of the edges array.
    /// 
    /// Returns:
    ///     str: HTML iframe code for embedding in Jupyter
    pub fn interactive_embed(&self) -> PyResult<String> {
        // Convert array to table format for streaming
        let mut table = self.to_table()?;
        
        // Use the table's interactive_embed method
        table.interactive_embed()
    }
    
    /// Rich HTML representation for Jupyter notebooks
    /// 
    /// Returns beautiful HTML table representation that displays automatically
    /// in Jupyter notebook cells for EdgesArray data.
    pub fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        // Convert to table and get its HTML representation
        let table = self.to_table()?;
        
        // Use the core table's _repr_html_ method for proper HTML output
        let html = table.table._repr_html_();
        Ok(html)
    }
    
    /// Convert edges array to table format for streaming visualization
    fn to_table(&self) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;
        
        // Create columns for the table
        let mut columns = HashMap::new();
        
        // Index column (0, 1, 2, ...)
        let indices: Vec<AttrValue> = (0..self.inner.len())
            .map(|i| AttrValue::SmallInt(i as i32))
            .collect();
        columns.insert("index".to_string(), groggy::storage::array::BaseArray::new(indices));
        
        // Edge ID column
        let edge_ids: Vec<AttrValue> = self.inner.edge_ids()
            .iter()
            .map(|&edge_id| AttrValue::SmallInt(edge_id as i32))
            .collect();
        columns.insert("edge_id".to_string(), groggy::storage::array::BaseArray::new(edge_ids));
        
        // Create the BaseTable
        let base_table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create table: {}", e)))?;
        
        Ok(crate::ffi::storage::table::PyBaseTable::from_table(base_table))
    }
}

/// Python wrapper for ArrayIterator<EdgeId> with edge-specific chaining operations
#[pyclass(name = "EdgesArrayIterator", unsendable)]
pub struct PyEdgesArrayIterator {
    inner: ArrayIterator<EdgeId>,
}

#[pymethods]
impl PyEdgesArrayIterator {
    /// Filter edges by minimum weight
    /// Enables: edges.iter().filter_by_weight(0.5)
    fn filter_by_weight(slf: PyRefMut<Self>, min_weight: f64) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_by_weight(min_weight),
        })
    }
    
    /// Group edges by source node
    /// Enables: edges.iter().group_by_source()
    fn group_by_source(slf: PyRefMut<Self>) -> PyResult<PyEdgeGroupsIterator> {
        let inner = slf.inner.clone();
        let groups_iterator = inner.group_by_source();
        
        Ok(PyEdgeGroupsIterator {
            inner: groups_iterator,
        })
    }
    
    /// Collect back into an EdgesArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyEdgesArray> {
        let inner = slf.inner.clone();
        let edge_ids = inner.into_vec();
        
        Ok(PyEdgesArray {
            inner: EdgesArray::new(edge_ids),
        })
    }
}

/// Python wrapper for MetaNodeArray - specialized array for MetaNode collections
#[pyclass(name = "MetaNodeArray", unsendable)]
pub struct PyMetaNodeArray {
    pub inner: MetaNodeArray,
}

#[pymethods]
impl PyMetaNodeArray {
    /// Get the number of meta-nodes
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("MetaNodeArray[{}]", self.inner.len())
    }
    
    /// Enable fluent chaining with .iter() method for meta-node-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyMetaNodeArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyMetaNodeArrayIterator {
            inner: array_iterator,
        })
    }
}

/// Python wrapper for ArrayIterator<MetaNode> with meta-node-specific chaining operations
#[pyclass(name = "MetaNodeArrayIterator", unsendable)]
pub struct PyMetaNodeArrayIterator {
    inner: ArrayIterator<MetaNode>,
}

#[pymethods]
impl PyMetaNodeArrayIterator {
    /// Expand meta-nodes back into subgraphs
    /// Enables: meta_nodes.iter().expand()
    fn expand(slf: PyRefMut<Self>) -> PyResult<PySubgraphArrayIterator> {
        let inner = slf.inner.clone();
        let subgraph_iterator = inner.expand();
        
        Ok(PySubgraphArrayIterator {
            inner: subgraph_iterator,
        })
    }
    
    /// Re-aggregate meta-nodes with new aggregation functions
    fn re_aggregate(slf: PyRefMut<Self>, aggs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.re_aggregate(aggs),
        })
    }
    
    /// Collect back into a MetaNodeArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyMetaNodeArray> {
        let inner = slf.inner.clone();
        let meta_nodes = inner.into_vec();
        
        Ok(PyMetaNodeArray {
            inner: MetaNodeArray::new(meta_nodes),
        })
    }
}

// =============================================================================
// Supporting iterator types for complex return values
// =============================================================================

#[pyclass(name = "NeighborsArrayIterator", unsendable)]
pub struct PyNeighborsArrayIterator {
    inner: ArrayIterator<Vec<NodeId>>,
}

#[pymethods]
impl PyNeighborsArrayIterator {
    /// Flatten neighbor lists into a single NodesArray
    fn flatten(slf: PyRefMut<Self>) -> PyResult<PyNodesArray> {
        let inner = slf.inner.clone();
        let neighbor_lists = inner.into_vec();
        let flattened: Vec<NodeId> = neighbor_lists.into_iter().flatten().collect();
        
        Ok(PyNodesArray {
            inner: NodesArray::new(flattened),
        })
    }
    
    /// Collect into a list of lists
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<Vec<usize>>> {
        let inner = slf.inner.clone();
        let neighbor_lists = inner.into_vec();
        let py_lists: Vec<Vec<usize>> = neighbor_lists
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect())
            .collect();
        
        Ok(py_lists)
    }
}

#[pyclass(name = "SubgraphArrayIterator", unsendable)]
pub struct PySubgraphArrayIterator {
    inner: ArrayIterator<groggy::subgraphs::subgraph::Subgraph>,
}

#[pymethods]
impl PySubgraphArrayIterator {
    /// Filter nodes within subgraphs (inherits from SubgraphLike)
    fn filter_nodes(slf: PyRefMut<Self>, query: &str) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_nodes(query),
        })
    }
    
    /// Collect into a list of subgraphs
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<String>> {
        let inner = slf.inner.clone();
        let _subgraphs = inner.into_vec();
        
        // Placeholder: return string representations
        Ok(vec!["placeholder_subgraph".to_string()])
    }
}

#[pyclass(name = "EdgeGroupsIterator", unsendable)]
pub struct PyEdgeGroupsIterator {
    inner: ArrayIterator<Vec<EdgeId>>,
}

#[pymethods]
impl PyEdgeGroupsIterator {
    /// Collect into a list of edge groups
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<Vec<usize>>> {
        let inner = slf.inner.clone();
        let edge_groups = inner.into_vec();
        let py_groups: Vec<Vec<usize>> = edge_groups
            .into_iter()
            .map(|group| group.into_iter().collect())
            .collect();
        
        Ok(py_groups)
    }
}