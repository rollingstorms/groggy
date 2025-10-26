//! Unified NumArray - Statistical array operations with comprehensive dtype support
//!
//! This module provides a single NumArray class that handles all numeric types internally,
//! eliminating the need for separate IntArray and BoolArray classes.

use groggy::storage::array::{BaseArray, BoolArray, NumArray};
use groggy::types::AttrValue;
use pyo3::prelude::*;

/// Unified statistical array supporting all numeric types internally
#[pyclass(name = "NumArray", unsendable)]
#[derive(Clone)]
pub struct PyNumArray {
    inner: NumericArrayData,
}

/// Internal storage supporting all numeric types
#[derive(Clone)]
enum NumericArrayData {
    Bool(BoolArray),
    Int32(NumArray<i32>),
    Int64(NumArray<i64>),
    Float32(NumArray<f32>),
    Float64(NumArray<f64>),
}

impl NumericArrayData {
    /// Get the string representation of the dtype
    fn dtype(&self) -> &'static str {
        match self {
            NumericArrayData::Bool(_) => "bool",
            NumericArrayData::Int32(_) => "int32",
            NumericArrayData::Int64(_) => "int64",
            NumericArrayData::Float32(_) => "float32",
            NumericArrayData::Float64(_) => "float64",
        }
    }

    /// Get the length of the array
    fn len(&self) -> usize {
        match self {
            NumericArrayData::Bool(arr) => arr.len(),
            NumericArrayData::Int32(arr) => arr.len(),
            NumericArrayData::Int64(arr) => arr.len(),
            NumericArrayData::Float32(arr) => arr.len(),
            NumericArrayData::Float64(arr) => arr.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to float64 for statistical operations
    fn to_float64_vec(&self) -> Vec<f64> {
        match self {
            NumericArrayData::Bool(arr) => arr.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
            NumericArrayData::Int32(arr) => arr.iter().map(|&x| x as f64).collect(),
            NumericArrayData::Int64(arr) => arr.iter().map(|&x| x as f64).collect(),
            NumericArrayData::Float32(arr) => arr.iter().map(|&x| x as f64).collect(),
            NumericArrayData::Float64(arr) => arr.iter().copied().collect(),
        }
    }

    /// Convert to int64 vector
    fn to_int64_vec(&self) -> Vec<i64> {
        match self {
            NumericArrayData::Bool(arr) => {
                arr.iter().map(|&b| if b { 1i64 } else { 0i64 }).collect()
            }
            NumericArrayData::Int32(arr) => arr.iter().map(|&x| x as i64).collect(),
            NumericArrayData::Int64(arr) => arr.iter().copied().collect(),
            NumericArrayData::Float32(arr) => arr.iter().map(|&x| x as i64).collect(),
            NumericArrayData::Float64(arr) => arr.iter().map(|&x| x as i64).collect(),
        }
    }

    /// Convert to bool vector
    fn to_bool_vec(&self) -> Vec<bool> {
        match self {
            NumericArrayData::Bool(arr) => arr.iter().copied().collect(),
            NumericArrayData::Int32(arr) => arr.iter().map(|&x| x != 0).collect(),
            NumericArrayData::Int64(arr) => arr.iter().map(|&x| x != 0).collect(),
            NumericArrayData::Float32(arr) => arr.iter().map(|&x| x != 0.0).collect(),
            NumericArrayData::Float64(arr) => arr.iter().map(|&x| x != 0.0).collect(),
        }
    }

    /// Get element as a Python object
    fn get_element(&self, index: usize, py: Python) -> Option<PyObject> {
        match self {
            NumericArrayData::Bool(arr) => arr.get(index).map(|b| b.to_object(py)),
            NumericArrayData::Int32(arr) => arr.get(index).map(|&x| x.to_object(py)),
            NumericArrayData::Int64(arr) => arr.get(index).map(|&x| x.to_object(py)),
            NumericArrayData::Float32(arr) => arr.get(index).map(|&x| x.to_object(py)),
            NumericArrayData::Float64(arr) => arr.get(index).map(|&x| x.to_object(py)),
        }
    }

    /// Convert to Python list
    fn to_list(&self, py: Python) -> PyResult<PyObject> {
        match self {
            NumericArrayData::Bool(arr) => {
                let list: Vec<bool> = arr.iter().copied().collect();
                Ok(list.to_object(py))
            }
            NumericArrayData::Int32(arr) => {
                let list: Vec<i32> = arr.iter().copied().collect();
                Ok(list.to_object(py))
            }
            NumericArrayData::Int64(arr) => {
                let list: Vec<i64> = arr.iter().copied().collect();
                Ok(list.to_object(py))
            }
            NumericArrayData::Float32(arr) => {
                let list: Vec<f32> = arr.iter().copied().collect();
                Ok(list.to_object(py))
            }
            NumericArrayData::Float64(arr) => {
                let list: Vec<f64> = arr.iter().copied().collect();
                Ok(list.to_object(py))
            }
        }
    }

    /// Check whether the array contains the provided value
    fn contains_py(&self, item: &PyAny) -> PyResult<bool> {
        match self {
            NumericArrayData::Bool(arr) => {
                let value = item.extract::<bool>()?;
                Ok(arr.iter().copied().any(|candidate| candidate == value))
            }
            NumericArrayData::Int32(arr) => {
                let raw = item.extract::<i64>()?;
                let value = i32::try_from(raw).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Value {} out of range for int32",
                        raw
                    ))
                })?;
                Ok(arr.iter().copied().any(|candidate| candidate == value))
            }
            NumericArrayData::Int64(arr) => {
                let value = item.extract::<i64>()?;
                Ok(arr.iter().copied().any(|candidate| candidate == value))
            }
            NumericArrayData::Float32(arr) => {
                let value = item.extract::<f64>()? as f32;
                Ok(arr
                    .iter()
                    .copied()
                    .any(|candidate| (candidate - value).abs() <= f32::EPSILON))
            }
            NumericArrayData::Float64(arr) => {
                let value = item.extract::<f64>()?;
                Ok(arr
                    .iter()
                    .copied()
                    .any(|candidate| (candidate - value).abs() <= f64::EPSILON))
            }
        }
    }
}

impl PyNumArray {
    /// Create new NumArray with automatic type inference
    pub fn new_with_inference(values: &[PyObject], py: Python) -> PyResult<Self> {
        // Try to infer the best dtype from the values
        let mut has_float = false;
        let mut has_int = false;
        let mut has_bool = false;
        let mut max_int_size = 0;

        for value in values {
            if value.extract::<bool>(py).is_ok() {
                has_bool = true;
            } else if let Ok(i_val) = value.extract::<i64>(py) {
                has_int = true;
                if i_val.abs() > i32::MAX as i64 {
                    max_int_size = 64;
                } else if max_int_size < 32 {
                    max_int_size = 32;
                }
            } else if value.extract::<f64>(py).is_ok() {
                has_float = true;
            }
        }

        // Determine the best dtype
        let dtype = if has_float {
            "float64"
        } else if has_int {
            if max_int_size == 64 {
                "int64"
            } else {
                "int32"
            }
        } else if has_bool {
            "bool"
        } else {
            "float64" // Default fallback
        };

        Self::new_with_dtype(values, dtype, py)
    }

    /// Create new NumArray with explicit dtype
    pub fn new_with_dtype(values: &[PyObject], dtype: &str, py: Python) -> PyResult<Self> {
        match dtype {
            "bool" => {
                let bool_values: Result<Vec<bool>, _> =
                    values.iter().map(|v| v.extract::<bool>(py)).collect();
                match bool_values {
                    Ok(vals) => Ok(PyNumArray {
                        inner: NumericArrayData::Bool(BoolArray::new(vals)),
                    }),
                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                        "Cannot convert values to bool dtype",
                    )),
                }
            }
            "int32" => {
                let int_values: Result<Vec<i32>, _> = values
                    .iter()
                    .map(|v| {
                        if let Ok(i) = v.extract::<i32>(py) {
                            Ok(i)
                        } else if let Ok(i) = v.extract::<i64>(py) {
                            Ok(i as i32)
                        } else if let Ok(f) = v.extract::<f64>(py) {
                            Ok(f.round() as i32)
                        } else if let Ok(b) = v.extract::<bool>(py) {
                            Ok(if b { 1 } else { 0 })
                        } else {
                            Err(())
                        }
                    })
                    .collect();
                match int_values {
                    Ok(vals) => Ok(PyNumArray {
                        inner: NumericArrayData::Int32(NumArray::new(vals)),
                    }),
                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                        "Cannot convert values to int32 dtype",
                    )),
                }
            }
            "int64" => {
                let int_values: Result<Vec<i64>, _> = values
                    .iter()
                    .map(|v| {
                        if let Ok(i) = v.extract::<i64>(py) {
                            Ok(i)
                        } else if let Ok(i) = v.extract::<i32>(py) {
                            Ok(i as i64)
                        } else if let Ok(f) = v.extract::<f64>(py) {
                            Ok(f.round() as i64)
                        } else if let Ok(b) = v.extract::<bool>(py) {
                            Ok(if b { 1 } else { 0 })
                        } else {
                            Err(())
                        }
                    })
                    .collect();
                match int_values {
                    Ok(vals) => Ok(PyNumArray {
                        inner: NumericArrayData::Int64(NumArray::new(vals)),
                    }),
                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                        "Cannot convert values to int64 dtype",
                    )),
                }
            }
            "float32" => {
                let float_values: Result<Vec<f32>, _> = values
                    .iter()
                    .map(|v| {
                        if let Ok(f) = v.extract::<f32>(py) {
                            Ok(f)
                        } else if let Ok(f) = v.extract::<f64>(py) {
                            Ok(f as f32)
                        } else if let Ok(i) = v.extract::<i64>(py) {
                            Ok(i as f32)
                        } else if let Ok(b) = v.extract::<bool>(py) {
                            Ok(if b { 1.0 } else { 0.0 })
                        } else {
                            Err(())
                        }
                    })
                    .collect();
                match float_values {
                    Ok(vals) => Ok(PyNumArray {
                        inner: NumericArrayData::Float32(NumArray::new(vals)),
                    }),
                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                        "Cannot convert values to float32 dtype",
                    )),
                }
            }
            "float64" => {
                let float_values: Result<Vec<f64>, _> = values
                    .iter()
                    .map(|v| {
                        if let Ok(f) = v.extract::<f64>(py) {
                            Ok(f)
                        } else if let Ok(f) = v.extract::<f32>(py) {
                            Ok(f as f64)
                        } else if let Ok(i) = v.extract::<i64>(py) {
                            Ok(i as f64)
                        } else if let Ok(b) = v.extract::<bool>(py) {
                            Ok(if b { 1.0 } else { 0.0 })
                        } else {
                            Err(())
                        }
                    })
                    .collect();
                match float_values {
                    Ok(vals) => Ok(PyNumArray {
                        inner: NumericArrayData::Float64(NumArray::new(vals)),
                    }),
                    Err(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                        "Cannot convert values to float64 dtype",
                    )),
                }
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype: {}. Supported: 'bool', 'int32', 'int64', 'float32', 'float64'",
                dtype
            ))),
        }
    }

    /// Create from AttrValues with type inference
    pub fn from_attr_values(attr_values: Vec<AttrValue>) -> PyResult<Self> {
        // Convert AttrValues to determine best internal type
        let mut has_float = false;
        let mut has_large_int = false;

        for attr in &attr_values {
            match attr {
                AttrValue::Float(_) => has_float = true,
                AttrValue::Int(i) if i.abs() > i32::MAX as i64 => has_large_int = true,
                _ => {}
            }
        }

        if has_float {
            // Use float64 for any float data
            let mut float_values = Vec::new();
            for attr in attr_values {
                match attr {
                    AttrValue::Int(i) => float_values.push(i as f64),
                    AttrValue::SmallInt(i) => float_values.push(i as f64),
                    AttrValue::Float(f) => float_values.push(f as f64),
                    AttrValue::Bool(b) => float_values.push(if b { 1.0 } else { 0.0 }),
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "NumArray can only contain numerical values",
                        ))
                    }
                }
            }
            Ok(PyNumArray {
                inner: NumericArrayData::Float64(NumArray::new(float_values)),
            })
        } else if has_large_int {
            // Use int64 for large integers
            let mut int_values = Vec::new();
            for attr in attr_values {
                match attr {
                    AttrValue::Int(i) => int_values.push(i),
                    AttrValue::SmallInt(i) => int_values.push(i as i64),
                    AttrValue::Bool(b) => int_values.push(if b { 1 } else { 0 }),
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "NumArray can only contain numerical values",
                        ))
                    }
                }
            }
            Ok(PyNumArray {
                inner: NumericArrayData::Int64(NumArray::new(int_values)),
            })
        } else {
            // Use int32 for small integers
            let mut int_values = Vec::new();
            for attr in attr_values {
                match attr {
                    AttrValue::Int(i) => int_values.push(i as i32),
                    AttrValue::SmallInt(i) => int_values.push(i),
                    AttrValue::Bool(b) => int_values.push(if b { 1 } else { 0 }),
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "NumArray can only contain numerical values",
                        ))
                    }
                }
            }
            Ok(PyNumArray {
                inner: NumericArrayData::Int32(NumArray::new(int_values)),
            })
        }
    }

    /// Create a NumArray optimized for boolean operations
    pub fn new_bool(values: Vec<bool>) -> Self {
        PyNumArray {
            inner: NumericArrayData::Bool(BoolArray::new(values)),
        }
    }

    /// Create a NumArray optimized for integer operations
    pub fn new_int64(values: Vec<i64>) -> Self {
        PyNumArray {
            inner: NumericArrayData::Int64(NumArray::new(values)),
        }
    }

    /// Create a NumArray optimized for float operations
    pub fn new_float64(values: Vec<f64>) -> Self {
        PyNumArray {
            inner: NumericArrayData::Float64(NumArray::new(values)),
        }
    }

    /// Legacy constructor for backward compatibility (PyNumArray::new)
    pub fn new(values: Vec<f64>) -> Self {
        Self::new_float64(values)
    }

    /// Create from node IDs (backward compatibility for PyIntArray::from_node_ids)
    pub fn from_node_ids(node_ids: Vec<usize>) -> Self {
        let values: Vec<i64> = node_ids.into_iter().map(|id| id as i64).collect();
        Self::new_int64(values)
    }

    /// Get access to the underlying data for backward compatibility
    /// This is only used internally by the FFI layer
    pub fn as_float64_array(&self) -> Option<&NumArray<f64>> {
        match &self.inner {
            NumericArrayData::Float64(arr) => Some(arr),
            _ => None,
        }
    }

    /// Get length for backward compatibility
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get values as f64 vector (used by matrix operations)
    pub fn to_float64_vec(&self) -> Vec<f64> {
        self.inner.to_float64_vec()
    }

    /// Get element at index as f64 (used by matrix operations)
    pub fn get_f64(&self, index: usize) -> Option<f64> {
        let f64_values = self.inner.to_float64_vec();
        f64_values.get(index).copied()
    }

    /// Get the dtype as a string (public method for FFI utilities)
    pub fn get_dtype(&self) -> &str {
        self.inner.dtype()
    }

    /// Convert to Python list (public method for FFI utilities)
    pub fn get_list(&self, py: Python) -> PyResult<PyObject> {
        self.inner.to_list(py)
    }

    /// Convert to int64 vector (public method for indexing)
    pub fn to_int64_vec(&self) -> Vec<i64> {
        self.inner.to_int64_vec()
    }

    /// Convert to bool vector (public method for indexing)
    pub fn to_bool_vec(&self) -> Vec<bool> {
        self.inner.to_bool_vec()
    }
}

#[pymethods]
impl PyNumArray {
    /// Create new NumArray from a list of values with optional dtype
    #[new]
    #[pyo3(signature = (data, *, dtype = None))]
    fn __new__(data: Vec<PyObject>, dtype: Option<&str>, py: Python) -> PyResult<Self> {
        match dtype {
            Some(dt) => Self::new_with_dtype(&data, dt, py),
            None => Self::new_with_inference(&data, py),
        }
    }

    /// Get the dtype of this array
    #[getter]
    fn dtype(&self) -> &str {
        self.inner.dtype()
    }

    /// Get the number of elements
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check whether the array contains the provided value
    fn contains(&self, item: &PyAny) -> PyResult<bool> {
        self.inner.contains_py(item)
    }

    /// Get element at index or perform advanced indexing
    fn __getitem__(&self, py: Python, index: &PyAny) -> PyResult<PyObject> {
        use crate::ffi::utils::indexing::python_index_to_slice_index;
        use groggy::storage::array::AdvancedIndexing;

        // Handle simple integer case directly for performance
        if let Ok(int_val) = index.extract::<isize>() {
            let len = self.inner.len() as isize;
            let actual_index = if int_val < 0 {
                (len + int_val) as usize
            } else {
                int_val as usize
            };

            return self.inner.get_element(actual_index, py).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!("Index {} out of range", int_val))
            });
        }

        // Handle advanced indexing - delegate to the appropriate internal array
        let slice_index = python_index_to_slice_index(py, index)?;

        match &self.inner {
            NumericArrayData::Bool(arr) => match arr.get_slice(&slice_index) {
                Ok(sliced_array) => {
                    let py_array = PyNumArray {
                        inner: NumericArrayData::Bool(sliced_array),
                    };
                    Ok(py_array.into_py(py))
                }
                Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
            },
            NumericArrayData::Int32(arr) => match arr.get_slice(&slice_index) {
                Ok(sliced_array) => {
                    let py_array = PyNumArray {
                        inner: NumericArrayData::Int32(sliced_array),
                    };
                    Ok(py_array.into_py(py))
                }
                Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
            },
            NumericArrayData::Int64(arr) => match arr.get_slice(&slice_index) {
                Ok(sliced_array) => {
                    let py_array = PyNumArray {
                        inner: NumericArrayData::Int64(sliced_array),
                    };
                    Ok(py_array.into_py(py))
                }
                Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
            },
            NumericArrayData::Float32(arr) => match arr.get_slice(&slice_index) {
                Ok(sliced_array) => {
                    let py_array = PyNumArray {
                        inner: NumericArrayData::Float32(sliced_array),
                    };
                    Ok(py_array.into_py(py))
                }
                Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
            },
            NumericArrayData::Float64(arr) => match arr.get_slice(&slice_index) {
                Ok(sliced_array) => {
                    let py_array = PyNumArray {
                        inner: NumericArrayData::Float64(sliced_array),
                    };
                    Ok(py_array.into_py(py))
                }
                Err(e) => Err(pyo3::exceptions::PyIndexError::new_err(format!("{}", e))),
            },
        }
    }

    /// Convert to Python list
    fn to_list(&self, py: Python) -> PyResult<PyObject> {
        self.inner.to_list(py)
    }

    /// String representation
    fn __repr__(&self) -> String {
        let len = self.inner.len();
        let dtype = self.inner.dtype();

        if len == 0 {
            return format!("NumArray(dtype: {})\n[]", dtype);
        }

        let max_display = 10;
        let show_all = len <= max_display;

        let mut output = format!("NumArray(dtype: {})\n", dtype);

        // Get a sample of values for display
        let sample_values = match &self.inner {
            NumericArrayData::Bool(arr) => {
                if show_all {
                    arr.iter()
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {}", i, v))
                        .collect::<Vec<_>>()
                } else {
                    let mut vals = arr
                        .iter()
                        .take(5)
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {}", i, v))
                        .collect::<Vec<_>>();
                    vals.push("      ...".to_string());
                    let skip_count = len.saturating_sub(3);
                    vals.extend(
                        arr.iter()
                            .skip(skip_count)
                            .enumerate()
                            .map(|(i, &v)| format!("[{:>3}] {}", skip_count + i, v)),
                    );
                    vals
                }
            }
            NumericArrayData::Int32(arr) => {
                if show_all {
                    arr.iter()
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {}", i, v))
                        .collect::<Vec<_>>()
                } else {
                    let mut vals = arr
                        .iter()
                        .take(5)
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {}", i, v))
                        .collect::<Vec<_>>();
                    vals.push("      ...".to_string());
                    let skip_count = len.saturating_sub(3);
                    vals.extend(
                        arr.iter()
                            .skip(skip_count)
                            .enumerate()
                            .map(|(i, &v)| format!("[{:>3}] {}", skip_count + i, v)),
                    );
                    vals
                }
            }
            NumericArrayData::Int64(arr) => {
                if show_all {
                    arr.iter()
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {}", i, v))
                        .collect::<Vec<_>>()
                } else {
                    let mut vals = arr
                        .iter()
                        .take(5)
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {}", i, v))
                        .collect::<Vec<_>>();
                    vals.push("      ...".to_string());
                    let skip_count = len.saturating_sub(3);
                    vals.extend(
                        arr.iter()
                            .skip(skip_count)
                            .enumerate()
                            .map(|(i, &v)| format!("[{:>3}] {}", skip_count + i, v)),
                    );
                    vals
                }
            }
            NumericArrayData::Float32(arr) => {
                if show_all {
                    arr.iter()
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {:.6}", i, v))
                        .collect::<Vec<_>>()
                } else {
                    let mut vals = arr
                        .iter()
                        .take(5)
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {:.6}", i, v))
                        .collect::<Vec<_>>();
                    vals.push("      ...".to_string());
                    let skip_count = len.saturating_sub(3);
                    vals.extend(
                        arr.iter()
                            .skip(skip_count)
                            .enumerate()
                            .map(|(i, &v)| format!("[{:>3}] {:.6}", skip_count + i, v)),
                    );
                    vals
                }
            }
            NumericArrayData::Float64(arr) => {
                if show_all {
                    arr.iter()
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {:.6}", i, v))
                        .collect::<Vec<_>>()
                } else {
                    let mut vals = arr
                        .iter()
                        .take(5)
                        .enumerate()
                        .map(|(i, &v)| format!("[{:>3}] {:.6}", i, v))
                        .collect::<Vec<_>>();
                    vals.push("      ...".to_string());
                    let skip_count = len.saturating_sub(3);
                    vals.extend(
                        arr.iter()
                            .skip(skip_count)
                            .enumerate()
                            .map(|(i, &v)| format!("[{:>3}] {:.6}", skip_count + i, v)),
                    );
                    vals
                }
            }
        };

        for val in sample_values {
            output.push_str(&val);
            output.push('\n');
        }

        output.trim_end().to_string()
    }

    /// Convert to different numeric type
    fn to_type(&self, dtype: &str) -> PyResult<PyObject> {
        match dtype {
            "bool" => {
                let bool_values = match &self.inner {
                    NumericArrayData::Bool(arr) => arr.iter().copied().collect(),
                    NumericArrayData::Int32(arr) => arr.iter().map(|&x| x != 0).collect(),
                    NumericArrayData::Int64(arr) => arr.iter().map(|&x| x != 0).collect(),
                    NumericArrayData::Float32(arr) => arr.iter().map(|&x| x != 0.0).collect(),
                    NumericArrayData::Float64(arr) => arr.iter().map(|&x| x != 0.0).collect(),
                };
                let result = PyNumArray::new_bool(bool_values);
                Python::with_gil(|py| Ok(result.into_py(py)))
            },
            "int32" => {
                let int_values = match &self.inner {
                    NumericArrayData::Bool(arr) => arr.iter().map(|&x| if x { 1i32 } else { 0i32 }).collect(),
                    NumericArrayData::Int32(arr) => arr.iter().copied().collect(),
                    NumericArrayData::Int64(arr) => arr.iter().map(|&x| x as i32).collect(),
                    NumericArrayData::Float32(arr) => arr.iter().map(|&x| x.round() as i32).collect(),
                    NumericArrayData::Float64(arr) => arr.iter().map(|&x| x.round() as i32).collect(),
                };
                let result = PyNumArray {
                    inner: NumericArrayData::Int32(NumArray::new(int_values))
                };
                Python::with_gil(|py| Ok(result.into_py(py)))
            },
            "int64" => {
                let int_values = match &self.inner {
                    NumericArrayData::Bool(arr) => arr.iter().map(|&x| if x { 1i64 } else { 0i64 }).collect(),
                    NumericArrayData::Int32(arr) => arr.iter().map(|&x| x as i64).collect(),
                    NumericArrayData::Int64(arr) => arr.iter().copied().collect(),
                    NumericArrayData::Float32(arr) => arr.iter().map(|&x| x.round() as i64).collect(),
                    NumericArrayData::Float64(arr) => arr.iter().map(|&x| x.round() as i64).collect(),
                };
                let result = PyNumArray::new_int64(int_values);
                Python::with_gil(|py| Ok(result.into_py(py)))
            },
            "float32" => {
                let float_values = match &self.inner {
                    NumericArrayData::Bool(arr) => arr.iter().map(|&x| if x { 1.0f32 } else { 0.0f32 }).collect(),
                    NumericArrayData::Int32(arr) => arr.iter().map(|&x| x as f32).collect(),
                    NumericArrayData::Int64(arr) => arr.iter().map(|&x| x as f32).collect(),
                    NumericArrayData::Float32(arr) => arr.iter().copied().collect(),
                    NumericArrayData::Float64(arr) => arr.iter().map(|&x| x as f32).collect(),
                };
                let result = PyNumArray {
                    inner: NumericArrayData::Float32(NumArray::new(float_values))
                };
                Python::with_gil(|py| Ok(result.into_py(py)))
            },
            "float64" => {
                let float_values = self.inner.to_float64_vec();
                let result = PyNumArray::new_float64(float_values);
                Python::with_gil(|py| Ok(result.into_py(py)))
            },
            "basearray" => {
                let attr_values: Vec<AttrValue> = match &self.inner {
                    NumericArrayData::Bool(arr) => arr.iter().map(|&x| AttrValue::Bool(x)).collect(),
                    NumericArrayData::Int32(arr) => arr.iter().map(|&x| AttrValue::SmallInt(x)).collect(),
                    NumericArrayData::Int64(arr) => arr.iter().map(|&x| AttrValue::Int(x)).collect(),
                    NumericArrayData::Float32(arr) => arr.iter().map(|&x| AttrValue::Float(x)).collect(),
                    NumericArrayData::Float64(arr) => arr.iter().map(|&x| AttrValue::Float(x as f32)).collect(),
                };
                let base_array = crate::ffi::storage::array::PyBaseArray {
                    inner: BaseArray::new(attr_values)
                };
                Python::with_gil(|py| Ok(base_array.into_py(py)))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported dtype: {}. Supported: 'bool', 'int32', 'int64', 'float32', 'float64', 'basearray'", dtype)
            ))
        }
    }

    // Statistical Methods (work on all numeric types)

    /// Calculate the mean (average)
    fn mean(&self) -> Option<f64> {
        let values = self.inner.to_float64_vec();
        if values.is_empty() {
            None
        } else {
            let sum: f64 = values.iter().sum();
            Some(sum / values.len() as f64)
        }
    }

    /// Calculate the sum
    fn sum(&self) -> f64 {
        let values = self.inner.to_float64_vec();
        values.iter().sum()
    }

    /// Find the minimum value
    fn min(&self) -> Option<f64> {
        let values = self.inner.to_float64_vec();
        values.iter().copied().fold(None, |acc, x| match acc {
            None => Some(x),
            Some(y) => Some(x.min(y)),
        })
    }

    /// Find the maximum value
    fn max(&self) -> Option<f64> {
        let values = self.inner.to_float64_vec();
        values.iter().copied().fold(None, |acc, x| match acc {
            None => Some(x),
            Some(y) => Some(x.max(y)),
        })
    }

    /// Calculate standard deviation
    fn std(&self) -> Option<f64> {
        let values = self.inner.to_float64_vec();
        if values.len() <= 1 {
            return Some(0.0);
        }

        let mean = self.mean()?;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        Some(variance.sqrt())
    }

    /// Calculate variance
    fn var(&self) -> Option<f64> {
        let values = self.inner.to_float64_vec();
        if values.len() <= 1 {
            return Some(0.0);
        }

        let mean = self.mean()?;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        Some(variance)
    }

    /// Get first element
    fn first(&self, py: Python) -> Option<PyObject> {
        self.inner.get_element(0, py)
    }

    /// Get last element  
    fn last(&self, py: Python) -> Option<PyObject> {
        let len = self.inner.len();
        if len > 0 {
            self.inner.get_element(len - 1, py)
        } else {
            None
        }
    }

    /// Count non-null values (for NumArray, all values are non-null by default)
    fn count(&self) -> usize {
        self.inner.len()
    }

    /// Detect missing/null values in the array
    /// Returns a boolean NumArray where True indicates a null value
    /// Similar to pandas Series.isna()
    /// For NumArray, all values are non-null, so this returns all False
    fn isna(&self) -> PyNumArray {
        let len = self.inner.len();
        let bool_values = vec![false; len];
        PyNumArray::new_bool(bool_values)
    }

    /// Detect non-missing/non-null values in the array
    /// Returns a boolean NumArray where True indicates a non-null value
    /// Similar to pandas Series.notna()
    /// For NumArray, all values are non-null, so this returns all True
    fn notna(&self) -> PyNumArray {
        let len = self.inner.len();
        let bool_values = vec![true; len];
        PyNumArray::new_bool(bool_values)
    }

    /// Count unique values
    fn nunique(&self) -> usize {
        // Convert to string representation for uniqueness check
        use std::collections::HashSet;
        let mut unique = HashSet::new();

        for i in 0..self.inner.len() {
            Python::with_gil(|py| {
                if let Some(val) = self.inner.get_element(i, py) {
                    unique.insert(format!("{:?}", val));
                }
            });
        }

        unique.len()
    }

    /// Get unique values as a new NumArray
    fn unique(&self) -> PyResult<PyNumArray> {
        use std::collections::HashSet;
        let mut unique_values = Vec::new();

        // For numeric types, we can use the actual values for uniqueness
        match &self.inner {
            NumericArrayData::Float64(_) => {
                let mut seen = HashSet::new();
                let values = self.inner.to_float64_vec();
                for val in values {
                    if seen.insert(val.to_bits()) {
                        // Use bits representation for float comparison
                        unique_values.push(val);
                    }
                }
            }
            NumericArrayData::Int64(_) => {
                let mut seen = HashSet::new();
                let values = self.inner.to_int64_vec();
                for val in values {
                    if seen.insert(val) {
                        unique_values.push(val as f64); // Convert to f64 for NumArray
                    }
                }
            }
            NumericArrayData::Bool(_) => {
                let mut seen = HashSet::new();
                let values = self.inner.to_bool_vec();
                for val in values {
                    if seen.insert(val) {
                        unique_values.push(if val { 1.0 } else { 0.0 }); // Convert bool to f64
                    }
                }
            }
            _ => {
                // For other types, fall back to string representation
                let mut seen = HashSet::new();
                for i in 0..self.inner.len() {
                    Python::with_gil(|py| {
                        if let Some(val) = self.inner.get_element(i, py) {
                            let str_repr = format!("{:?}", val);
                            if seen.insert(str_repr.clone()) {
                                // Try to convert back to f64, use index if not possible
                                let numeric_val = str_repr.parse::<f64>().unwrap_or(i as f64);
                                unique_values.push(numeric_val);
                            }
                        }
                    });
                }
            }
        }

        Ok(PyNumArray::new(unique_values))
    }

    // Comparison Operators (return BoolArray)

    /// Greater than comparison (>) - returns NumArray with bool dtype
    fn __gt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let scalar = other.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("NumArray comparison requires a numeric value")
        })?;

        let values = self.inner.to_float64_vec();
        let result: Vec<bool> = values.iter().map(|&x| x > scalar).collect();
        let bool_array = PyNumArray::new_bool(result);
        Ok(bool_array.into_py(py))
    }

    /// Less than comparison (<) - returns NumArray with bool dtype
    fn __lt__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let scalar = other.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("NumArray comparison requires a numeric value")
        })?;

        let values = self.inner.to_float64_vec();
        let result: Vec<bool> = values.iter().map(|&x| x < scalar).collect();
        let bool_array = PyNumArray::new_bool(result);
        Ok(bool_array.into_py(py))
    }

    /// Greater than or equal comparison (>=) - returns NumArray with bool dtype
    fn __ge__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let scalar = other.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("NumArray comparison requires a numeric value")
        })?;

        let values = self.inner.to_float64_vec();
        let result: Vec<bool> = values.iter().map(|&x| x >= scalar).collect();
        let bool_array = PyNumArray::new_bool(result);
        Ok(bool_array.into_py(py))
    }

    /// Less than or equal comparison (<=) - returns NumArray with bool dtype  
    fn __le__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let scalar = other.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("NumArray comparison requires a numeric value")
        })?;

        let values = self.inner.to_float64_vec();
        let result: Vec<bool> = values.iter().map(|&x| x <= scalar).collect();
        let bool_array = PyNumArray::new_bool(result);
        Ok(bool_array.into_py(py))
    }

    /// Equality comparison (==) - returns NumArray with bool dtype
    fn __eq__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let scalar = other.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("NumArray comparison requires a numeric value")
        })?;

        let values = self.inner.to_float64_vec();
        let result: Vec<bool> = values
            .iter()
            .map(|&x| (x - scalar).abs() < f64::EPSILON)
            .collect();
        let bool_array = PyNumArray::new_bool(result);
        Ok(bool_array.into_py(py))
    }

    /// Not equal comparison (!=) - returns NumArray with bool dtype
    fn __ne__(&self, py: Python, other: &PyAny) -> PyResult<PyObject> {
        let scalar = other.extract::<f64>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("NumArray comparison requires a numeric value")
        })?;

        let values = self.inner.to_float64_vec();
        let result: Vec<bool> = values
            .iter()
            .map(|&x| (x - scalar).abs() >= f64::EPSILON)
            .collect();
        let bool_array = PyNumArray::new_bool(result);
        Ok(bool_array.into_py(py))
    }

    /// Matrix Integration: Reshape NumArray into a GraphMatrix
    fn reshape(&self, py: Python, rows: usize, cols: usize) -> PyResult<PyObject> {
        // Convert to float64 for matrix operations (matrices are typically float)
        let float_array = self.to_type("float64")?;
        let float_array = float_array.extract::<PyNumArray>(py)?;

        let matrix = crate::ffi::storage::matrix::PyGraphMatrix::from_flattened(
            py,
            &float_array,
            rows,
            cols,
        )?;
        Ok(matrix.into_py(py))
    }

    /// Create iterator
    fn __iter__(slf: PyRef<Self>) -> PyNumArrayIterator {
        PyNumArrayIterator {
            array: slf.into(),
            index: 0,
        }
    }
}

/// Iterator for PyNumArray
#[pyclass]
pub struct PyNumArrayIterator {
    array: Py<PyNumArray>,
    index: usize,
}

#[pymethods]
impl PyNumArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        let array = self.array.borrow(py);
        if self.index < array.inner.len() {
            let result = array.inner.get_element(self.index, py);
            self.index += 1;
            Ok(result)
        } else {
            Ok(None)
        }
    }
}

// Backward Compatibility Aliases and Factory Functions

/// Create NumArray with int64 dtype (replaces IntArray)
#[pyfunction]
/// Future feature: direct array construction
#[allow(dead_code)]
pub fn int_array(values: Vec<i64>) -> PyNumArray {
    PyNumArray::new_int64(values)
}

/// Create NumArray with bool dtype (replaces BoolArray)  
#[pyfunction]
/// Future feature: direct array construction
#[allow(dead_code)]
pub fn bool_array(values: Vec<bool>) -> PyNumArray {
    PyNumArray::new_bool(values)
}

/// Create NumArray with float64 dtype
#[pyfunction]
/// Future feature: direct array construction
#[allow(dead_code)]
pub fn num_array(values: Vec<f64>) -> PyNumArray {
    PyNumArray::new_float64(values)
}

// Type aliases for backward compatibility
pub type PyIntArray = PyNumArray; // Deprecated - use NumArray(dtype='int64')
pub type PyStatsArray = PyNumArray; // Backward compatibility alias
pub type PyStatsArrayIterator = PyNumArrayIterator;
