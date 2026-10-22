//! Unified NumArray - Statistical array operations with comprehensive dtype support
//!
//! This module provides a single NumArray class that handles all numeric types internally,
//! eliminating the need for separate IntArray and BoolArray classes.

use groggy::storage::array::{BaseArray, BoolArray, NumArray};
use groggy::types::AttrValue;
use pyo3::prelude::*;

use groggy::storage::array::numeric_arithmetic::{
    self as numeric, ConversionPolicy, NumericBinary, NumericInput,
};
pub(crate) fn input(value: &PyAny) -> PyResult<NumericInput> {
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(NumericInput::Bool(value.extract()?));
    }
    if value.is_instance_of::<pyo3::types::PyLong>() {
        return value.extract::<i128>().map(NumericInput::Integer);
    }
    if let Ok(value) = value.extract::<i64>() {
        return Ok(NumericInput::Integer(i128::from(value)));
    }
    value.extract::<f64>().map(NumericInput::Float)
}
fn numeric_error(error: groggy::GraphError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}

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
    fn inputs(&self) -> Vec<NumericInput> {
        match self {
            Self::Bool(a) => numeric::inputs(a.as_slice()),
            Self::Int32(a) => numeric::inputs(a.as_slice()),
            Self::Int64(a) => numeric::inputs(a.as_slice()),
            Self::Float32(a) => numeric::inputs(a.as_slice()),
            Self::Float64(a) => numeric::inputs(a.as_slice()),
        }
    }
    fn converted(values: &[NumericInput], dtype: &str, policy: ConversionPolicy) -> PyResult<Self> {
        match dtype {
            "bool" => NumArray::<bool>::from_inputs(values, policy)
                .map(|a| Self::Bool(BoolArray::new(a.as_slice().to_vec()))),
            "int32" => NumArray::<i32>::from_inputs(values, policy).map(Self::Int32),
            "int64" => NumArray::<i64>::from_inputs(values, policy).map(Self::Int64),
            "float32" => NumArray::<f32>::from_inputs(values, policy).map(Self::Float32),
            "float64" => NumArray::<f64>::from_inputs(values, policy).map(Self::Float64),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported numeric dtype",
                ))
            }
        }
        .map_err(numeric_error)
    }
    fn binary(&self, other: &Self, op: NumericBinary) -> PyResult<Self> {
        match (self, other) {
            (Self::Int32(a), Self::Int32(b)) => a.checked_binary(b, op).map(Self::Int32),
            (Self::Int64(a), Self::Int64(b)) => a.checked_binary(b, op).map(Self::Int64),
            (Self::Float32(a), Self::Float32(b)) => a.checked_binary(b, op).map(Self::Float32),
            (Self::Float64(a), Self::Float64(b)) => a.checked_binary(b, op).map(Self::Float64),
            (Self::Bool(a), Self::Bool(b)) => NumArray::new(a.as_slice().to_vec())
                .checked_binary(&NumArray::new(b.as_slice().to_vec()), op)
                .map(|a| Self::Bool(BoolArray::new(a.as_slice().to_vec()))),
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Vector arithmetic requires matching dtypes; cast explicitly",
                ))
            }
        }
        .map_err(numeric_error)
    }
    fn scalar(&self, value: NumericInput, op: NumericBinary, reverse: bool) -> PyResult<Self> {
        match self {
            Self::Int32(a) => a.scalar_binary(value, op, reverse).map(Self::Int32),
            Self::Int64(a) => a.scalar_binary(value, op, reverse).map(Self::Int64),
            Self::Float32(a) => a.scalar_binary(value, op, reverse).map(Self::Float32),
            Self::Float64(a) => a.scalar_binary(value, op, reverse).map(Self::Float64),
            Self::Bool(a) => NumArray::new(a.as_slice().to_vec())
                .scalar_binary(value, op, reverse)
                .map(|a| Self::Bool(BoolArray::new(a.as_slice().to_vec()))),
        }
        .map_err(numeric_error)
    }
    fn scalar_divide(&self, value: NumericInput, reverse: bool) -> PyResult<Self> {
        match self {
            Self::Int32(a) => numeric::divide_scalar(a.as_slice(), value, reverse),
            Self::Int64(a) => numeric::divide_scalar(a.as_slice(), value, reverse),
            Self::Float32(a) => numeric::divide_scalar(a.as_slice(), value, reverse),
            Self::Float64(a) => numeric::divide_scalar(a.as_slice(), value, reverse),
            Self::Bool(a) => numeric::divide_scalar(a.as_slice(), value, reverse),
        }
        .map(Self::Float64)
        .map_err(numeric_error)
    }
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

    /// Exact signed integer positions; no implicit truncation or saturation.
    fn to_int64_vec(&self) -> PyResult<Vec<i64>> {
        match self {
            Self::Bool(a) => numeric::integer_indices(a.as_slice()),
            Self::Int32(a) => numeric::integer_indices(a.as_slice()),
            Self::Int64(a) => numeric::integer_indices(a.as_slice()),
            Self::Float32(a) => numeric::integer_indices(a.as_slice()),
            Self::Float64(a) => numeric::integer_indices(a.as_slice()),
        }
        .map_err(numeric_error)
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
        let scalar = input(item)?;
        Ok(
            numeric::compare_scalar(&self.inputs(), scalar, numeric::NumericComparison::Equal)
                .into_iter()
                .any(|v| v),
        )
    }
}

impl PyNumArray {
    fn comparison(&self, other: &PyAny, op: numeric::NumericComparison) -> PyResult<Self> {
        let values = if let Ok(other) = other.extract::<PyRef<PyNumArray>>() {
            numeric::compare(&self.inner.inputs(), &other.inner.inputs(), op)
                .map_err(numeric_error)?
        } else {
            numeric::compare_scalar(&self.inner.inputs(), input(other)?, op)
        };
        Ok(Self {
            inner: NumericArrayData::Bool(BoolArray::new(values)),
        })
    }
    fn binary_operand(&self, other: &PyAny, op: NumericBinary, reverse: bool) -> PyResult<Self> {
        let inner = if let Ok(other) = other.extract::<PyRef<PyNumArray>>() {
            if reverse {
                other.inner.binary(&self.inner, op)
            } else {
                self.inner.binary(&other.inner, op)
            }
        } else {
            self.inner.scalar(input(other)?, op, reverse)
        }?;
        Ok(Self { inner })
    }
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
                if i32::try_from(i_val).is_err() {
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
        if dtype == "bool" {
            return values
                .iter()
                .map(|v| v.extract::<bool>(py))
                .collect::<PyResult<Vec<_>>>()
                .map(|v| Self {
                    inner: NumericArrayData::Bool(BoolArray::new(v)),
                });
        }
        let values = values
            .iter()
            .map(|v| input(v.as_ref(py)))
            .collect::<PyResult<Vec<_>>>()?;
        NumericArrayData::converted(&values, dtype, ConversionPolicy::Round)
            .map(|inner| Self { inner })
    }

    /// Create from AttrValues with type inference
    pub fn from_attr_values(attr_values: Vec<AttrValue>) -> PyResult<Self> {
        let (dtype, values) = numeric::attribute_inputs(&attr_values)
            .map_err(|e| pyo3::exceptions::PyTypeError::new_err(e.to_string()))?;
        let dtype = match dtype {
            groggy::storage::advanced_matrix::DType::Float64 => "float64",
            groggy::storage::advanced_matrix::DType::Int64 => "int64",
            _ => "int32",
        };
        NumericArrayData::converted(&values, dtype, ConversionPolicy::Round)
            .map(|inner| Self { inner })
    }

    pub fn from_base_attributes(values: &[AttrValue]) -> PyResult<Self> {
        let (dtype, values) = numeric::promoted_attribute_inputs(values)
            .map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("BaseArray contains non-numeric or missing data and cannot be converted to numeric array: {e}")))?;
        let dtype = match dtype {
            groggy::storage::advanced_matrix::DType::Bool => "bool",
            groggy::storage::advanced_matrix::DType::Int64 => "int64",
            _ => "float64",
        };
        NumericArrayData::converted(&values, dtype, ConversionPolicy::Round)
            .map(|inner| Self { inner })
    }
    pub fn from_attributes_with_dtype(
        values: &[AttrValue],
        dtype: &str,
        exact: bool,
    ) -> PyResult<Self> {
        let (_, values) = numeric::attribute_inputs(values).map_err(numeric_error)?;
        NumericArrayData::converted(
            &values,
            dtype,
            if exact {
                ConversionPolicy::Exact
            } else {
                ConversionPolicy::Round
            },
        )
        .map(|inner| Self { inner })
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
    pub fn from_node_ids(node_ids: Vec<usize>) -> PyResult<Self> {
        NumArray::<i64>::from_usize_ids(&node_ids)
            .map(|values| Self {
                inner: NumericArrayData::Int64(values),
            })
            .map_err(numeric_error)
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
        match &self.inner {
            NumericArrayData::Bool(a) => a
                .get(index)
                .map(numeric::from_scalar)
                .map(NumericInput::as_f64),
            NumericArrayData::Int32(a) => a
                .get(index)
                .copied()
                .map(numeric::from_scalar)
                .map(NumericInput::as_f64),
            NumericArrayData::Int64(a) => a
                .get(index)
                .copied()
                .map(numeric::from_scalar)
                .map(NumericInput::as_f64),
            NumericArrayData::Float32(a) => a
                .get(index)
                .copied()
                .map(numeric::from_scalar)
                .map(NumericInput::as_f64),
            NumericArrayData::Float64(a) => a.get(index).copied(),
        }
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
    pub fn to_int64_vec(&self) -> PyResult<Vec<i64>> {
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
    #[pyo3(signature=(dtype, *, exact=false))]
    fn to_type(&self, py: Python, dtype: &str, exact: bool) -> PyResult<PyObject> {
        let policy = if exact {
            ConversionPolicy::Exact
        } else {
            ConversionPolicy::Round
        };
        if dtype == "basearray" {
            let values = match &self.inner {
                NumericArrayData::Bool(a) => {
                    NumArray::new(a.as_slice().to_vec()).to_attributes_with_policy(policy)
                }
                NumericArrayData::Int32(a) => a.to_attributes_with_policy(policy),
                NumericArrayData::Int64(a) => a.to_attributes_with_policy(policy),
                NumericArrayData::Float32(a) => a.to_attributes_with_policy(policy),
                NumericArrayData::Float64(a) => a.to_attributes_with_policy(policy),
            }
            .map_err(numeric_error)?;
            return Ok(crate::ffi::storage::array::PyBaseArray {
                inner: BaseArray::new(values),
            }
            .into_py(py));
        }
        let policy = if exact {
            ConversionPolicy::Exact
        } else {
            ConversionPolicy::Round
        };
        Ok(Self {
            inner: NumericArrayData::converted(&self.inner.inputs(), dtype, policy)?,
        }
        .into_py(py))
    }
    fn __add__(&self, other: &PyAny) -> PyResult<Self> {
        self.binary_operand(other, NumericBinary::Add, false)
    }
    fn __radd__(&self, other: &PyAny) -> PyResult<Self> {
        self.binary_operand(other, NumericBinary::Add, true)
    }
    fn __sub__(&self, other: &PyAny) -> PyResult<Self> {
        self.binary_operand(other, NumericBinary::Subtract, false)
    }
    fn __rsub__(&self, other: &PyAny) -> PyResult<Self> {
        self.binary_operand(other, NumericBinary::Subtract, true)
    }
    fn __mul__(&self, other: &PyAny) -> PyResult<Self> {
        self.binary_operand(other, NumericBinary::Multiply, false)
    }
    fn __rmul__(&self, other: &PyAny) -> PyResult<Self> {
        self.binary_operand(other, NumericBinary::Multiply, true)
    }
    fn __truediv__(&self, other: &PyAny) -> PyResult<Self> {
        let inner = if let Ok(other) = other.extract::<PyRef<PyNumArray>>() {
            NumericArrayData::Float64(
                numeric::divide(&self.inner.inputs(), &other.inner.inputs())
                    .map_err(numeric_error)?,
            )
        } else {
            self.inner.scalar_divide(input(other)?, false)?
        };
        Ok(Self { inner })
    }
    fn __rtruediv__(&self, other: &PyAny) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.scalar_divide(input(other)?, true)?,
        })
    }
    fn norm(&self) -> f64 {
        match &self.inner {
            NumericArrayData::Bool(a) => numeric::norm(a.as_slice()),
            NumericArrayData::Int32(a) => a.norm(),
            NumericArrayData::Int64(a) => a.norm(),
            NumericArrayData::Float32(a) => a.norm(),
            NumericArrayData::Float64(a) => a.norm(),
        }
    }
    fn dot(&self, py: Python, other: &Self) -> PyResult<PyObject> {
        match (&self.inner, &other.inner) {
            (NumericArrayData::Int32(a), NumericArrayData::Int32(b)) => {
                a.dot(b).map(|v| v.to_object(py))
            }
            (NumericArrayData::Int64(a), NumericArrayData::Int64(b)) => {
                a.dot(b).map(|v| v.to_object(py))
            }
            (NumericArrayData::Float32(a), NumericArrayData::Float32(b)) => {
                a.dot(b).map(|v| v.to_object(py))
            }
            (NumericArrayData::Float64(a), NumericArrayData::Float64(b)) => {
                a.dot(b).map(|v| v.to_object(py))
            }
            (NumericArrayData::Bool(a), NumericArrayData::Bool(b)) => {
                NumArray::new(a.as_slice().to_vec())
                    .dot(&NumArray::new(b.as_slice().to_vec()))
                    .map(|v| v.to_object(py))
            }
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Dot requires matching dtypes",
                ))
            }
        }
        .map_err(numeric_error)
    }

    /// Delegate mean to shared typed native kernels.
    fn mean(&self) -> Option<f64> {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::mean(arr.as_slice())
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::mean(arr.as_slice())
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::mean(arr.as_slice())
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::mean(arr.as_slice())
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::mean(arr.as_slice())
            }
        }
    }

    /// Delegate sum to shared typed native kernels.
    fn sum(&self) -> f64 {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::sum(arr.as_slice())
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::sum(arr.as_slice())
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::sum(arr.as_slice())
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::sum(arr.as_slice())
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::sum(arr.as_slice())
            }
        }
    }

    /// Delegate min to shared typed native kernels.
    fn min(&self) -> Option<f64> {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), true)
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), true)
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), true)
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), true)
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), true)
            }
        }
    }

    /// Delegate max to shared typed native kernels.
    fn max(&self) -> Option<f64> {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), false)
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), false)
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), false)
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), false)
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::extremum(arr.as_slice(), false)
            }
        }
    }

    /// Delegate std to shared typed native kernels.
    fn std(&self) -> Option<f64> {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::std_dev(arr.as_slice())
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::std_dev(arr.as_slice())
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::std_dev(arr.as_slice())
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::std_dev(arr.as_slice())
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::std_dev(arr.as_slice())
            }
        }
    }

    /// Delegate var to shared typed native kernels.
    fn var(&self) -> Option<f64> {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::variance(arr.as_slice())
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::variance(arr.as_slice())
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::variance(arr.as_slice())
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::variance(arr.as_slice())
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::variance(arr.as_slice())
            }
        }
    }

    /// Delegate median to the typed native kernel.
    fn median(&self) -> Option<f64> {
        match &self.inner {
            NumericArrayData::Bool(arr) => {
                groggy::storage::array::numeric_reductions::median(arr.as_slice())
            }
            NumericArrayData::Int32(arr) => {
                groggy::storage::array::numeric_reductions::median(arr.as_slice())
            }
            NumericArrayData::Int64(arr) => {
                groggy::storage::array::numeric_reductions::median(arr.as_slice())
            }
            NumericArrayData::Float32(arr) => {
                groggy::storage::array::numeric_reductions::median(arr.as_slice())
            }
            NumericArrayData::Float64(arr) => {
                groggy::storage::array::numeric_reductions::median(arr.as_slice())
            }
        }
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
    fn unique(&self) -> PyResult<Self> {
        let inner = match &self.inner {
            NumericArrayData::Bool(a) => NumericArrayData::Bool(BoolArray::new(
                NumArray::new(a.as_slice().to_vec()).unique().into_vec(),
            )),
            NumericArrayData::Int32(a) => NumericArrayData::Int32(a.unique()),
            NumericArrayData::Int64(a) => NumericArrayData::Int64(a.unique()),
            NumericArrayData::Float32(a) => NumericArrayData::Float32(a.unique()),
            NumericArrayData::Float64(a) => NumericArrayData::Float64(a.unique()),
        };
        Ok(Self { inner })
    }

    fn __gt__(&self, other: &PyAny) -> PyResult<Self> {
        self.comparison(other, numeric::NumericComparison::Greater)
    }
    fn __lt__(&self, other: &PyAny) -> PyResult<Self> {
        self.comparison(other, numeric::NumericComparison::Less)
    }
    fn __ge__(&self, other: &PyAny) -> PyResult<Self> {
        self.comparison(other, numeric::NumericComparison::GreaterEqual)
    }
    fn __le__(&self, other: &PyAny) -> PyResult<Self> {
        self.comparison(other, numeric::NumericComparison::LessEqual)
    }
    fn __eq__(&self, other: &PyAny) -> PyResult<Self> {
        self.comparison(other, numeric::NumericComparison::Equal)
    }
    fn __ne__(&self, other: &PyAny) -> PyResult<Self> {
        self.comparison(other, numeric::NumericComparison::NotEqual)
    }

    /// Matrix Integration: Reshape NumArray into a GraphMatrix
    fn reshape(&self, py: Python, rows: usize, cols: usize) -> PyResult<PyObject> {
        // Convert to float64 for matrix operations (matrices are typically float)
        let float_array = self.to_type(py, "float64", true)?;
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
