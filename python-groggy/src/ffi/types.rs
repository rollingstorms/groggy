//! FFI Type Wrappers
//!
//! This module provides Python wrappers for core Groggy types.

use groggy::{AttrValue as RustAttrValue, EdgeId, NodeId};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python wrapper for AttrValue
#[pyclass(name = "AttrValue")]
#[derive(Clone)]
pub struct PyAttrValue {
    pub inner: RustAttrValue,
}

impl PyAttrValue {
    pub fn new(inner: RustAttrValue) -> Self {
        Self { inner }
    }

    /// Create PyAttrValue from RustAttrValue (for FFI integration)
    pub fn from_attr_value(attr_value: RustAttrValue) -> Self {
        Self { inner: attr_value }
    }

    /// Convert PyAttrValue to RustAttrValue (for FFI integration)
    pub fn to_attr_value(&self) -> RustAttrValue {
        self.inner.clone()
    }

    /// Get the raw Python value for internal FFI usage
    pub fn to_python_value(&self, py: Python) -> PyResult<PyObject> {
        crate::ffi::utils::attr_value_to_python_value(py, &self.inner)
    }

    /// Create PyAttrValue from Python value (public constructor)
    pub fn from_py_value(value: &PyAny) -> PyResult<Self> {
        crate::ffi::utils::python_value_to_attr_value(value).map(Self::new)
    }
}

#[pymethods]
impl PyAttrValue {
    #[new]
    fn py_new(value: &PyAny) -> PyResult<Self> {
        Self::from_py_value(value)
    }

    /// Explicit utc_instant property from the corresponding Python datetime type.
    #[staticmethod]
    fn utc_instant(value: &PyAny) -> PyResult<Self> {
        let result = Self::from_py_value(value)?;
        if matches!(result.inner, RustAttrValue::UtcInstant(_)) {
            Ok(result)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected utc_instant input",
            ))
        }
    }
    /// Explicit date property from the corresponding Python datetime type.
    #[staticmethod]
    fn date(value: &PyAny) -> PyResult<Self> {
        let result = Self::from_py_value(value)?;
        if matches!(result.inner, RustAttrValue::Date(_)) {
            Ok(result)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected date input",
            ))
        }
    }
    /// Explicit duration property from the corresponding Python datetime type.
    #[staticmethod]
    fn duration(value: &PyAny) -> PyResult<Self> {
        let result = Self::from_py_value(value)?;
        if matches!(result.inner, RustAttrValue::Duration(_)) {
            Ok(result)
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected duration input",
            ))
        }
    }

    /// Explicit f32 storage with native checked rounding; finite overflow rejects.
    #[staticmethod]
    fn float32(value: f64) -> PyResult<Self> {
        RustAttrValue::Float64(value)
            .to_numeric_type(groggy::NumericType::Float32)
            .map(Self::new)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Value exceeds float32 range"))
    }
    /// Explicit lossless f64 scalar property; legacy f32 constructors remain typed.
    #[staticmethod]
    fn float64(value: f64) -> Self {
        Self::new(RustAttrValue::Float64(value))
    }
    /// Explicit lossless f64 vector property.
    #[staticmethod]
    fn float64_vec(values: Vec<f64>) -> Self {
        Self::new(RustAttrValue::Float64Vec(values))
    }

    /// Explicit float32 vector constructor, including empty vectors and numeric casts.
    #[staticmethod]
    fn float_vec(values: Vec<f64>) -> PyResult<Self> {
        groggy::storage::array::NumArray::new(values)
            .cast_with_policy::<f32>(groggy::storage::numeric::scalar::ConversionPolicy::Round)
            .map(|array| Self::new(RustAttrValue::FloatVec(array.as_slice().to_vec())))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Explicit int64 vector constructor.
    #[staticmethod]
    fn int_vec(values: Vec<i64>) -> Self {
        Self::new(RustAttrValue::IntVec(values))
    }

    /// Explicit text vector constructor.
    #[staticmethod]
    fn text_vec(values: Vec<String>) -> Self {
        Self::new(RustAttrValue::TextVec(values))
    }

    /// Explicit boolean vector constructor.
    #[staticmethod]
    fn bool_vec(values: Vec<bool>) -> Self {
        Self::new(RustAttrValue::BoolVec(values))
    }

    #[getter]
    fn value(&self, py: Python) -> PyResult<PyObject> {
        self.to_python_value(py)
    }

    #[getter]
    fn type_name(&self) -> &'static str {
        match &self.inner {
            RustAttrValue::UtcInstant(_) => "utc_instant",
            RustAttrValue::Date(_) => "date",
            RustAttrValue::Duration(_) => "duration",
            RustAttrValue::Int(_) => "int",
            RustAttrValue::Float(_) => "float",
            RustAttrValue::Float64(_) => "float64",
            RustAttrValue::Float64Vec(_) => "float64_vec",
            RustAttrValue::Text(_) => "text",
            RustAttrValue::Bool(_) => "bool",
            RustAttrValue::FloatVec(_) => "float_vec",
            RustAttrValue::IntVec(_) => "int_vec",
            RustAttrValue::TextVec(_) => "text_vec",
            RustAttrValue::BoolVec(_) => "bool_vec",
            RustAttrValue::Bytes(_) => "bytes",
            RustAttrValue::CompactText(_) => "text",
            RustAttrValue::SmallInt(_) => "int",
            RustAttrValue::CompressedText(_) => "text",
            RustAttrValue::CompressedFloatVec(_) => "float_vec",
            RustAttrValue::SubgraphRef(_) => "subgraph_ref",
            RustAttrValue::NodeArray(_) => "node_array",
            RustAttrValue::EdgeArray(_) => "edge_array",
            RustAttrValue::Null => "null",
            RustAttrValue::Json(_) => "json",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AttrValue({})",
            match &self.inner {
                RustAttrValue::Int(i) => i.to_string(),
                RustAttrValue::Float(f) => f.to_string(),
                RustAttrValue::Float64(f) => f.to_string(),
                RustAttrValue::UtcInstant(value) => value.to_string(),
                RustAttrValue::Date(value) => value.to_string(),
                RustAttrValue::Duration(value) => value.to_string(),
                RustAttrValue::Float64Vec(v) => format!("{:?}", v),
                RustAttrValue::Text(s) => format!("\"{}\"", s),
                RustAttrValue::Bool(b) => b.to_string(),
                RustAttrValue::FloatVec(v) => format!("{:?}", v),
                RustAttrValue::IntVec(v) => format!("{:?}", v),
                RustAttrValue::TextVec(v) => format!("{:?}", v),
                RustAttrValue::BoolVec(v) => format!("{:?}", v),
                RustAttrValue::Bytes(b) => format!("b\"{:?}\"", b),
                RustAttrValue::CompactText(cs) => format!("\"{}\"", cs.as_str()),
                RustAttrValue::SmallInt(i) => i.to_string(),
                RustAttrValue::CompressedText(cd) => {
                    match cd.decompress_text() {
                        Ok(data) => format!("\"{}\"", data),
                        Err(_) => "compressed(error)".to_string(),
                    }
                }
                RustAttrValue::CompressedFloatVec(cd) => {
                    match cd.decompress_float_vec() {
                        Ok(data) => format!("{:?}", data),
                        Err(_) => "compressed(error)".to_string(),
                    }
                }
                RustAttrValue::SubgraphRef(subgraph_id) => format!("SubgraphRef({})", subgraph_id),
                RustAttrValue::NodeArray(node_ids) => format!("{:?}", node_ids),
                RustAttrValue::EdgeArray(edge_ids) => format!("{:?}", edge_ids),
                RustAttrValue::Null => "None".to_string(),
                RustAttrValue::Json(json_str) => format!("Json({})", json_str),
            }
        )
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(match &self.inner {
            RustAttrValue::Int(i) => i.to_string(),
            RustAttrValue::Float(f) => f.to_string(),
            RustAttrValue::Float64(f) => f.to_string(),
            RustAttrValue::UtcInstant(value) => value.to_string(),
            RustAttrValue::Date(value) => value.to_string(),
            RustAttrValue::Duration(value) => value.to_string(),
            RustAttrValue::Float64Vec(v) => format!("{:?}", v),
            RustAttrValue::Text(s) => s.clone(),
            RustAttrValue::Bool(b) => b.to_string(),
            RustAttrValue::FloatVec(v) => format!("{:?}", v),
            RustAttrValue::Bytes(b) => format!("{:?}", b),
            RustAttrValue::CompactText(cs) => cs.as_str().to_string(),
            RustAttrValue::SmallInt(i) => i.to_string(),
            RustAttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(data) => data,
                Err(_) => "compressed(error)".to_string(),
            },
            RustAttrValue::CompressedFloatVec(cd) => match cd.decompress_float_vec() {
                Ok(data) => format!("{:?}", data),
                Err(_) => "compressed(error)".to_string(),
            },
            RustAttrValue::SubgraphRef(subgraph_id) => format!("SubgraphRef({})", subgraph_id),
            RustAttrValue::NodeArray(node_ids) => format!("{:?}", node_ids),
            RustAttrValue::EdgeArray(edge_ids) => format!("{:?}", edge_ids),
            RustAttrValue::IntVec(v) => format!("{:?}", v),
            RustAttrValue::TextVec(v) => format!("{:?}", v),
            RustAttrValue::BoolVec(v) => format!("{:?}", v),
            RustAttrValue::Json(s) => s.clone(),
            RustAttrValue::Null => "None".to_string(),
        })
    }

    fn __eq__(&self, other: &PyAttrValue) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// Native result handle that keeps data in Rust
#[pyclass]
pub struct PyResultHandle {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub result_type: String,
}

#[pymethods]
impl PyResultHandle {
    #[getter]
    fn nodes(&self) -> Vec<NodeId> {
        self.nodes.clone()
    }

    #[getter]
    fn edges(&self) -> Vec<EdgeId> {
        self.edges.clone()
    }

    #[getter]
    fn result_type(&self) -> String {
        self.result_type.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ResultHandle(nodes={}, edges={}, type='{}')",
            self.nodes.len(),
            self.edges.len(),
            self.result_type
        )
    }
}

/// Python wrapper for high-performance attribute collections
#[pyclass(unsendable)]
pub struct PyAttributeCollection {
    pub graph_ref: *const groggy::Graph, // Unsafe but controlled access
    pub node_ids: Vec<NodeId>,
    pub attr_name: String,
}

#[pymethods]
impl PyAttributeCollection {
    /// Get count of attributes without converting
    fn len(&self) -> usize {
        self.node_ids.len()
    }

    /// Compute statistics directly in Rust
    fn compute_stats(&self, py: Python) -> PyResult<PyObject> {
        // Safe because we control the lifetime
        let graph = unsafe { &*self.graph_ref };

        let mut values = Vec::new();
        for &node_id in &self.node_ids {
            if let Ok(Some(attr)) = graph.get_node_attr(node_id, &self.attr_name) {
                values.push(attr);
            }
        }

        // Compute statistics in Rust
        {
            let dict = PyDict::new(py);

            // Count
            dict.set_item("count", values.len())?;

            let column = groggy::storage::array::BaseArray::new(values);
            for (name, operation) in [
                ("sum", "sum"),
                ("min", "min"),
                ("max", "max"),
                ("average", "mean"),
            ] {
                let value = column
                    .aggregate_numeric(operation)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                if !matches!(value, RustAttrValue::Null) {
                    dict.set_item(
                        name,
                        crate::ffi::utils::attr_value_to_python_value(py, &value)?,
                    )?;
                }
            }

            Ok(dict.to_object(py))
        }
    }

    /// Get sample values without converting all
    fn sample_values(&self, count: usize) -> PyResult<Vec<PyAttrValue>> {
        let graph = unsafe { &*self.graph_ref };
        let mut results = Vec::new();

        let step = if self.node_ids.len() <= count {
            1
        } else {
            self.node_ids.len() / count
        };

        for &node_id in self.node_ids.iter().step_by(step).take(count) {
            if let Ok(Some(attr)) = graph.get_node_attr(node_id, &self.attr_name) {
                results.push(PyAttrValue { inner: attr });
            }
        }

        Ok(results)
    }
}

// We'll add more types as needed during modularization
