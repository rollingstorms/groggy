//! FFI Type Wrappers
//!
//! This module provides Python wrappers for core Groggy types.

use groggy::{AttrValue as RustAttrValue, EdgeId, NodeId};
use pyo3::exceptions::PyTypeError;
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
}

#[pymethods]
impl PyAttrValue {
    #[new]
    fn py_new(value: &PyAny) -> PyResult<Self> {
        let rust_value = if let Ok(b) = value.extract::<bool>() {
            RustAttrValue::Bool(b)
        } else if let Ok(i) = value.extract::<i64>() {
            RustAttrValue::Int(i)
        } else if let Ok(f) = value.extract::<f64>() {
            RustAttrValue::Float(f as f32) // Convert f64 to f32
        } else if let Ok(f) = value.extract::<f32>() {
            RustAttrValue::Float(f)
        } else if let Ok(s) = value.extract::<String>() {
            RustAttrValue::Text(s)
        } else if let Ok(vec) = value.extract::<Vec<f32>>() {
            RustAttrValue::FloatVec(vec)
        } else if let Ok(vec) = value.extract::<Vec<f64>>() {
            // Convert Vec<f64> to Vec<f32>
            let f32_vec: Vec<f32> = vec.into_iter().map(|f| f as f32).collect();
            RustAttrValue::FloatVec(f32_vec)
        } else if let Ok(bytes) = value.extract::<Vec<u8>>() {
            RustAttrValue::Bytes(bytes)
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Unsupported attribute value type. Supported types: int, float, str, bool, List[float], bytes"
            ));
        };

        Ok(Self { inner: rust_value })
    }

    #[getter]
    fn value(&self, py: Python) -> PyObject {
        match &self.inner {
            RustAttrValue::Int(i) => i.to_object(py),
            RustAttrValue::Float(f) => f.to_object(py),
            RustAttrValue::Text(s) => s.to_object(py),
            RustAttrValue::Bool(b) => b.to_object(py),
            RustAttrValue::FloatVec(v) => v.to_object(py),
            RustAttrValue::Bytes(b) => b.to_object(py),
            // Handle optimized variants by extracting their underlying value
            RustAttrValue::CompactText(cs) => cs.as_str().to_object(py),
            RustAttrValue::SmallInt(i) => i.to_object(py),
            RustAttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::CompressedFloatVec(cd) => match cd.decompress_float_vec() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::Null => py.None(),
        }
    }

    #[getter]
    fn type_name(&self) -> &'static str {
        match &self.inner {
            RustAttrValue::Int(_) => "int",
            RustAttrValue::Float(_) => "float",
            RustAttrValue::Text(_) => "text",
            RustAttrValue::Bool(_) => "bool",
            RustAttrValue::FloatVec(_) => "float_vec",
            RustAttrValue::Bytes(_) => "bytes",
            RustAttrValue::CompactText(_) => "text",
            RustAttrValue::SmallInt(_) => "int",
            RustAttrValue::CompressedText(_) => "text",
            RustAttrValue::CompressedFloatVec(_) => "float_vec",
            RustAttrValue::Null => "null",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AttrValue({})",
            match &self.inner {
                RustAttrValue::Int(i) => i.to_string(),
                RustAttrValue::Float(f) => f.to_string(),
                RustAttrValue::Text(s) => format!("\"{}\"", s),
                RustAttrValue::Bool(b) => b.to_string(),
                RustAttrValue::FloatVec(v) => format!("{:?}", v),
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
                RustAttrValue::Null => "None".to_string(),
            }
        )
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(match &self.inner {
            RustAttrValue::Int(i) => i.to_string(),
            RustAttrValue::Float(f) => f.to_string(),
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
            RustAttrValue::Null => "None".to_string(),
        })
    }

    fn __eq__(&self, other: &PyAttrValue) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // Create a hash based on the variant and value
        match &self.inner {
            RustAttrValue::Int(i) => {
                0u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            RustAttrValue::Float(f) => {
                1u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            RustAttrValue::Text(s) => {
                2u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            RustAttrValue::Bool(b) => {
                3u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            RustAttrValue::FloatVec(v) => {
                4u8.hash(&mut hasher);
                for f in v {
                    f.to_bits().hash(&mut hasher);
                }
            }
            RustAttrValue::Bytes(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            RustAttrValue::CompactText(cs) => {
                6u8.hash(&mut hasher);
                cs.as_str().hash(&mut hasher);
            }
            RustAttrValue::SmallInt(i) => {
                7u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            RustAttrValue::CompressedText(cd) => {
                8u8.hash(&mut hasher);
                if let Ok(text) = cd.decompress_text() {
                    text.hash(&mut hasher);
                }
            }
            RustAttrValue::CompressedFloatVec(cd) => {
                9u8.hash(&mut hasher);
                if let Ok(vec) = cd.decompress_float_vec() {
                    for f in vec {
                        f.to_bits().hash(&mut hasher);
                    }
                }
            }
            RustAttrValue::Null => {
                10u8.hash(&mut hasher);
                // No additional data to hash for Null
            }
        }
        hasher.finish()
    }
}

// ToPyObject implementation for PyAttrValue
impl pyo3::ToPyObject for PyAttrValue {
    fn to_object(&self, py: pyo3::Python<'_>) -> pyo3::PyObject {
        match &self.inner {
            RustAttrValue::Int(i) => i.to_object(py),
            RustAttrValue::Float(f) => f.to_object(py),
            RustAttrValue::Text(s) => s.to_object(py),
            RustAttrValue::Bool(b) => b.to_object(py),
            RustAttrValue::FloatVec(v) => v.to_object(py),
            RustAttrValue::Bytes(b) => b.to_object(py),
            RustAttrValue::CompactText(cs) => cs.as_str().to_object(py),
            RustAttrValue::SmallInt(i) => (*i as i64).to_object(py),
            RustAttrValue::CompressedText(cd) => match cd.decompress_text() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::CompressedFloatVec(cd) => match cd.decompress_float_vec() {
                Ok(data) => data.to_object(py),
                Err(_) => py.None(),
            },
            RustAttrValue::Null => py.None(),
        }
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

            // Type-specific statistics
            if !values.is_empty() {
                match &values[0] {
                    RustAttrValue::Int(_) => {
                        let int_values: Vec<i64> = values
                            .iter()
                            .filter_map(|v| {
                                if let RustAttrValue::Int(i) = v {
                                    Some(*i)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !int_values.is_empty() {
                            let sum: i64 = int_values.iter().sum();
                            let avg = sum as f64 / int_values.len() as f64;
                            let min = *int_values.iter().min().unwrap();
                            let max = *int_values.iter().max().unwrap();

                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                        }
                    }
                    RustAttrValue::Float(_) => {
                        let float_values: Vec<f32> = values
                            .iter()
                            .filter_map(|v| {
                                if let RustAttrValue::Float(f) = v {
                                    Some(*f)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !float_values.is_empty() {
                            let sum: f32 = float_values.iter().sum();
                            let avg = sum / float_values.len() as f32;
                            let min = float_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let max = float_values
                                .iter()
                                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                        }
                    }
                    _ => {
                        // For other types, just provide count
                    }
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
