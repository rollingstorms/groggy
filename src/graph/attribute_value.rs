// src/graph/attribute_value.rs
//! Native attribute value system to replace JSON serialization

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyFloat, PyLong, PyBool, PyString};
use std::collections::HashMap;

/// Native attribute value that can be directly converted from Python without JSON
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    None,
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<AttributeValue>),
    Dict(HashMap<String, AttributeValue>),
}

impl AttributeValue {
    /// Convert to Python object
    pub fn to_python(&self, py: Python) -> PyResult<PyObject> {
        match self {
            AttributeValue::None => Ok(py.None()),
            AttributeValue::String(s) => Ok(s.to_object(py)),
            AttributeValue::Integer(i) => Ok(i.to_object(py)),
            AttributeValue::Float(f) => Ok(f.to_object(py)),
            AttributeValue::Boolean(b) => Ok(b.to_object(py)),
            AttributeValue::List(list) => {
                let py_list = PyList::empty(py);
                for item in list {
                    py_list.append(item.to_python(py)?)?;
                }
                Ok(py_list.to_object(py))
            }
            AttributeValue::Dict(dict) => {
                let py_dict = PyDict::new(py);
                for (key, value) in dict {
                    py_dict.set_item(key, value.to_python(py)?)?;
                }
                Ok(py_dict.to_object(py))
            }
        }
    }
    
    /// Get memory size estimate for this value
    pub fn memory_size(&self) -> usize {
        match self {
            AttributeValue::None => 0,
            AttributeValue::String(s) => s.len(),
            AttributeValue::Integer(_) => 8,
            AttributeValue::Float(_) => 8,
            AttributeValue::Boolean(_) => 1,
            AttributeValue::List(list) => {
                8 + list.iter().map(|v| v.memory_size()).sum::<usize>()
            }
            AttributeValue::Dict(dict) => {
                8 + dict.iter()
                    .map(|(k, v)| k.len() + v.memory_size())
                    .sum::<usize>()
            }
        }
    }
}

/// Convert from Python object to AttributeValue
impl<'source> FromPyObject<'source> for AttributeValue {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        // Try different types in order of likelihood
        if ob.is_none() {
            Ok(AttributeValue::None)
        } else if let Ok(s) = ob.extract::<String>() {
            Ok(AttributeValue::String(s))
        } else if let Ok(i) = ob.extract::<i64>() {
            Ok(AttributeValue::Integer(i))
        } else if let Ok(f) = ob.extract::<f64>() {
            Ok(AttributeValue::Float(f))
        } else if let Ok(b) = ob.extract::<bool>() {
            Ok(AttributeValue::Boolean(b))
        } else if let Ok(list) = ob.downcast::<PyList>() {
            let mut values = Vec::new();
            for item in list.iter() {
                values.push(AttributeValue::extract(item)?);
            }
            Ok(AttributeValue::List(values))
        } else if let Ok(dict) = ob.downcast::<PyDict>() {
            let mut values = HashMap::new();
            for (key, value) in dict.iter() {
                let key_str = key.extract::<String>()?;
                let value_attr = AttributeValue::extract(value)?;
                values.insert(key_str, value_attr);
            }
            Ok(AttributeValue::Dict(values))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                format!("Cannot convert {} to AttributeValue", ob.get_type().name()?)
            ))
        }
    }
}

/// Convert AttributeValue to Python object
impl ToPyObject for AttributeValue {
    fn to_object(&self, py: Python) -> PyObject {
        self.to_python(py).unwrap_or_else(|_| py.None())
    }
}

/// For backwards compatibility with existing JSON-based code
impl AttributeValue {
    /// Create from JSON string (for migration)
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        let value: serde_json::Value = serde_json::from_str(json_str)?;
        Ok(Self::from_json_value(value))
    }
    
    /// Create from serde_json::Value (for migration)
    pub fn from_json_value(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => AttributeValue::None,
            serde_json::Value::Bool(b) => AttributeValue::Boolean(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    AttributeValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    AttributeValue::Float(f)
                } else {
                    AttributeValue::String(n.to_string())
                }
            }
            serde_json::Value::String(s) => AttributeValue::String(s),
            serde_json::Value::Array(arr) => {
                let values = arr.into_iter().map(Self::from_json_value).collect();
                AttributeValue::List(values)
            }
            serde_json::Value::Object(obj) => {
                let values = obj.into_iter()
                    .map(|(k, v)| (k, Self::from_json_value(v)))
                    .collect();
                AttributeValue::Dict(values)
            }
        }
    }
    
    /// Convert to JSON string (for migration)
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.to_json_value()).unwrap_or_default()
    }
    
    /// Convert to serde_json::Value (for migration)
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            AttributeValue::None => serde_json::Value::Null,
            AttributeValue::String(s) => serde_json::Value::String(s.clone()),
            AttributeValue::Integer(i) => serde_json::Value::Number((*i).into()),
            AttributeValue::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0))
            ),
            AttributeValue::Boolean(b) => serde_json::Value::Bool(*b),
            AttributeValue::List(list) => {
                let values = list.iter().map(|v| v.to_json_value()).collect();
                serde_json::Value::Array(values)
            }
            AttributeValue::Dict(dict) => {
                let values = dict.iter()
                    .map(|(k, v)| (k.clone(), v.to_json_value()))
                    .collect();
                serde_json::Value::Object(values)
            }
        }
    }
}