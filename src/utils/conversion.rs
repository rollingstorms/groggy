use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyAny};
use std::collections::HashMap;
use serde_json::Value as JsonValue;

/// Convert Python dictionary to JSON map
pub fn python_dict_to_json_map(py_dict: &PyDict) -> PyResult<HashMap<String, JsonValue>> {
    let mut map = HashMap::new();
    
    for (key, value) in py_dict {
        let key_str: String = key.extract()?;
        let json_value = python_to_json_value(value)?;
        map.insert(key_str, json_value);
    }
    
    Ok(map)
}

/// Convert Python value to JSON value
pub fn python_to_json_value(py_value: &PyAny) -> PyResult<JsonValue> {
    if py_value.is_none() {
        Ok(JsonValue::Null)
    } else if let Ok(b) = py_value.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(i) = py_value.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = py_value.extract::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            Ok(JsonValue::Number(num))
        } else {
            Ok(JsonValue::Null)
        }
    } else if let Ok(s) = py_value.extract::<String>() {
        Ok(JsonValue::String(s))
    } else if let Ok(py_list) = py_value.downcast::<PyList>() {
        let mut vec = Vec::new();
        for item in py_list {
            vec.push(python_to_json_value(item)?);
        }
        Ok(JsonValue::Array(vec))
    } else if let Ok(py_dict) = py_value.downcast::<PyDict>() {
        let map = python_dict_to_json_map(py_dict)?;
        Ok(JsonValue::Object(map.into_iter().collect()))
    } else {
        // Fallback: convert to string
        let s: String = py_value.str()?.extract()?;
        Ok(JsonValue::String(s))
    }
}

/// Convert JSON value to Python value
pub fn json_value_to_python(py: Python, json_value: &JsonValue) -> PyResult<PyObject> {
    match json_value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok(b.to_object(py)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        JsonValue::String(s) => Ok(s.to_object(py)),
        JsonValue::Array(arr) => {
            let py_list = pyo3::types::PyList::empty(py);
            for item in arr {
                let py_item = json_value_to_python(py, item)?;
                py_list.append(py_item)?;
            }
            Ok(py_list.to_object(py))
        }
        JsonValue::Object(obj) => {
            let py_dict = pyo3::types::PyDict::new(py);
            for (key, value) in obj {
                let py_value = json_value_to_python(py, value)?;
                py_dict.set_item(key, py_value)?;
            }
            Ok(py_dict.to_object(py))
        }
    }
}
