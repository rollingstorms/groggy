// src_new/utils/json.rs
// Utilities for Python <-> JSON conversion using pyo3 and serde_json.
// Adapted from src_old/utils/conversion.rs for robust, idiomatic interoperability.
/// Python <-> JSON conversion utilities
use pyo3::ToPyObject;
use pyo3::types::{PyDict, PyList};

/// Convert Python dictionary to JSON map
pub fn python_dict_to_json_map(py_dict: &pyo3::types::PyDict) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (key, value) in py_dict.iter() {
        let key_str: String = key.extract().unwrap_or_else(|_| key.str().unwrap().to_string());
        let json_value = python_to_json_value(value);
        map.insert(key_str, json_value);
    }
    map
}

/// Convert Python value to JSON value
pub fn python_to_json_value(py_value: &pyo3::types::PyAny) -> serde_json::Value {
    use pyo3::types::{PyDict, PyList};
    if py_value.is_none() {
        serde_json::Value::Null
    } else if let Ok(b) = py_value.extract::<bool>() {
        serde_json::Value::Bool(b)
    } else if let Ok(i) = py_value.extract::<i64>() {
        serde_json::Value::Number(i.into())
    } else if let Ok(f) = py_value.extract::<f64>() {
        serde_json::Number::from_f64(f).map_or(serde_json::Value::Null, serde_json::Value::Number)
    } else if let Ok(s) = py_value.extract::<String>() {
        serde_json::Value::String(s)
    } else if let Ok(py_list) = py_value.downcast::<PyList>() {
        let vec = py_list.iter().map(python_to_json_value).collect();
        serde_json::Value::Array(vec)
    } else if let Ok(py_dict) = py_value.downcast::<PyDict>() {
        let map = python_dict_to_json_map(py_dict);
        serde_json::Value::Object(map)
    } else {
        // Fallback: convert to string
        let s: String = py_value.str().unwrap().to_string();
        serde_json::Value::String(s)
    }
}

/// Convert JSON value to Python value
pub fn json_value_to_python(json_value: &serde_json::Value, py: pyo3::Python) -> pyo3::PyObject {
    use pyo3::types::{PyDict, PyList};
    match json_value {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.to_object(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else if let Some(f) = n.as_f64() {
                f.to_object(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.to_object(py),
        serde_json::Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                let py_item = json_value_to_python(item, py);
                py_list.append(py_item).unwrap();
            }
            py_list.to_object(py)
        }
        serde_json::Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, value) in obj {
                let py_value = json_value_to_python(value, py);
                py_dict.set_item(key, py_value).unwrap();
            }
            py_dict.to_object(py)
        }
    }
}
