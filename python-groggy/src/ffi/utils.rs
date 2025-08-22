use groggy::{AttrValue as RustAttrValue, GraphError};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Convert Python value to Rust AttrValue (reusing PyAttrValue conversion logic)
pub fn python_value_to_attr_value(value: &PyAny) -> PyResult<RustAttrValue> {
    // Check for None/null first
    if value.is_none() {
        return Ok(RustAttrValue::Null);
    }

    // Fast path optimization: Check most common types first

    // Check booleans FIRST (Python bool is a subtype of int, so this must come before int check)
    if let Ok(b) = value.extract::<bool>() {
        return Ok(RustAttrValue::Bool(b));
    }

    // Check integers second (very common in benchmarks)
    if let Ok(i) = value.extract::<i64>() {
        return Ok(RustAttrValue::Int(i));
    }

    // Check strings third (also very common)
    if let Ok(s) = value.extract::<String>() {
        return Ok(RustAttrValue::Text(s));
    }

    // Check floats fourth
    if let Ok(f) = value.extract::<f64>() {
        return Ok(RustAttrValue::Float(f as f32)); // Convert f64 to f32
    }

    // Less common types
    if let Ok(f) = value.extract::<f32>() {
        return Ok(RustAttrValue::Float(f));
    } else if let Ok(vec) = value.extract::<Vec<f32>>() {
        return Ok(RustAttrValue::FloatVec(vec));
    } else if let Ok(vec) = value.extract::<Vec<f64>>() {
        // Convert Vec<f64> to Vec<f32>
        let f32_vec: Vec<f32> = vec.into_iter().map(|f| f as f32).collect();
        return Ok(RustAttrValue::FloatVec(f32_vec));
    } else if let Ok(bytes) = value.extract::<Vec<u8>>() {
        return Ok(RustAttrValue::Bytes(bytes));
    }

    // Check for generic list types that we can convert
    if value.hasattr("__iter__").unwrap_or(false)
        && !value.is_instance_of::<pyo3::types::PyString>()
    {
        // This is some kind of iterable (list, tuple, etc.) but not a string

        // Try to extract as a list and convert all elements to the same type
        if let Ok(list) = value.extract::<Vec<&PyAny>>() {
            if list.is_empty() {
                // Empty list - default to empty float vector
                return Ok(RustAttrValue::FloatVec(vec![]));
            }

            // Try to determine the type by sampling the first element
            let first_elem = list[0];

            // Try as integers first
            if first_elem.extract::<i64>().is_ok() {
                let mut int_vec = Vec::with_capacity(list.len());
                for item in list {
                    match item.extract::<i64>() {
                        Ok(i) => int_vec.push(i),
                        Err(_) => {
                            return Err(PyErr::new::<PyTypeError, _>(
                                format!("Mixed types in list not supported. Expected all integers but found: {}", 
                                    item.get_type().name()?)
                            ));
                        }
                    }
                }
                // Store as comma-separated string for now (could add IntVec to AttrValue later)
                let int_str = int_vec
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                return Ok(RustAttrValue::Text(format!("[{}]", int_str)));
            }

            // Try as floats
            if first_elem.extract::<f64>().is_ok() || first_elem.extract::<f32>().is_ok() {
                let mut float_vec = Vec::with_capacity(list.len());
                for item in list {
                    if let Ok(f) = item.extract::<f64>() {
                        float_vec.push(f as f32);
                    } else if let Ok(f) = item.extract::<f32>() {
                        float_vec.push(f);
                    } else {
                        return Err(PyErr::new::<PyTypeError, _>(format!(
                            "Mixed types in list not supported. Expected all floats but found: {}",
                            item.get_type().name()?
                        )));
                    }
                }
                return Ok(RustAttrValue::FloatVec(float_vec));
            }

            // Try as strings
            if first_elem.extract::<String>().is_ok() {
                let mut string_vec = Vec::with_capacity(list.len());
                for item in list {
                    match item.extract::<String>() {
                        Ok(s) => string_vec.push(s),
                        Err(_) => {
                            return Err(PyErr::new::<PyTypeError, _>(
                                format!("Mixed types in list not supported. Expected all strings but found: {}", 
                                    item.get_type().name()?)
                            ));
                        }
                    }
                }
                // Store as JSON-like string representation
                let strings_json = format!(
                    "[{}]",
                    string_vec
                        .iter()
                        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                return Ok(RustAttrValue::Text(strings_json));
            }

            // If we get here, it's a list of an unsupported type
            return Err(PyErr::new::<PyTypeError, _>(format!(
                "Unsupported list element type: {}. Supported list types: [int], [float], [str]",
                first_elem.get_type().name()?
            )));
        }
    }

    // If we get here, it's a completely unsupported type
    Err(PyErr::new::<PyTypeError, _>(
        format!("Unsupported attribute value type: {}. Supported types: int, float, str, bool, bytes, [int], [float], [str]", 
            value.get_type().name()?)
    ))
}

/// Convert Rust AttrValue to Python object
pub fn attr_value_to_python_value(py: Python, attr_value: &RustAttrValue) -> PyResult<PyObject> {
    let py_value = match attr_value {
        RustAttrValue::Int(i) => i.to_object(py),
        RustAttrValue::Float(f) => f.to_object(py),
        RustAttrValue::Text(s) => s.to_object(py),
        RustAttrValue::Bool(b) => b.to_object(py),
        RustAttrValue::FloatVec(vec) => vec.to_object(py),
        RustAttrValue::Bytes(bytes) => bytes.to_object(py),
        RustAttrValue::CompactText(compact_str) => compact_str.as_str().to_object(py),
        RustAttrValue::SmallInt(i) => i.to_object(py),
        RustAttrValue::CompressedText(_) => "compressed_text".to_object(py), // Placeholder
        RustAttrValue::CompressedFloatVec(_) => vec!["compressed_floats"].to_object(py), // Placeholder
        RustAttrValue::Null => py.None(),
    };
    Ok(py_value)
}

/// Convert Rust GraphError to Python exception
pub fn graph_error_to_py_err(error: GraphError) -> PyErr {
    match error {
        GraphError::NodeNotFound {
            node_id,
            operation,
            suggestion,
        } => PyErr::new::<PyValueError, _>(format!(
            "Node {} not found during {}. {}",
            node_id, operation, suggestion
        )),
        GraphError::EdgeNotFound {
            edge_id,
            operation,
            suggestion,
        } => PyErr::new::<PyValueError, _>(format!(
            "Edge {} not found during {}. {}",
            edge_id, operation, suggestion
        )),
        GraphError::InvalidInput(message) => PyErr::new::<PyValueError, _>(message),
        GraphError::NotImplemented {
            feature,
            tracking_issue,
        } => {
            let mut message = format!("Feature '{}' is not yet implemented", feature);
            if let Some(issue) = tracking_issue {
                message.push_str(&format!(". See: {}", issue));
            }
            PyErr::new::<PyRuntimeError, _>(message)
        }
        _ => PyErr::new::<PyRuntimeError, _>(format!("Graph error: {}", error)),
    }
}
