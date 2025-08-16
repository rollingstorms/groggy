use pyo3::prelude::*;
use pyo3::exceptions::{PyTypeError, PyValueError, PyRuntimeError};
use groggy::{AttrValue as RustAttrValue, GraphError};

/// Convert Python value to Rust AttrValue (reusing PyAttrValue conversion logic)
pub fn python_value_to_attr_value(value: &PyAny) -> PyResult<RustAttrValue> {
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
        return Ok(RustAttrValue::Float(f as f32));  // Convert f64 to f32
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
        return Ok(RustAttrValue::Bytes(bytes))
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "Unsupported attribute value type"
        ))
    }
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
    };
    Ok(py_value)
}

/// Convert Rust GraphError to Python exception
pub fn graph_error_to_py_err(error: GraphError) -> PyErr {
    match error {
        GraphError::NodeNotFound { node_id, operation, suggestion } => {
            PyErr::new::<PyValueError, _>(format!(
                "Node {} not found during {}. {}",
                node_id, operation, suggestion
            ))
        },
        GraphError::EdgeNotFound { edge_id, operation, suggestion } => {
            PyErr::new::<PyValueError, _>(format!(
                "Edge {} not found during {}. {}",
                edge_id, operation, suggestion
            ))
        },
        GraphError::InvalidInput(message) => {
            PyErr::new::<PyValueError, _>(message)
        },
        GraphError::NotImplemented { feature, tracking_issue } => {
            let mut message = format!("Feature '{}' is not yet implemented", feature);
            if let Some(issue) = tracking_issue {
                message.push_str(&format!(". See: {}", issue));
            }
            PyErr::new::<PyRuntimeError, _>(message)
        },
        _ => PyErr::new::<PyRuntimeError, _>(format!("Graph error: {}", error))
    }
}