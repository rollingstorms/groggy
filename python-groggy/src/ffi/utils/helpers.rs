use groggy::{AttrValue as RustAttrValue, GraphError};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Convert every property before a creation call can allocate an entity.
pub fn parse_creation_attrs(
    attrs: Option<&PyDict>,
    excluded: &[&str],
) -> (HashMap<String, RustAttrValue>, Vec<String>) {
    let mut parsed = HashMap::new();
    let mut errors = Vec::new();
    if let Some(attrs) = attrs {
        for (position, (key, value)) in attrs.iter().enumerate() {
            let name = key.extract::<String>();
            if name
                .as_ref()
                .map(|name| excluded.contains(&name.as_str()))
                .unwrap_or(false)
            {
                continue;
            }
            let converted = python_value_to_attr_value(value);
            match name {
                Ok(name) => match converted {
                    Ok(value) => {
                        parsed.insert(name, value);
                    }
                    Err(error) => {
                        errors.push(format!("property[{position}].value: {error}"));
                        // Preserve the property key for core validation; errors prevent publication.
                        parsed.insert(name, RustAttrValue::Null);
                    }
                },
                Err(error) => {
                    errors.push(format!("property[{position}].name: {error}"));
                    if let Err(error) = converted {
                        errors.push(format!("property[{position}].value: {error}"));
                    }
                }
            }
        }
    }
    (parsed, errors)
}

/// Validate JSON object keys without silently coercing their types.
fn validate_json_container(
    value: &PyAny,
    path: &mut std::collections::HashSet<usize>,
    depth: usize,
) -> PyResult<()> {
    use pyo3::types::{PyDict, PyList, PyTuple};
    if depth > 128 {
        return Err(PyValueError::new_err(
            "Structured properties exceed 128 nesting levels",
        ));
    }
    if !(value.is_instance_of::<PyDict>()
        || value.is_instance_of::<PyList>()
        || value.is_instance_of::<PyTuple>())
    {
        return Ok(());
    }
    let identity = value.as_ptr() as usize;
    if !path.insert(identity) {
        return Err(PyValueError::new_err("Circular structured property"));
    }
    if let Ok(dict) = value.downcast::<PyDict>() {
        for (key, value) in dict.iter() {
            key.extract::<String>()
                .map_err(|_| PyTypeError::new_err("Structured property keys must be strings"))?;
            validate_json_container(value, path, depth + 1)?;
        }
    } else {
        for item in value.iter()? {
            validate_json_container(item?, path, depth + 1)?;
        }
    }
    path.remove(&identity);
    Ok(())
}

fn structured_property(value: &PyAny) -> PyResult<RustAttrValue> {
    use pyo3::types::PyDict;
    validate_json_container(value, &mut std::collections::HashSet::new(), 0)?;
    let py = value.py();
    let options = PyDict::new(py);
    options.set_item("allow_nan", false)?;
    options.set_item("sort_keys", true)?;
    let json = py
        .import("json")?
        .call_method("dumps", (value,), Some(options))
        .map_err(|error| {
            PyTypeError::new_err(format!(
                "Unsupported attribute value type {}: {error}",
                value.get_type().name().unwrap_or("unknown")
            ))
        })?;
    Ok(RustAttrValue::Json(json.extract()?))
}

/// Preserve an explicitly typed NumPy float32 scalar/array without importing NumPy.
fn numpy_float32(value: &PyAny) -> bool {
    if value.is_instance_of::<pyo3::types::PyFloat>()
        || value.is_instance_of::<pyo3::types::PyList>()
        || value.is_instance_of::<pyo3::types::PyTuple>()
    {
        return false;
    }
    let kind = value.get_type();
    let numpy = kind
        .getattr("__module__")
        .and_then(|v| v.extract::<String>())
        .is_ok_and(|v| v == "numpy");
    if !numpy {
        return false;
    }
    kind.getattr("__name__")
        .and_then(|v| v.extract::<String>())
        .is_ok_and(|v| v == "float32")
        || value
            .getattr("dtype")
            .and_then(|v| v.getattr("name"))
            .and_then(|v| v.extract::<String>())
            .is_ok_and(|v| v == "float32")
}

/// One translation policy shared by constructors, graph writers and table adapters.
/// Homogeneous lists retain their element type; heterogeneous containers use Json.
pub fn python_value_to_attr_value(value: &PyAny) -> PyResult<RustAttrValue> {
    use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyLong, PyString};
    if let Ok(typed) = value.extract::<PyRef<crate::ffi::types::PyAttrValue>>() {
        return Ok(typed.inner.clone());
    }
    if value.is_none() {
        return Ok(RustAttrValue::Null);
    }
    if let Some(value) = super::temporal::from_python(value)? {
        return Ok(value);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(RustAttrValue::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyLong>() {
        return Ok(RustAttrValue::Int(value.extract()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(RustAttrValue::Text(value.extract()?));
    }
    if let Ok(bytes) = value.downcast::<PyBytes>() {
        return Ok(RustAttrValue::Bytes(bytes.as_bytes().to_vec()));
    }
    if let Ok(bytes) = value.downcast::<PyByteArray>() {
        return Ok(RustAttrValue::Bytes(bytes.to_vec()));
    }
    if value.is_instance_of::<PyDict>() {
        return structured_property(value);
    }
    if let Ok(items) = value.extract::<Vec<&PyAny>>() {
        if items.is_empty() {
            return Ok(if numpy_float32(value) {
                RustAttrValue::FloatVec(Vec::new())
            } else {
                RustAttrValue::Float64Vec(Vec::new())
            });
        }
        if items.iter().all(|item| item.is_instance_of::<PyBool>()) {
            return Ok(RustAttrValue::BoolVec(
                items
                    .iter()
                    .map(|item| item.extract())
                    .collect::<PyResult<_>>()?,
            ));
        }
        if items.iter().all(|item| {
            !item.is_instance_of::<PyBool>()
                && (item.is_instance_of::<PyLong>() || item.hasattr("__index__").unwrap_or(false))
        }) {
            return Ok(RustAttrValue::IntVec(
                items
                    .iter()
                    .map(|item| item.extract())
                    .collect::<PyResult<_>>()?,
            ));
        }
        if items.iter().all(|item| item.is_instance_of::<PyString>()) {
            return Ok(RustAttrValue::TextVec(
                items
                    .iter()
                    .map(|item| item.extract())
                    .collect::<PyResult<_>>()?,
            ));
        }
        if items
            .iter()
            .all(|item| !item.is_instance_of::<PyBool>() && item.extract::<f64>().is_ok())
        {
            if numpy_float32(value) || items.iter().all(|item| numpy_float32(item)) {
                return Ok(RustAttrValue::FloatVec(
                    items
                        .iter()
                        .map(|item| item.extract::<f32>())
                        .collect::<PyResult<_>>()?,
                ));
            }
            return Ok(RustAttrValue::Float64Vec(
                items
                    .iter()
                    .map(|item| item.extract::<f64>())
                    .collect::<PyResult<_>>()?,
            ));
        }
        return structured_property(value);
    }
    if value.hasattr("__index__")? {
        return Ok(RustAttrValue::Int(value.extract()?));
    }
    if let Ok(number) = value.extract::<f64>() {
        return Ok(if numpy_float32(value) {
            RustAttrValue::Float(number as f32)
        } else {
            RustAttrValue::Float64(number)
        });
    }
    structured_property(value)
}

/// Decode storage optimizations and structured values before exposing Python data.
pub fn attr_value_to_python_value(py: Python, attr_value: &RustAttrValue) -> PyResult<PyObject> {
    use pyo3::types::PyBytes;
    Ok(match attr_value {
        RustAttrValue::UtcInstant(value) => super::temporal::instant_to_python(py, *value)?,
        RustAttrValue::Date(value) => super::temporal::date_to_python(py, *value)?,
        RustAttrValue::Duration(value) => super::temporal::duration_to_python(py, *value)?,
        RustAttrValue::Int(value) => value.to_object(py),
        RustAttrValue::SmallInt(value) => value.to_object(py),
        RustAttrValue::Float(value) => value.to_object(py),
        RustAttrValue::Float64(value) => value.to_object(py),
        RustAttrValue::Float64Vec(value) => value.to_object(py),
        RustAttrValue::Bool(value) => value.to_object(py),
        RustAttrValue::Text(value) => value.to_object(py),
        RustAttrValue::CompactText(value) => value.as_str().to_object(py),
        RustAttrValue::CompressedText(value) => value
            .decompress_text()
            .map_err(PyValueError::new_err)?
            .to_object(py),
        RustAttrValue::FloatVec(value) => value.to_object(py),
        RustAttrValue::CompressedFloatVec(value) => value
            .decompress_float_vec()
            .map_err(PyValueError::new_err)?
            .to_object(py),
        RustAttrValue::IntVec(value) => value.to_object(py),
        RustAttrValue::TextVec(value) => value.to_object(py),
        RustAttrValue::BoolVec(value) => value.to_object(py),
        RustAttrValue::Bytes(value) => PyBytes::new(py, value).to_object(py),
        RustAttrValue::Json(value) => py
            .import("json")?
            .call_method1("loads", (value,))?
            .to_object(py),
        RustAttrValue::SubgraphRef(value) => value.to_object(py),
        RustAttrValue::NodeArray(value) => value.to_object(py),
        RustAttrValue::EdgeArray(value) => value.to_object(py),
        RustAttrValue::Null => py.None(),
    })
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
