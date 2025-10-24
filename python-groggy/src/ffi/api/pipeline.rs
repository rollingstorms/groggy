use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyNone;

use crate::ffi::types::PyAttrValue;

#[pyclass]
pub struct PyPipelineHandle {
    handle_id: usize,
}

// Global pipeline registry using OnceLock for thread-safe initialization
static PIPELINE_REGISTRY: OnceLock<Mutex<HashMap<usize, groggy::algorithms::Pipeline>>> =
    OnceLock::new();
static NEXT_ID: OnceLock<Mutex<usize>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<usize, groggy::algorithms::Pipeline>> {
    PIPELINE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_id() -> usize {
    let next_id_mutex = NEXT_ID.get_or_init(|| Mutex::new(1));
    let mut id = next_id_mutex.lock().unwrap();
    let current = *id;
    *id += 1;
    current
}

#[pyfunction]
#[pyo3(name = "build_pipeline")]
pub fn py_build_pipeline(py: Python, spec: &PyAny) -> PyResult<PyPipelineHandle> {
    groggy::algorithms::ensure_algorithms_registered();
    let mut builder = groggy::algorithms::PipelineBuilder::new();

    // Expect spec as iterable of dicts {"id": str, "params": dict}
    for item in spec.iter()? {
        let item = item?;
        let id: String = item.get_item("id")?.extract()?;
        let params_any = item
            .get_item("params")
            .unwrap_or_else(|_| PyNone::get(py).into());
        let params_dict: HashMap<String, PyAttrValue> = if params_any.is_none() {
            HashMap::new()
        } else {
            params_any.extract()?
        };
        builder = builder.with_algorithm(id, |params| {
            for (key, value) in &params_dict {
                // Convert PyAttrValue (wrapping AttrValue) to AlgorithmParamValue
                let param_value = match &value.inner {
                    groggy::types::AttrValue::Int(i) => {
                        groggy::algorithms::AlgorithmParamValue::Int(*i)
                    }
                    groggy::types::AttrValue::SmallInt(i) => {
                        groggy::algorithms::AlgorithmParamValue::Int(*i as i64)
                    }
                    groggy::types::AttrValue::Float(f) => {
                        groggy::algorithms::AlgorithmParamValue::Float(*f as f64)
                    }
                    groggy::types::AttrValue::Bool(b) => {
                        groggy::algorithms::AlgorithmParamValue::Bool(*b)
                    }
                    groggy::types::AttrValue::Text(s) => {
                        groggy::algorithms::AlgorithmParamValue::Text(s.clone())
                    }
                    groggy::types::AttrValue::CompactText(s) => {
                        groggy::algorithms::AlgorithmParamValue::Text(s.as_str().to_string())
                    }
                    groggy::types::AttrValue::CompressedText(data) => {
                        // Decompress and convert
                        if let Ok(text) = data.decompress_text() {
                            groggy::algorithms::AlgorithmParamValue::Text(text)
                        } else {
                            continue; // Skip decompression errors
                        }
                    }
                    groggy::types::AttrValue::IntVec(v) => {
                        groggy::algorithms::AlgorithmParamValue::IntList(v.clone())
                    }
                    groggy::types::AttrValue::FloatVec(v) => {
                        groggy::algorithms::AlgorithmParamValue::FloatList(
                            v.iter().map(|&f| f as f64).collect(),
                        )
                    }
                    groggy::types::AttrValue::CompressedFloatVec(data) => {
                        // Decompress float vec
                        if let Ok(floats) = data.decompress_float_vec() {
                            groggy::algorithms::AlgorithmParamValue::FloatList(
                                floats.iter().map(|&f| f as f64).collect(),
                            )
                        } else {
                            continue; // Skip decompression errors
                        }
                    }
                    groggy::types::AttrValue::BoolVec(v) => {
                        groggy::algorithms::AlgorithmParamValue::BoolList(v.clone())
                    }
                    groggy::types::AttrValue::TextVec(v) => {
                        groggy::algorithms::AlgorithmParamValue::TextList(v.clone())
                    }
                    groggy::types::AttrValue::Json(json_str) => {
                        match serde_json::from_str::<serde_json::Value>(json_str) {
                            Ok(value) => groggy::algorithms::AlgorithmParamValue::Json(value),
                            Err(_) => continue,
                        }
                    }
                    // Skip types that don't have algorithm param equivalents
                    groggy::types::AttrValue::Bytes(_)
                    | groggy::types::AttrValue::Null
                    | groggy::types::AttrValue::SubgraphRef(_)
                    | groggy::types::AttrValue::NodeArray(_)
                    | groggy::types::AttrValue::EdgeArray(_) => {
                        continue;
                    }
                };
                params.insert(key.clone(), param_value);
            }
        });
    }

    let pipeline = builder
        .build(groggy::algorithms::global_registry())
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let id = next_id();
    registry().lock().unwrap().insert(id, pipeline);
    Ok(PyPipelineHandle { handle_id: id })
}

#[pyfunction]
#[pyo3(name = "run_pipeline")]
pub fn py_run_pipeline(
    _py: Python,
    handle: &PyPipelineHandle,
    subgraph: &crate::ffi::subgraphs::subgraph::PySubgraph,
) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph> {
    // Get the pipeline from registry
    let registry_guard = registry().lock().unwrap();
    let pipeline = registry_guard
        .get(&handle.handle_id)
        .ok_or_else(|| PyRuntimeError::new_err("invalid pipeline handle"))?;

    // Clone the inner subgraph for processing
    let subgraph_inner = subgraph.inner.clone();

    // NOTE: GIL Release Limitation
    // We cannot release the GIL here because Subgraph contains Rc<RefCell<Graph>>,
    // which is not Send. To enable GIL release for long-running algorithms:
    // 1. Refactor Graph to use Arc<RwLock<GraphInner>> instead of Rc<RefCell<Graph>>
    // 2. Ensure all Subgraph clones use Arc for thread-safe reference counting
    // 3. Update all graph operations to use RwLock instead of RefCell
    // This would allow: py.allow_threads(|| pipeline.run(...))
    // For now, Python threads will be blocked during algorithm execution.

    let mut context = groggy::algorithms::Context::new();
    let result = pipeline
        .run(&mut context, subgraph_inner)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    // Wrap result back in PySubgraph
    // Attribute updates from algorithms are automatically included in the result
    crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(result)
}

#[pyfunction]
#[pyo3(name = "get_pipeline_context_info")]
pub fn py_get_pipeline_context_info() -> PyResult<HashMap<String, PyAttrValue>> {
    // Return information about pipeline execution capabilities
    let mut info = HashMap::new();

    info.insert(
        "supports_gil_release".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Bool(false)),
    );
    info.insert(
        "reason".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Text(
            "Subgraph uses Rc<RefCell<Graph>> which is not Send. \
             Refactor to Arc<RwLock<>> to enable parallel execution."
                .to_string(),
        )),
    );
    info.insert(
        "incremental_updates".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Bool(true)),
    );
    info.insert(
        "bulk_attribute_optimization".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Bool(true)),
    );

    Ok(info)
}

#[pyfunction]
#[pyo3(name = "drop_pipeline")]
pub fn py_drop_pipeline(handle: &PyPipelineHandle) {
    registry().lock().unwrap().remove(&handle.handle_id);
}

#[pyfunction]
#[pyo3(name = "get_algorithm_metadata")]
pub fn py_get_algorithm_metadata(algorithm_id: &str) -> PyResult<HashMap<String, PyAttrValue>> {
    groggy::algorithms::ensure_algorithms_registered();
    let registry = groggy::algorithms::global_registry();

    let metadata = registry.metadata(algorithm_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!("Algorithm '{}' not found", algorithm_id))
    })?;

    let mut map = HashMap::new();
    map.insert(
        "id".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Text(metadata.id)),
    );
    map.insert(
        "name".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Text(metadata.name)),
    );
    map.insert(
        "description".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Text(metadata.description)),
    );
    map.insert(
        "version".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Text(metadata.version)),
    );
    map.insert(
        "supports_cancellation".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Bool(
            metadata.supports_cancellation,
        )),
    );
    map.insert(
        "cost_hint".to_string(),
        PyAttrValue::new(groggy::types::AttrValue::Text(format!(
            "{:?}",
            metadata.cost_hint
        ))),
    );

    // Return structured parameter metadata
    if !metadata.parameters.is_empty() {
        // Build a JSON representation of parameters manually
        let params_json_parts: Vec<String> = metadata
            .parameters
            .iter()
            .map(|param| {
                let mut fields = vec![
                    format!(r#""name":"{}""#, param.name),
                    format!(
                        r#""description":"{}""#,
                        param.description.replace('"', r#"\""#)
                    ),
                    format!(r#""type":"{}""#, format!("{:?}", param.value_type)),
                    format!(r#""required":{}"#, param.required),
                ];

                // Include default value if present
                if let Some(ref default) = param.default_value {
                    let default_text = match default {
                        groggy::algorithms::AlgorithmParamValue::Int(i) => i.to_string(),
                        groggy::algorithms::AlgorithmParamValue::Float(f) => f.to_string(),
                        groggy::algorithms::AlgorithmParamValue::Bool(b) => b.to_string(),
                        groggy::algorithms::AlgorithmParamValue::Text(s) => {
                            format!(r#""{}""#, s.replace('"', r#"\""#))
                        }
                        groggy::algorithms::AlgorithmParamValue::IntList(v) => format!("{:?}", v),
                        groggy::algorithms::AlgorithmParamValue::FloatList(v) => format!("{:?}", v),
                        groggy::algorithms::AlgorithmParamValue::BoolList(v) => format!("{:?}", v),
                        groggy::algorithms::AlgorithmParamValue::TextList(v) => format!("{:?}", v),
                        groggy::algorithms::AlgorithmParamValue::Json(v) => v.to_string(),
                        groggy::algorithms::AlgorithmParamValue::None => "null".to_string(),
                    };
                    fields.push(format!(r#""default":{}"#, default_text));
                }

                format!("{{{}}}", fields.join(","))
            })
            .collect();

        let params_json = format!("[{}]", params_json_parts.join(","));
        map.insert(
            "parameters".to_string(),
            PyAttrValue::new(groggy::types::AttrValue::Text(params_json)),
        );
    }

    Ok(map)
}

#[pyfunction]
#[pyo3(name = "validate_algorithm_params")]
pub fn py_validate_algorithm_params(
    algorithm_id: &str,
    params: HashMap<String, PyAttrValue>,
) -> PyResult<Vec<String>> {
    groggy::algorithms::ensure_algorithms_registered();
    let registry = groggy::algorithms::global_registry();

    let metadata = registry.metadata(algorithm_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!("Algorithm '{}' not found", algorithm_id))
    })?;

    let mut errors = Vec::new();

    // Check for missing required parameters
    for param_meta in &metadata.parameters {
        if param_meta.required && !params.contains_key(&param_meta.name) {
            errors.push(format!(
                "Missing required parameter '{}' (type: {:?})",
                param_meta.name, param_meta.value_type
            ));
        }
    }

    // Check for unknown parameters
    let valid_param_names: std::collections::HashSet<String> =
        metadata.parameters.iter().map(|p| p.name.clone()).collect();

    for param_name in params.keys() {
        if !valid_param_names.contains(param_name) {
            errors.push(format!("Unknown parameter '{}'", param_name));
        }
    }

    // Type validation (basic)
    for param_meta in &metadata.parameters {
        if let Some(value) = params.get(&param_meta.name) {
            let type_matches = matches!(
                (&param_meta.value_type, &value.inner),
                (
                    groggy::algorithms::ParameterType::Int,
                    groggy::types::AttrValue::Int(_)
                ) | (
                    groggy::algorithms::ParameterType::Int,
                    groggy::types::AttrValue::SmallInt(_)
                ) | (
                    groggy::algorithms::ParameterType::Float,
                    groggy::types::AttrValue::Float(_)
                ) | (
                    groggy::algorithms::ParameterType::Bool,
                    groggy::types::AttrValue::Bool(_)
                ) | (
                    groggy::algorithms::ParameterType::Text,
                    groggy::types::AttrValue::Text(_)
                ) | (
                    groggy::algorithms::ParameterType::Text,
                    groggy::types::AttrValue::CompactText(_)
                ) | (
                    groggy::algorithms::ParameterType::IntList,
                    groggy::types::AttrValue::IntVec(_)
                ) | (
                    groggy::algorithms::ParameterType::FloatList,
                    groggy::types::AttrValue::FloatVec(_)
                ) | (
                    groggy::algorithms::ParameterType::BoolList,
                    groggy::types::AttrValue::BoolVec(_)
                ) | (
                    groggy::algorithms::ParameterType::TextList,
                    groggy::types::AttrValue::TextVec(_)
                ) | (
                    groggy::algorithms::ParameterType::Json,
                    groggy::types::AttrValue::Json(_)
                )
            );

            if !type_matches {
                errors.push(format!(
                    "Parameter '{}' has wrong type: expected {:?}, got {:?}",
                    param_meta.name,
                    param_meta.value_type,
                    value.inner.type_name()
                ));
            }
        }
    }

    Ok(errors)
}

#[pyfunction]
#[pyo3(name = "list_algorithm_categories")]
pub fn py_list_algorithm_categories() -> PyResult<HashMap<String, Vec<String>>> {
    groggy::algorithms::ensure_algorithms_registered();
    let registry = groggy::algorithms::global_registry();

    let mut categories: HashMap<String, Vec<String>> = HashMap::new();

    for metadata in registry.list() {
        // Extract category from algorithm ID (e.g., "centrality.pagerank" -> "centrality")
        let parts: Vec<&str> = metadata.id.split('.').collect();
        if parts.len() >= 2 {
            let category = parts[0].to_string();
            categories
                .entry(category)
                .or_default()
                .push(metadata.id.clone());
        } else {
            // Uncategorized algorithms go into "other"
            categories
                .entry("other".to_string())
                .or_default()
                .push(metadata.id.clone());
        }
    }

    Ok(categories)
}

#[pyfunction]
#[pyo3(name = "list_algorithms")]
pub fn py_list_algorithms() -> PyResult<Vec<HashMap<String, PyAttrValue>>> {
    groggy::algorithms::ensure_algorithms_registered();
    let registry = groggy::algorithms::global_registry();
    let mut result = Vec::new();
    for metadata in registry.list() {
        let mut map = HashMap::new();
        map.insert(
            "id".to_string(),
            PyAttrValue::new(groggy::types::AttrValue::Text(metadata.id.to_string())),
        );
        map.insert(
            "name".to_string(),
            PyAttrValue::new(groggy::types::AttrValue::Text(metadata.name.to_string())),
        );
        map.insert(
            "description".to_string(),
            PyAttrValue::new(groggy::types::AttrValue::Text(
                metadata.description.to_string(),
            )),
        );
        map.insert(
            "version".to_string(),
            PyAttrValue::new(groggy::types::AttrValue::Text(metadata.version.to_string())),
        );
        map.insert(
            "supports_cancellation".to_string(),
            PyAttrValue::new(groggy::types::AttrValue::Bool(
                metadata.supports_cancellation,
            )),
        );

        if !metadata.parameters.is_empty() {
            let params_text = metadata
                .parameters
                .iter()
                .map(|param| {
                    format!(
                        "{}:{:?} ({}required)",
                        param.name,
                        param.value_type,
                        if param.required { "" } else { "not " }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            map.insert(
                "parameters".to_string(),
                PyAttrValue::new(groggy::types::AttrValue::Text(params_text)),
            );
        }
        result.push(map);
    }
    Ok(result)
}

#[pymodule]
pub fn pipeline(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPipelineHandle>()?;
    m.add_function(wrap_pyfunction!(py_build_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(py_drop_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_algorithms, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_algorithm_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_validate_algorithm_params, m)?)?;
    m.add_function(wrap_pyfunction!(py_list_algorithm_categories, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_pipeline_context_info, m)?)?;
    Ok(())
}
