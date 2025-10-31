use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyNone};

use crate::ffi::types::PyAttrValue;

#[pyclass]
pub struct PyPipelineHandle {
    handle_id: usize,
}

struct PipelineEntry {
    pipeline: Arc<groggy::algorithms::Pipeline>,
    build_time_seconds: f64,
}

// Global pipeline registry using OnceLock for thread-safe initialization
static PIPELINE_REGISTRY: OnceLock<Mutex<HashMap<usize, PipelineEntry>>> = OnceLock::new();
static NEXT_ID: OnceLock<Mutex<usize>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<usize, PipelineEntry>> {
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

    let algo_registry = groggy::algorithms::global_registry();
    let build_start = Instant::now();
    let pipeline = builder
        .build(algo_registry)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let build_elapsed = build_start.elapsed();
    let build_time_seconds = build_elapsed.as_secs_f64();
    let id = next_id();
    registry().lock().unwrap().insert(
        id,
        PipelineEntry {
            pipeline: Arc::new(pipeline),
            build_time_seconds,
        },
    );
    Ok(PyPipelineHandle { handle_id: id })
}

#[pyfunction]
#[pyo3(name = "run_pipeline", signature = (handle, subgraph, *, persist_results = true))]
pub fn py_run_pipeline(
    py: Python,
    handle: &PyPipelineHandle,
    subgraph: &crate::ffi::subgraphs::subgraph::PySubgraph,
    persist_results: bool,
) -> PyResult<(crate::ffi::subgraphs::subgraph::PySubgraph, PyObject)> {
    let total_start = Instant::now();

    // Get the pipeline from registry
    let (pipeline, build_time_seconds) = {
        let registry_guard = registry().lock().unwrap();
        let entry = registry_guard
            .get(&handle.handle_id)
            .ok_or_else(|| PyRuntimeError::new_err("invalid pipeline handle"))?;
        (Arc::clone(&entry.pipeline), entry.build_time_seconds)
    };

    // Clone the inner subgraph for processing
    let clone_start = Instant::now();
    let subgraph_inner = subgraph.inner.clone();
    let clone_elapsed = clone_start.elapsed();

    let context_start = Instant::now();
    // NOTE: GIL Release Limitation
    // We cannot release the GIL here because Subgraph contains Rc<RefCell<Graph>>,
    // which is not Send. To enable GIL release for long-running algorithms:
    // 1. Refactor Graph to use Arc<RwLock<GraphInner>> instead of Rc<RefCell<Graph>>
    // 2. Ensure all Subgraph clones use Arc for thread-safe reference counting
    // 3. Update all graph operations to use RwLock instead of RefCell
    // This would allow: py.allow_threads(|| pipeline.run(...))
    // For now, Python threads will be blocked during algorithm execution.

    let mut context = groggy::algorithms::Context::new();
    context.set_persist_results(persist_results);
    let context_elapsed = context_start.elapsed();

    let run_start = Instant::now();
    let result = pipeline
        .run(&mut context, subgraph_inner)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let run_elapsed = run_start.elapsed();
    let run_time_seconds = run_elapsed.as_secs_f64();

    let post_run_start = Instant::now();

    let timers = context.timer_snapshot();
    let py_timers = PyDict::new(py);
    for (key, duration) in timers {
        py_timers.set_item(key, duration.as_secs_f64())?;
    }

    let call_counters = context.call_counter_snapshot();
    let py_calls = PyDict::new(py);
    for (key, counter) in call_counters {
        let sub = PyDict::new(py);
        sub.set_item("count", counter.count())?;
        sub.set_item("total", counter.total_duration().as_secs_f64())?;
        if let Some(avg) = counter.avg_duration() {
            sub.set_item("avg", avg.as_secs_f64())?;
        }
        py_calls.set_item(key, sub)?;
    }

    let stats_map = context.stat_snapshot();
    let py_stats = PyDict::new(py);
    for (key, value) in stats_map {
        py_stats.set_item(key, value)?;
    }

    let stats = PyDict::new(py);
    stats.set_item("build_time", build_time_seconds)?;
    stats.set_item("run_time", run_time_seconds)?;
    stats.set_item("timers", py_timers)?;
    stats.set_item("call_counters", py_calls)?;
    stats.set_item("stats", py_stats)?;
    stats.set_item("subgraph_clone_time", clone_elapsed.as_secs_f64())?;
    stats.set_item("persist_results", persist_results)?;

    let outputs = context.take_outputs();
    let py_outputs = PyDict::new(py);
    for (key, output) in outputs {
        match output {
            groggy::algorithms::AlgorithmOutput::Components(components) => {
                let py_components = PyList::empty(py);
                for component in components {
                    let py_nodes = PyList::empty(py);
                    for node in component {
                        py_nodes.append(node)?;
                    }
                    py_components.append(py_nodes)?;
                }
                py_outputs.set_item(key, py_components)?;
            }
        }
    }
    stats.set_item("outputs", py_outputs)?;

    let py_subgraph = crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(result)?;
    let post_run_elapsed = post_run_start.elapsed();

    let total_elapsed = total_start.elapsed();
    let total_secs = total_elapsed.as_secs_f64();
    let clone_secs = clone_elapsed.as_secs_f64();
    let context_secs = context_elapsed.as_secs_f64();
    let outputs_secs = post_run_elapsed.as_secs_f64();
    let overhead_secs =
        (total_secs - clone_secs - context_secs - run_time_seconds - outputs_secs).max(0.0);

    let ffi_timers = PyDict::new(py);
    ffi_timers.set_item("ffi.total", total_secs)?;
    ffi_timers.set_item("ffi.clone_subgraph", clone_secs)?;
    ffi_timers.set_item("ffi.context_setup", context_secs)?;
    ffi_timers.set_item("ffi.pipeline_run", run_time_seconds)?;
    ffi_timers.set_item("ffi.outputs_marshalling", outputs_secs)?;
    ffi_timers.set_item("ffi.overhead_misc", overhead_secs)?;
    stats.set_item("ffi_timers", ffi_timers)?;

    Ok((py_subgraph, stats.into()))
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
