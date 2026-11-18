//! FFI wrappers for the Rust display system
//!
//! This module provides Python access to the pure Rust display formatters
//! without any circular dependencies.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyLong};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Python wrapper for DisplayConfig
#[pyclass(name = "DisplayConfig", module = "groggy.display")]
#[derive(Clone, Default)]
pub struct PyDisplayConfig {
    config: groggy::display::DisplayConfig,
}

#[pymethods]
impl PyDisplayConfig {
    /// Create new DisplayConfig with default settings
    #[new]
    #[pyo3(signature = (max_rows=10, max_cols=50, max_width=120, precision=6, use_color=true))]
    pub fn new(
        max_rows: usize,
        max_cols: usize,
        max_width: usize,
        precision: usize,
        use_color: bool,
    ) -> Self {
        Self {
            config: groggy::display::DisplayConfig {
                max_rows,
                max_cols,
                max_width,
                precision,
                use_color,
            },
        }
    }

    /// Create DisplayConfig with default values
    #[staticmethod]
    #[allow(clippy::should_implement_trait)] // Intentional: static method for Python API
    pub fn default() -> Self {
        Self {
            config: groggy::display::DisplayConfig::default(),
        }
    }

    /// Get max_rows setting
    #[getter]
    pub fn max_rows(&self) -> usize {
        self.config.max_rows
    }

    /// Set max_rows setting
    #[setter]
    pub fn set_max_rows(&mut self, value: usize) {
        self.config.max_rows = value;
    }

    /// Get max_cols setting
    #[getter]
    pub fn max_cols(&self) -> usize {
        self.config.max_cols
    }

    /// Set max_cols setting
    #[setter]
    pub fn set_max_cols(&mut self, value: usize) {
        self.config.max_cols = value;
    }

    /// Get precision setting
    #[getter]
    pub fn precision(&self) -> usize {
        self.config.precision
    }

    /// Set precision setting
    #[setter]
    pub fn set_precision(&mut self, value: usize) {
        self.config.precision = value;
    }

    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "DisplayConfig(max_rows={}, max_cols={}, max_width={}, precision={}, use_color={})",
            self.config.max_rows,
            self.config.max_cols,
            self.config.max_width,
            self.config.precision,
            self.config.use_color
        )
    }
}

impl PyDisplayConfig {
    /// Get reference to internal config (for internal use)
    pub(crate) fn get_config(&self) -> &groggy::display::DisplayConfig {
        &self.config
    }
}

/// Python wrapper for table formatting functions
#[pyclass(name = "TableFormatter", module = "groggy.display")]
pub struct PyTableFormatter;

#[pymethods]
impl PyTableFormatter {
    /// Format table data using Rust formatter
    #[staticmethod]
    pub fn format_table(
        py: Python<'_>,
        table_data: HashMap<String, PyObject>,
        config: Option<&PyDisplayConfig>,
    ) -> PyResult<String> {
        let default_config = groggy::display::DisplayConfig::default();
        let config = config
            .map(|c| c.get_config())
            .unwrap_or(&default_config)
            .clone();

        // Convert Python data to Rust format
        let rust_data = python_dict_to_rust_hashmap(table_data)?;

        // Call the Rust formatter
        Ok(py.allow_threads(move || groggy::display::format_table(rust_data, &config)))
    }

    /// Format any data structure automatically
    #[staticmethod]
    pub fn format_data_structure(
        py: Python<'_>,
        data: HashMap<String, PyObject>,
        data_type: Option<String>,
        config: Option<&PyDisplayConfig>,
    ) -> PyResult<String> {
        let default_config = groggy::display::DisplayConfig::default();
        let config = config
            .map(|c| c.get_config())
            .unwrap_or(&default_config)
            .clone();
        let rust_data = python_dict_to_rust_hashmap(data)?;

        Ok(py.allow_threads(move || {
            groggy::display::format_data_structure(rust_data, data_type.as_deref(), &config)
        }))
    }
}

/// Helper function to convert Python dict to Rust HashMap<String, serde_json::Value>
fn python_dict_to_rust_hashmap(
    py_dict: HashMap<String, PyObject>,
) -> PyResult<HashMap<String, Value>> {
    let mut rust_map = HashMap::new();

    for (key, py_value) in py_dict {
        let json_value = python_to_json_value(py_value)?;
        rust_map.insert(key, json_value);
    }

    Ok(rust_map)
}

/// Helper function to convert Python objects to serde_json::Value
fn python_to_json_value(py_obj: PyObject) -> PyResult<Value> {
    Python::with_gil(|py| {
        let py_any = py_obj.as_ref(py);

        if py_any.is_none() {
            Ok(Value::Null)
        } else if let Ok(s) = py_any.extract::<String>() {
            Ok(Value::String(s))
        } else if py_any.is_instance_of::<PyBool>() {
            Ok(Value::Bool(py_any.extract()?))
        } else if py_any.is_instance_of::<PyLong>() {
            if let Ok(i) = py_any.extract::<i64>() {
                Ok(Value::Number(i.into()))
            } else if let Ok(i) = py_any.extract::<u64>() {
                Ok(Value::Number(i.into()))
            } else {
                // JSON has no portable arbitrary-width integer. Preserve its
                // exact Python spelling rather than silently rounding via f64.
                Ok(Value::String(py_any.str()?.to_str()?.to_owned()))
            }
        } else if py_any.is_instance_of::<PyFloat>() {
            Ok(serde_json::Number::from_f64(py_any.extract()?)
                .map(Value::Number)
                .unwrap_or(Value::Null))
        } else if let Ok(i) = py_any.extract::<i64>() {
            // Retain the numeric protocol accepted by historical NumPy scalars.
            Ok(Value::Number(i.into()))
        } else if let Ok(i) = py_any.extract::<u64>() {
            Ok(Value::Number(i.into()))
        } else if let Ok(f) = py_any.extract::<f64>() {
            Ok(serde_json::Number::from_f64(f)
                .map(Value::Number)
                .unwrap_or(Value::Null))
        } else if let Ok(list) = py_any.extract::<Vec<PyObject>>() {
            let mut json_array = Vec::new();
            for item in list {
                json_array.push(python_to_json_value(item)?);
            }
            Ok(Value::Array(json_array))
        } else if let Ok(dict) = py_any.extract::<HashMap<String, PyObject>>() {
            let mut json_object = Map::new();
            for (k, v) in dict {
                json_object.insert(k, python_to_json_value(v)?);
            }
            Ok(Value::Object(json_object))
        } else {
            // Fallback: convert to string
            Ok(Value::String(py_any.str()?.to_str()?.to_owned()))
        }
    })
}

/// Helper trait to convert tables to display data
/// Future feature for display data conversion
#[allow(dead_code)]
pub trait ToDisplayData {
    fn to_display_data(&self) -> HashMap<String, Value>;
}

/// Register display functions with the Python module
pub fn register_display_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_display_positions, m)?)?;
    m.add_function(wrap_pyfunction!(_display_clip, m)?)?;
    m.add_function(wrap_pyfunction!(_display_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(_display_kind, m)?)?;
    m.add_function(wrap_pyfunction!(_display_widths, m)?)?;
    m.add_function(wrap_pyfunction!(_display_array, m)?)?;
    m.add_function(wrap_pyfunction!(_display_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(_display_table, m)?)?;
    // Add format_table function
    m.add_function(wrap_pyfunction!(format_table_function, m)?)?;
    m.add_function(wrap_pyfunction!(format_data_structure, m)?)?;
    m.add_function(wrap_pyfunction!(detect_display_type, m)?)?;

    Ok(())
}

/// Route a display dictionary through the canonical native formatter.
#[pyfunction]
#[pyo3(signature = (data, data_type=None, config=None))]
fn format_data_structure(
    py: Python<'_>,
    data: HashMap<String, PyObject>,
    data_type: Option<String>,
    config: Option<&PyDisplayConfig>,
) -> PyResult<String> {
    PyTableFormatter::format_data_structure(py, data, data_type, config)
}

/// Detect the native display kind without duplicating its classification policy.
#[pyfunction]
fn detect_display_type(data: HashMap<String, PyObject>) -> PyResult<&'static str> {
    let data = python_dict_to_rust_hashmap(data)?;
    Ok(groggy::display::detect_display_type(&data))
}

/// Standalone function for formatting tables (exported to Python)
#[pyfunction]
pub fn format_table_function(
    py: Python<'_>,
    table_data: HashMap<String, PyObject>,
    config: Option<PyDisplayConfig>,
) -> PyResult<String> {
    let default_config = groggy::display::DisplayConfig::default();
    let rust_config = config
        .as_ref()
        .map(|c| c.get_config())
        .unwrap_or(&default_config)
        .clone();

    // Convert Python data to Rust format
    let rust_data = python_dict_to_rust_hashmap(table_data)?;

    // Call the Rust formatter
    Ok(py.allow_threads(move || groggy::display::format_table(rust_data, &rust_config)))
}

/// Owned display plans and text cross the bridge; all layout work stays native.
#[pyfunction]
fn _display_positions(length: usize, budget: usize) -> Vec<Option<usize>> {
    groggy::display::render::interactive::positions(length, budget)
}

#[pyfunction]
fn _display_clip(text: &str, width: usize, ellipsis: &str) -> String {
    groggy::display::render::interactive::clip(text, width, ellipsis)
}

#[pyfunction]
fn _display_widths(headers: Vec<String>, rows: Vec<Vec<String>>, budget: usize) -> Vec<usize> {
    groggy::display::render::interactive::widths(&headers, &rows, budget)
}

#[pyfunction]
fn _display_array(
    py: Python<'_>,
    cells: Vec<(Option<usize>, String, bool)>,
    length: usize,
    name: String,
    dtype: String,
    shape: String,
) -> String {
    py.allow_threads(move || {
        groggy::display::render::interactive::array(&cells, length, &name, &dtype, &shape)
    })
}

#[pyfunction]
fn _display_matrix(
    py: Python<'_>,
    rows: Vec<Vec<String>>,
    empty: bool,
    shape: String,
    dtype: String,
) -> String {
    py.allow_threads(move || {
        groggy::display::render::interactive::matrix(&rows, empty, &shape, &dtype)
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Thin translation of one native display request.
fn _display_table(
    py: Python<'_>,
    headers: Vec<String>,
    types: Vec<String>,
    rows: Vec<Vec<String>>,
    empty: bool,
    summary: String,
    budget: usize,
    color: bool,
) -> String {
    py.allow_threads(move || {
        groggy::display::render::interactive::table(
            &headers, &types, &rows, empty, &summary, budget, color,
        )
    })
}

#[pyfunction]
fn _display_dtype(dtype: &str, hints: bool) -> String {
    groggy::display::format_dtype(dtype, hints)
}

#[pyfunction]
fn _display_kind(table: bool, sequence: bool, nested: bool) -> &'static str {
    groggy::display::detect_display_shape(table, sequence, nested)
}
