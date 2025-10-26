//! FFI wrappers for the Rust display system
//!
//! This module provides Python access to the pure Rust display formatters
//! without any circular dependencies.

use pyo3::prelude::*;
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
        table_data: HashMap<String, PyObject>,
        config: Option<&PyDisplayConfig>,
    ) -> PyResult<String> {
        let default_config = groggy::display::DisplayConfig::default();
        let config = config.map(|c| c.get_config()).unwrap_or(&default_config);

        // Convert Python data to Rust format
        let rust_data = python_dict_to_rust_hashmap(table_data)?;

        // Call the Rust formatter
        Ok(groggy::display::format_table(rust_data, config))
    }

    /// Format any data structure automatically
    #[staticmethod]
    pub fn format_data_structure(
        data: HashMap<String, PyObject>,
        data_type: Option<String>,
        config: Option<&PyDisplayConfig>,
    ) -> PyResult<String> {
        let default_config = groggy::display::DisplayConfig::default();
        let config = config.map(|c| c.get_config()).unwrap_or(&default_config);
        let rust_data = python_dict_to_rust_hashmap(data)?;

        Ok(groggy::display::format_data_structure(
            rust_data,
            data_type.as_deref(),
            config,
        ))
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

        if let Ok(s) = py_any.extract::<String>() {
            Ok(Value::String(s))
        } else if let Ok(i) = py_any.extract::<i64>() {
            Ok(Value::Number(serde_json::Number::from(i)))
        } else if let Ok(f) = py_any.extract::<f64>() {
            Ok(Value::Number(
                serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)),
            ))
        } else if let Ok(b) = py_any.extract::<bool>() {
            Ok(Value::Bool(b))
        } else if py_any.is_none() {
            Ok(Value::Null)
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
            Ok(Value::String(format!("{}", py_any)))
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
    // Add format_table function
    m.add_function(wrap_pyfunction!(format_table_function, m)?)?;

    Ok(())
}

/// Standalone function for formatting tables (exported to Python)
#[pyfunction]
pub fn format_table_function(
    table_data: HashMap<String, PyObject>,
    config: Option<PyDisplayConfig>,
) -> PyResult<String> {
    let default_config = groggy::display::DisplayConfig::default();
    let rust_config = config
        .as_ref()
        .map(|c| c.get_config())
        .unwrap_or(&default_config);

    // Convert Python data to Rust format
    let rust_data = python_dict_to_rust_hashmap(table_data)?;

    // Call the Rust formatter
    Ok(groggy::display::format_table(rust_data, rust_config))
}
