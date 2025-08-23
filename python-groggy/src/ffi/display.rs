//! Display FFI Module
//!
//! Python bindings for display formatting functionality.

use groggy::display::{
    detect_display_type, format_array, format_data_structure, format_matrix, format_table,
    DisplayConfig,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::collections::HashMap;

/// Python wrapper for DisplayConfig
#[pyclass(name = "DisplayConfig")]
#[derive(Clone)]
pub struct PyDisplayConfig {
    pub inner: DisplayConfig,
}

#[pymethods]
impl PyDisplayConfig {
    #[new]
    #[pyo3(signature = (max_rows = None, max_cols = None, max_width = None, precision = None, use_color = None))]
    fn new(
        max_rows: Option<usize>,
        max_cols: Option<usize>,
        max_width: Option<usize>,
        precision: Option<usize>,
        use_color: Option<bool>,
    ) -> Self {
        let mut config = DisplayConfig::default();

        if let Some(rows) = max_rows {
            config.max_rows = rows;
        }
        if let Some(cols) = max_cols {
            config.max_cols = cols;
        }
        if let Some(width) = max_width {
            config.max_width = width;
        }
        if let Some(prec) = precision {
            config.precision = prec;
        }
        if let Some(color) = use_color {
            config.use_color = color;
        }

        Self { inner: config }
    }

    #[getter]
    fn max_rows(&self) -> usize {
        self.inner.max_rows
    }

    #[setter]
    fn set_max_rows(&mut self, value: usize) {
        self.inner.max_rows = value;
    }

    #[getter]
    fn max_cols(&self) -> usize {
        self.inner.max_cols
    }

    #[setter]
    fn set_max_cols(&mut self, value: usize) {
        self.inner.max_cols = value;
    }

    #[getter]
    fn max_width(&self) -> usize {
        self.inner.max_width
    }

    #[setter]
    fn set_max_width(&mut self, value: usize) {
        self.inner.max_width = value;
    }

    #[getter]
    fn precision(&self) -> usize {
        self.inner.precision
    }

    #[setter]
    fn set_precision(&mut self, value: usize) {
        self.inner.precision = value;
    }

    #[getter]
    fn use_color(&self) -> bool {
        self.inner.use_color
    }

    #[setter]
    fn set_use_color(&mut self, value: bool) {
        self.inner.use_color = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "DisplayConfig(max_rows={}, max_cols={}, max_width={}, precision={}, use_color={})",
            self.inner.max_rows,
            self.inner.max_cols,
            self.inner.max_width,
            self.inner.precision,
            self.inner.use_color
        )
    }
}

/// Convert PyDict to HashMap<String, Value> for display functions
fn pydict_to_hashmap(py_dict: &PyDict) -> PyResult<HashMap<String, Value>> {
    let mut map = HashMap::new();

    for (key, value) in py_dict.iter() {
        let key_str: String = key.extract()?;
        let json_value = python_to_json_value(value)?;
        map.insert(key_str, json_value);
    }

    Ok(map)
}

/// Convert Python value to serde_json::Value
fn python_to_json_value(py_value: &PyAny) -> PyResult<Value> {
    if let Ok(s) = py_value.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(i) = py_value.extract::<i64>() {
        Ok(Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = py_value.extract::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            Ok(Value::Number(num))
        } else {
            Ok(Value::Null)
        }
    } else if let Ok(b) = py_value.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if py_value.is_none() {
        Ok(Value::Null)
    } else if let Ok(list) = py_value.extract::<Vec<&PyAny>>() {
        let mut json_array = Vec::new();
        for item in list {
            json_array.push(python_to_json_value(item)?);
        }
        Ok(Value::Array(json_array))
    } else {
        // Fallback: convert to string
        Ok(Value::String(format!("{:?}", py_value)))
    }
}

/// Python function to format array data
#[pyfunction]
#[pyo3(signature = (data, config = None))]
pub fn py_format_array(data: &PyDict, config: Option<&PyDisplayConfig>) -> PyResult<String> {
    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_array(data_map, display_config))
}

/// Python function to format matrix data
#[pyfunction]
#[pyo3(signature = (data, config = None))]
pub fn py_format_matrix(data: &PyDict, config: Option<&PyDisplayConfig>) -> PyResult<String> {
    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_matrix(data_map, display_config))
}

/// Python function to format table data
#[pyfunction]
#[pyo3(signature = (data, config = None))]
pub fn py_format_table(data: &PyDict, config: Option<&PyDisplayConfig>) -> PyResult<String> {
    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_table(data_map, display_config))
}

/// Python function to auto-detect and format data structure
#[pyfunction]
#[pyo3(signature = (data, data_type = None, config = None))]
pub fn py_format_data_structure(
    data: &PyDict,
    data_type: Option<&str>,
    config: Option<&PyDisplayConfig>,
) -> PyResult<String> {
    let data_map = pydict_to_hashmap(data)?;
    let default_config = DisplayConfig::default();
    let display_config = config.map(|c| &c.inner).unwrap_or(&default_config);

    Ok(format_data_structure(data_map, data_type, display_config))
}

/// Python function to detect display type
#[pyfunction]
pub fn py_detect_display_type(data: &PyDict) -> PyResult<String> {
    let data_map = pydict_to_hashmap(data)?;
    Ok(detect_display_type(&data_map).to_string())
}

/// Register display functions with the Python module
pub fn register_display_functions(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register display config class
    m.add_class::<PyDisplayConfig>()?;

    // Register formatting functions
    m.add_function(wrap_pyfunction!(py_format_array, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_table, m)?)?;
    m.add_function(wrap_pyfunction!(py_format_data_structure, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_display_type, m)?)?;

    // Add aliases for Python compatibility
    m.add("format_array", m.getattr("py_format_array")?)?;
    m.add("format_matrix", m.getattr("py_format_matrix")?)?;
    m.add("format_table", m.getattr("py_format_table")?)?;
    m.add(
        "format_data_structure",
        m.getattr("py_format_data_structure")?,
    )?;
    m.add("detect_display_type", m.getattr("py_detect_display_type")?)?;

    Ok(())
}
