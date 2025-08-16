//! FFI Type Wrappers
//! 
//! This module provides Python wrappers for core Groggy types.

use pyo3::prelude::*;
use groggy::{AttrValue as RustAttrValue, NodeId, EdgeId};

/// Python wrapper for AttrValue
#[pyclass(name = "AttrValue")]
#[derive(Debug, Clone)]
pub struct PyAttrValue {
    pub inner: RustAttrValue,
}

impl PyAttrValue {
    pub fn new(inner: RustAttrValue) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAttrValue {
    #[new]
    fn py_new(value: &PyAny) -> PyResult<Self> {
        // Use existing utility function
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self::new(attr_value))
    }
    
    fn __repr__(&self) -> String {
        match &self.inner {
            RustAttrValue::None => "None".to_string(),
            RustAttrValue::Bool(b) => b.to_string(),
            RustAttrValue::Int(i) => i.to_string(),
            RustAttrValue::Float(f) => f.to_string(),
            RustAttrValue::Text(s) => format!("'{}'", s),
            RustAttrValue::Bytes(b) => format!("bytes(len={})", b.len()),
            RustAttrValue::List(items) => {
                let item_reprs: Vec<String> = items.iter().map(|v| format!("{:?}", v)).collect();
                format!("[{}]", item_reprs.join(", "))
            }
        }
    }
}

// We'll add more types as needed during modularization
