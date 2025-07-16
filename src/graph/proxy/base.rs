//! ProxyAttributeManager: Concrete per-entity attribute interface for fine-grained get/set operations.
//! Used by both NodeProxy and EdgeProxy.

use crate::graph::managers::attributes::AttributeManager;
use crate::graph::types::{NodeId, EdgeId};
use serde_json::Value;
use pyo3::prelude::*;
use crate::utils::json::{python_to_json_value, json_value_to_python};
use pyo3::{Python, PyObject, PyAny};
#[pyclass]
pub struct NodeProxyAttributeManager {
    #[pyo3(get)]
    pub id: NodeId,
    #[pyo3(get)]
    pub attribute_manager: AttributeManager,
}

#[pymethods]
impl NodeProxyAttributeManager {
    #[new]
    pub fn new(id: NodeId, attribute_manager: AttributeManager) -> Self {
        Self { id, attribute_manager }
    }

    pub fn get(&self, attr_name: String, py: Python) -> Option<PyObject> {
        self.attribute_manager.get(&self.id.to_string(), &attr_name)
            .map(|v| json_value_to_python(&v, py))
    }

    pub fn set(&mut self, attr_name: String, py_value: &PyAny) -> bool {
        let value = python_to_json_value(py_value);
        self.attribute_manager.set(&self.id.to_string(), &attr_name, value).is_ok()
    }
}

#[pyclass]
pub struct EdgeProxyAttributeManager {
    #[pyo3(get)]
    pub id: EdgeId,
    #[pyo3(get)]
    pub attribute_manager: AttributeManager,
}

#[pymethods]
impl EdgeProxyAttributeManager {
    #[new]
    pub fn new(id: EdgeId, attribute_manager: AttributeManager) -> Self {
        Self { id, attribute_manager }
    }

    pub fn get(&self, attr_name: String, py: Python) -> Option<PyObject> {
        self.attribute_manager.get(&self.id.to_string(), &attr_name)
            .map(|v| json_value_to_python(&v, py))
    }

    pub fn set(&mut self, attr_name: String, py_value: &PyAny) -> bool {
        let value = python_to_json_value(py_value);
        self.attribute_manager.set(&self.id.to_string(), &attr_name, value).is_ok()
    }
}
