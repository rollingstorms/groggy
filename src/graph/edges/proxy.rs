// src_new/graph/edges/proxy.rs
//! EdgeProxy: Per-edge interface for attribute access, endpoints, and graph operations in Groggy.
//! Designed for agent/LLM workflows and backend extensibility.

use pyo3::prelude::*;
use pyo3::types::PyString;
use crate::graph::types::{EdgeId, NodeId};
use crate::graph::managers::attributes::AttributeManager;
use crate::graph::proxy::base::EdgeProxyAttributeManager;
// use serde_json::Value; // Currently unused

#[pyclass]
pub struct EdgeProxy {
    #[pyo3(get)]
    pub edge_id: EdgeId,
    #[pyo3(get)]
    pub source: NodeId,
    #[pyo3(get)]
    pub target: NodeId,
    #[pyo3(get)]
    pub attribute_manager: AttributeManager,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}

#[pymethods]
impl EdgeProxy {
    /// Create a new EdgeProxy from Python (simplified constructor)
    #[new]
    pub fn py_new(_py_edge_id: &PyAny, py_source: &PyAny, py_target: &PyAny) -> PyResult<Self> {
        // Extract source and target NodeId
        let source: NodeId = if let Ok(id_str) = py_source.extract::<String>() {
            NodeId::new(id_str)
        } else if let Ok(id_int) = py_source.extract::<i64>() {
            NodeId::new(id_int.to_string())
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Source NodeId must be str or int"));
        };
        let target: NodeId = if let Ok(id_str) = py_target.extract::<String>() {
            NodeId::new(id_str)
        } else if let Ok(id_int) = py_target.extract::<i64>() {
            NodeId::new(id_int.to_string())
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Target NodeId must be str or int"));
        };
        let edge_id = EdgeId::new(source.clone(), target.clone());
        // For demo/test, create new GraphStore and AttributeManager internally
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let attribute_manager = AttributeManager::new_with_graph_store(graph_store.clone());
        Ok(Self { edge_id, source, target, attribute_manager, graph_store })
    }

    /// Returns a ProxyAttributeManager for this edge (per-attribute API).
    pub fn attr_manager(&self) -> EdgeProxyAttributeManager {
        EdgeProxyAttributeManager::new(self.edge_id.clone(), self.attribute_manager.clone())
    }

    /// Get the value of a single attribute for this edge (JSON string).
    pub fn get_attr(&self, py_attr_name: &PyAny, py: Python) -> PyResult<Option<PyObject>> {
        let attr_name: String = py_attr_name.extract()?;
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            if let Some(val) = self.attribute_manager.get_edge_value(attr_name, index) {
                let py_val = PyString::new(py, &serde_json::to_string(&val).unwrap_or_default()).into_py(py);
                Ok(Some(py_val))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Set the value of a single attribute for this edge (JSON string).
    pub fn set_attr(&mut self, py_attr_name: &PyAny, py_value: &PyAny, py: Python) -> PyResult<()> {
        let attr_name: String = py_attr_name.extract()?;
        // Convert Python value to serde_json::Value
        let json_value = if let Ok(s) = py_value.extract::<String>() {
            serde_json::Value::String(s)
        } else if let Ok(i) = py_value.extract::<i64>() {
            serde_json::Value::Number(i.into())
        } else if let Ok(f) = py_value.extract::<f64>() {
            serde_json::Number::from_f64(f).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
        } else if let Ok(b) = py_value.extract::<bool>() {
            serde_json::Value::Bool(b)
        } else {
            // Fallback: use Python's json.dumps
            let json_mod = py.import("json")?;
            let json_str: String = json_mod.call_method1("dumps", (py_value,))?.extract()?;
            serde_json::from_str(&json_str).unwrap_or(serde_json::Value::Null)
        };
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            self.attribute_manager.set_edge_value(attr_name, index, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Edge not found in graph"))
        }
    }

    /// Get all attributes for this edge as a map (JSON string).
    pub fn attrs(&self, py: Python) -> PyResult<PyObject> {
        if let Some(index) = self.graph_store.edge_index(&self.edge_id) {
            let mut map = serde_json::Map::new();
            for attr in self.attribute_manager.columnar.edge_attr_names() {
                if let Some(val) = self.attribute_manager.get_edge_value(attr.clone(), index) {
                    map.insert(attr, val);
                }
            }
            let py_dict = pyo3::types::PyDict::new(py);
            for (k, v) in map {
                let py_val = pyo3::types::PyString::new(py, &serde_json::to_string(&v).unwrap_or_default());
                py_dict.set_item(k, py_val)?;
            }
            Ok(py_dict.into())
        } else {
            Ok(pyo3::types::PyDict::new(py).into())
        }
    }

    /// Returns a string representation of this edge (for debugging or display).
    pub fn __str__(&self) -> String {
        format!("EdgeProxy({}, {} -> {})", self.edge_id, self.source, self.target)
    }
}

impl EdgeProxy {
    /// Regular Rust constructor - not exposed to Python
    pub fn new(edge_id: EdgeId, source: NodeId, target: NodeId, attribute_manager: AttributeManager, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>) -> Self {
        Self { edge_id, source, target, attribute_manager, graph_store }
    }
}

// Helper: EdgeProxy factory methods: (create new edges with attribute/type-guided construction)
