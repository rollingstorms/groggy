// src_new/graph/nodes/proxy.rs
//! NodeProxy: Per-node interface for attribute access, neighbors, and graph operations in Groggy.
//! Designed for agent/LLM workflows and backend extensibility.

use pyo3::prelude::*;
use pyo3::types::PyString;
use crate::graph::types::NodeId;
use crate::graph::managers::attributes::AttributeManager;

#[pyclass]
pub struct NodeProxy {
    #[pyo3(get)]
    pub node_id: NodeId,
    pub attribute_manager: std::sync::Arc<AttributeManager>,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}


#[pymethods]
impl NodeProxy {
    /// Create a new NodeProxy from Python (simplified constructor)
    #[new]
    pub fn py_new(py_node_id: &PyAny) -> PyResult<Self> {
        // Convert node_id from Python (str/int)
        let node_id: NodeId = if let Ok(id_str) = py_node_id.extract::<String>() {
            NodeId::new(id_str)
        } else if let Ok(id_int) = py_node_id.extract::<i64>() {
            NodeId::new(id_int.to_string())
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("NodeId must be str or int"));
        };
        // For demo/test, create a new GraphStore internally
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let attribute_manager = std::sync::Arc::new(AttributeManager::new_with_graph_store(graph_store.clone()));
        Ok(Self { node_id, attribute_manager, graph_store })
    }

    /// Returns a ProxyAttributeManager for this node (per-attribute API).
    pub fn attr_manager(&self) -> crate::graph::nodes::proxy::ProxyAttributeManager {
        crate::graph::nodes::proxy::ProxyAttributeManager {
            node_id: self.node_id.clone(),
            attribute_manager: self.attribute_manager.clone(),
            graph_store: self.graph_store.clone(),
        }
    }

    /// Get the value of a single attribute for this node (JSON).
    pub fn get_attr(&self, py_attr_name: &PyAny, py: Python) -> PyResult<Option<PyObject>> {
        let attr_name: String = py_attr_name.extract()?;
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            if let Some(val) = self.attribute_manager.get_node_value(attr_name, index) {
                let py_val = PyString::new(py, &serde_json::to_string(&val).unwrap_or_default()).into_py(py);
                Ok(Some(py_val))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Set the value of a single attribute for this node (JSON).
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
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.set_node_value(attr_name, index, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }

    /// Get all attributes for this node as a map (JSON string).
    pub fn attrs(&self, py: Python) -> PyResult<PyObject> {
        let mut map = serde_json::Map::new();
        for attr in self.attribute_manager.columnar.node_attr_names() {
            if let Some(val) = self.attribute_manager.get_node_value(attr.clone(), self.graph_store.node_index(&self.node_id).unwrap_or(0)) {
                map.insert(attr, val);
            }
        }
        let py_dict = pyo3::types::PyDict::new(py);
        for (k, v) in map {
            let py_val = pyo3::types::PyString::new(py, &serde_json::to_string(&v).unwrap_or_default());
            py_dict.set_item(k, py_val)?;
        }
        Ok(py_dict.into())
    }

    /// Returns a string representation of this node (for debugging or display).
    pub fn __str__(&self) -> String {
        format!("NodeProxy({})", self.node_id)
    }
}

impl NodeProxy {
    /// Regular Rust constructor - not exposed to Python
    pub fn new(node_id: NodeId, graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>) -> Self {
        let attribute_manager = std::sync::Arc::new(AttributeManager::new_with_graph_store(graph_store.clone()));
        Self { node_id, attribute_manager, graph_store }
    }
}

/// ProxyAttributeManager: Per-node attribute interface for fine-grained get/set operations.
#[pyclass]
pub struct ProxyAttributeManager {
    pub node_id: NodeId,
    pub attribute_manager: std::sync::Arc<AttributeManager>,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
}

#[pymethods]
impl ProxyAttributeManager {
    /// Returns the value of the specified attribute for this node as JSON string.
    pub fn get(&self, attr_name: String) -> Option<String> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.get_node_value(attr_name, index)
                .map(|v| serde_json::to_string(&v).unwrap_or_default())
        } else {
            None
        }
    }

    /// Sets the value of the specified attribute for this node (JSON string, type-checked).
    pub fn set(&mut self, attr_name: String, value: String) -> PyResult<()> {
        let json_value: serde_json::Value = serde_json::from_str(&value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e)))?;
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.set_node_value(attr_name, index, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }

    /// Returns the value of the specified int attribute for this node.
    pub fn get_int(&self, attr_name: String) -> Option<i64> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.get_node_int(attr_name, index)
        } else {
            None
        }
    }
    /// Sets the value of the specified int attribute for this node.
    pub fn set_int(&mut self, attr_name: String, value: i64) -> PyResult<()> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.set_node_int(attr_name, index, value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }
    /// Returns the value of the specified float attribute for this node.
    pub fn get_float(&self, attr_name: String) -> Option<f64> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.get_node_float(attr_name, index)
        } else {
            None
        }
    }
    /// Sets the value of the specified float attribute for this node.
    pub fn set_float(&mut self, attr_name: String, value: f64) -> PyResult<()> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.set_node_float(attr_name, index, value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }
    /// Returns the value of the specified bool attribute for this node.
    pub fn get_bool(&self, attr_name: String) -> Option<bool> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.get_node_bool(attr_name, index)
        } else {
            None
        }
    }
    /// Sets the value of the specified bool attribute for this node.
    pub fn set_bool(&mut self, attr_name: String, value: bool) -> PyResult<()> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.set_node_bool(attr_name, index, value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }
    /// Returns the value of the specified string attribute for this node.
    pub fn get_str(&self, attr_name: String) -> Option<String> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.get_node_str(attr_name, index)
        } else {
            None
        }
    }
    /// Sets the value of the specified string attribute for this node.
    pub fn set_str(&mut self, attr_name: String, value: String) -> PyResult<()> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            self.attribute_manager.columnar.set_node_str(attr_name, index, value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }
    /// Checks if the specified attribute exists for this node.
    pub fn has(&self, attr_name: String) -> bool {
        self.attribute_manager.columnar.node_attr_names().contains(&attr_name)
    }
    /// Removes the specified attribute for this node (sets to None if present).
    pub fn remove(&mut self, attr_name: String) -> PyResult<()> {
        if let Some(index) = self.graph_store.node_index(&self.node_id) {
            // Remove value for this node (set to None in the column, do not drop schema)
            let uid = match self.attribute_manager.columnar.attr_name_to_uid.get(&attr_name) {
                Some(u) => u.clone(),
                None => return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Attribute not found")),
            };
            let mut col = match self.attribute_manager.columnar.columns.get_mut(&(crate::storage::columnar::ColumnKind::Node, uid.clone())) {
                Some(c) => c,
                None => return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Column not found")),
            };
            match &mut *col {
                crate::storage::columnar::ColumnData::Int(vec) => {
                    if index < vec.len() { vec[index] = None; }
                }
                crate::storage::columnar::ColumnData::Float(vec) => {
                    if index < vec.len() { vec[index] = None; }
                }
                crate::storage::columnar::ColumnData::Bool(vec) => {
                    if index < vec.len() { vec[index] = None; }
                }
                crate::storage::columnar::ColumnData::Str(vec) => {
                    if index < vec.len() { vec[index] = None; }
                }
                crate::storage::columnar::ColumnData::Json(map) => {
                    map.remove(&index);
                }
            }
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>("Node not found in graph"))
        }
    }
}

