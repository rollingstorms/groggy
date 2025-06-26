use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use petgraph::{Graph as PetGraph, Directed};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value as JsonValue;
use dashmap::DashMap;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    pub id: String,
    pub attributes: HashMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub source: String,
    pub target: String,
    pub attributes: HashMap<String, JsonValue>,
}

/// High-performance graph structure implemented in Rust
#[pyclass]
pub struct FastGraph {
    pub graph: PetGraph<NodeData, EdgeData, Directed>,
    pub node_id_to_index: DashMap<String, NodeIndex>,
    pub node_index_to_id: DashMap<NodeIndex, String>,
    pub graph_attributes: HashMap<String, JsonValue>,
}

#[pymethods]
impl FastGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            graph: PetGraph::new(),
            node_id_to_index: DashMap::new(),
            node_index_to_id: DashMap::new(),
            graph_attributes: HashMap::new(),
        }
    }

    /// Add a single node to the graph
    pub fn add_node(&mut self, node_id: String, attributes: Option<&PyDict>) -> PyResult<()> {
        if self.node_id_to_index.contains_key(&node_id) {
            return Ok(()); // Node already exists
        }

        let attrs = if let Some(py_attrs) = attributes {
            python_dict_to_json_map(py_attrs)?
        } else {
            HashMap::new()
        };

        let node_data = NodeData { 
            id: node_id.clone(), 
            attributes: attrs 
        };
        
        let node_index = self.graph.add_node(node_data);
        
        self.node_id_to_index.insert(node_id.clone(), node_index);
        self.node_index_to_id.insert(node_index, node_id);
        
        Ok(())
    }

    /// Add multiple nodes efficiently
    fn batch_add_nodes(&mut self, node_data: &PyList) -> PyResult<()> {
        let mut nodes_to_add = Vec::new();
        
        // Parse all nodes first
        for item in node_data {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            let node_id: String = tuple.get_item(0)?.extract()?;
            
            if self.node_id_to_index.contains_key(&node_id) {
                continue; // Skip existing nodes
            }
            
            let attributes = if tuple.len() > 1 {
                let py_attrs = tuple.get_item(1)?.downcast::<PyDict>()?;
                python_dict_to_json_map(py_attrs)?
            } else {
                HashMap::new()
            };
            
            nodes_to_add.push((node_id, attributes));
        }
        
        // Add all nodes in batch
        for (node_id, attributes) in nodes_to_add {
            let node_data = NodeData { 
                id: node_id.clone(), 
                attributes 
            };
            
            let node_index = self.graph.add_node(node_data);
            self.node_id_to_index.insert(node_id.clone(), node_index);
            self.node_index_to_id.insert(node_index, node_id);
        }
        
        Ok(())
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: String, target: String, attributes: Option<&PyDict>) -> PyResult<()> {
        // Ensure both nodes exist, create if they don't
        if !self.node_id_to_index.contains_key(&source) {
            self.add_node(source.clone(), None)?;
        }
        if !self.node_id_to_index.contains_key(&target) {
            self.add_node(target.clone(), None)?;
        }
        
        let source_idx = *self.node_id_to_index.get(&source).unwrap();
        let target_idx = *self.node_id_to_index.get(&target).unwrap();
        
        // Check if edge already exists
        if self.graph.find_edge(source_idx, target_idx).is_some() {
            return Ok(()); // Edge already exists, skip
        }
        
        let attrs = if let Some(py_attrs) = attributes {
            python_dict_to_json_map(py_attrs)?
        } else {
            HashMap::new()
        };
        
        let edge_data = EdgeData { 
            source, 
            target, 
            attributes: attrs 
        };
        
        self.graph.add_edge(source_idx, target_idx, edge_data);
        
        Ok(())
    }

    /// Add multiple edges efficiently
    fn batch_add_edges(&mut self, edge_data: &PyList) -> PyResult<()> {
        let mut edges_to_add = Vec::new();
        let mut nodes_to_create = std::collections::HashSet::new();
        
        // First pass: collect edges and nodes to create
        for item in edge_data {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;
            let source: String = tuple.get_item(0)?.extract()?;
            let target: String = tuple.get_item(1)?.extract()?;
            
            if !self.node_id_to_index.contains_key(&source) {
                nodes_to_create.insert(source.clone());
            }
            if !self.node_id_to_index.contains_key(&target) {
                nodes_to_create.insert(target.clone());
            }
            
            let attributes = if tuple.len() > 2 {
                let py_attrs = tuple.get_item(2)?.downcast::<PyDict>()?;
                python_dict_to_json_map(py_attrs)?
            } else {
                HashMap::new()
            };
            
            edges_to_add.push((source, target, attributes));
        }
        
        // Create missing nodes
        for node_id in nodes_to_create {
            self.add_node(node_id, None)?;
        }
        
        // Add all edges
        for (source, target, attributes) in edges_to_add {
            let source_idx = *self.node_id_to_index.get(&source).unwrap();
            let target_idx = *self.node_id_to_index.get(&target).unwrap();
            
            let edge_data = EdgeData { 
                source, 
                target, 
                attributes 
            };
            
            self.graph.add_edge(source_idx, target_idx, edge_data);
        }
        
        Ok(())
    }

    /// Get node count
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get neighbors of a node (both incoming and outgoing)
    fn get_neighbors(&self, node_id: String) -> PyResult<Vec<String>> {
        use petgraph::visit::EdgeRef;
        
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        let mut neighbors: Vec<String> = Vec::new();
        
        // Get outgoing neighbors
        for neighbor_idx in self.graph.neighbors(*node_idx) {
            if let Some(id) = self.node_index_to_id.get(&neighbor_idx) {
                neighbors.push(id.clone());
            }
        }
        
        // Get incoming neighbors
        for edge_ref in self.graph.edges_directed(*node_idx, petgraph::Direction::Incoming) {
            let source_idx = edge_ref.source();
            if let Some(id) = self.node_index_to_id.get(&source_idx) {
                let id_str = id.clone();
                if !neighbors.contains(&id_str) {  // Avoid duplicates
                    neighbors.push(id_str);
                }
            }
        }
        
        Ok(neighbors)
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.node_id_to_index.iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get all edge IDs (formatted as "source->target")
    pub fn get_edge_ids(&self) -> Vec<String> {
        let mut edge_ids = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some(edge_data) = self.graph.edge_weight(edge_idx) {
                let edge_id = format!("{}->{}", edge_data.source, edge_data.target);
                edge_ids.push(edge_id);
            }
        }
        edge_ids
    }

    /// Get node attributes
    pub fn get_node_attributes(&self, node_id: String) -> PyResult<PyObject> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        if let Some(node_data) = self.graph.node_weight(*node_idx) {
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for (key, value) in &node_data.attributes {
                    let py_value = json_value_to_python(py, value)?;
                    dict.set_item(key, py_value)?;
                }
                Ok(dict.into_py(py))
            })
        } else {
            Python::with_gil(|py| Ok(PyDict::new(py).into_py(py)))
        }
    }

    /// Get edge attributes
    pub fn get_edge_attributes(&self, source: String, target: String) -> PyResult<PyObject> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;
        
        if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
            if let Some(edge_data) = self.graph.edge_weight(edge_idx) {
                Python::with_gil(|py| {
                    let dict = PyDict::new(py);
                    for (key, value) in &edge_data.attributes {
                        let py_value = json_value_to_python(py, value)?;
                        dict.set_item(key, py_value)?;
                    }
                    Ok(dict.into_py(py))
                })
            } else {
                Python::with_gil(|py| Ok(PyDict::new(py).into_py(py)))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Edge from '{}' to '{}' not found", source, target)
            ))
        }
    }

    /// Set node attribute
    pub fn set_node_attribute(&mut self, node_id: String, key: String, value: &PyAny) -> PyResult<()> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        if let Some(node_data) = self.graph.node_weight_mut(*node_idx) {
            let json_value = python_to_json_value(value)?;
            node_data.attributes.insert(key, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))
        }
    }

    /// Set edge attribute
    pub fn set_edge_attribute(&mut self, source: String, target: String, key: String, value: &PyAny) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;
        
        if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
            if let Some(edge_data) = self.graph.edge_weight_mut(edge_idx) {
                let json_value = python_to_json_value(value)?;
                edge_data.attributes.insert(key, json_value);
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    format!("Edge from '{}' to '{}' not found", source, target)
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Edge from '{}' to '{}' not found", source, target)
            ))
        }
    }

    /// Create a deep copy of the graph
    fn copy(&self) -> Self {
        FastGraph {
            graph: self.graph.clone(),
            node_id_to_index: self.node_id_to_index.clone(),
            node_index_to_id: self.node_index_to_id.clone(),
            graph_attributes: self.graph_attributes.clone(),
        }
    }

    /// Get graph statistics
    fn get_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("nodes".to_string(), self.graph.node_count());
        stats.insert("edges".to_string(), self.graph.edge_count());
        stats
    }
}

/// Helper function to convert Python dict to JSON HashMap
fn python_dict_to_json_map(py_dict: &PyDict) -> PyResult<HashMap<String, JsonValue>> {
    let mut map = HashMap::new();
    
    for (key, value) in py_dict {
        let key_str: String = key.extract()?;
        let json_value = python_to_json_value(value)?;
        map.insert(key_str, json_value);
    }
    
    Ok(map)
}

/// Convert Python value to JSON value
fn python_to_json_value(py_value: &PyAny) -> PyResult<JsonValue> {
    if py_value.is_none() {
        Ok(JsonValue::Null)
    } else if let Ok(b) = py_value.extract::<bool>() {
        Ok(JsonValue::Bool(b))
    } else if let Ok(i) = py_value.extract::<i64>() {
        Ok(JsonValue::Number(i.into()))
    } else if let Ok(f) = py_value.extract::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(f) {
            Ok(JsonValue::Number(num))
        } else {
            Ok(JsonValue::Null)
        }
    } else if let Ok(s) = py_value.extract::<String>() {
        Ok(JsonValue::String(s))
    } else if let Ok(py_list) = py_value.downcast::<pyo3::types::PyList>() {
        let mut vec = Vec::new();
        for item in py_list {
            vec.push(python_to_json_value(item)?);
        }
        Ok(JsonValue::Array(vec))
    } else if let Ok(py_dict) = py_value.downcast::<pyo3::types::PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, value) in py_dict {
            let key_str: String = key.extract()?;
            map.insert(key_str, python_to_json_value(value)?);
        }
        Ok(JsonValue::Object(map))
    } else {
        // Fallback: convert to string
        let s: String = py_value.str()?.extract()?;
        Ok(JsonValue::String(s))
    }
}

/// Convert JSON value back to Python object
fn json_value_to_python(py: Python, value: &JsonValue) -> PyResult<PyObject> {
    match value {
        JsonValue::Null => Ok(py.None()),
        JsonValue::Bool(b) => Ok(b.to_object(py)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        JsonValue::String(s) => Ok(s.to_object(py)),
        JsonValue::Array(arr) => {
            let py_list = pyo3::types::PyList::empty(py);
            for item in arr {
                py_list.append(json_value_to_python(py, item)?)?;
            }
            Ok(py_list.to_object(py))
        }
        JsonValue::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                py_dict.set_item(key, json_value_to_python(py, val)?)?;
            }
            Ok(py_dict.to_object(py))
        }
    }
}
