use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use serde_json::Value as JsonValue;
use crate::graph::hybrid_types::{HybridGraphType, HybridNodeData, HybridEdgeData};
use crate::storage::columnar::ColumnarStore;
use crate::graph::views::ViewManager;
use crate::utils::{python_dict_to_json_map, python_to_json_value, json_value_to_python};

/// Next-generation graph structure with hybrid node/columnar storage
#[pyclass]
pub struct HybridFastGraph {
    /// Lightweight graph structure (topology + attribute UIDs only)
    pub graph: HybridGraphType,
    /// Node ID mappings
    pub node_id_to_index: HashMap<String, petgraph::graph::NodeIndex>,
    pub node_index_to_id: HashMap<petgraph::graph::NodeIndex, String>,
    /// Edge index to (source, target) mappings
    pub edge_index_to_endpoints: HashMap<petgraph::graph::EdgeIndex, (String, String)>,
    /// Columnar attribute storage
    pub columnar_store: ColumnarStore,
    /// Multi-view interface
    pub view_manager: ViewManager,
    /// Graph metadata
    pub graph_attributes: HashMap<String, JsonValue>,
    pub is_directed: bool,
}

#[pymethods]
impl HybridFastGraph {
    #[new]
    #[pyo3(signature = (directed = true))]
    pub fn new(directed: bool) -> Self {
        let graph = if directed {
            HybridGraphType::new_directed()
        } else {
            HybridGraphType::new_undirected()
        };
        
        Self {
            graph,
            node_id_to_index: HashMap::new(),
            node_index_to_id: HashMap::new(),
            edge_index_to_endpoints: HashMap::new(),
            columnar_store: ColumnarStore::new(),
            view_manager: ViewManager::new(directed),
            graph_attributes: HashMap::new(),
            is_directed: directed,
        }
    }

    /// Get whether this graph is directed
    pub fn is_directed(&self) -> bool {
        self.is_directed
    }

    /// Add a single node to the graph
    pub fn add_node(&mut self, node_id: String, attributes: Option<&PyDict>) -> PyResult<()> {
        if self.node_id_to_index.contains_key(&node_id) {
            return Ok(()); // Node already exists
        }

        // Create hybrid node data (lightweight)
        let hybrid_node = HybridNodeData {
            id: node_id.clone(),
            attr_uids: std::collections::HashSet::new(),
        };
        
        // Add to graph topology
        let node_index = self.graph.add_node(hybrid_node);
        self.node_id_to_index.insert(node_id.clone(), node_index);
        self.node_index_to_id.insert(node_index, node_id.clone());
        
        // Store attributes in columnar format
        if let Some(py_attrs) = attributes {
            let attrs = python_dict_to_json_map(py_attrs)?;
            
            for (attr_name, attr_value) in attrs {
                self.columnar_store.set_node_attribute(
                    node_index.index(),
                    &attr_name,
                    attr_value
                );
            }
        }
        
        Ok(())
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: String) -> PyResult<bool> {
        if let Some(node_idx) = self.node_id_to_index.remove(&node_id) {
            // Remove from columnar storage
            self.columnar_store.remove_node(node_idx.index());
            
            // Remove from graph topology
            self.node_index_to_id.remove(&node_idx);
            self.graph.remove_node(node_idx);
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Add an edge between two nodes
    pub fn add_edge(&mut self, source: String, target: String, attributes: Option<&PyDict>) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;

        // Create hybrid edge data (lightweight)
        let hybrid_edge = HybridEdgeData {
            source: source.clone(),
            target: target.clone(),
            attr_uids: std::collections::HashSet::new(),
        };

        // Add to graph topology
        let edge_index = self.graph.add_edge(*source_idx, *target_idx, hybrid_edge);
        self.edge_index_to_endpoints.insert(edge_index, (source.clone(), target.clone()));
        
        // Store attributes in columnar format
        if let Some(py_attrs) = attributes {
            let attrs = python_dict_to_json_map(py_attrs)?;
            
            for (attr_name, attr_value) in attrs {
                self.columnar_store.set_edge_attribute(
                    edge_index.index(),
                    &attr_name,
                    attr_value
                );
            }
        }
        
        Ok(())
    }

    /// Check if an edge exists between two nodes
    pub fn has_edge(&self, source: String, target: String) -> PyResult<bool> {
        let source_idx = match self.node_id_to_index.get(&source) {
            Some(idx) => idx,
            None => return Ok(false),
        };
        let target_idx = match self.node_id_to_index.get(&target) {
            Some(idx) => idx,
            None => return Ok(false),
        };

        Ok(self.graph.find_edge(*source_idx, *target_idx).is_some())
    }

    /// Remove an edge between two nodes
    pub fn remove_edge(&mut self, source: String, target: String) -> PyResult<bool> {
        let source_idx = self.node_id_to_index.get(&source)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Source node '{}' not found", source)
            ))?;
        let target_idx = self.node_id_to_index.get(&target)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Target node '{}' not found", target)
            ))?;

        if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
            // Remove from columnar storage
            // TODO: Implement edge removal from columnar store
            
            // Remove from graph topology
            self.edge_index_to_endpoints.remove(&edge_idx);
            self.graph.remove_edge(edge_idx);
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get node count
    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.node_id_to_index.keys().cloned().collect()
    }

    /// Get all edge IDs (formatted as "source->target")
    pub fn get_edge_ids(&self) -> Vec<String> {
        self.edge_index_to_endpoints
            .values()
            .map(|(source, target)| format!("{}->{}", source, target))
            .collect()
    }

    /// Get node attributes using columnar storage
    pub fn get_node_attributes(&self, node_id: String) -> PyResult<PyObject> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        let attributes = self.columnar_store.get_node_attributes(node_idx.index());
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (key, value) in attributes {
                let py_value = json_value_to_python(py, &value)?;
                dict.set_item(key, py_value)?;
            }
            Ok(dict.into_py(py))
        })
    }

    /// Get edge attributes using columnar storage
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
            let attributes = self.columnar_store.get_edge_attributes(edge_idx.index());
            
            Python::with_gil(|py| {
                let dict = PyDict::new(py);
                for (key, value) in attributes {
                    let py_value = json_value_to_python(py, &value)?;
                    dict.set_item(key, py_value)?;
                }
                Ok(dict.into_py(py))
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Edge from '{}' to '{}' not found", source, target)
            ))
        }
    }

    /// Set node attribute using columnar storage
    pub fn set_node_attribute(&mut self, node_id: String, key: String, value: &PyAny) -> PyResult<()> {
        let node_idx = self.node_id_to_index.get(&node_id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Node '{}' not found", node_id)
            ))?;
        
        let json_value = python_to_json_value(value)?;
        self.columnar_store.set_node_attribute(node_idx.index(), &key, json_value);
        
        Ok(())
    }

    /// Set edge attribute using columnar storage
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
            let json_value = python_to_json_value(value)?;
            self.columnar_store.set_edge_attribute(edge_idx.index(), &key, json_value);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Edge from '{}' to '{}' not found", source, target)
            ))
        }
    }

    /// Ultra-fast attribute filtering using columnar storage
    #[pyo3(name = "filter_nodes_by_attributes")]
    fn filter_nodes_by_attributes(&self, filters: &PyDict) -> PyResult<Vec<String>> {
        let filter_map = python_dict_to_json_map(filters)?;
        let matching_indices = self.columnar_store.filter_nodes_by_attributes(&filter_map);
        
        // Convert indices back to node IDs
        let mut result = Vec::new();
        for node_index in matching_indices {
            if let Some(node_id) = self.node_index_to_id.get(&petgraph::graph::NodeIndex::new(node_index)) {
                result.push(node_id.clone());
            }
        }
        
        Ok(result)
    }

    /// Ultra-fast edge filtering using columnar storage
    #[pyo3(name = "filter_edges_by_attributes")]
    fn filter_edges_by_attributes(&self, filters: &PyDict) -> PyResult<Vec<(String, String)>> {
        let filter_map = python_dict_to_json_map(filters)?;
        let matching_indices = self.columnar_store.filter_edges_by_attributes(&filter_map);
        
        // Convert indices back to edge endpoints
        let mut result = Vec::new();
        for edge_index in matching_indices {
            if let Some((source, target)) = self.edge_index_to_endpoints.get(&petgraph::graph::EdgeIndex::new(edge_index)) {
                result.push((source.clone(), target.clone()));
            }
        }
        
        Ok(result)
    }

    /// Get storage statistics
    pub fn get_storage_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        // Graph topology stats
        stats.insert("nodes".to_string(), self.graph.node_count());
        stats.insert("edges".to_string(), self.graph.edge_count());
        
        // Columnar storage stats
        stats.extend(self.columnar_store.get_stats());
        
        stats
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Calculate storage efficiency
        let node_count = self.graph.node_count() as f64;
        let attribute_count = *self.columnar_store.get_stats().get("attributes_registered").unwrap_or(&0) as f64;
        
        if node_count > 0.0 {
            metrics.insert("attributes_per_node".to_string(), attribute_count / node_count);
        }
        
        // Calculate memory efficiency (placeholder)
        metrics.insert("memory_efficiency".to_string(), 0.85); // Would be calculated based on actual memory usage
        
        metrics
    }

    /// Export capabilities for numerical analysis
    pub fn export_node_attributes_as_vectors(&self, py: Python, attr_names: Vec<String>) -> PyResult<PyObject> {
        // This would export specified attributes as numerical vectors
        // For integration with NumPy, Arrow, etc.
        let py_dict = PyDict::new(py);
        
        for attr_name in attr_names {
            let values = self.columnar_store.get_attribute_values(&attr_name);
            // Convert to appropriate Python type based on the attribute type
            let py_values = values.iter()
                .map(|v| json_value_to_python(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            py_dict.set_item(&attr_name, py_values)?;
        }
        
        Ok(py_dict.into_py(py))
    }

    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: String) -> Vec<String> {
        if let Some(node_idx) = self.node_id_to_index.get(&node_id) {
            self.graph.neighbors(*node_idx)
                .into_iter()
                .filter_map(|neighbor_idx| self.node_index_to_id.get(&neighbor_idx).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Create a deep copy of the graph
    fn copy(&self) -> Self {
        HybridFastGraph {
            graph: self.graph.clone(),
            node_id_to_index: self.node_id_to_index.clone(),
            node_index_to_id: self.node_index_to_id.clone(),
            edge_index_to_endpoints: self.edge_index_to_endpoints.clone(),
            columnar_store: self.columnar_store.clone(),
            view_manager: ViewManager::new(self.is_directed),
            graph_attributes: self.graph_attributes.clone(),
            is_directed: self.is_directed,
        }
    }
}
