// src/graph/columnar_graph.rs
//! Columnar graph storage for maximum memory efficiency

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::graph::attribute_value::AttributeValue;
use crate::graph::columnar_collections::{NodeCollection, EdgeCollection};

/// Columnar storage for a single attribute type
#[derive(Debug, Clone)]
pub struct ColumnStore {
    /// Attribute name
    name: String,
    /// Values stored in a contiguous vector
    values: Vec<Option<AttributeValue>>,
    /// Mapping from entity ID to vector index
    id_to_index: HashMap<String, usize>,
    /// Reverse mapping from index to entity ID
    index_to_id: Vec<String>,
    /// Track which indices are valid
    valid_indices: Vec<bool>,
}

impl ColumnStore {
    pub fn new(name: String) -> Self {
        Self {
            name,
            values: Vec::new(),
            id_to_index: HashMap::new(),
            index_to_id: Vec::new(),
            valid_indices: Vec::new(),
        }
    }
    
    /// Add or update a value for an entity
    pub fn set(&mut self, entity_id: &str, value: AttributeValue) {
        if let Some(&index) = self.id_to_index.get(entity_id) {
            // Update existing value
            self.values[index] = Some(value);
            self.valid_indices[index] = true;
        } else {
            // Add new value
            let index = self.values.len();
            self.values.push(Some(value));
            self.valid_indices.push(true);
            self.id_to_index.insert(entity_id.to_string(), index);
            self.index_to_id.push(entity_id.to_string());
        }
    }
    
    /// Get value for an entity
    pub fn get(&self, entity_id: &str) -> Option<&AttributeValue> {
        if let Some(&index) = self.id_to_index.get(entity_id) {
            if self.valid_indices[index] {
                self.values[index].as_ref()
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Remove value for an entity
    pub fn remove(&mut self, entity_id: &str) {
        if let Some(&index) = self.id_to_index.get(entity_id) {
            self.valid_indices[index] = false;
            self.values[index] = None;
        }
    }
    
    /// Get all entity IDs that have this attribute
    pub fn get_entity_ids(&self) -> Vec<String> {
        self.valid_indices
            .iter()
            .enumerate()
            .filter_map(|(i, &valid)| {
                if valid {
                    Some(self.index_to_id[i].clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Filter entities based on a predicate
    pub fn filter<F>(&self, predicate: F) -> Vec<String>
    where
        F: Fn(&AttributeValue) -> bool,
    {
        self.valid_indices
            .iter()
            .enumerate()
            .filter_map(|(i, &valid)| {
                if valid {
                    if let Some(ref value) = self.values[i] {
                        if predicate(value) {
                            return Some(self.index_to_id[i].clone());
                        }
                    }
                }
                None
            })
            .collect()
    }
    
    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let values_size = self.values.iter()
            .map(|v| v.as_ref().map_or(0, |val| val.memory_size()))
            .sum::<usize>();
        
        let id_to_index_size = self.id_to_index.keys()
            .map(|k| k.len())
            .sum::<usize>();
        
        let index_to_id_size = self.index_to_id.iter()
            .map(|s| s.len())
            .sum::<usize>();
        
        values_size + id_to_index_size + index_to_id_size + 
        self.valid_indices.len() + self.name.len()
    }
}

/// Columnar graph storage system - replaces the original Graph
#[pyclass(name = "Graph")]
pub struct ColumnarGraph {
    /// Node storage: just IDs
    node_ids: Vec<String>,
    node_id_to_index: HashMap<String, usize>,
    
    /// Edge storage: pairs of node indices
    edges: Vec<(usize, usize)>,
    edge_to_index: HashMap<(usize, usize), usize>,
    
    /// Node attribute columns
    node_columns: HashMap<String, ColumnStore>,
    
    /// Edge attribute columns  
    edge_columns: HashMap<String, ColumnStore>,
    
    /// Graph metadata
    pub directed: bool,
    
}

#[pymethods]
impl ColumnarGraph {
    #[new]
    pub fn new(directed: Option<bool>) -> Self {
        Self {
            node_ids: Vec::new(),
            node_id_to_index: HashMap::new(),
            edges: Vec::new(),
            edge_to_index: HashMap::new(),
            node_columns: HashMap::new(),
            edge_columns: HashMap::new(),
            directed: directed.unwrap_or(true),
        }
    }
    
    /// Add nodes efficiently
    pub fn add_nodes(&mut self, node_ids: Vec<String>) -> PyResult<()> {
        for node_id in node_ids {
            if !self.node_id_to_index.contains_key(&node_id) {
                let index = self.node_ids.len();
                self.node_id_to_index.insert(node_id.clone(), index);
                self.node_ids.push(node_id);
            }
        }
        Ok(())
    }
    
    /// Add edges efficiently
    pub fn add_edges(&mut self, edges: Vec<(String, String)>) -> PyResult<()> {
        for (source, target) in edges {
            // Ensure nodes exist
            if !self.node_id_to_index.contains_key(&source) {
                self.add_nodes(vec![source.clone()])?;
            }
            if !self.node_id_to_index.contains_key(&target) {
                self.add_nodes(vec![target.clone()])?;
            }
            
            // Add edge by index
            let source_idx = self.node_id_to_index[&source];
            let target_idx = self.node_id_to_index[&target];
            let edge_pair = (source_idx, target_idx);
            
            if !self.edge_to_index.contains_key(&edge_pair) {
                let edge_index = self.edges.len();
                self.edge_to_index.insert(edge_pair, edge_index);
                self.edges.push(edge_pair);
            }
        }
        Ok(())
    }
    
    /// Set node attributes in batch (columnar)
    pub fn set_node_attributes(&mut self, py: Python, attrs: &PyDict) -> PyResult<()> {
        for (node_id_obj, node_attrs_obj) in attrs.iter() {
            let node_id = node_id_obj.extract::<String>()?;
            let node_attrs = node_attrs_obj.downcast::<PyDict>()?;
            
            // Ensure node exists
            if !self.node_id_to_index.contains_key(&node_id) {
                self.add_nodes(vec![node_id.clone()])?;
            }
            
            // Set each attribute in its column
            for (attr_name_obj, attr_value_obj) in node_attrs.iter() {
                let attr_name = attr_name_obj.extract::<String>()?;
                let attr_value = AttributeValue::extract(attr_value_obj)?;
                
                // Get or create column
                let column = self.node_columns.entry(attr_name.clone())
                    .or_insert_with(|| ColumnStore::new(attr_name));
                
                // Set value in column
                column.set(&node_id, attr_value);
            }
        }
        Ok(())
    }
    
    /// Set edge attributes in batch (columnar)
    pub fn set_edge_attributes(&mut self, py: Python, attrs: &PyDict) -> PyResult<()> {
        for (edge_id_obj, edge_attrs_obj) in attrs.iter() {
            let edge_id = edge_id_obj.extract::<String>()?;
            let edge_attrs = edge_attrs_obj.downcast::<PyDict>()?;
            
            // Set each attribute in its column
            for (attr_name_obj, attr_value_obj) in edge_attrs.iter() {
                let attr_name = attr_name_obj.extract::<String>()?;
                let attr_value = AttributeValue::extract(attr_value_obj)?;
                
                // Get or create column
                let column = self.edge_columns.entry(attr_name.clone())
                    .or_insert_with(|| ColumnStore::new(attr_name));
                
                // Set value in column
                column.set(&edge_id, attr_value);
            }
        }
        Ok(())
    }
    
    /// Get node attribute value
    pub fn get_node_attribute(&self, py: Python, node_id: &str, attr_name: &str) -> PyResult<PyObject> {
        if let Some(column) = self.node_columns.get(attr_name) {
            if let Some(value) = column.get(node_id) {
                return value.to_python(py);
            }
        }
        Ok(py.None())
    }
    
    /// Get edge attribute value
    pub fn get_edge_attribute(&self, py: Python, edge_id: &str, attr_name: &str) -> PyResult<PyObject> {
        if let Some(column) = self.edge_columns.get(attr_name) {
            if let Some(value) = column.get(edge_id) {
                return value.to_python(py);
            }
        }
        Ok(py.None())
    }
    
    /// FAST: Filter nodes by attribute value (pure Rust)
    pub fn filter_nodes_by_string(&self, attr_name: &str, target_value: &str) -> Vec<String> {
        if let Some(column) = self.node_columns.get(attr_name) {
            column.filter(|value| {
                if let AttributeValue::String(s) = value {
                    s == target_value
                } else {
                    false
                }
            })
        } else {
            Vec::new()
        }
    }
    
    /// FAST: Filter nodes by integer comparison (pure Rust)
    pub fn filter_nodes_by_int_gt(&self, attr_name: &str, target_value: i64) -> Vec<String> {
        if let Some(column) = self.node_columns.get(attr_name) {
            column.filter(|value| {
                if let AttributeValue::Integer(i) = value {
                    *i > target_value
                } else {
                    false
                }
            })
        } else {
            Vec::new()
        }
    }
    
    /// FAST: Filter nodes by boolean value (pure Rust)
    pub fn filter_nodes_by_bool(&self, attr_name: &str, target_value: bool) -> Vec<String> {
        if let Some(column) = self.node_columns.get(attr_name) {
            column.filter(|value| {
                if let AttributeValue::Boolean(b) = value {
                    *b == target_value
                } else {
                    false
                }
            })
        } else {
            Vec::new()
        }
    }
    
    /// FAST: Complex filter with multiple conditions (pure Rust)
    pub fn filter_nodes_complex(&self, 
        role: Option<&str>, 
        min_salary: Option<i64>, 
        active: Option<bool>
    ) -> Vec<String> {
        let mut candidates: Option<Vec<String>> = None;
        
        // Apply role filter
        if let Some(role_val) = role {
            let role_matches = self.filter_nodes_by_string("role", role_val);
            candidates = Some(role_matches);
        }
        
        // Apply salary filter
        if let Some(salary_val) = min_salary {
            let salary_matches = self.filter_nodes_by_int_gt("salary", salary_val);
            candidates = match candidates {
                Some(existing) => {
                    Some(existing.into_iter()
                        .filter(|id| salary_matches.contains(id))
                        .collect())
                }
                None => Some(salary_matches),
            };
        }
        
        // Apply active filter
        if let Some(active_val) = active {
            let active_matches = self.filter_nodes_by_bool("active", active_val);
            candidates = match candidates {
                Some(existing) => {
                    Some(existing.into_iter()
                        .filter(|id| active_matches.contains(id))
                        .collect())
                }
                None => Some(active_matches),
            };
        }
        
        candidates.unwrap_or_else(|| self.node_ids.clone())
    }
    
    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.node_ids.clone()
    }
    
    /// Get all edges as (source, target) pairs
    pub fn get_edge_list(&self) -> Vec<(String, String)> {
        self.edges.iter().map(|&(src_idx, tgt_idx)| {
            (self.node_ids[src_idx].clone(), self.node_ids[tgt_idx].clone())
        }).collect()
    }
    
    /// Get node count
    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }
    
    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        // Node storage
        let node_storage = self.node_ids.iter().map(|s| s.len()).sum::<usize>();
        stats.insert("node_storage_bytes".to_string(), node_storage);
        
        // Edge storage (just indices)
        let edge_storage = self.edges.len() * 16; // 2 * usize
        stats.insert("edge_storage_bytes".to_string(), edge_storage);
        
        // Node attribute columns
        let node_attr_storage = self.node_columns.values()
            .map(|col| col.memory_usage())
            .sum::<usize>();
        stats.insert("node_attr_storage_bytes".to_string(), node_attr_storage);
        
        // Edge attribute columns
        let edge_attr_storage = self.edge_columns.values()
            .map(|col| col.memory_usage())
            .sum::<usize>();
        stats.insert("edge_attr_storage_bytes".to_string(), edge_attr_storage);
        
        // Total
        let total = node_storage + edge_storage + node_attr_storage + edge_attr_storage;
        stats.insert("total_bytes".to_string(), total);
        
        stats
    }
    
    /// Add nodes directly - collection API
    pub fn nodes_add(&mut self, node_data: &PyAny) -> PyResult<()> {
        // Same logic as NodeCollection::add but operating directly on self
        if let Ok(node_list) = node_data.downcast::<PyList>() {
            let mut node_ids = Vec::new();
            let mut all_attributes = std::collections::HashMap::new();
            
            for item in node_list.iter() {
                if let Ok(node_dict) = item.downcast::<PyDict>() {
                    let node_id = if let Ok(Some(id_obj)) = node_dict.get_item("id") {
                        id_obj.extract::<String>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Node must have 'id' field"
                        ));
                    };
                    
                    node_ids.push(node_id.clone());
                    
                    let mut node_attrs = std::collections::HashMap::new();
                    for (key, value) in node_dict.iter() {
                        let key_str = key.extract::<String>()?;
                        if key_str != "id" {
                            node_attrs.insert(key_str, value);
                        }
                    }
                    
                    if !node_attrs.is_empty() {
                        all_attributes.insert(node_id, node_attrs);
                    }
                } else if let Ok(node_id) = item.extract::<String>() {
                    node_ids.push(node_id);
                }
            }
            
            self.add_nodes(node_ids)?;
            
            if !all_attributes.is_empty() {
                Python::with_gil(|py| {
                    let attrs_dict = PyDict::new(py);
                    for (node_id, node_attrs) in all_attributes {
                        let node_dict = PyDict::new(py);
                        for (attr_name, attr_value) in node_attrs {
                            node_dict.set_item(attr_name, attr_value)?;
                        }
                        attrs_dict.set_item(node_id, node_dict)?;
                    }
                    self.set_node_attributes(py, attrs_dict)
                })?;
            }
        }
        Ok(())
    }
    
    /// Add edges directly - collection API
    pub fn edges_add(&mut self, edge_data: &PyAny) -> PyResult<()> {
        // Same logic as EdgeCollection::add but operating directly on self
        if let Ok(edge_list) = edge_data.downcast::<PyList>() {
            let mut edge_tuples = Vec::new();
            let mut all_attributes = std::collections::HashMap::new();
            
            for item in edge_list.iter() {
                if let Ok(edge_dict) = item.downcast::<PyDict>() {
                    let source = if let Ok(Some(src_obj)) = edge_dict.get_item("source") {
                        src_obj.extract::<String>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Edge must have 'source' field"
                        ));
                    };
                    
                    let target = if let Ok(Some(tgt_obj)) = edge_dict.get_item("target") {
                        tgt_obj.extract::<String>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Edge must have 'target' field"
                        ));
                    };
                    
                    edge_tuples.push((source.clone(), target.clone()));
                    
                    let mut edge_attrs = std::collections::HashMap::new();
                    for (key, value) in edge_dict.iter() {
                        let key_str = key.extract::<String>()?;
                        if key_str != "source" && key_str != "target" {
                            edge_attrs.insert(key_str, value);
                        }
                    }
                    
                    if !edge_attrs.is_empty() {
                        let edge_id = format!("{}->{}", source, target);
                        all_attributes.insert(edge_id, edge_attrs);
                    }
                }
            }
            
            self.add_edges(edge_tuples)?;
            
            if !all_attributes.is_empty() {
                Python::with_gil(|py| {
                    let attrs_dict = PyDict::new(py);
                    for (edge_id, edge_attrs) in all_attributes {
                        let edge_dict = PyDict::new(py);
                        for (attr_name, attr_value) in edge_attrs {
                            edge_dict.set_item(attr_name, attr_value)?;
                        }
                        attrs_dict.set_item(edge_id, edge_dict)?;
                    }
                    self.set_edge_attributes(py, attrs_dict)
                })?;
            }
        }
        Ok(())
    }
    
    /// Get nodes collection (API compatibility) - returns a lightweight wrapper
    #[getter]
    pub fn nodes(&self) -> NodeCollection {
        NodeCollection::new()
    }
    
    /// Get edges collection (API compatibility) - returns a lightweight wrapper 
    #[getter]
    pub fn edges(&self) -> EdgeCollection {
        EdgeCollection::new()
    }
    
    /// Get graph info
    pub fn info(&self) -> PyResult<HashMap<String, PyObject>> {
        let mut info = HashMap::new();
        Python::with_gil(|py| {
            info.insert("directed".to_string(), self.directed.to_object(py));
            info.insert("node_count".to_string(), self.node_count().to_object(py));
            info.insert("edge_count".to_string(), self.edge_count().to_object(py));
            info.insert("node_attribute_columns".to_string(), 
                       self.node_columns.len().to_object(py));
            info.insert("edge_attribute_columns".to_string(), 
                       self.edge_columns.len().to_object(py));
            
            let mem_stats = self.memory_stats();
            info.insert("memory_total_bytes".to_string(), 
                       mem_stats.get("total_bytes").unwrap_or(&0).to_object(py));
        });
        Ok(info)
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!("ColumnarGraph(nodes={}, edges={}, directed={}, node_attrs={}, edge_attrs={})", 
                self.node_count(), self.edge_count(), self.directed,
                self.node_columns.len(), self.edge_columns.len())
    }
}