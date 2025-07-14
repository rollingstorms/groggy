#![allow(non_local_definitions)]
#![allow(clippy::uninlined_format_args)]
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::graph::types::{EdgeData, GraphType, NodeData};
use crate::storage::ColumnarStore;
use crate::utils::{python_dict_to_json_map, python_pyobject_to_json, python_value_to_json};

/// Entity type for filtering operations (minimal version for columnar store)
#[derive(Debug, Clone, Copy)]
pub enum EntityType {
    Node,
    Edge,
}

/// Filter criteria for columnar operations (minimal version for columnar store) 
#[derive(Debug, Clone)]
pub enum FilterCriteria {
    Attributes(HashMap<String, serde_json::Value>),
    Numeric(String, String, f64),
    String(String, String, String), 
    MultiCriteria {
        exact: HashMap<String, serde_json::Value>,
        numeric: Vec<(String, String, f64)>,
        string: Vec<(String, String, String)>,
    },
}

/// FastGraph struct with hybrid/columnar storage (MAIN IMPLEMENTATION)
#[pyclass]
pub struct FastGraph {
    /// The actual graph structure using the new lightweight types
    pub graph: GraphType,

    /// Whether this is a directed graph
    pub is_directed: bool,

    /// Columnar storage for attributes
    pub columnar_store: ColumnarStore,

    /// Bidirectional mappings for node IDs
    pub node_id_to_index: HashMap<String, NodeIndex>,
    pub node_index_to_id: HashMap<NodeIndex, String>,

    /// Edge tracking
    pub edge_index_to_endpoints: HashMap<petgraph::graph::EdgeIndex, (String, String)>,
}

#[pymethods]
impl FastGraph {
    #[new]
    pub fn new(directed: bool) -> Self {
        let graph = if directed {
            GraphType::new_directed()
        } else {
            GraphType::new_undirected()
        };

        Self {
            graph,
            is_directed: directed,
            columnar_store: ColumnarStore::new(),
            node_id_to_index: HashMap::new(),
            node_index_to_id: HashMap::new(),
            edge_index_to_endpoints: HashMap::new(),
        }
    }

    /// Add a single node - ONLY public method for single node addition
    pub fn add_node(&mut self, node_id: String, attributes: Option<&PyDict>) -> PyResult<()> {
        let attrs = if let Some(py_attrs) = attributes {
            Some(python_dict_to_json_map(py_attrs)?)
        } else {
            None
        };
        self.add_single_node_internal(node_id, attrs)
    }

    /// Add multiple nodes - ONLY public method for batch node addition
    pub fn add_nodes(&mut self, nodes_data: Vec<(String, Option<&PyDict>)>) -> PyResult<()> {
        let mut bulk_nodes = Vec::new();
        for (node_id, attributes) in nodes_data {
            let attrs = if let Some(py_attrs) = attributes {
                python_dict_to_json_map(py_attrs)?
            } else {
                HashMap::new()
            };
            bulk_nodes.push((node_id, attrs));
        }
        self.bulk_add_nodes_internal_call(bulk_nodes);
        Ok(())
    }

    /// Add a single edge - ONLY public method for single edge addition
    pub fn add_edge(&mut self, source: String, target: String, attributes: Option<&PyDict>) -> PyResult<()> {
        let attrs = if let Some(py_attrs) = attributes {
            Some(python_dict_to_json_map(py_attrs)?)
        } else {
            None
        };
        self.add_single_edge_internal(source, target, attrs)
    }

    /// Add multiple edges - ONLY public method for batch edge addition
    pub fn add_edges(&mut self, edges_data: Vec<(String, String, Option<&PyDict>)>) -> PyResult<()> {
        let mut bulk_edges = Vec::new();
        for (source, target, attributes) in edges_data {
            let attrs = if let Some(py_attrs) = attributes {
                python_dict_to_json_map(py_attrs)?
            } else {
                HashMap::new()
            };
            bulk_edges.push((source, target, attrs));
        }
        self.bulk_add_edges_internal_call(bulk_edges);
        Ok(())
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get all node IDs
    pub fn get_node_ids(&self) -> Vec<String> {
        self.node_id_to_index.keys().cloned().collect()
    }

    /// Check if node exists
    pub fn has_node(&self, node_id: &str) -> bool {
        self.node_id_to_index.contains_key(node_id)
    }

    /// Check if edge exists
    pub fn has_edge(&self, source: &str, target: &str) -> bool {
        if let (Some(source_idx), Some(target_idx)) = (
            self.node_id_to_index.get(source),
            self.node_id_to_index.get(target),
        ) {
            self.graph.find_edge(*source_idx, *target_idx).is_some()
        } else {
            false
        }
    }

    /// Get node attributes
    pub fn get_node_attributes(
        &self,
        node_id: &str,
    ) -> PyResult<Option<HashMap<String, pyo3::PyObject>>> {
        use crate::utils::json_value_to_python;
        use pyo3::Python;

        let node_idx = self.node_id_to_index.get(node_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id))
        })?;

        if let Some(_node_data) = self.graph.node_weight(*node_idx) {
            Python::with_gil(|py| {
                // Retrieve attributes from columnar store
                let json_attributes = self.columnar_store.get_node_attributes(node_idx.index());

                // Convert JsonValue to PyObject
                let mut py_attributes = HashMap::new();
                for (attr_name, json_value) in json_attributes {
                    let py_value = json_value_to_python(py, &json_value)?;
                    py_attributes.insert(attr_name, py_value);
                }

                Ok(Some(py_attributes))
            })
        } else {
            Ok(None)
        }
    }

    // === UNIFIED FILTERING METHODS ===

    /// Filter nodes by attribute dictionary - unified filtering interface
    pub fn filter_nodes_by_attributes(
        &self,
        filters: HashMap<String, pyo3::PyObject>,
    ) -> PyResult<Vec<String>> {
        use pyo3::Python;

        if filters.is_empty() {
            return Ok(self.get_node_ids());
        }

        Python::with_gil(|py| {
            // Convert PyObjects to JsonValues
            let mut json_filters = HashMap::new();
            for (attr_name, py_value) in filters {
                let json_value = python_pyobject_to_json(py, &py_value)?;
                json_filters.insert(attr_name, json_value);
            }

            // Use columnar store's optimized exact match filtering
            let matching_indices = self.columnar_store.filter_nodes_by_attributes(&json_filters);
            
            // Convert indices back to node IDs
            let mut result = Vec::new();
            for node_index in matching_indices {
                if let Some(node_id) = self
                    .node_index_to_id
                    .get(&petgraph::graph::NodeIndex::new(node_index))
                {
                    result.push(node_id.clone());
                }
            }
            
            Ok(result)
        })
    }

    /// Filter edges by attribute dictionary - unified filtering interface  
    pub fn filter_edges_by_attributes(
        &self,
        filters: HashMap<String, pyo3::PyObject>,
    ) -> PyResult<Vec<String>> {
        use pyo3::Python;

        if filters.is_empty() {
            return Ok(self.get_edge_ids());
        }

        Python::with_gil(|py| {
            // Convert PyObjects to JsonValues
            let mut json_filters = HashMap::new();
            for (attr_name, py_value) in filters {
                let json_value = python_pyobject_to_json(py, &py_value)?;
                json_filters.insert(attr_name, json_value);
            }

            // Use columnar store's optimized exact match filtering
            let matching_indices = self.columnar_store.filter_edges_by_attributes(&json_filters);
            
            // Convert indices back to edge IDs
            let mut result = Vec::new();
            for edge_index in matching_indices {
                if let Some((source_idx, target_idx)) = self
                    .graph
                    .edge_endpoints(petgraph::graph::EdgeIndex::new(edge_index))
                {
                    if let (Some(source_id), Some(target_id)) = (
                        self.node_index_to_id.get(&source_idx),
                        self.node_index_to_id.get(&target_idx),
                    ) {
                        result.push(format!("{}->{}", source_id, target_id));
                    }
                }
            }
            
            Ok(result)
        })
    }

    /// Filter nodes by numeric comparison - unified filtering interface
    pub fn filter_nodes_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_by_numeric_comparison("node", attr_name, operator, value)
    }

    /// Filter edges by numeric comparison - unified filtering interface
    pub fn filter_edges_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_by_numeric_comparison("edge", attr_name, operator, value)
    }

    /// Filter nodes by string comparison - unified filtering interface  
    pub fn filter_nodes_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_by_string_comparison("node", attr_name, operator, value)
    }

    /// Filter edges by string comparison - unified filtering interface
    pub fn filter_edges_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_by_string_comparison("edge", attr_name, operator, value)
    }

    /// Filter nodes with multi-criteria - unified filtering interface
    pub fn filter_nodes_multi_criteria(
        &self,
        exact_matches: HashMap<String, String>,
        numeric_comparisons: Vec<(String, String, f64)>,
        string_comparisons: Vec<(String, String, String)>,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_multi_criteria("node", exact_matches, numeric_comparisons, string_comparisons)
    }

    /// Filter edges with multi-criteria - unified filtering interface
    pub fn filter_edges_multi_criteria(
        &self,
        exact_matches: HashMap<String, String>,
        numeric_comparisons: Vec<(String, String, f64)>,
        string_comparisons: Vec<(String, String, String)>,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_multi_criteria("edge", exact_matches, numeric_comparisons, string_comparisons)
    }

    /// Filter nodes with sparse algorithm - unified filtering interface
    pub fn filter_nodes_by_attributes_sparse(
        &self,
        filters: HashMap<String, pyo3::PyObject>,
    ) -> PyResult<Vec<String>> {
        self.filter_entities_by_attributes_sparse("node", filters)
    }
}

impl FastGraph {
    // === SIMPLIFIED HELPER METHODS ===
    
    fn add_single_node_internal(&mut self, id: String, attributes: Option<HashMap<String, serde_json::Value>>) -> PyResult<()> {
        if self.node_id_to_index.contains_key(&id) {
            return Ok(()); // Already exists
        }
        
        // 1. Graph State: add address to topology
        let node_data = NodeData { id: id.clone(), attr_uids: std::collections::HashSet::new() };
        let node_idx = self.graph.add_node(node_data);
        
        // 2. Graph Pool: update address mappings  
        self.node_id_to_index.insert(id.clone(), node_idx);
        self.node_index_to_id.insert(node_idx, id);
        
        // 3. Attr Tables: store attributes via columnar
        if let Some(attrs) = attributes {
            for (attr_name, attr_value) in attrs {
                self.columnar_store.set_node_attribute(node_idx.index(), &attr_name, attr_value);
            }
        }
        
        Ok(())
    }
    
    fn add_single_edge_internal(&mut self, source: String, target: String, attributes: Option<HashMap<String, serde_json::Value>>) -> PyResult<()> {
        // Get node indices
        let source_idx = self.node_id_to_index.get(&source).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Source node '{}' not found", source))
        })?;
        let target_idx = self.node_id_to_index.get(&target).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Target node '{}' not found", target))
        })?;

        // 1. Graph State: add address to topology
        let edge_data = EdgeData {
            source: source.clone(),
            target: target.clone(),
            attr_uids: std::collections::HashSet::new(),
        };
        let edge_idx = self.graph.add_edge(*source_idx, *target_idx, edge_data);

        // 2. Graph Pool: update address mappings
        self.edge_index_to_endpoints.insert(edge_idx, (source, target));

        // 3. Attr Tables: store attributes via columnar
        if let Some(attrs) = attributes {
            for (attr_name, attr_value) in attrs {
                self.columnar_store.set_edge_attribute(edge_idx.index(), &attr_name, attr_value);
            }
        }

        Ok(())
    }    
    /// Get neighbors of a node (for directed graphs, returns only outgoing neighbors)
    pub fn get_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(node_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id))
        })?;

        let neighbor_indices = if self.is_directed {
            // For directed graphs, only return outgoing neighbors
            self.graph
                .neighbors_directed(*node_idx, petgraph::Direction::Outgoing)
        } else {
            // For undirected graphs, neighbors() returns all connected nodes
            self.graph.neighbors(*node_idx)
        };

        let neighbor_ids: Vec<String> = neighbor_indices
            .into_iter()
            .filter_map(|idx| self.node_index_to_id.get(&idx).cloned())
            .collect();

        Ok(neighbor_ids)
    }

    /// Get outgoing neighbors of a node (for directed graphs)
    pub fn get_outgoing_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(node_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id))
        })?;

        let neighbor_indices = self
            .graph
            .neighbors_directed(*node_idx, petgraph::Direction::Outgoing);
        let neighbor_ids: Vec<String> = neighbor_indices
            .into_iter()
            .filter_map(|idx| self.node_index_to_id.get(&idx).cloned())
            .collect();

        Ok(neighbor_ids)
    }

    /// Get incoming neighbors of a node (for directed graphs)
    pub fn get_incoming_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(node_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id))
        })?;

        let neighbor_indices = self
            .graph
            .neighbors_directed(*node_idx, petgraph::Direction::Incoming);
        let neighbor_ids: Vec<String> = neighbor_indices
            .into_iter()
            .filter_map(|idx| self.node_index_to_id.get(&idx).cloned())
            .collect();

        Ok(neighbor_ids)
    }

    /// Get all neighbors of a node (both incoming and outgoing)
    pub fn get_all_neighbors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let node_idx = self.node_id_to_index.get(node_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id))
        })?;

        let mut neighbor_ids = std::collections::HashSet::new();

        // Get outgoing neighbors
        for idx in self
            .graph
            .neighbors_directed(*node_idx, petgraph::Direction::Outgoing)
        {
            if let Some(id) = self.node_index_to_id.get(&idx) {
                neighbor_ids.insert(id.clone());
            }
        }

        // Get incoming neighbors
        for idx in self
            .graph
            .neighbors_directed(*node_idx, petgraph::Direction::Incoming)
        {
            if let Some(id) = self.node_index_to_id.get(&idx) {
                neighbor_ids.insert(id.clone());
            }
        }

        Ok(neighbor_ids.into_iter().collect())
    }

    /// Get edge attributes - retrieve from columnar store  
    pub fn get_edge_attributes(
        &self,
        source: &str,
        target: &str,
    ) -> PyResult<Option<HashMap<String, pyo3::PyObject>>> {
        use pyo3::Python;
        if let (Some(source_idx), Some(target_idx)) = (
            self.node_id_to_index.get(source),
            self.node_id_to_index.get(target),
        ) {
            if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
                if let Some(_edge_data) = self.graph.edge_weight(edge_idx) {
                    Python::with_gil(|py| {
                        use crate::utils::json_value_to_python;

                        // Retrieve attributes from columnar store
                        let json_attributes =
                            self.columnar_store.get_edge_attributes(edge_idx.index());

                        // Convert JsonValue to PyObject
                        let mut py_attributes = HashMap::new();
                        for (attr_name, json_value) in json_attributes {
                            let py_value = json_value_to_python(py, &json_value)?;
                            py_attributes.insert(attr_name, py_value);
                        }

                        Ok(Some(py_attributes))
                    })
                } else {
                    Ok(None)
                }
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Check if the graph is directed
    pub fn is_directed(&self) -> bool {
        self.is_directed
    }

    /// Get edge IDs
    pub fn get_edge_ids(&self) -> Vec<String> {
        let mut edge_ids = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some((source_idx, target_idx)) = self.graph.edge_endpoints(edge_idx) {
                if let (Some(source_id), Some(target_id)) = (
                    self.node_index_to_id.get(&source_idx),
                    self.node_index_to_id.get(&target_idx),
                ) {
                    edge_ids.push(format!("{}->{}", source_id, target_id));
                }
            }
        }
        edge_ids
    }

    /// Set a single node attribute
    pub fn set_node_attribute(
        &mut self,
        node_id: &str,
        attribute: &str,
        value: &PyAny,
    ) -> PyResult<()> {
        let node_idx = self.node_id_to_index.get(node_id).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node '{}' not found", node_id))
        })?;

        // Convert Python value to JsonValue
        let json_value = python_value_to_json(value)?;

        // Store in columnar format
        self.columnar_store
            .set_node_attribute(node_idx.index(), attribute, json_value);

        Ok(())
    }

    /// Set a single edge attribute
    pub fn set_edge_attribute(
        &mut self,
        source: &str,
        target: &str,
        attribute: &str,
        value: &PyAny,
    ) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(source).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Source node '{}' not found",
                source
            ))
        })?;
        let target_idx = self.node_id_to_index.get(target).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Target node '{}' not found",
                target
            ))
        })?;

        let edge_idx = self
            .graph
            .find_edge(*source_idx, *target_idx)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Edge from '{}' to '{}' not found",
                    source, target
                ))
            })?;

        // Convert Python value to JsonValue
        let json_value = python_value_to_json(value)?;

        // Store in columnar format
        self.columnar_store
            .set_edge_attribute(edge_idx.index(), attribute, json_value);

        Ok(())
    }

    /// Set multiple edge attributes at once
    pub fn set_edge_attributes(
        &mut self,
        source: &str,
        target: &str,
        attributes: &PyDict,
    ) -> PyResult<()> {
        let source_idx = self.node_id_to_index.get(source).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Source node '{}' not found",
                source
            ))
        })?;
        let target_idx = self.node_id_to_index.get(target).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Target node '{}' not found",
                target
            ))
        })?;

        let edge_idx = self
            .graph
            .find_edge(*source_idx, *target_idx)
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Edge from '{}' to '{}' not found",
                    source, target
                ))
            })?;

        // Convert all attributes
        let attr_map = python_dict_to_json_map(attributes)?;

        // Store each attribute in columnar format
        for (attr_name, attr_value) in attr_map {
            self.columnar_store
                .set_edge_attribute(edge_idx.index(), &attr_name, attr_value);
        }

        Ok(())
    }

    /// Set batch node attributes - efficiently set attributes for multiple nodes
    pub fn set_nodes_attributes_batch(
        &mut self,
        node_attrs: HashMap<String, HashMap<String, pyo3::PyObject>>,
    ) -> PyResult<()> {
        use pyo3::Python;

        Python::with_gil(|py| {
            for (node_id, attributes) in node_attrs {
                let node_idx = self.node_id_to_index.get(&node_id).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Node '{}' not found",
                        node_id
                    ))
                })?;

                for (attr_name, py_value) in attributes {
                    let json_value = python_pyobject_to_json(py, &py_value)?;
                    self.columnar_store.set_node_attribute(
                        node_idx.index(),
                        &attr_name,
                        json_value,
                    );
                }
            }
            Ok(())
        })
    }

    /// Set batch edge attributes - efficiently set attributes for multiple edges
    pub fn set_edges_attributes_batch(
        &mut self,
        edge_attrs: HashMap<(String, String), HashMap<String, pyo3::PyObject>>,
    ) -> PyResult<()> {
        use pyo3::Python;

        Python::with_gil(|py| {
            for ((source, target), attributes) in edge_attrs {
                let source_idx = self.node_id_to_index.get(&source).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Source node '{}' not found",
                        source
                    ))
                })?;
                let target_idx = self.node_id_to_index.get(&target).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Target node '{}' not found",
                        target
                    ))
                })?;

                let edge_idx = self
                    .graph
                    .find_edge(*source_idx, *target_idx)
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                            "Edge from '{}' to '{}' not found",
                            source, target
                        ))
                    })?;

                for (attr_name, py_value) in attributes {
                    let json_value = python_pyobject_to_json(py, &py_value)?;
                    self.columnar_store.set_edge_attribute(
                        edge_idx.index(),
                        &attr_name,
                        json_value,
                    );
                }
            }
            Ok(())
        })
    }

    /// Remove a node and all its edges
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if let Some(node_idx) = self.node_id_to_index.get(node_id).cloned() {
            // Remove all edges connected to this node first
            let edges_to_remove: Vec<_> = self
                .graph
                .edge_indices()
                .into_iter()
                .filter(|&edge_idx| {
                    if let Some((source, target)) = self.graph.edge_endpoints(edge_idx) {
                        source == node_idx || target == node_idx
                    } else {
                        false
                    }
                })
                .collect();

            for edge_idx in edges_to_remove {
                // Clean up edge mappings and columnar storage
                if let Some((_source, _target)) = self.edge_index_to_endpoints.remove(&edge_idx) {
                    // Remove from columnar store (implement if needed)
                    // self.columnar_store.remove_edge(edge_idx.index());
                }
                self.graph.remove_edge(edge_idx);
            }

            // Remove the node from columnar store
            self.columnar_store.remove_node_legacy(node_idx.index());

            // Remove the node from the graph
            self.graph.remove_node(node_idx);

            // Clean up mappings
            self.node_id_to_index.remove(node_id);
            self.node_index_to_id.remove(&node_idx);

            true
        } else {
            false
        }
    }

    /// Remove an edge
    pub fn remove_edge(&mut self, source: &str, target: &str) -> bool {
        if let (Some(source_idx), Some(target_idx)) = (
            self.node_id_to_index.get(source),
            self.node_id_to_index.get(target),
        ) {
            if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
                // Remove from edge mappings
                self.edge_index_to_endpoints.remove(&edge_idx);

                // Remove from columnar store (if needed)
                // self.columnar_store.remove_edge(edge_idx.index());

                // Remove from graph
                self.graph.remove_edge(edge_idx);

                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Remove multiple nodes - delegates to single node removal
    pub fn remove_nodes(&mut self, node_ids: Vec<String>) -> usize {
        let mut removed_count = 0;
        for node_id in node_ids {
            if self.remove_node(&node_id) {
                removed_count += 1;
            }
        }
        removed_count
    }

    /// Remove multiple edges - delegates to single edge removal
    pub fn remove_edges(&mut self, edge_pairs: Vec<(String, String)>) -> usize {
        let mut removed_count = 0;
        for (source, target) in edge_pairs {
            if self.remove_edge(&source, &target) {
                removed_count += 1;
            }
        }
        removed_count
    }

    // === UNIFIED FILTERING IMPLEMENTATION ===

    /// Unified entity filtering by attributes - eliminates duplication between nodes and edges
    fn filter_entities_by_attributes(
        &self,
        entity_type: &str,
        filters: HashMap<String, pyo3::PyObject>,
    ) -> PyResult<Vec<String>> {
        use pyo3::Python;

        if filters.is_empty() {
            return Ok(match entity_type {
                "node" => self.get_node_ids(),
                "edge" => self.get_edge_ids(),
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
            });
        }

        Python::with_gil(|py| {
            // Convert PyObjects to JsonValues
            let mut json_filters = HashMap::new();
            for (attr_name, py_value) in filters {
                let json_value = python_pyobject_to_json(py, &py_value)?;
                json_filters.insert(attr_name, json_value);
            }

            // Use columnar store's unified filtering
            let entity_enum = match entity_type {
                "node" => EntityType::Node,
                "edge" => EntityType::Edge,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
            };

            let criteria = FilterCriteria::Attributes(json_filters);
            let matching_indices = self.columnar_store.filter_entities(&entity_enum, criteria);

            // Convert indices back to IDs based on entity type
            self.indices_to_ids(entity_type, matching_indices)
        })
    }

    /// Unified entity filtering by numeric comparison
    fn filter_entities_by_numeric_comparison(
        &self,
        entity_type: &str,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> PyResult<Vec<String>> {
        let entity_enum = match entity_type {
            "node" => EntityType::Node,
            "edge" => EntityType::Edge,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
        };

        let criteria = FilterCriteria::Numeric(attr_name.to_string(), operator.to_string(), value);
        let matching_indices = self.columnar_store.filter_entities(&entity_enum, criteria);

        self.indices_to_ids(entity_type, matching_indices)
    }

    /// Unified entity filtering by string comparison
    fn filter_entities_by_string_comparison(
        &self,
        entity_type: &str,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> PyResult<Vec<String>> {
        let entity_enum = match entity_type {
            "node" => EntityType::Node,
            "edge" => EntityType::Edge,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
        };

        let criteria = FilterCriteria::String(attr_name.to_string(), operator.to_string(), value.to_string());
        let matching_indices = self.columnar_store.filter_entities(&entity_enum, criteria);

        self.indices_to_ids(entity_type, matching_indices)
    }

    /// Unified entity filtering with multi-criteria
    fn filter_entities_multi_criteria(
        &self,
        entity_type: &str,
        exact_matches: HashMap<String, String>,
        numeric_comparisons: Vec<(String, String, f64)>,
        string_comparisons: Vec<(String, String, String)>,
    ) -> PyResult<Vec<String>> {
        let entity_enum = match entity_type {
            "node" => EntityType::Node,
            "edge" => EntityType::Edge,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
        };

        // Convert exact matches to JSON values
        let mut exact_json = HashMap::new();
        for (key, value) in exact_matches {
            exact_json.insert(key, serde_json::Value::String(value));
        }

        let criteria = FilterCriteria::MultiCriteria {
            exact: exact_json,
            numeric: numeric_comparisons,
            string: string_comparisons,
        };

        let matching_indices = self.columnar_store.filter_entities(&entity_enum, criteria);
        self.indices_to_ids(entity_type, matching_indices)
    }

    /// Unified entity filtering by attributes with sparse algorithm
    fn filter_entities_by_attributes_sparse(
        &self,
        entity_type: &str,
        filters: HashMap<String, pyo3::PyObject>,
    ) -> PyResult<Vec<String>> {
        use pyo3::Python;

        if filters.is_empty() {
            return Ok(match entity_type {
                "node" => self.get_node_ids(),
                "edge" => self.get_edge_ids(),
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
            });
        }

        Python::with_gil(|py| {
            // Convert PyObjects to JsonValues
            let mut json_filters = HashMap::new();
            for (attr_name, py_value) in filters {
                let json_value = python_pyobject_to_json(py, &py_value)?;
                json_filters.insert(attr_name, json_value);
            }

            // Use columnar store's sparse filtering (only for nodes currently)
            let matching_indices = match entity_type {
                "node" => self.columnar_store.filter_nodes_sparse(&json_filters),
                "edge" => {
                    // For edges, fall back to regular filtering since sparse is only implemented for nodes
                    self.columnar_store.filter_edges_by_attributes(&json_filters)
                },
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
            };

            self.indices_to_ids(entity_type, matching_indices)
        })
    }

    /// Helper method to convert indices to IDs based on entity type
    fn indices_to_ids(&self, entity_type: &str, indices: Vec<usize>) -> PyResult<Vec<String>> {
        let mut result = Vec::new();

        match entity_type {
            "node" => {
                for node_index in indices {
                    if let Some(node_id) = self
                        .node_index_to_id
                        .get(&petgraph::graph::NodeIndex::new(node_index))
                    {
                        result.push(node_id.clone());
                    }
                }
            },
            "edge" => {
                for edge_index in indices {
                    if let Some((source_idx, target_idx)) = self
                        .graph
                        .edge_endpoints(petgraph::graph::EdgeIndex::new(edge_index))
                    {
                        if let (Some(source_id), Some(target_id)) = (
                            self.node_index_to_id.get(&source_idx),
                            self.node_index_to_id.get(&target_idx),
                        ) {
                            result.push(format!("{}->{}", source_id, target_id));
                        }
                    }
                }
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid entity type")),
        }

        Ok(result)
    }

    /// Get a specific attribute for multiple nodes efficiently  
    pub fn get_nodes_attribute(
        &self,
        attr_name: &str,
        node_ids: Vec<String>
    ) -> PyResult<HashMap<String, pyo3::PyObject>> {
        use crate::utils::json_value_to_python;
        use pyo3::Python;
        
        Python::with_gil(|py| {
            let mut result = HashMap::new();
            
            for node_id in node_ids {
                if let Some(node_idx) = self.node_id_to_index.get(&node_id) {
                    if let Some(json_value) = self.columnar_store.get_node_attribute(node_idx.index(), attr_name) {
                        let py_value = json_value_to_python(py, &json_value)?;
                        result.insert(node_id, py_value);
                    }
                }
            }
            
            Ok(result)
        })
    }

    /// Get all attributes for multiple nodes efficiently
    pub fn get_nodes_attributes(
        &self,
        node_ids: Vec<String>
    ) -> PyResult<HashMap<String, HashMap<String, pyo3::PyObject>>> {
        use crate::utils::json_value_to_python;
        use pyo3::Python;
        
        Python::with_gil(|py| {
            let mut result = HashMap::new();
            
            for node_id in node_ids {
                if let Some(node_idx) = self.node_id_to_index.get(&node_id) {
                    let json_attributes = self.columnar_store.get_node_attributes(node_idx.index());
                    if !json_attributes.is_empty() {
                        let mut py_attributes = HashMap::new();
                        for (attr_name, json_value) in json_attributes {
                            let py_value = json_value_to_python(py, &json_value)?;
                            py_attributes.insert(attr_name, py_value);
                        }
                        result.insert(node_id, py_attributes);
                    }
                }
            }
            
            Ok(result)
        })
    }

    /// Get a specific attribute for all nodes efficiently (useful for statistics)
    pub fn get_all_nodes_attribute(
        &self,
        attr_name: &str
    ) -> PyResult<HashMap<String, pyo3::PyObject>> {
        use crate::utils::json_value_to_python;
        use pyo3::Python;
        
        Python::with_gil(|py| {
            let mut result = HashMap::new();
            
            for (node_id, node_idx) in &self.node_id_to_index {
                if let Some(json_value) = self.columnar_store.get_node_attribute(node_idx.index(), attr_name) {
                    let py_value = json_value_to_python(py, &json_value)?;
                    result.insert(node_id.clone(), py_value);
                }
            }
            
            Ok(result)
        })
    }

    /// Get a specific attribute for multiple edges efficiently
    pub fn get_edges_attribute(
        &self,
        attr_name: &str,
        edge_endpoints: Vec<(String, String)>
    ) -> PyResult<HashMap<(String, String), pyo3::PyObject>> {
        use crate::utils::json_value_to_python;
        use pyo3::Python;
        
        Python::with_gil(|py| {
            let mut result = HashMap::new();
            
            for (source, target) in edge_endpoints {
                if let (Some(source_idx), Some(target_idx)) = (
                    self.node_id_to_index.get(&source),
                    self.node_id_to_index.get(&target)
                ) {
                    if let Some(edge_idx) = self.graph.find_edge(*source_idx, *target_idx) {
                        if let Some(json_value) = self.columnar_store.get_edge_attribute(edge_idx.index(), attr_name) {
                            let py_value = json_value_to_python(py, &json_value)?;
                            result.insert((source, target), py_value);
                        }
                    }
                }
            }
            
            Ok(result)
        })
    }

    /// Get a specific attribute for all edges efficiently (useful for statistics)
    pub fn get_all_edges_attribute(
        &self,
        attr_name: &str
    ) -> PyResult<HashMap<(String, String), pyo3::PyObject>> {
        use crate::utils::json_value_to_python;
        use pyo3::Python;
        
        Python::with_gil(|py| {
            let mut result = HashMap::new();
            
            for edge_idx in self.graph.edge_indices() {
                if let Some((source_idx, target_idx)) = self.graph.edge_endpoints(edge_idx) {
                    // Get source and target node IDs from the mapping
                    if let (Some(source_id), Some(target_id)) = (
                        self.node_index_to_id.get(&source_idx),
                        self.node_index_to_id.get(&target_idx)
                    ) {
                        if let Some(json_value) = self.columnar_store.get_edge_attribute(edge_idx.index(), attr_name) {
                            let py_value = json_value_to_python(py, &json_value)?;
                            result.insert((source_id.clone(), target_id.clone()), py_value);
                        }
                    }
                }
            }
            
            Ok(result)
        })
    }
}

impl FastGraph {
    /// Get node weight by index (internal use)
    pub fn get_node_weight(&self, node_idx: petgraph::graph::NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(node_idx)
    }

    /// Get edge weight by index (internal use)
    pub fn get_edge_weight(&self, edge_idx: petgraph::graph::EdgeIndex) -> Option<&EdgeData> {
        self.graph.edge_weight(edge_idx)
    }

    /// Get all edge indices (internal use)
    pub fn get_edge_indices(&self) -> Vec<petgraph::graph::EdgeIndex> {
        self.graph.edge_indices()
    }

    /// Get edge endpoints (internal use)
    pub fn get_edge_endpoints(
        &self,
        edge_idx: petgraph::graph::EdgeIndex,
    ) -> Option<(petgraph::graph::NodeIndex, petgraph::graph::NodeIndex)> {
        self.graph.edge_endpoints(edge_idx)
    }



    /// Get neighbors by index (internal use)
    pub fn get_neighbors_public(
        &self,
        node_idx: petgraph::graph::NodeIndex,
    ) -> Vec<petgraph::graph::NodeIndex> {
        self.graph.neighbors(node_idx).collect()
    }

    /// Get directed edges (internal use)
    pub fn get_edges_directed(
        &self,
        node_idx: petgraph::graph::NodeIndex,
        direction: petgraph::Direction,
    ) -> Vec<petgraph::graph::EdgeReference<EdgeData>> {
        self.graph.edges_directed(node_idx, direction)
    }

    /// Add node directly to graph (internal use only - not part of public API)
    pub(crate) fn add_node_to_graph_public(&mut self, node_data: NodeData) -> petgraph::graph::NodeIndex {
        self.graph.add_node(node_data)
    }

    /// Add edge directly to graph (internal use only - not part of public API)
    pub(crate) fn add_edge_to_graph_public(
        &mut self,
        source_idx: petgraph::graph::NodeIndex,
        target_idx: petgraph::graph::NodeIndex,
        edge_data: EdgeData,
    ) -> petgraph::graph::EdgeIndex {
        self.graph.add_edge(source_idx, target_idx, edge_data)
    }
    
    /// Internal bulk add nodes (delegates to bulk_operations.rs)
    fn bulk_add_nodes_internal_call(&mut self, nodes: Vec<(String, HashMap<String, serde_json::Value>)>) {
        // Use the existing optimized bulk operations
        self.bulk_add_nodes_internal_impl(nodes);
    }
    
    /// Internal bulk add edges (delegates to bulk_operations.rs) 
    fn bulk_add_edges_internal_call(&mut self, edges: Vec<(String, String, HashMap<String, serde_json::Value>)>) {
        // Use the existing optimized bulk operations
        self.bulk_add_edges_internal_impl(edges);
    }
    

    
    /// Wrapper for the bulk operations implementation
    pub(crate) fn bulk_add_nodes_internal_impl(
        &mut self, 
        nodes_data: Vec<(String, HashMap<String, serde_json::Value>)>
    ) -> Vec<petgraph::graph::NodeIndex> {
        let mut node_indices = Vec::with_capacity(nodes_data.len());
        
        // Prepare bulk columnar operations
        let mut bulk_attributes: HashMap<String, Vec<(usize, serde_json::Value)>> = HashMap::new();
        
        // Add nodes to graph structure first
        for (node_id, attributes) in nodes_data {
            // Skip if node already exists
            if self.node_id_to_index.contains_key(&node_id) {
                continue;
            }
            
            // Create lightweight node data
            let node_data = NodeData {
                id: node_id.clone(),
                attr_uids: std::collections::HashSet::new(),
            };
            
            // Add to graph topology
            let node_index = self.graph.add_node(node_data);
            
            // Update mappings
            self.node_id_to_index.insert(node_id.clone(), node_index);
            self.node_index_to_id.insert(node_index, node_id);
            
            node_indices.push(node_index);
            
            // Prepare attributes for bulk columnar insert
            for (attr_name, attr_value) in attributes {
                bulk_attributes
                    .entry(attr_name)
                    .or_insert_with(Vec::new)
                    .push((node_index.index(), attr_value));
            }
        }
        
        // Bulk insert attributes into columnar store
        for (attr_name, attr_data) in bulk_attributes {
            self.columnar_store.bulk_set_node_attributes(&attr_name, attr_data);
        }
        
        node_indices
    }
    
    /// Wrapper for the bulk edge operations implementation
    pub(crate) fn bulk_add_edges_internal_impl(
        &mut self,
        edges_data: Vec<(String, String, HashMap<String, serde_json::Value>)>
    ) -> Vec<petgraph::graph::EdgeIndex> {
        let mut edge_indices = Vec::with_capacity(edges_data.len());
        
        // Prepare bulk columnar operations
        let mut bulk_attributes: HashMap<String, Vec<(usize, serde_json::Value)>> = HashMap::new();
        
        // Add edges to graph structure first
        for (source, target, attributes) in edges_data {
            // Get node indices
            let source_idx = match self.node_id_to_index.get(&source) {
                Some(idx) => *idx,
                None => continue, // Skip if source doesn't exist
            };
            let target_idx = match self.node_id_to_index.get(&target) {
                Some(idx) => *idx,
                None => continue, // Skip if target doesn't exist
            };
            
            // Create lightweight edge data
            let edge_data = EdgeData {
                source: source.clone(),
                target: target.clone(),
                attr_uids: std::collections::HashSet::new(),
            };
            
            // Add to graph topology
            let edge_index = self.graph.add_edge(source_idx, target_idx, edge_data);
            
            // Store edge mapping
            self.edge_index_to_endpoints.insert(edge_index, (source, target));
            
            edge_indices.push(edge_index);
            
            // Prepare attributes for bulk columnar insert
            for (attr_name, attr_value) in attributes {
                bulk_attributes
                    .entry(attr_name)
                    .or_insert_with(Vec::new)
                    .push((edge_index.index(), attr_value));
            }
        }
        
        // Bulk insert attributes into columnar store
        for (attr_name, attr_data) in bulk_attributes {
            self.columnar_store.bulk_set_edge_attributes(&attr_name, attr_data);
        }
        
        edge_indices
    }
    

}
