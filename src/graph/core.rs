#![allow(non_local_definitions)]
#![allow(clippy::uninlined_format_args)]
use petgraph::graph::NodeIndex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::graph::types::{EdgeData, GraphType, NodeData};
use crate::storage::ColumnarStore;
use crate::utils::{python_dict_to_json_map, python_pyobject_to_json, python_value_to_json};

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

    /// Add a single node with optional attributes
    pub fn add_node(&mut self, node_id: String, attributes: Option<&PyDict>) -> PyResult<()> {
        // Check if node already exists
        if self.node_id_to_index.contains_key(&node_id) {
            return Ok(()); // Node already exists
        }

        // Create node data (lightweight)
        let node_data = NodeData {
            id: node_id.clone(),
            attr_uids: std::collections::HashSet::new(),
        };

        // Add to graph topology
        let node_index = self.graph.add_node(node_data);

        // Update mappings
        self.node_id_to_index.insert(node_id.clone(), node_index);
        self.node_index_to_id.insert(node_index, node_id.clone());

        // Store attributes in columnar format
        if let Some(attrs) = attributes {
            let attr_map = python_dict_to_json_map(attrs)?;
            for (attr_name, attr_value) in attr_map {
                self.columnar_store
                    .set_node_attribute(node_index.index(), &attr_name, attr_value);
            }
        }

        Ok(())
    }

    /// Add a single edge with optional attributes
    pub fn add_edge(
        &mut self,
        source: String,
        target: String,
        attributes: Option<&PyDict>,
    ) -> PyResult<()> {
        // Get node indices
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

        // Create edge data (lightweight)
        let edge_data = EdgeData {
            source: source.clone(),
            target: target.clone(),
            attr_uids: std::collections::HashSet::new(),
        };

        // Add to graph topology
        let edge_index = self.graph.add_edge(*source_idx, *target_idx, edge_data);

        // Store edge mapping
        self.edge_index_to_endpoints
            .insert(edge_index, (source, target));

        // Store attributes in columnar format
        if let Some(attrs) = attributes {
            let attr_map = python_dict_to_json_map(attrs)?;
            for (attr_name, attr_value) in attr_map {
                self.columnar_store
                    .set_edge_attribute(edge_index.index(), &attr_name, attr_value);
            }
        }

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

    /// Get node attributes - retrieve from columnar store
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

    /// Add multiple nodes in batch - now uses optimized bulk operations internally
    pub fn add_nodes(&mut self, nodes_data: Vec<(String, Option<&PyDict>)>) -> PyResult<()> {
        // Convert to bulk format
        let mut bulk_nodes = Vec::new();
        for (node_id, attributes) in nodes_data {
            let attrs = if let Some(py_attrs) = attributes {
                python_dict_to_json_map(py_attrs)?
            } else {
                std::collections::HashMap::new()
            };
            bulk_nodes.push((node_id, attrs));
        }
        
        // Use optimized bulk method
        self.bulk_add_nodes_internal(bulk_nodes);
        Ok(())
    }

    /// Add multiple edges in batch - now uses optimized bulk operations internally
    pub fn add_edges(
        &mut self,
        edges_data: Vec<(String, String, Option<&PyDict>)>,
    ) -> PyResult<()> {
        // Convert to bulk format
        let mut bulk_edges = Vec::new();
        for (source, target, attributes) in edges_data {
            let attrs = if let Some(py_attrs) = attributes {
                python_dict_to_json_map(py_attrs)?
            } else {
                std::collections::HashMap::new()
            };
            bulk_edges.push((source, target, attrs));
        }
        
        // Use optimized bulk method
        self.bulk_add_edges_internal(bulk_edges);
        Ok(())
    }

    /// Filter nodes by attribute dictionary - Python interface to columnar filtering
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

            // Use columnar store's optimized filtering ONLY
            let matching_indices = self
                .columnar_store
                .filter_nodes_by_attributes(&json_filters);

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

    /// Filter edges by attribute dictionary - Python interface to columnar filtering  
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

            // Use columnar store's optimized filtering ONLY
            let matching_indices = self
                .columnar_store
                .filter_edges_by_attributes(&json_filters);

            // Convert indices back to edge IDs (source->target format)
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

    /// Filter nodes by numeric comparison - Python interface to columnar filtering
    pub fn filter_nodes_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> PyResult<Vec<String>> {
        let matching_indices = self
            .columnar_store
            .filter_nodes_by_numeric_comparison(attr_name, operator, value);

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
    }

    /// Filter nodes by string comparison - Python interface to columnar filtering  
    pub fn filter_nodes_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> PyResult<Vec<String>> {
        let matching_indices = self
            .columnar_store
            .filter_nodes_by_string_comparison(attr_name, operator, value);

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
    }

    /// Filter edges by numeric comparison - Python interface to columnar filtering
    pub fn filter_edges_by_numeric_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: f64,
    ) -> PyResult<Vec<String>> {
        let matching_indices = self
            .columnar_store
            .filter_edges_by_numeric_comparison(attr_name, operator, value);

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
    }

    /// Filter edges by string comparison - Python interface to columnar filtering
    pub fn filter_edges_by_string_comparison(
        &self,
        attr_name: &str,
        operator: &str,
        value: &str,
    ) -> PyResult<Vec<String>> {
        let matching_indices = self
            .columnar_store
            .filter_edges_by_string_comparison(attr_name, operator, value);

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
    }

    /// Filter nodes with sparse algorithm - Python interface to optimized sparse filtering  
    pub fn filter_nodes_by_attributes_sparse(
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

            // Use columnar store's optimized sparse filtering ONLY (no bitmap creation)
            let matching_indices = self
                .columnar_store
                .filter_nodes_sparse(&json_filters);

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

    /// Optimized multi-criteria node filtering - all intersection logic in Rust
    pub fn filter_nodes_multi_criteria(
        &self,
        exact_matches: HashMap<String, String>,
        numeric_comparisons: Vec<(String, String, f64)>,
        string_comparisons: Vec<(String, String, String)>,
    ) -> PyResult<Vec<String>> {
        // Convert exact matches to JsonValue
        let json_exact: HashMap<String, serde_json::Value> = exact_matches
            .into_iter()
            .map(|(k, v)| {
                // Smart type conversion
                let json_value = if v == "true" || v == "True" {
                    serde_json::Value::Bool(true)
                } else if v == "false" || v == "False" {
                    serde_json::Value::Bool(false)
                } else if let Ok(num) = v.parse::<i64>() {
                    serde_json::Value::Number(serde_json::Number::from(num))
                } else if let Ok(num) = v.parse::<f64>() {
                    serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap_or(serde_json::Number::from(0)))
                } else {
                    serde_json::Value::String(v)
                };
                (k, json_value)
            })
            .collect();

        // Use the optimized multi-criteria filtering (all intersection logic in Rust)
        let matching_indices = self.columnar_store.filter_nodes_multi_criteria(
            &json_exact,
            &numeric_comparisons,
            &string_comparisons,
        );

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
    }

    /// Optimized multi-criteria edge filtering - all intersection logic in Rust
    pub fn filter_edges_multi_criteria(
        &self,
        exact_matches: HashMap<String, String>,
        numeric_comparisons: Vec<(String, String, f64)>,
        string_comparisons: Vec<(String, String, String)>,
    ) -> PyResult<Vec<String>> {
        // Convert exact matches to JsonValue
        let json_exact: HashMap<String, serde_json::Value> = exact_matches
            .into_iter()
            .map(|(k, v)| {
                // Smart type conversion
                let json_value = if v == "true" || v == "True" {
                    serde_json::Value::Bool(true)
                } else if v == "false" || v == "False" {
                    serde_json::Value::Bool(false)
                } else if let Ok(num) = v.parse::<i64>() {
                    serde_json::Value::Number(serde_json::Number::from(num))
                } else if let Ok(num) = v.parse::<f64>() {
                    serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap_or(serde_json::Number::from(0)))
                } else {
                    serde_json::Value::String(v)
                };
                (k, json_value)
            })
            .collect();

        // Use the optimized multi-criteria filtering (all intersection logic in Rust)
        let matching_indices = self.columnar_store.filter_edges_multi_criteria(
            &json_exact,
            &numeric_comparisons,
            &string_comparisons,
        );

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

    /// Add node directly to graph (internal use)
    pub fn add_node_to_graph_public(&mut self, node_data: NodeData) -> petgraph::graph::NodeIndex {
        self.graph.add_node(node_data)
    }

    /// Add edge directly to graph (internal use)
    pub fn add_edge_to_graph_public(
        &mut self,
        source_idx: petgraph::graph::NodeIndex,
        target_idx: petgraph::graph::NodeIndex,
        edge_data: EdgeData,
    ) -> petgraph::graph::EdgeIndex {
        self.graph.add_edge(source_idx, target_idx, edge_data)
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
}
