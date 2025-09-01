//! Graph Query Module
//!
//! Python bindings for advanced query operations on graphs.

use crate::ffi::core::query::{PyEdgeFilter, PyNodeFilter};
use crate::ffi::core::subgraph::PySubgraph;
use crate::ffi::utils::graph_error_to_py_err;
use groggy::core::subgraph::Subgraph;
use groggy::{AttrName, EdgeId, NodeId};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

/// Query operations for graphs
#[pyclass(name = "GraphQuery")]
pub struct PyGraphQuery {
    /// Reference to the parent graph
    pub graph: Py<crate::ffi::api::graph::PyGraph>,
}

#[pymethods]
impl PyGraphQuery {
    /// Filter nodes by criteria - DELEGATED to core Graph.find_nodes()
    pub fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Extract filter and delegate to core Graph.find_nodes()
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            filter_obj.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object",
            ));
        };

        // Pure delegation to core Graph methods
        let graph = self.graph.borrow(py);
        let filtered_nodes = graph
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // Create subgraph using core Subgraph constructor
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let induced_edges = Subgraph::calculate_induced_edges(&graph.inner, &node_set)
            .map_err(graph_error_to_py_err)?;

        let subgraph = Subgraph::new(
            graph.inner.clone(),
            node_set,
            induced_edges,
            "filtered_nodes".to_string(),
        );

        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Filter edges by criteria - DELEGATED to core Graph.find_edges()
    pub fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Extract filter and delegate to core Graph.find_edges()
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            filter_obj.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object",
            ));
        };

        // Pure delegation to core Graph methods
        let graph = self.graph.borrow(py);
        let filtered_edges = graph
            .inner
            .borrow_mut()
            .find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;

        // Calculate incident nodes using core Graph.edge_endpoints()
        let mut incident_nodes = HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                incident_nodes.insert(source);
                incident_nodes.insert(target);
            }
        }

        // Create subgraph using core Subgraph constructor
        let edge_set: HashSet<EdgeId> = filtered_edges.iter().copied().collect();
        let subgraph = Subgraph::new(
            graph.inner.clone(),
            incident_nodes,
            edge_set,
            "filtered_edges".to_string(),
        );

        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Filter nodes within a subgraph
    fn filter_subgraph_nodes(
        &self,
        py: Python,
        subgraph: &PySubgraph,
        filter: &PyAny,
    ) -> PyResult<PySubgraph> {
        let graph = self.graph.borrow(py);

        // Parse filter same way as filter_nodes
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser (FIXED: no circular dependency)
            let mut parser = groggy::core::query_parser::QueryParser::new();
            parser.parse_node_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query",
            ));
        };

        // Apply filter only to nodes in the subgraph
        let subgraph_node_set = subgraph.inner.nodes().clone();
        let all_filtered_nodes = graph
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // Intersect with subgraph nodes
        let filtered_nodes: Vec<NodeId> = all_filtered_nodes
            .into_iter()
            .filter(|node_id| subgraph_node_set.contains(node_id))
            .collect();

        // Calculate induced edges within the filtered nodes
        let filtered_node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let filtered_edges: Vec<EdgeId> = subgraph
            .inner
            .edges()
            .iter()
            .filter(|&&edge_id| {
                if let Ok((source, target)) = graph.inner.borrow().edge_endpoints(edge_id) {
                    filtered_node_set.contains(&source) && filtered_node_set.contains(&target)
                } else {
                    false
                }
            })
            .copied()
            .collect();

        // Create subgraph using core Subgraph constructor
        let filtered_node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let filtered_edge_set: HashSet<EdgeId> = filtered_edges.iter().copied().collect();

        let subgraph = Subgraph::new(
            graph.inner.clone(),
            filtered_node_set,
            filtered_edge_set,
            "filtered_subgraph".to_string(),
        );

        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Aggregate attribute values
    #[pyo3(signature = (attribute, operation, target = None, node_ids = None))]
    fn aggregate(
        &self,
        py: Python,
        attribute: AttrName,
        operation: String,
        target: Option<String>,
        node_ids: Option<Vec<NodeId>>,
    ) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let target = target.unwrap_or_else(|| "nodes".to_string());

        match target.as_str() {
            "nodes" => {
                if let Some(node_list) = node_ids {
                    // Custom node list aggregation
                    self.aggregate_custom_nodes(py, &graph, node_list, attribute)
                } else {
                    // All nodes aggregation
                    let result = graph
                        .inner
                        .borrow()
                        .aggregate_node_attribute(&attribute, &operation)
                        .map_err(graph_error_to_py_err)?;

                    let dict = PyDict::new(py);
                    dict.set_item("value", result.value)?;
                    dict.set_item("operation", &operation)?;
                    dict.set_item("attribute", &attribute)?;
                    dict.set_item("target", "nodes")?;
                    Ok(dict.to_object(py))
                }
            }
            "edges" => {
                // Edge aggregation
                let result = graph
                    .inner
                    .borrow()
                    .aggregate_edge_attribute(&attribute, &operation)
                    .map_err(graph_error_to_py_err)?;

                let dict = PyDict::new(py);
                dict.set_item("value", result.value)?;
                dict.set_item("operation", &operation)?;
                dict.set_item("attribute", &attribute)?;
                dict.set_item("target", "edges")?;
                Ok(dict.to_object(py))
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid target '{}'. Use 'nodes' or 'edges'",
                target
            ))),
        }
    }

    /// Execute a graph query with filters
    #[pyo3(signature = (query, **_kwargs))]
    fn execute(
        &self,
        py: Python,
        query: &str,
        _kwargs: Option<&pyo3::types::PyDict>,
    ) -> PyResult<PyObject> {
        // Parse and execute complex graph queries
        let _graph = self.graph.borrow(py);

        // For now, support basic query patterns
        if let Some(filter_str) = query.strip_prefix("nodes where ") {
            let filter_py_str = filter_str.to_string().into_py(py);
            self.filter_nodes(py, filter_py_str.as_ref(py))
                .map(|subgraph| Py::new(py, subgraph).unwrap().to_object(py))
        } else if let Some(filter_str) = query.strip_prefix("edges where ") {
            let filter_py_str = filter_str.to_string().into_py(py);
            self.filter_edges(py, filter_py_str.as_ref(py))
                .map(|subgraph| Py::new(py, subgraph).unwrap().to_object(py))
        } else {
            Err(PyValueError::new_err(format!(
                "Unsupported query pattern: {}",
                query
            )))
        }
    }

    /// Get query statistics
    fn get_stats(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let node_count = graph.get_node_count();
        let edge_count = graph.get_edge_count();

        Ok(format!(
            "Query module ready: {} nodes, {} edges available",
            node_count, edge_count
        ))
    }
}

impl PyGraphQuery {
    /// Helper method for custom node list aggregation
    fn aggregate_custom_nodes(
        &self,
        py: Python,
        graph: &crate::ffi::api::graph::PyGraph,
        node_ids: Vec<NodeId>,
        attribute: AttrName,
    ) -> PyResult<PyObject> {
        // Use bulk attribute retrieval for much better performance
        let bulk_attributes = graph
            .inner
            .borrow()
            ._get_node_attributes_for_nodes(&node_ids, &attribute)
            .map_err(graph_error_to_py_err)?;
        let mut values = Vec::new();

        // Extract values from bulk result
        for value in bulk_attributes.into_iter().flatten() {
            values.push(value);
        }

        // Compute statistics
        let dict = PyDict::new(py);
        dict.set_item("count", values.len())?;

        if !values.is_empty() {
            // Convert first value to determine type for aggregation
            if let Some(first_val) = values.first() {
                match first_val {
                    groggy::AttrValue::Int(_) | groggy::AttrValue::SmallInt(_) => {
                        let int_values: Vec<i64> = values
                            .iter()
                            .filter_map(|v| match v {
                                groggy::AttrValue::Int(i) => Some(*i),
                                groggy::AttrValue::SmallInt(i) => Some(*i as i64),
                                _ => None,
                            })
                            .collect();

                        if !int_values.is_empty() {
                            dict.set_item("sum", int_values.iter().sum::<i64>())?;
                            dict.set_item("min", *int_values.iter().min().unwrap())?;
                            dict.set_item("max", *int_values.iter().max().unwrap())?;
                            dict.set_item(
                                "mean",
                                int_values.iter().sum::<i64>() as f64 / int_values.len() as f64,
                            )?;
                        }
                    }
                    groggy::AttrValue::Float(_) => {
                        let float_values: Vec<f64> = values
                            .iter()
                            .filter_map(|v| match v {
                                groggy::AttrValue::Float(f) => Some(*f as f64),
                                _ => None,
                            })
                            .collect();

                        if !float_values.is_empty() {
                            dict.set_item("sum", float_values.iter().sum::<f64>())?;
                            dict.set_item(
                                "min",
                                float_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                            )?;
                            dict.set_item(
                                "max",
                                float_values
                                    .iter()
                                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                            )?;
                            dict.set_item(
                                "mean",
                                float_values.iter().sum::<f64>() / float_values.len() as f64,
                            )?;
                        }
                    }
                    _ => {
                        // For non-numeric types, just provide count
                    }
                }
            }
        }

        Ok(dict.to_object(py))
    }
}
