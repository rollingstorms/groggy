//! Traversal result types
//!
//! Python bindings for traversal result structures.

use groggy::{EdgeId, NodeId};
use pyo3::prelude::*;

/// Result of a graph traversal operation
#[pyclass(name = "TraversalResult")]
pub struct PyTraversalResult {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub distances: Option<Vec<usize>>,
    pub traversal_type: String,
}

#[pymethods]
impl PyTraversalResult {
    #[getter]
    fn nodes(&self) -> Vec<NodeId> {
        self.nodes.clone()
    }

    #[getter]
    fn edges(&self) -> Vec<EdgeId> {
        self.edges.clone()
    }

    #[getter]
    fn distances(&self) -> Option<Vec<usize>> {
        self.distances.clone()
    }

    #[getter]
    fn traversal_type(&self) -> String {
        self.traversal_type.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TraversalResult(nodes={}, edges={}, type='{}')",
            self.nodes.len(),
            self.edges.len(),
            self.traversal_type
        )
    }
}

impl PyTraversalResult {
    pub fn new(
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        distances: Option<Vec<usize>>,
        traversal_type: String,
    ) -> Self {
        Self {
            nodes,
            edges,
            distances,
            traversal_type,
        }
    }
}

/// Result of an aggregation operation
#[pyclass(name = "AggregationResult")]
pub struct PyAggregationResult {
    pub value: f64,
    pub operation: String,
    pub attribute: String,
    pub count: usize,
}

#[pymethods]
impl PyAggregationResult {
    #[getter]
    fn value(&self) -> f64 {
        self.value
    }

    #[getter]
    fn operation(&self) -> String {
        self.operation.clone()
    }

    #[getter]
    fn attribute(&self) -> String {
        self.attribute.clone()
    }

    #[getter]
    fn count(&self) -> usize {
        self.count
    }

    fn __repr__(&self) -> String {
        format!(
            "AggregationResult(value={}, operation='{}', attribute='{}')",
            self.value, self.operation, self.attribute
        )
    }
}

impl PyAggregationResult {
    pub fn new(value: f64, operation: String, attribute: String, count: usize) -> Self {
        Self {
            value,
            operation,
            attribute,
            count,
        }
    }
}

/// Result of a grouped aggregation operation
#[pyclass(name = "GroupedAggregationResult")]
pub struct PyGroupedAggregationResult {
    pub groups: PyObject,
    pub operation: String,
    pub attribute: String,
}

#[pymethods]
impl PyGroupedAggregationResult {
    #[getter]
    fn groups(&self) -> PyObject {
        self.groups.clone()
    }

    #[getter]
    fn operation(&self) -> String {
        self.operation.clone()
    }

    #[getter]
    fn attribute(&self) -> String {
        self.attribute.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupedAggregationResult(operation='{}', attribute='{}')",
            self.operation, self.attribute
        )
    }
}

impl PyGroupedAggregationResult {
    pub fn new(groups: PyObject, operation: String, attribute: String) -> Self {
        Self {
            groups,
            operation,
            attribute,
        }
    }
}
