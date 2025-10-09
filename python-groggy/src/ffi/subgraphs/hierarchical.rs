//! Python FFI bindings for hierarchical subgraph operations
//!
//! This module provides Python bindings for collapsing subgraphs into meta-nodes,
//! attribute aggregation, and hierarchical graph navigation.

use groggy::subgraphs::{AggregationFunction, MetaNode};
use groggy::traits::subgraph_operations::{
    EdgeAggregationConfig, EdgeAggregationFunction, ExternalEdgeStrategy, MetaEdgeStrategy,
};
use groggy::traits::GraphEntity;
use groggy::{AttrName, AttrValue, NodeId};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;

/// Python wrapper for AggregationFunction
#[pyclass(name = "AggregationFunction", unsendable)]
#[derive(Clone)]
pub struct PyAggregationFunction {
    pub inner: AggregationFunction,
}

#[pymethods]
impl PyAggregationFunction {
    /// Create a Sum aggregation function
    #[classmethod]
    fn sum(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::Sum,
        }
    }

    /// Create a Mean aggregation function
    #[classmethod]
    fn mean(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::Mean,
        }
    }

    /// Create a Max aggregation function
    #[classmethod]
    fn max(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::Max,
        }
    }

    /// Create a Min aggregation function
    #[classmethod]
    fn min(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::Min,
        }
    }

    /// Create a Count aggregation function
    #[classmethod]
    fn count(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::Count,
        }
    }

    /// Create a First aggregation function
    #[classmethod]
    fn first(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::First,
        }
    }

    /// Create a Last aggregation function
    #[classmethod]
    fn last(_cls: &PyType) -> Self {
        Self {
            inner: AggregationFunction::Last,
        }
    }

    /// Create a Concat aggregation function with custom separator
    #[classmethod]
    #[pyo3(signature = (separator = ",".to_string()))]
    fn concat(_cls: &PyType, separator: String) -> Self {
        Self {
            inner: AggregationFunction::Concat(separator),
        }
    }

    /// Parse aggregation function from string
    #[classmethod]
    fn from_string(_cls: &PyType, s: String) -> PyResult<Self> {
        let inner = AggregationFunction::from_string(&s)
            .map_err(|e| PyValueError::new_err(format!("Invalid aggregation function: {}", e)))?;
        Ok(Self { inner })
    }

    /// String representation
    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("AggregationFunction.{}", self.inner)
    }

    /// Apply aggregation to a list of values
    fn aggregate(&self, py: Python, values: Vec<PyObject>) -> PyResult<PyObject> {
        // Convert Python values to AttrValues
        let attr_values: Result<Vec<AttrValue>, PyErr> = values
            .iter()
            .map(|py_val| crate::ffi::utils::python_value_to_attr_value(py_val.as_ref(py)))
            .collect();

        let attr_values = attr_values?;

        // Apply aggregation
        let result = self
            .inner
            .aggregate(&attr_values)
            .map_err(|e| PyValueError::new_err(format!("Aggregation failed: {}", e)))?;

        // Convert result back to Python
        crate::ffi::utils::attr_value_to_python_value(py, &result)
    }
}

/// Python wrapper for ExternalEdgeStrategy
#[pyclass(name = "ExternalEdgeStrategy", unsendable)]
#[derive(Clone)]
pub struct PyExternalEdgeStrategy {
    pub inner: ExternalEdgeStrategy,
}

#[pymethods]
impl PyExternalEdgeStrategy {
    /// Create separate meta-edges for each original edge (preserves all attributes)
    #[classmethod]
    fn copy(_cls: &PyType) -> Self {
        Self {
            inner: ExternalEdgeStrategy::Copy,
        }
    }

    /// Create single meta-edge with aggregated attributes (default)
    #[classmethod]
    fn aggregate(_cls: &PyType) -> Self {
        Self {
            inner: ExternalEdgeStrategy::Aggregate,
        }
    }

    /// Create single meta-edge with only count information
    #[classmethod]
    fn count(_cls: &PyType) -> Self {
        Self {
            inner: ExternalEdgeStrategy::Count,
        }
    }

    /// No meta-edges to external nodes created
    #[classmethod]
    fn none(_cls: &PyType) -> Self {
        Self {
            inner: ExternalEdgeStrategy::None,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("ExternalEdgeStrategy.{:?}", self.inner)
    }
}

/// Python wrapper for MetaEdgeStrategy
#[pyclass(name = "MetaEdgeStrategy", unsendable)]
#[derive(Clone)]
pub struct PyMetaEdgeStrategy {
    pub inner: MetaEdgeStrategy,
}

#[pymethods]
impl PyMetaEdgeStrategy {
    /// Automatically create meta-to-meta edges based on subgraph connections (default)
    #[classmethod]
    fn auto(_cls: &PyType) -> Self {
        Self {
            inner: MetaEdgeStrategy::Auto,
        }
    }

    /// Only create meta-to-meta edges when explicitly requested
    #[classmethod]
    fn explicit(_cls: &PyType) -> Self {
        Self {
            inner: MetaEdgeStrategy::Explicit,
        }
    }

    /// No meta-to-meta edges created automatically
    #[classmethod]
    fn none(_cls: &PyType) -> Self {
        Self {
            inner: MetaEdgeStrategy::None,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("MetaEdgeStrategy.{:?}", self.inner)
    }
}

/// Python wrapper for EdgeAggregationFunction
#[pyclass(name = "EdgeAggregationFunction", unsendable)]
#[derive(Clone)]
pub struct PyEdgeAggregationFunction {
    pub inner: EdgeAggregationFunction,
}

#[pymethods]
impl PyEdgeAggregationFunction {
    #[classmethod]
    fn sum(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Sum,
        }
    }

    #[classmethod]
    fn mean(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Mean,
        }
    }

    #[classmethod]
    fn max(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Max,
        }
    }

    #[classmethod]
    fn min(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Min,
        }
    }

    #[classmethod]
    fn count(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Count,
        }
    }

    #[classmethod]
    fn concat(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Concat,
        }
    }

    #[classmethod]
    fn concat_unique(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::ConcatUnique,
        }
    }

    #[classmethod]
    fn first(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::First,
        }
    }

    #[classmethod]
    fn last(_cls: &PyType) -> Self {
        Self {
            inner: EdgeAggregationFunction::Last,
        }
    }

    /// Parse aggregation function from string
    #[classmethod]
    fn from_string(_cls: &PyType, s: String) -> PyResult<Self> {
        let inner = EdgeAggregationFunction::from_string(&s).map_err(|e| {
            PyValueError::new_err(format!("Invalid edge aggregation function: {}", e))
        })?;
        Ok(Self { inner })
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("EdgeAggregationFunction.{:?}", self.inner)
    }
}

/// Python wrapper for EdgeAggregationConfig
#[pyclass(name = "EdgeAggregationConfig", unsendable)]
#[derive(Clone, Default)]
pub struct PyEdgeAggregationConfig {
    pub inner: EdgeAggregationConfig,
}

#[pymethods]
impl PyEdgeAggregationConfig {
    #[new]
    #[pyo3(signature = (
        edge_to_external = None,
        edge_to_meta = None,
        edge_aggregation = None,
        default_aggregation = None,
        min_edge_count = None,
        include_edge_count = None,
        mark_entity_type = None
    ))]
    fn new(
        edge_to_external: Option<PyExternalEdgeStrategy>,
        edge_to_meta: Option<PyMetaEdgeStrategy>,
        edge_aggregation: Option<&PyDict>,
        default_aggregation: Option<PyEdgeAggregationFunction>,
        min_edge_count: Option<u32>,
        include_edge_count: Option<bool>,
        mark_entity_type: Option<bool>,
    ) -> PyResult<Self> {
        let mut config = EdgeAggregationConfig::default();

        if let Some(strategy) = edge_to_external {
            config.edge_to_external = strategy.inner;
        }

        if let Some(strategy) = edge_to_meta {
            config.edge_to_meta = strategy.inner;
        }

        if let Some(agg_dict) = edge_aggregation {
            for (key, value) in agg_dict.iter() {
                let attr_name: AttrName = key.extract::<String>()?;
                if let Ok(func) = value.extract::<PyEdgeAggregationFunction>() {
                    config.edge_aggregation.insert(attr_name, func.inner);
                } else if let Ok(func_str) = value.extract::<String>() {
                    let func = EdgeAggregationFunction::from_string(&func_str).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Invalid aggregation function '{}': {}",
                            func_str, e
                        ))
                    })?;
                    config.edge_aggregation.insert(attr_name, func);
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "Invalid aggregation function type for attribute '{}'",
                        attr_name
                    )));
                }
            }
        }

        if let Some(default_agg) = default_aggregation {
            config.default_aggregation = default_agg.inner;
        }

        if let Some(min_count) = min_edge_count {
            config.min_edge_count = min_count;
        }

        if let Some(include_count) = include_edge_count {
            config.include_edge_count = include_count;
        }

        if let Some(mark_type) = mark_entity_type {
            config.mark_entity_type = mark_type;
        }

        Ok(Self { inner: config })
    }

    /// Create default configuration
    #[classmethod]
    fn default(_cls: &PyType) -> Self {
        Default::default()
    }

    fn __repr__(&self) -> String {
        format!("EdgeAggregationConfig({:?})", self.inner)
    }
}

/// Parse edge aggregation configuration from Python dict
#[allow(dead_code)]
pub fn parse_edge_config(edge_config_dict: Option<&PyDict>) -> PyResult<EdgeAggregationConfig> {
    if let Some(config_dict) = edge_config_dict {
        // Try to extract as EdgeAggregationConfig object first
        if let Ok(Some(config_obj)) = config_dict.get_item("_config") {
            if let Ok(py_config) = config_obj.extract::<PyEdgeAggregationConfig>() {
                return Ok(py_config.inner);
            }
        }

        // Otherwise, parse individual parameters
        PyEdgeAggregationConfig::new(
            config_dict
                .get_item("edge_to_external")?
                .and_then(|v| v.extract().ok()),
            config_dict
                .get_item("edge_to_meta")?
                .and_then(|v| v.extract().ok()),
            config_dict
                .get_item("edge_aggregation")?
                .and_then(|v| v.extract().ok()),
            config_dict
                .get_item("default_aggregation")?
                .and_then(|v| v.extract().ok()),
            config_dict
                .get_item("min_edge_count")?
                .and_then(|v| v.extract().ok()),
            config_dict
                .get_item("include_edge_count")?
                .and_then(|v| v.extract().ok()),
            config_dict
                .get_item("mark_entity_type")?
                .and_then(|v| v.extract().ok()),
        )
        .map(|config| config.inner)
    } else {
        Ok(EdgeAggregationConfig::default())
    }
}

/// DEPRECATED: Old Python wrapper for MetaNode - use crate::ffi::entities::PyMetaNode instead
#[pyclass(name = "OldMetaNode", unsendable)]
pub struct PyMetaNodeOld {
    pub inner: MetaNode,
}

impl PyMetaNodeOld {
    /// Create from core MetaNode
    pub fn from_meta_node(meta_node: MetaNode) -> Self {
        Self { inner: meta_node }
    }
}

#[pymethods]
impl PyMetaNodeOld {
    /// Get the node ID of this meta-node
    #[getter]
    fn node_id(&self) -> NodeId {
        self.inner.node_id()
    }

    /// Check if this meta-node contains a subgraph
    #[getter]
    fn has_subgraph(&self) -> bool {
        self.inner.has_contained_subgraph()
    }

    /// Get the contained subgraph ID
    #[getter]
    fn subgraph_id(&self) -> Option<usize> {
        self.inner.contained_subgraph_id()
    }

    /// Get the contained subgraph as a PySubgraph
    #[getter]
    fn subgraph(&self, py: Python) -> PyResult<Option<PyObject>> {
        use crate::ffi::subgraphs::subgraph::PySubgraph;
        use groggy::subgraphs::Subgraph;

        if let Some(subgraph_id) = self.inner.contained_subgraph_id() {
            // Get the graph reference
            let graph_ref = self.inner.graph_ref();

            // Reconstruct the subgraph from stored data
            let (nodes, edges, subgraph_type) = {
                let graph = graph_ref.borrow();
                let result = match graph.pool().get_subgraph(subgraph_id) {
                    Ok(data) => data,
                    Err(e) => {
                        return Err(PyValueError::new_err(format!(
                            "Failed to get subgraph data: {}",
                            e
                        )))
                    }
                };
                result
            };

            // Create new Subgraph with the stored nodes and edges
            let subgraph = Subgraph::new(graph_ref.clone(), nodes, edges, subgraph_type);

            // Wrap in PySubgraph
            let py_subgraph = PySubgraph::from_core_subgraph(subgraph)?;
            Ok(Some(Py::new(py, py_subgraph)?.to_object(py)))
        } else {
            Ok(None)
        }
    }

    /// Expand this meta-node back into its contained subgraph
    fn expand(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.inner.expand_to_subgraph() {
            Ok(Some(subgraph)) => {
                // Convert to PySubgraph
                use crate::ffi::subgraphs::subgraph::PySubgraph;
                let py_subgraph = PySubgraph::from_trait_object(subgraph)?;
                Ok(Some(Py::new(py, py_subgraph)?.to_object(py)))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to expand meta-node: {}",
                e
            ))),
        }
    }

    /// Get aggregated attributes of the contained subgraph
    fn attributes(&self, py: Python) -> PyResult<PyObject> {
        let attributes = self
            .inner
            .aggregated_attributes()
            .map_err(|e| PyValueError::new_err(format!("Failed to get attributes: {}", e)))?;

        let py_dict = PyDict::new(py);
        for (attr_name, attr_value) in attributes {
            let py_value = crate::ffi::utils::attr_value_to_python_value(py, &attr_value)?;
            py_dict.set_item(attr_name.as_str(), py_value)?;
        }

        Ok(py_dict.to_object(py))
    }

    /// Re-aggregate attributes from the contained subgraph
    fn re_aggregate(
        &self,
        aggregation_functions: HashMap<String, Py<PyAggregationFunction>>,
    ) -> PyResult<()> {
        let mut agg_functions = HashMap::new();

        for (attr_name, py_agg_func) in aggregation_functions {
            Python::with_gil(|py| {
                let agg_func = py_agg_func.borrow(py);
                agg_functions.insert(AttrName::from(attr_name), agg_func.inner.clone());
            });
        }

        self.inner
            .re_aggregate(agg_functions)
            .map_err(|e| PyValueError::new_err(format!("Re-aggregation failed: {}", e)))?;

        Ok(())
    }

    /// String representation
    fn __str__(&self) -> String {
        self.inner.summary()
    }

    fn __repr__(&self) -> String {
        format!("MetaNode({})", self.inner.summary())
    }

    /// Get entity type
    #[getter]
    fn entity_type(&self) -> &'static str {
        "meta_node"
    }

    /// Check if this is a subgraph node (always True for MetaNode)
    #[getter]
    fn is_subgraph_node(&self) -> bool {
        true
    }
}

/// Convert aggregation functions dict from Python to Rust
#[allow(dead_code)]
pub fn parse_aggregation_functions(
    _py: Python,
    agg_dict: &PyDict,
) -> PyResult<HashMap<AttrName, AggregationFunction>> {
    let mut functions = HashMap::new();

    for (key, value) in agg_dict {
        let attr_name = AttrName::from(key.extract::<String>()?);

        // Support both string and AggregationFunction objects
        let agg_func = if let Ok(py_agg_func) = value.extract::<PyRef<PyAggregationFunction>>() {
            py_agg_func.inner.clone()
        } else if let Ok(func_str) = value.extract::<String>() {
            AggregationFunction::from_string(&func_str).map_err(|e| {
                PyValueError::new_err(format!(
                    "Invalid aggregation function '{}': {}",
                    func_str, e
                ))
            })?
        } else {
            return Err(PyTypeError::new_err(
                "Aggregation function must be string or AggregationFunction object",
            ));
        };

        functions.insert(attr_name, agg_func);
    }

    Ok(functions)
}

/// Helper functions for hierarchical operations on subgraphs
#[allow(dead_code)]
pub trait PyHierarchicalOperations {
    /// Collapse subgraph to meta-node with attribute aggregation
    fn add_to_graph(&self, py: Python, agg_functions: Option<&PyDict>) -> PyResult<PyObject>;

    /// Get parent meta-node if contained within one
    fn parent_meta_node(&self, py: Python) -> PyResult<Option<PyObject>>;

    /// Get child meta-nodes if this contains them
    fn child_meta_nodes(&self, py: Python) -> PyResult<Vec<PyObject>>;

    /// Get hierarchy level
    fn hierarchy_level(&self) -> PyResult<usize>;
}
