//! Python FFI bindings for hierarchical subgraph operations
//!
//! This module provides Python bindings for collapsing subgraphs into meta-nodes,
//! attribute aggregation, and hierarchical graph navigation.

use groggy::subgraphs::{AggregationFunction, HierarchicalOperations, MetaNode};
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
        let result = self.inner.aggregate(&attr_values).map_err(|e| {
            PyValueError::new_err(format!("Aggregation failed: {}", e))
        })?;

        // Convert result back to Python
        crate::ffi::utils::attr_value_to_python_value(py, &result)
    }
}

/// Python wrapper for MetaNode
#[pyclass(name = "MetaNode", unsendable)]
pub struct PyMetaNode {
    pub inner: MetaNode,
}

impl PyMetaNode {
    /// Create from core MetaNode
    pub fn from_meta_node(meta_node: MetaNode) -> Self {
        Self { inner: meta_node }
    }
}

#[pymethods]
impl PyMetaNode {
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
        use std::collections::HashSet;
        
        if let Some(subgraph_id) = self.inner.contained_subgraph_id() {
            // Get the graph reference
            let graph_ref = self.inner.graph_ref();
            
            // Reconstruct the subgraph from stored data
            let (nodes, edges, subgraph_type) = {
                let graph = graph_ref.borrow();
                let result = match graph.pool().get_subgraph(subgraph_id) {
                    Ok(data) => data,
                    Err(e) => return Err(PyValueError::new_err(format!(
                        "Failed to get subgraph data: {}", e
                    ))),
                };
                result
            };
            
            // Create new Subgraph with the stored nodes and edges
            let subgraph = Subgraph::new(
                graph_ref.clone(),
                nodes,
                edges,
                subgraph_type,
            );
            
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
        let attributes = self.inner.aggregated_attributes().map_err(|e| {
            PyValueError::new_err(format!("Failed to get attributes: {}", e))
        })?;

        let py_dict = PyDict::new(py);
        for (attr_name, attr_value) in attributes {
            let py_value = crate::ffi::utils::attr_value_to_python_value(py, &attr_value)?;
            py_dict.set_item(attr_name.as_str(), py_value)?;
        }

        Ok(py_dict.to_object(py))
    }

    /// Re-aggregate attributes from the contained subgraph
    fn re_aggregate(&self, aggregation_functions: HashMap<String, Py<PyAggregationFunction>>) -> PyResult<()> {
        let mut agg_functions = HashMap::new();
        
        for (attr_name, py_agg_func) in aggregation_functions {
            Python::with_gil(|py| {
                let agg_func = py_agg_func.borrow(py);
                agg_functions.insert(AttrName::from(attr_name), agg_func.inner.clone());
            });
        }

        self.inner.re_aggregate(agg_functions).map_err(|e| {
            PyValueError::new_err(format!("Re-aggregation failed: {}", e))
        })?;

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
pub fn parse_aggregation_functions(py: Python, agg_dict: &PyDict) -> PyResult<HashMap<AttrName, AggregationFunction>> {
    let mut functions = HashMap::new();
    
    for (key, value) in agg_dict {
        let attr_name = AttrName::from(key.extract::<String>()?);
        
        // Support both string and AggregationFunction objects
        let agg_func = if let Ok(py_agg_func) = value.extract::<PyRef<PyAggregationFunction>>() {
            py_agg_func.inner.clone()
        } else if let Ok(func_str) = value.extract::<String>() {
            AggregationFunction::from_string(&func_str).map_err(|e| {
                PyValueError::new_err(format!("Invalid aggregation function '{}': {}", func_str, e))
            })?
        } else {
            return Err(PyTypeError::new_err(
                "Aggregation function must be string or AggregationFunction object"
            ));
        };
        
        functions.insert(attr_name, agg_func);
    }
    
    Ok(functions)
}

/// Helper functions for hierarchical operations on subgraphs
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