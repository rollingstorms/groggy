//! Query FFI Bindings
//!
//! Python bindings for query and filter operations.

use groggy::core::query::{AttributeFilter, EdgeFilter, NodeFilter};
use groggy::{AttrName, EdgeId, NodeId};
use pyo3::prelude::*;

// Import types from our FFI modules

/// Python wrapper for AttributeFilter
#[pyclass(name = "AttributeFilter")]
#[derive(Clone)]
pub struct PyAttributeFilter {
    pub inner: AttributeFilter,
}

#[pymethods]
impl PyAttributeFilter {
    #[staticmethod]
    fn equals(value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::Equals(attr_value),
        })
    }

    #[staticmethod]
    fn greater_than(value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::GreaterThan(attr_value),
        })
    }

    #[staticmethod]
    fn less_than(value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::LessThan(attr_value),
        })
    }

    #[staticmethod]
    fn not_equals(value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::NotEquals(attr_value),
        })
    }

    #[staticmethod]
    fn greater_than_or_equal(value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::GreaterThanOrEqual(attr_value),
        })
    }

    #[staticmethod]
    fn less_than_or_equal(value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: AttributeFilter::LessThanOrEqual(attr_value),
        })
    }
}

/// Python wrapper for NodeFilter
#[pyclass(name = "NodeFilter")]
#[derive(Clone)]
pub struct PyNodeFilter {
    pub inner: NodeFilter,
}

#[pymethods]
impl PyNodeFilter {
    #[staticmethod]
    fn has_attribute(name: AttrName) -> Self {
        Self {
            inner: NodeFilter::HasAttribute { name },
        }
    }

    #[staticmethod]
    fn attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: NodeFilter::AttributeEquals {
                name,
                value: attr_value,
            },
        })
    }

    #[staticmethod]
    fn attribute_filter(name: AttrName, filter: &PyAttributeFilter) -> Self {
        Self {
            inner: NodeFilter::AttributeFilter {
                name,
                filter: filter.inner.clone(),
            },
        }
    }

    #[staticmethod]
    fn and_filters(filters: Vec<PyRef<PyNodeFilter>>) -> Self {
        let rust_filters: Vec<NodeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: NodeFilter::And(rust_filters),
        }
    }

    #[staticmethod]
    fn or_filters(filters: Vec<PyRef<PyNodeFilter>>) -> Self {
        let rust_filters: Vec<NodeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: NodeFilter::Or(rust_filters),
        }
    }

    #[staticmethod]
    fn not_filter(filter: &PyNodeFilter) -> Self {
        Self {
            inner: NodeFilter::Not(Box::new(filter.inner.clone())),
        }
    }
}

/// Python wrapper for EdgeFilter
#[pyclass(name = "EdgeFilter")]
#[derive(Clone)]
pub struct PyEdgeFilter {
    pub inner: EdgeFilter,
}

#[pymethods]
impl PyEdgeFilter {
    #[staticmethod]
    fn has_attribute(name: AttrName) -> Self {
        Self {
            inner: EdgeFilter::HasAttribute { name },
        }
    }

    #[staticmethod]
    fn attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        Ok(Self {
            inner: EdgeFilter::AttributeEquals {
                name,
                value: attr_value,
            },
        })
    }

    #[staticmethod]
    fn attribute_filter(name: AttrName, filter: &PyAttributeFilter) -> Self {
        Self {
            inner: EdgeFilter::AttributeFilter {
                name,
                filter: filter.inner.clone(),
            },
        }
    }

    #[staticmethod]
    fn and_filters(filters: Vec<PyRef<PyEdgeFilter>>) -> Self {
        let rust_filters: Vec<EdgeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: EdgeFilter::And(rust_filters),
        }
    }

    #[staticmethod]
    fn or_filters(filters: Vec<PyRef<PyEdgeFilter>>) -> Self {
        let rust_filters: Vec<EdgeFilter> = filters.iter().map(|f| f.inner.clone()).collect();
        Self {
            inner: EdgeFilter::Or(rust_filters),
        }
    }

    #[staticmethod]
    fn not_filter(filter: &PyEdgeFilter) -> Self {
        Self {
            inner: EdgeFilter::Not(Box::new(filter.inner.clone())),
        }
    }

    #[staticmethod]
    fn connects_nodes(source: NodeId, target: NodeId) -> Self {
        Self {
            inner: EdgeFilter::ConnectsNodes { source, target },
        }
    }

    #[staticmethod]
    fn connects_any(node_ids: Vec<NodeId>) -> Self {
        Self {
            inner: EdgeFilter::ConnectsAny(node_ids),
        }
    }

    /// Filter edges where the source node has a specific attribute value
    #[staticmethod]
    fn source_attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        // This is a convenience method that will need to be implemented in the core
        // For now, we'll create a combination filter (this is a placeholder)
        Ok(Self {
            inner: EdgeFilter::AttributeEquals {
                name: format!("source_{}", name),
                value: attr_value,
            },
        })
    }

    /// Filter edges where the target node has a specific attribute value
    #[staticmethod]
    fn target_attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        // This is a convenience method that will need to be implemented in the core
        // For now, we'll create a combination filter (this is a placeholder)
        Ok(Self {
            inner: EdgeFilter::AttributeEquals {
                name: format!("target_{}", name),
                value: attr_value,
            },
        })
    }

    /// Filter edges where source OR target node has the specified attribute value
    /// This addresses the user's complaint about complex OR conditions
    #[staticmethod]
    fn source_or_target_attribute_equals(name: AttrName, value: &PyAny) -> PyResult<Self> {
        let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
        // Create an OR filter combining source and target attribute filters
        let source_filter = EdgeFilter::AttributeEquals {
            name: format!("source_{}", name),
            value: attr_value.clone(),
        };
        let target_filter = EdgeFilter::AttributeEquals {
            name: format!("target_{}", name),
            value: attr_value,
        };

        Ok(Self {
            inner: EdgeFilter::Or(vec![source_filter, target_filter]),
        })
    }

    /// Filter edges where source OR target node attribute matches any of the provided values
    /// This makes the "contains an attribute as either the target or source" use case much simpler
    #[staticmethod]
    fn source_or_target_attribute_in(name: AttrName, values: Vec<&PyAny>) -> PyResult<Self> {
        let mut filters = Vec::new();

        for value in values {
            let attr_value = crate::ffi::utils::python_value_to_attr_value(value)?;
            let source_filter = EdgeFilter::AttributeEquals {
                name: format!("source_{}", name),
                value: attr_value.clone(),
            };
            let target_filter = EdgeFilter::AttributeEquals {
                name: format!("target_{}", name),
                value: attr_value,
            };
            filters.push(EdgeFilter::Or(vec![source_filter, target_filter]));
        }

        // Combine all value filters with OR
        Ok(Self {
            inner: EdgeFilter::Or(filters),
        })
    }
}

/// Python wrapper for TraversalResult
#[pyclass(name = "TraversalResult")]
pub struct PyTraversalResult {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
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
    fn algorithm(&self) -> String {
        "NotImplemented".to_string()
    }
}

/// Python wrapper for AggregationResult
#[pyclass(name = "AggregationResult")]
pub struct PyAggregationResult {
    pub value: f64,
}

#[pymethods]
impl PyAggregationResult {
    #[getter]
    fn value(&self) -> f64 {
        self.value
    }

    fn __repr__(&self) -> String {
        format!("AggregationResult({})", self.value)
    }
}

/// Python wrapper for GroupedAggregationResult
#[pyclass(name = "GroupedAggregationResult")]
pub struct PyGroupedAggregationResult {
    pub value: PyObject,
}

#[pymethods]
impl PyGroupedAggregationResult {
    #[getter]
    fn value(&self) -> PyObject {
        self.value.clone()
    }

    fn __repr__(&self) -> String {
        "GroupedAggregationResult(...)".to_string()
    }
}
