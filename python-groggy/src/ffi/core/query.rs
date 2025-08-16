//! Query FFI Bindings
//! 
//! Python bindings for query and filter operations.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError, PyImportError, PyNotImplementedError};
use groggy::{NodeId, EdgeId, AttrName};
use groggy::core::query::{AttributeFilter, NodeFilter, EdgeFilter};

// Import types from our FFI modules
use crate::ffi::types::PyAttrValue;

/// Python wrapper for AttributeFilter
#[pyclass(name = "AttributeFilter")]
#[derive(Clone)]
pub struct PyAttributeFilter {
    pub inner: AttributeFilter,
}

#[pymethods]
impl PyAttributeFilter {
    #[staticmethod]
    fn equals(value: &PyAttrValue) -> Self {
        Self { inner: AttributeFilter::Equals(value.inner.clone()) }
    }
    
    #[staticmethod]
    fn greater_than(value: &PyAttrValue) -> Self {
        Self { inner: AttributeFilter::GreaterThan(value.inner.clone()) }
    }
    
    #[staticmethod]
    fn less_than(value: &PyAttrValue) -> Self {
        Self { inner: AttributeFilter::LessThan(value.inner.clone()) }
    }
    
    #[staticmethod]
    fn not_equals(value: &PyAttrValue) -> Self {
        Self { inner: AttributeFilter::NotEquals(value.inner.clone()) }
    }
    
    #[staticmethod]
    fn greater_than_or_equal(value: &PyAttrValue) -> Self {
        Self { inner: AttributeFilter::GreaterThanOrEqual(value.inner.clone()) }
    }
    
    #[staticmethod]
    fn less_than_or_equal(value: &PyAttrValue) -> Self {
        Self { inner: AttributeFilter::LessThanOrEqual(value.inner.clone()) }
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
        Self { inner: NodeFilter::HasAttribute { name } }
    }
    
    #[staticmethod]
    fn attribute_equals(name: AttrName, value: &PyAttrValue) -> Self {
        Self { 
            inner: NodeFilter::AttributeEquals { 
                name, 
                value: value.inner.clone() 
            } 
        }
    }
    
    #[staticmethod]
    fn attribute_filter(name: AttrName, filter: &PyAttributeFilter) -> Self {
        Self { 
            inner: NodeFilter::AttributeFilter { 
                name, 
                filter: filter.inner.clone() 
            } 
        }
    }
    
    #[staticmethod]
    fn and_filters(filters: Vec<PyRef<PyNodeFilter>>) -> Self {
        let rust_filters: Vec<NodeFilter> = filters.iter()
            .map(|f| f.inner.clone())
            .collect();
        Self { inner: NodeFilter::And(rust_filters) }
    }
    
    #[staticmethod]
    fn or_filters(filters: Vec<PyRef<PyNodeFilter>>) -> Self {
        let rust_filters: Vec<NodeFilter> = filters.iter()
            .map(|f| f.inner.clone())
            .collect();
        Self { inner: NodeFilter::Or(rust_filters) }
    }
    
    #[staticmethod]
    fn not_filter(filter: &PyNodeFilter) -> Self {
        Self { inner: NodeFilter::Not(Box::new(filter.inner.clone())) }
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
        Self { inner: EdgeFilter::HasAttribute { name } }
    }
    
    #[staticmethod]
    fn attribute_equals(name: AttrName, value: &PyAttrValue) -> Self {
        Self { 
            inner: EdgeFilter::AttributeEquals { 
                name, 
                value: value.inner.clone() 
            } 
        }
    }
    
    #[staticmethod]
    fn attribute_filter(name: AttrName, filter: &PyAttributeFilter) -> Self {
        Self { 
            inner: EdgeFilter::AttributeFilter { 
                name, 
                filter: filter.inner.clone() 
            } 
        }
    }
    
    #[staticmethod]
    fn and_filters(filters: Vec<PyRef<PyEdgeFilter>>) -> Self {
        let rust_filters: Vec<EdgeFilter> = filters.iter()
            .map(|f| f.inner.clone())
            .collect();
        Self { inner: EdgeFilter::And(rust_filters) }
    }
    
    #[staticmethod]
    fn or_filters(filters: Vec<PyRef<PyEdgeFilter>>) -> Self {
        let rust_filters: Vec<EdgeFilter> = filters.iter()
            .map(|f| f.inner.clone())
            .collect();
        Self { inner: EdgeFilter::Or(rust_filters) }
    }
    
    #[staticmethod]
    fn not_filter(filter: &PyEdgeFilter) -> Self {
        Self { inner: EdgeFilter::Not(Box::new(filter.inner.clone())) }
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
