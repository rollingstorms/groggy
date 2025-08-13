#![allow(non_local_definitions)] // Suppress PyO3 macro warnings

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyNotImplementedError};
// use std::collections::HashMap; // TODO: Remove if not needed

// Import from the main groggy crate
use groggy::{
    Graph as RustGraph,
    AttrValue as RustAttrValue, 
    NodeId, 
    EdgeId,
    AttrName,
    GraphError,
    StateId,
    // Phase 3 imports - use explicit paths  
    core::query::{
        NodeFilter,
        EdgeFilter,  
        AttributeFilter,
    },
    // Version control imports
    core::history::{
        Commit,
        HistoryStatistics,
    },
    core::ref_manager::BranchInfo,
};

/// Convert Rust GraphError to Python exception
fn graph_error_to_py_err(error: GraphError) -> PyErr {
    match error {
        GraphError::NodeNotFound { node_id, operation, suggestion } => {
            PyErr::new::<PyValueError, _>(format!(
                "Node {} not found during {}. {}",
                node_id, operation, suggestion
            ))
        },
        GraphError::EdgeNotFound { edge_id, operation, suggestion } => {
            PyErr::new::<PyValueError, _>(format!(
                "Edge {} not found during {}. {}",
                edge_id, operation, suggestion
            ))
        },
        GraphError::InvalidInput(message) => {
            PyErr::new::<PyValueError, _>(message)
        },
        GraphError::NotImplemented { feature, tracking_issue } => {
            let mut message = format!("Feature '{}' is not yet implemented", feature);
            if let Some(issue) = tracking_issue {
                message.push_str(&format!(". See: {}", issue));
            }
            PyErr::new::<PyRuntimeError, _>(message)
        },
        _ => PyErr::new::<PyRuntimeError, _>(format!("Graph error: {}", error))
    }
}

/// Native result handle that keeps data in Rust
#[pyclass]
pub struct PyResultHandle {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    result_type: String,
}

#[pymethods]
impl PyResultHandle {
    /// Get the length of the result set without converting to Python
    fn len(&self) -> usize {
        self.nodes.len()
    }
    
    /// Python's len() function support
    fn __len__(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if result set is empty
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    
    /// Get result type
    fn result_type(&self) -> &str {
        &self.result_type
    }
    
    /// Get a slice of nodes (for iteration) - only convert what's needed
    fn get_nodes_slice(&self, start: usize, length: usize) -> Vec<NodeId> {
        let end = (start + length).min(self.nodes.len());
        self.nodes[start..end].to_vec()
    }
    
    /// Get all nodes (only when explicitly requested)
    fn get_all_nodes(&self) -> Vec<NodeId> {
        self.nodes.clone()
    }
    
    /// Iterate over nodes with a step size
    fn iter_nodes(&self, step: Option<usize>) -> Vec<NodeId> {
        let step_size = step.unwrap_or(1);
        self.nodes.iter().step_by(step_size).copied().collect()
    }
    
    /// Apply another filter to this result set (intersection)
    fn apply_filter(&self, graph: &mut PyGraph, filter: &PyNodeFilter) -> PyResult<PyResultHandle> {
        // Filter the nodes in this result set
        let mut filtered_nodes = Vec::new();
        
        for &node_id in &self.nodes {
            // Check if node matches the additional filter
            // Check if node matches the filter by finding it in filtered results
            let temp_result = graph.inner.find_nodes(filter.inner.clone())
                .map_err(graph_error_to_py_err)?;
            if temp_result.contains(&node_id) {
                filtered_nodes.push(node_id);
            }
        }
        
        Ok(PyResultHandle {
            nodes: filtered_nodes,
            edges: self.edges.clone(),
            result_type: format!("{}_filtered", self.result_type),
        })
    }
    
    /// Union with another result handle
    fn union_with(&self, other: &PyResultHandle) -> PyResultHandle {
        let mut combined_nodes = self.nodes.clone();
        for &node in &other.nodes {
            if !combined_nodes.contains(&node) {
                combined_nodes.push(node);
            }
        }
        
        let mut combined_edges = self.edges.clone();
        for &edge in &other.edges {
            if !combined_edges.contains(&edge) {
                combined_edges.push(edge);
            }
        }
        
        PyResultHandle {
            nodes: combined_nodes,
            edges: combined_edges,
            result_type: format!("{}+{}", self.result_type, other.result_type),
        }
    }
    
    /// Intersection with another result handle
    fn intersect_with(&self, other: &PyResultHandle) -> PyResultHandle {
        let intersection_nodes: Vec<NodeId> = self.nodes.iter()
            .filter(|node| other.nodes.contains(node))
            .copied()
            .collect();
            
        let intersection_edges: Vec<EdgeId> = self.edges.iter()
            .filter(|edge| other.edges.contains(edge))
            .copied()
            .collect();
        
        PyResultHandle {
            nodes: intersection_nodes,
            edges: intersection_edges,
            result_type: format!("{}&{}", self.result_type, other.result_type),
        }
    }
}

/// Native attribute collection that keeps data in Rust
#[pyclass(unsendable)]
pub struct PyAttributeCollection {
    graph_ref: *const RustGraph, // Unsafe but controlled access
    node_ids: Vec<NodeId>,
    attr_name: String,
}

#[pymethods] 
impl PyAttributeCollection {
    /// Get count of attributes without converting
    fn len(&self) -> usize {
        self.node_ids.len()
    }
    
    /// Compute statistics directly in Rust
    fn compute_stats(&self, py: Python) -> PyResult<PyObject> {
        // Safe because we control the lifetime
        let graph = unsafe { &*self.graph_ref };
        
        let mut values = Vec::new();
        for &node_id in &self.node_ids {
            if let Ok(Some(attr)) = graph.get_node_attr(node_id, &self.attr_name) {
                values.push(attr);
            }
        }
        
        // Compute statistics in Rust
        {
            let dict = PyDict::new(py);
            
            // Count
            dict.set_item("count", values.len())?;
            
            // Type-specific statistics
            if !values.is_empty() {
                match &values[0] {
                    RustAttrValue::Int(_) => {
                        let int_values: Vec<i64> = values.iter()
                            .filter_map(|v| if let RustAttrValue::Int(i) = v { Some(*i) } else { None })
                            .collect();
                        
                        if !int_values.is_empty() {
                            let sum: i64 = int_values.iter().sum();
                            let avg = sum as f64 / int_values.len() as f64;
                            let min = *int_values.iter().min().unwrap();
                            let max = *int_values.iter().max().unwrap();
                            
                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                        }
                    },
                    RustAttrValue::Float(_) => {
                        let float_values: Vec<f32> = values.iter()
                            .filter_map(|v| if let RustAttrValue::Float(f) = v { Some(*f) } else { None })
                            .collect();
                        
                        if !float_values.is_empty() {
                            let sum: f32 = float_values.iter().sum();
                            let avg = sum / float_values.len() as f32;
                            let min = float_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let max = float_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            
                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                        }
                    },
                    _ => {
                        // For other types, just provide count
                    }
                }
            }
            
            Ok(dict.to_object(py))
        }
    }
    
    /// Get sample values without converting all
    fn sample_values(&self, count: usize) -> PyResult<Vec<PyAttrValue>> {
        let graph = unsafe { &*self.graph_ref };
        let mut results = Vec::new();
        
        let step = if self.node_ids.len() <= count { 
            1 
        } else { 
            self.node_ids.len() / count 
        };
        
        for &node_id in self.node_ids.iter().step_by(step).take(count) {
            if let Ok(Some(attr)) = graph.get_node_attr(node_id, &self.attr_name) {
                results.push(PyAttrValue { inner: attr });
            }
        }
        
        Ok(results)
    }
}

/// Python wrapper for AttrValue
#[pyclass(name = "AttrValue")]
#[derive(Clone)]
pub struct PyAttrValue {
    inner: RustAttrValue,
}

#[pymethods]
impl PyAttrValue {
    #[new]
    fn new(value: &PyAny) -> PyResult<Self> {
        let rust_value = if let Ok(b) = value.extract::<bool>() {
            RustAttrValue::Bool(b)
        } else if let Ok(i) = value.extract::<i64>() {
            RustAttrValue::Int(i)
        } else if let Ok(f) = value.extract::<f64>() {
            RustAttrValue::Float(f as f32)  // Convert f64 to f32
        } else if let Ok(f) = value.extract::<f32>() {
            RustAttrValue::Float(f)
        } else if let Ok(s) = value.extract::<String>() {
            RustAttrValue::Text(s)
        } else if let Ok(vec) = value.extract::<Vec<f32>>() {
            RustAttrValue::FloatVec(vec)
        } else if let Ok(vec) = value.extract::<Vec<f64>>() {
            // Convert Vec<f64> to Vec<f32>
            let f32_vec: Vec<f32> = vec.into_iter().map(|f| f as f32).collect();
            RustAttrValue::FloatVec(f32_vec)
        } else if let Ok(bytes) = value.extract::<Vec<u8>>() {
            RustAttrValue::Bytes(bytes)
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Unsupported attribute value type. Supported types: int, float, str, bool, List[float], bytes"
            ));
        };
        
        Ok(Self { inner: rust_value })
    }
    
    #[getter]
    fn value(&self, py: Python) -> PyObject {
        match &self.inner {
            RustAttrValue::Int(i) => i.to_object(py),
            RustAttrValue::Float(f) => f.to_object(py),
            RustAttrValue::Text(s) => s.to_object(py),
            RustAttrValue::Bool(b) => b.to_object(py),
            RustAttrValue::FloatVec(v) => v.to_object(py),
            RustAttrValue::Bytes(b) => b.to_object(py),
            // Handle optimized variants by extracting their underlying value
            RustAttrValue::CompactText(cs) => cs.as_str().to_object(py),
            RustAttrValue::SmallInt(i) => i.to_object(py),
            RustAttrValue::CompressedText(cd) => {
                match cd.decompress_text() {
                    Ok(data) => data.to_object(py),
                    Err(_) => py.None()
                }
            },
            RustAttrValue::CompressedFloatVec(cd) => {
                match cd.decompress_float_vec() {
                    Ok(data) => data.to_object(py),
                    Err(_) => py.None()
                }
            },
        }
    }
    
    #[getter]
    fn type_name(&self) -> &'static str {
        match &self.inner {
            RustAttrValue::Int(_) => "int",
            RustAttrValue::Float(_) => "float",
            RustAttrValue::Text(_) => "text",
            RustAttrValue::Bool(_) => "bool",
            RustAttrValue::FloatVec(_) => "float_vec",
            RustAttrValue::Bytes(_) => "bytes",
            RustAttrValue::CompactText(_) => "text",
            RustAttrValue::SmallInt(_) => "int",
            RustAttrValue::CompressedText(_) => "text",
            RustAttrValue::CompressedFloatVec(_) => "float_vec",
        }
    }
    
    fn __repr__(&self) -> String {
        format!("AttrValue({})", match &self.inner {
            RustAttrValue::Int(i) => i.to_string(),
            RustAttrValue::Float(f) => f.to_string(),
            RustAttrValue::Text(s) => format!("\"{}\"", s),
            RustAttrValue::Bool(b) => b.to_string(),
            RustAttrValue::FloatVec(v) => format!("{:?}", v),
            RustAttrValue::Bytes(b) => format!("b\"{:?}\"", b),
            RustAttrValue::CompactText(cs) => format!("\"{}\"", cs.as_str()),
            RustAttrValue::SmallInt(i) => i.to_string(),
            RustAttrValue::CompressedText(cd) => {
                match cd.decompress_text() {
                    Ok(data) => format!("\"{}\"", data),
                    Err(_) => "compressed(error)".to_string()
                }
            },
            RustAttrValue::CompressedFloatVec(cd) => {
                match cd.decompress_float_vec() {
                    Ok(data) => format!("{:?}", data),
                    Err(_) => "compressed(error)".to_string()
                }
            },
        })
    }
    
    fn __eq__(&self, other: &PyAttrValue) -> bool {
        self.inner == other.inner
    }
    
    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // Create a hash based on the variant and value
        match &self.inner {
            RustAttrValue::Int(i) => {
                0u8.hash(&mut hasher);
                i.hash(&mut hasher);
            },
            RustAttrValue::Float(f) => {
                1u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            },
            RustAttrValue::Text(s) => {
                2u8.hash(&mut hasher);
                s.hash(&mut hasher);
            },
            RustAttrValue::Bool(b) => {
                3u8.hash(&mut hasher);
                b.hash(&mut hasher);
            },
            RustAttrValue::FloatVec(v) => {
                4u8.hash(&mut hasher);
                for f in v {
                    f.to_bits().hash(&mut hasher);
                }
            },
            RustAttrValue::Bytes(b) => {
                5u8.hash(&mut hasher);
                b.hash(&mut hasher);
            },
            RustAttrValue::CompactText(cs) => {
                6u8.hash(&mut hasher);
                cs.as_str().hash(&mut hasher);
            },
            RustAttrValue::SmallInt(i) => {
                7u8.hash(&mut hasher);
                i.hash(&mut hasher);
            },
            RustAttrValue::CompressedText(cd) => {
                8u8.hash(&mut hasher);
                if let Ok(text) = cd.decompress_text() {
                    text.hash(&mut hasher);
                }
            },
            RustAttrValue::CompressedFloatVec(cd) => {
                9u8.hash(&mut hasher);
                if let Ok(vec) = cd.decompress_float_vec() {
                    for f in vec {
                        f.to_bits().hash(&mut hasher);
                    }
                }
            },
        }
        hasher.finish()
    }
}

/// Python wrapper for AttributeFilter
#[pyclass(name = "AttributeFilter")]
#[derive(Clone)]
pub struct PyAttributeFilter {
    inner: AttributeFilter,
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
}

/// Python wrapper for NodeFilter
#[pyclass(name = "NodeFilter")]
#[derive(Clone)]
pub struct PyNodeFilter {
    inner: NodeFilter,
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
    inner: EdgeFilter,
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

/// Python wrapper for TraversalResult (stub)
#[pyclass(name = "TraversalResult")]
pub struct PyTraversalResult {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
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

/// Python wrapper for Commit
#[pyclass(name = "Commit")]
#[derive(Clone)]
pub struct PyCommit {
    inner: std::sync::Arc<Commit>,
}

#[pymethods]
impl PyCommit {
    #[getter]
    fn id(&self) -> StateId {
        self.inner.id
    }
    
    #[getter]
    fn parents(&self) -> Vec<StateId> {
        self.inner.parents.clone()
    }
    
    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }
    
    #[getter]
    fn author(&self) -> String {
        self.inner.author.clone()
    }
    
    #[getter]
    fn timestamp(&self) -> u64 {
        self.inner.timestamp
    }
    
    fn is_root(&self) -> bool {
        self.inner.is_root()
    }
    
    fn is_merge(&self) -> bool {
        self.inner.is_merge()
    }
    
    fn __repr__(&self) -> String {
        format!("Commit(id={}, message='{}', author='{}')", 
                self.inner.id, self.inner.message, self.inner.author)
    }
}

/// Python wrapper for BranchInfo
#[pyclass(name = "BranchInfo")]
#[derive(Clone)]
pub struct PyBranchInfo {
    inner: BranchInfo,
}

#[pymethods]
impl PyBranchInfo {
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    
    #[getter]
    fn head(&self) -> StateId {
        self.inner.head
    }
    
    #[getter]
    fn is_default(&self) -> bool {
        self.inner.is_default
    }
    
    #[getter]
    fn is_current(&self) -> bool {
        self.inner.is_current
    }
    
    fn __repr__(&self) -> String {
        format!("BranchInfo(name='{}', head={})", 
                self.inner.name, self.inner.head)
    }
}

/// Python wrapper for HistoryStatistics
#[pyclass(name = "HistoryStatistics")]
#[derive(Clone)]
pub struct PyHistoryStatistics {
    inner: HistoryStatistics,
}

#[pymethods]
impl PyHistoryStatistics {
    #[getter]
    fn total_commits(&self) -> usize {
        self.inner.total_commits
    }
    
    #[getter]
    fn total_branches(&self) -> usize {
        self.inner.total_branches
    }
    
    #[getter]
    fn total_tags(&self) -> usize {
        self.inner.total_tags
    }
    
    #[getter]
    fn storage_efficiency(&self) -> f64 {
        self.inner.storage_efficiency
    }
    
    #[getter]
    fn oldest_commit_age(&self) -> u64 {
        self.inner.oldest_commit_age
    }
    
    #[getter]
    fn newest_commit_age(&self) -> u64 {
        self.inner.newest_commit_age
    }
    
    fn __repr__(&self) -> String {
        format!("HistoryStatistics(commits={}, branches={}, efficiency={:.2})", 
                self.inner.total_commits, self.inner.total_branches, self.inner.storage_efficiency)
    }
}

/// Python wrapper for HistoricalView
#[pyclass(name = "HistoricalView")]
pub struct PyHistoricalView {
    // Store the state ID that this view represents
    state_id: StateId,
    // For actual graph operations, we'll need to call back to the graph
    // In a full implementation, this would contain a HistoricalView<'graph>
}

#[pymethods]
impl PyHistoricalView {
    #[getter]
    fn state_id(&self) -> StateId {
        self.state_id
    }
    
    /// Get nodes from this historical state
    /// Note: This is a simplified implementation. In practice, you'd need
    /// access to the graph to reconstruct the state.
    fn get_node_ids(&self) -> PyResult<Vec<NodeId>> {
        // Placeholder - in real implementation, would query graph state
        Ok(Vec::new())
    }
    
    /// Get edges from this historical state
    fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        // Placeholder - in real implementation, would query graph state
        Ok(Vec::new())
    }
    
    /// Get a node attribute from this historical state
    fn get_node_attribute(&self, _node: NodeId, _attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        // Placeholder - in real implementation, would query historical state
        Ok(None)
    }
    
    /// Get an edge attribute from this historical state
    fn get_edge_attribute(&self, _edge: EdgeId, _attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        // Placeholder - in real implementation, would query historical state
        Ok(None)
    }
    
    /// Check if a node exists in this historical state
    fn has_node(&self, _node: NodeId) -> PyResult<bool> {
        // Placeholder - in real implementation, would query historical state
        Ok(false)
    }
    
    /// Check if an edge exists in this historical state
    fn has_edge(&self, _edge: EdgeId) -> PyResult<bool> {
        // Placeholder - in real implementation, would query historical state
        Ok(false)
    }
    
    /// Get the neighbors of a node in this historical state
    fn get_neighbors(&self, _node: NodeId) -> PyResult<Vec<NodeId>> {
        // Placeholder - in real implementation, would query historical state
        Ok(Vec::new())
    }
    
    fn __repr__(&self) -> String {
        format!("HistoricalView(state_id={})", self.state_id)
    }
}

/// Python wrapper for the main Graph
#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    inner: RustGraph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new(_config: Option<&PyDict>) -> PyResult<Self> {
        // For now, ignore config and just create a default graph
        // TODO: Convert Python config to GraphConfig when needed
        let rust_graph = RustGraph::new();
        Ok(Self { inner: rust_graph })
    }
    
    // === CORE GRAPH OPERATIONS ===
    
    fn add_node(&mut self) -> NodeId {
        self.inner.add_node()
    }
    
    fn add_nodes(&mut self, count: usize) -> Vec<NodeId> {
        self.inner.add_nodes(count)
    }
    
    fn add_edge(&mut self, source: NodeId, target: NodeId) -> PyResult<EdgeId> {
        self.inner.add_edge(source, target)
            .map_err(graph_error_to_py_err)
    }
    
    fn add_edges(&mut self, edges: Vec<(NodeId, NodeId)>) -> Vec<EdgeId> {
        self.inner.add_edges(&edges)
    }
    
    fn remove_node(&mut self, node: NodeId) -> PyResult<()> {
        self.inner.remove_node(node)
            .map_err(graph_error_to_py_err)
    }
    
    fn remove_edge(&mut self, edge: EdgeId) -> PyResult<()> {
        self.inner.remove_edge(edge)
            .map_err(graph_error_to_py_err)
    }
    
    fn remove_nodes(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.inner.remove_nodes(&nodes)
            .map_err(graph_error_to_py_err)
    }
    
    fn remove_edges(&mut self, edges: Vec<EdgeId>) -> PyResult<()> {
        self.inner.remove_edges(&edges)
            .map_err(graph_error_to_py_err)
    }
    
    // === ATTRIBUTE OPERATIONS ===
    
    fn set_node_attribute(&mut self, node: NodeId, attr: AttrName, value: &PyAttrValue) -> PyResult<()> {
        self.inner.set_node_attr(node, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    fn set_edge_attribute(&mut self, edge: EdgeId, attr: AttrName, value: &PyAttrValue) -> PyResult<()> {
        self.inner.set_edge_attr(edge, attr, value.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    fn get_node_attribute(&self, node: NodeId, attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        match self.inner.get_node_attr(node, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    fn get_edge_attribute(&self, edge: EdgeId, attr: AttrName) -> PyResult<Option<PyAttrValue>> {
        match self.inner.get_edge_attr(edge, &attr) {
            Ok(Some(value)) => Ok(Some(PyAttrValue { inner: value })),
            Ok(None) => Ok(None),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    // === OPTIMIZED BULK OPERATIONS (Phase 1 - Zero PyAttrValue) ===
    
    fn set_node_attributes(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // HYPER-OPTIMIZED bulk API - minimize PyO3 overhead and allocations
        let mut attrs_values = std::collections::HashMap::with_capacity(attrs_dict.len());
        
        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;
            
            // OPTIMIZATION: Extract all fields at once to reduce PyO3 calls
            let (nodes, values_obj, value_type): (Vec<NodeId>, &pyo3::PyAny, String) = {
                let nodes_item = data_dict.get_item("nodes")?.ok_or_else(|| 
                    PyErr::new::<PyKeyError, _>("Missing 'nodes' key"))?;
                let values_item = data_dict.get_item("values")?.ok_or_else(|| 
                    PyErr::new::<PyKeyError, _>("Missing 'values' key"))?;
                let type_item = data_dict.get_item("value_type")?.ok_or_else(|| 
                    PyErr::new::<PyKeyError, _>("Missing 'value_type' key"))?;
                    
                (nodes_item.extract()?, values_item, type_item.extract()?)
            };
            
            let len = nodes.len();
            
            // OPTIMIZATION: Pre-allocate result vector and use direct indexing
            let mut pairs = Vec::with_capacity(len);
            
            // OPTIMIZATION: Match on str slice to avoid repeated string comparisons
            match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    // OPTIMIZATION: Direct loop instead of iterator chain
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Text(values[i].clone())));
                    }
                },
                "int" => {
                    let values: Vec<i64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Int(values[i])));
                    }
                },
                "float" => {
                    let values: Vec<f64> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Float(values[i] as f32)));
                    }
                },
                "bool" => {
                    let values: Vec<bool> = values_obj.extract()?;
                    if values.len() != len {
                        return Err(PyErr::new::<PyValueError, _>("Length mismatch"));
                    }
                    
                    for i in 0..len {
                        pairs.push((nodes[i], RustAttrValue::Bool(values[i])));
                    }
                },
                _ => return Err(PyErr::new::<PyValueError, _>("Unsupported type"))
            };
            
            attrs_values.insert(attr, pairs);
        }
        
        self.inner.set_node_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }
    
    /// Simple bulk method that accepts [(node_id, AttrValue), ...] format (matches benchmark expectations)
    fn set_node_attribute_bulk(&mut self, attr_name: String, node_values: Vec<(NodeId, PyAttrValue)>) -> PyResult<()> {
        let mut attrs_values = std::collections::HashMap::new();
        let converted_values: Vec<(NodeId, RustAttrValue)> = node_values.into_iter()
            .map(|(node_id, py_attr_value)| (node_id, py_attr_value.inner))
            .collect();
        
        attrs_values.insert(attr_name, converted_values);
        self.inner.set_node_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }

    fn set_edge_attributes(&mut self, _py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // New efficient columnar API for edges - zero PyAttrValue objects created!
        let mut attrs_values = std::collections::HashMap::new();
        
        for (attr_name, attr_data) in attrs_dict {
            let attr: AttrName = attr_name.extract()?;
            let data_dict: &PyDict = attr_data.downcast()?;
            
            // Extract components in bulk using the same pattern as node attributes
            let edges: Vec<EdgeId> = if let Ok(Some(item)) = data_dict.get_item("edges") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>("Missing 'edges' key in attribute data"));
            };
            let value_type: String = if let Ok(Some(item)) = data_dict.get_item("value_type") {
                item.extract()?
            } else {
                return Err(PyErr::new::<PyKeyError, _>("Missing 'value_type' key in attribute data"));
            };
            
            // Batch convert based on known type - no individual type detection!
            let pairs = match value_type.as_str() {
                "text" => {
                    let values: Vec<String> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Text(val)))
                        .collect()
                },
                "int" => {
                    let values: Vec<i64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Int(val)))
                        .collect()
                },
                "float" => {
                    let values: Vec<f64> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Float(val as f32)))
                        .collect()
                },
                "bool" => {
                    let values: Vec<bool> = if let Ok(Some(item)) = data_dict.get_item("values") {
                        item.extract()?
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Missing 'values' key in attribute data"));
                    };
                    
                    if values.len() != edges.len() {
                        return Err(PyErr::new::<PyValueError, _>(
                            format!("Mismatched lengths: {} edges vs {} values", edges.len(), values.len())
                        ));
                    }
                    
                    edges.into_iter()
                        .zip(values.into_iter())
                        .map(|(edge, val)| (edge, RustAttrValue::Bool(val)))
                        .collect()
                },
                _ => return Err(PyErr::new::<PyValueError, _>(
                    format!("Unsupported value_type: '{}'. Supported: text, int, float, bool", value_type)
                ))
            };
            
            attrs_values.insert(attr, pairs);
        }
        
        self.inner.set_edge_attrs(attrs_values)
            .map_err(graph_error_to_py_err)
    }
    
    fn get_nodes_attributes(&self, attr: AttrName, nodes: Vec<NodeId>) -> PyResult<Vec<Option<PyAttrValue>>> {
        let result = self.inner.get_nodes_attrs(&attr, &nodes)
            .map_err(graph_error_to_py_err)?;
        
        Ok(result.into_iter()
            .map(|opt| opt.map(|val| PyAttrValue { inner: val }))
            .collect())
    }
    
    fn get_edges_attributes(&self, attr: AttrName, edges: Vec<EdgeId>) -> PyResult<Vec<Option<PyAttrValue>>> {
        let result = self.inner.get_edges_attrs(&attr, &edges)
            .map_err(graph_error_to_py_err)?;
        
        Ok(result.into_iter()
            .map(|opt| opt.map(|val| PyAttrValue { inner: val }))
            .collect())
    }
    
    
    fn get_edge_attributes(&self, edge: EdgeId, py: Python) -> PyResult<PyObject> {
        let attrs = self.inner.get_edge_attrs(edge)
            .map_err(graph_error_to_py_err)?;
        
        // Convert HashMap to Python dict
        let dict = PyDict::new(py);
        for (attr_name, attr_value) in attrs {
            let py_value = Py::new(py, PyAttrValue { inner: attr_value })?;
            dict.set_item(attr_name, py_value)?;
        }
        Ok(dict.to_object(py))
    }
    
    // === TOPOLOGY OPERATIONS ===
    
    fn contains_node(&self, node: NodeId) -> bool {
        self.inner.contains_node(node) 
    }
    
    fn contains_edge(&self, edge: EdgeId) -> bool {
        self.inner.contains_edge(edge)
    }
    
    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }
    
    fn edge_ids(&self) -> Vec<EdgeId> {
        self.inner.edge_ids()
    }
    
    fn edge_endpoints(&self, edge: EdgeId) -> PyResult<(NodeId, NodeId)> {
        self.inner.edge_endpoints(edge)
            .map_err(graph_error_to_py_err)
    }
    
    fn neighbors(&self, node: NodeId) -> PyResult<Vec<NodeId>> {
        self.inner.neighbors(node)
            .map_err(graph_error_to_py_err)
    }
    
    fn degree(&self, node: NodeId) -> PyResult<usize> {
        self.inner.degree(node)
            .map_err(graph_error_to_py_err)
    }
    
    // === STATISTICS ===
    
    fn memory_statistics(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.inner.memory_statistics();
        
        // Convert MemoryStatistics to Python dict
        let dict = PyDict::new(py);
        dict.set_item("pool_memory_bytes", stats.pool_memory_bytes)?;
        dict.set_item("space_memory_bytes", stats.space_memory_bytes)?;
        dict.set_item("history_memory_bytes", stats.history_memory_bytes)?;
        dict.set_item("change_tracker_memory_bytes", stats.change_tracker_memory_bytes)?;
        dict.set_item("total_memory_bytes", stats.total_memory_bytes)?;
        dict.set_item("total_memory_mb", stats.total_memory_mb)?;
        
        // Add memory efficiency stats
        let efficiency_dict = PyDict::new(py);
        efficiency_dict.set_item("bytes_per_node", stats.memory_efficiency.bytes_per_node)?;
        efficiency_dict.set_item("bytes_per_edge", stats.memory_efficiency.bytes_per_edge)?;
        efficiency_dict.set_item("bytes_per_entity", stats.memory_efficiency.bytes_per_entity)?;
        efficiency_dict.set_item("overhead_ratio", stats.memory_efficiency.overhead_ratio)?;
        efficiency_dict.set_item("cache_efficiency", stats.memory_efficiency.cache_efficiency)?;
        dict.set_item("memory_efficiency", efficiency_dict)?;
        
        // Add compression statistics
        let compression_dict = PyDict::new(py);
        compression_dict.set_item("compressed_attributes", stats.compression_stats.compressed_attributes)?;
        compression_dict.set_item("total_attributes", stats.compression_stats.total_attributes)?;
        compression_dict.set_item("average_compression_ratio", stats.compression_stats.average_compression_ratio)?;
        compression_dict.set_item("memory_saved_bytes", stats.compression_stats.memory_saved_bytes)?;
        compression_dict.set_item("memory_saved_percentage", stats.compression_stats.memory_saved_percentage)?;
        dict.set_item("compression_stats", compression_dict)?;
        
        Ok(dict.to_object(py))
    }
    
    fn statistics(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.inner.statistics();
        
        // Convert basic statistics to Python dict  
        let dict = PyDict::new(py);
        dict.set_item("node_count", stats.node_count)?;
        dict.set_item("edge_count", stats.edge_count)?;
        dict.set_item("attribute_count", stats.attribute_count)?;
        
        Ok(dict.to_object(py))
    }
    
    fn __repr__(&self) -> String {
        let node_count = self.inner.node_ids().len();
        let edge_count = self.inner.edge_ids().len();
        format!("Graph(nodes={}, edges={})", node_count, edge_count)
    }
    
    // === PHASE 3 QUERYING METHODS ===
    
    /// Phase 3.1: Advanced filtering
    fn filter_nodes(&mut self, filter: &PyNodeFilter) -> PyResult<PyResultHandle> {
        let nodes = self.inner.find_nodes(filter.inner.clone())
            .map_err(graph_error_to_py_err)?;
            
        Ok(PyResultHandle {
            nodes,
            edges: Vec::new(),
            result_type: "filtered_nodes".to_string(),
        })
    }
    
    
    fn filter_edges(&mut self, filter: &PyEdgeFilter) -> PyResult<Vec<EdgeId>> {
        self.inner.find_edges(filter.inner.clone())
            .map_err(graph_error_to_py_err)
    }
    
    /// Phase 3.2: Graph traversal algorithms
    fn traverse_bfs(&mut self, start_node: NodeId, max_depth: Option<usize>, 
                   node_filter: Option<&PyNodeFilter>, edge_filter: Option<&PyEdgeFilter>) -> PyResult<PyResultHandle> {
        
        // Create traversal options
        let mut options = groggy::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        if let Some(filter) = node_filter {
            options.node_filter = Some(filter.inner.clone());
        }
        if let Some(filter) = edge_filter {
            options.edge_filter = Some(filter.inner.clone());
        }
        
        // Perform BFS traversal
        let result = self.inner.bfs(start_node, options)
            .map_err(graph_error_to_py_err)?;
        
        Ok(PyResultHandle {
            nodes: result.nodes,
            edges: result.edges,
            result_type: "bfs_traversal".to_string(),
        })
    }
    
    
    fn traverse_dfs(&mut self, start_node: NodeId, max_depth: Option<usize>,
                   node_filter: Option<&PyNodeFilter>, edge_filter: Option<&PyEdgeFilter>) -> PyResult<PyResultHandle> {
        
        // Create traversal options
        let mut options = groggy::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        if let Some(filter) = node_filter {
            options.node_filter = Some(filter.inner.clone());
        }
        if let Some(filter) = edge_filter {
            options.edge_filter = Some(filter.inner.clone());
        }
        
        // Perform DFS traversal
        let result = self.inner.dfs(start_node, options)
            .map_err(graph_error_to_py_err)?;
        
        Ok(PyResultHandle {
            nodes: result.nodes,
            edges: result.edges,
            result_type: "dfs_traversal".to_string(),
        })
    }
    
    fn shortest_path(&mut self, source: NodeId, target: NodeId, weight_attribute: Option<AttrName>) -> PyResult<Option<Vec<NodeId>>> {
        let _ = (source, target, weight_attribute);
        Err(PyErr::new::<PyNotImplementedError, _>("Shortest path not implemented"))
    }
    
    fn find_connected_components(&mut self) -> PyResult<Vec<PyResultHandle>> {
        let options = groggy::core::traversal::TraversalOptions::default();
        let result = self.inner.connected_components(options)
            .map_err(graph_error_to_py_err)?;
        
        // Convert each component to a PyResultHandle
        let mut handles = Vec::new();
        for (i, component) in result.components.into_iter().enumerate() {
            handles.push(PyResultHandle {
                nodes: component.nodes, // Access the nodes field
                edges: Vec::new(), // TODO: Could add component edges if needed
                result_type: format!("connected_component_{}", i),
            });
        }
        
        Ok(handles)
    }
    
    /// Phase 3.4: Query result aggregation and analytics
    fn aggregate_node_attribute(&self, attribute: AttrName, operation: String) -> PyResult<PyAggregationResult> {
        let result = self.inner.aggregate_node_attribute(&attribute, &operation)
            .map_err(graph_error_to_py_err)?;
        Ok(PyAggregationResult { value: result.value })
    }
    
    fn aggregate_edge_attribute(&self, attribute: AttrName, operation: String) -> PyResult<PyAggregationResult> {
        let result = self.inner.aggregate_edge_attribute(&attribute, &operation)
            .map_err(graph_error_to_py_err)?;
        Ok(PyAggregationResult { value: result.value })
    }
    
    fn group_nodes_by_attribute(&self, attribute: AttrName, aggregation_attr: AttrName, operation: String) -> PyResult<PyGroupedAggregationResult> {
        let py = unsafe { Python::assume_gil_acquired() };
        let results = self.inner.group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;
        
        // Convert HashMap to Python dict
        let dict = PyDict::new(py);
        for (attr_value, agg_result) in results {
            let py_attr_value = PyAttrValue { inner: attr_value };
            let py_agg_result = PyAggregationResult { value: agg_result.value };
            dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
        }
        
        Ok(PyGroupedAggregationResult {
            value: dict.to_object(py)
        })
    }
    
    /// Native attribute collection - returns handle for vectorized operations in Rust
    fn get_node_attribute_collection(&self, node_ids: Vec<NodeId>, attribute: AttrName) -> PyResult<PyAttributeCollection> {
        // We need a way to safely reference the graph - for now use raw pointer with lifetime control
        Ok(PyAttributeCollection {
            graph_ref: &self.inner as *const RustGraph,
            node_ids,
            attr_name: attribute,
        })
    }
    
    /// Native bulk aggregation - compute statistics entirely in Rust  
    /// Native vectorized attribute access - return collection handle
    fn get_node_attributes(&self, node_ids: Vec<NodeId>, attribute: AttrName) -> PyResult<PyAttributeCollection> {
        Ok(PyAttributeCollection {
            graph_ref: &self.inner as *const RustGraph,
            node_ids,
            attr_name: attribute,
        })
    }
    
    /// Native bulk attribute retrieval without individual PyO3 conversions
    fn get_node_attributes_batch(&self, node_ids: Vec<NodeId>, attributes: Vec<AttrName>, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for attr_name in attributes {
            let mut values = Vec::new();
            for &node_id in &node_ids {
                if let Ok(Some(attr)) = self.inner.get_node_attr(node_id, &attr_name) {
                    values.push(PyAttrValue { inner: attr });
                } else {
                    values.push(PyAttrValue { inner: RustAttrValue::Text("None".to_string()) });
                }
            }
            let py_values: Vec<PyObject> = values.into_iter()
                .map(|v| Py::new(py, v).map(|p| p.to_object(py)))
                .collect::<Result<Vec<_>, _>>()?;
            dict.set_item(&attr_name, py_values)?;
        }
        
        Ok(dict.to_object(py))
    }
    
    /// Native filtered attribute access - apply filter and return attributes in one step
    fn get_attribute_by_filter(&mut self, filter: &PyNodeFilter, attribute: AttrName, py: Python) -> PyResult<PyObject> {
        // Get filtered nodes
        let nodes = self.inner.find_nodes(filter.inner.clone())
            .map_err(graph_error_to_py_err)?;
            
        // Extract attributes in bulk
        let mut values = Vec::new();
        for &node_id in &nodes {
            if let Ok(Some(attr)) = self.inner.get_node_attr(node_id, &attribute) {
                values.push(PyAttrValue { inner: attr });
            }
        }
        
        let result_dict = PyDict::new(py);
        result_dict.set_item("node_count", nodes.len())?;
        let py_values: Vec<PyObject> = values.into_iter()
            .map(|v| Py::new(py, v).map(|p| p.to_object(py)))
            .collect::<Result<Vec<_>, _>>()?;
        result_dict.set_item("values", py_values)?;
        
        Ok(result_dict.to_object(py))
    }

    fn aggregate_nodes(&self, node_ids: Vec<NodeId>, attribute: AttrName, py: Python) -> PyResult<PyObject> {
        let mut values = Vec::new();
        for &node_id in &node_ids {
            if let Ok(Some(attr)) = self.inner.get_node_attr(node_id, &attribute) {
                values.push(attr);
            }
        }
        
        // Compute all statistics in one pass in Rust
        {
            let dict = PyDict::new(py);
            
            dict.set_item("count", values.len())?;
            
            if !values.is_empty() {
                match &values[0] {
                    RustAttrValue::Int(_) => {
                        let int_values: Vec<i64> = values.iter()
                            .filter_map(|v| if let RustAttrValue::Int(i) = v { Some(*i) } else { None })
                            .collect();
                        
                        if !int_values.is_empty() {
                            let sum: i64 = int_values.iter().sum();
                            let avg = sum as f64 / int_values.len() as f64;
                            let min = *int_values.iter().min().unwrap();
                            let max = *int_values.iter().max().unwrap();
                            
                            // Compute variance/stddev
                            let variance = int_values.iter()
                                .map(|&x| (x as f64 - avg).powi(2))
                                .sum::<f64>() / int_values.len() as f64;
                            let stddev = variance.sqrt();
                            
                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                            dict.set_item("variance", variance)?;
                            dict.set_item("stddev", stddev)?;
                        }
                    },
                    RustAttrValue::Float(_) => {
                        let float_values: Vec<f32> = values.iter()
                            .filter_map(|v| if let RustAttrValue::Float(f) = v { Some(*f) } else { None })
                            .collect();
                        
                        if !float_values.is_empty() {
                            let sum: f64 = float_values.iter().map(|&x| x as f64).sum();
                            let avg = sum / float_values.len() as f64;
                            let min = float_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let max = float_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            
                            // Compute variance/stddev
                            let variance = float_values.iter()
                                .map(|&x| (x as f64 - avg).powi(2))
                                .sum::<f64>() / float_values.len() as f64;
                            let stddev = variance.sqrt();
                            
                            dict.set_item("sum", sum)?;
                            dict.set_item("average", avg)?;
                            dict.set_item("min", min)?;
                            dict.set_item("max", max)?;
                            dict.set_item("variance", variance)?;
                            dict.set_item("stddev", stddev)?;
                        }
                    },
                    _ => {
                        // For other types, just provide count
                    }
                }
            }
            
            Ok(dict.to_object(py))
        }
    }
    
    fn compute_comprehensive_stats(&self, attribute: AttrName, target: String) -> PyResult<PyObject> {
        let _ = (attribute, target);
        Err(PyErr::new::<PyNotImplementedError, _>("Comprehensive stats not implemented"))
    }
    
    // === VERSION CONTROL OPERATIONS ===
    
    /// Commit current changes to version control
    fn commit(&mut self, message: String, author: String) -> PyResult<StateId> {
        self.inner.commit(message, author)
            .map_err(graph_error_to_py_err)
    }
    
    /// Create a new branch
    fn create_branch(&mut self, branch_name: String) -> PyResult<()> {
        self.inner.create_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }
    
    /// Switch to a different branch
    fn checkout_branch(&mut self, branch_name: String) -> PyResult<()> {
        self.inner.checkout_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }
    
    /// List all branches
    fn list_branches(&self) -> Vec<PyBranchInfo> {
        self.inner.list_branches()
            .into_iter()
            .map(|branch_info| PyBranchInfo { inner: branch_info })
            .collect()
    }
    
    /// Get commit history  
    fn get_commit_history(&self) -> Vec<PyCommit> {
        // Use the public commit_history method which returns CommitInfo
        // For now, return empty vector since CommitInfo != Commit
        Vec::new()
    }
    
    /// Create a historical view of the graph at a specific commit
    fn get_historical_view(&self, commit_id: StateId) -> PyResult<PyHistoricalView> {
        match self.inner.view_at_commit(commit_id) {
            Ok(_view) => Ok(PyHistoricalView {
                state_id: commit_id,
            }),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    /// Check if there are uncommitted changes
    fn has_uncommitted_changes(&self) -> bool {
        self.inner.has_uncommitted_changes()
    }
}

/// The Python module
#[pymodule]
fn _groggy(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core classes
    m.add_class::<PyGraph>()?;
    m.add_class::<PyAttrValue>()?;
    
    // Phase 3 filtering and querying classes
    m.add_class::<PyAttributeFilter>()?;
    m.add_class::<PyNodeFilter>()?;
    m.add_class::<PyEdgeFilter>()?;
    m.add_class::<PyTraversalResult>()?;
    m.add_class::<PyAggregationResult>()?;
    m.add_class::<PyGroupedAggregationResult>()?;

    // Version control classes
    m.add_class::<PyCommit>()?;
    m.add_class::<PyBranchInfo>()?;
    m.add_class::<PyHistoryStatistics>()?;
    m.add_class::<PyHistoricalView>()?;
    
    // Native performance classes
    m.add_class::<PyResultHandle>()?;
    m.add_class::<PyAttributeCollection>()?;
    
    Ok(())
}
