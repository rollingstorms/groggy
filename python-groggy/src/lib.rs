#![allow(non_local_definitions)] // Suppress PyO3 macro warnings

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError};
// use std::collections::HashMap; // TODO: Remove if not needed

// Import from the main groggy crate
use groggy::{
    Graph as RustGraph,
    AttrValue as RustAttrValue, 
    NodeId, 
    EdgeId,
    AttrName,
    StateId,
    // Phase 3 imports - use explicit paths  
    core::{
        array::{GraphArray, StatsSummary},
        subgraph::Subgraph as RustSubgraph,
        query::{
            NodeFilter,
            EdgeFilter,
            AttributeFilter,
        },
    },
    // Version control imports
    core::history::{
        Commit,
        HistoryStatistics,
    },
    core::ref_manager::BranchInfo,
};

mod utils;
use utils::{python_value_to_attr_value, attr_value_to_python_value, graph_error_to_py_err};




/// Native result handle that keeps data in Rust
#[pyclass]
pub struct PyResultHandle {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    result_type: String,
}

/// Subgraph - represents a filtered view of the main graph with same interface
#[pyclass(name = "Subgraph", unsendable)]
pub struct PySubgraph {
    // Use the core Rust Subgraph with proper Rc<RefCell<Graph>> architecture
    inner: Option<RustSubgraph>,
    // Fallback data for legacy compatibility (when inner is None)
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
    subgraph_type: String,
    graph: Option<Py<PyGraph>>,
}

impl PySubgraph {
    /// Create a PySubgraph from a core RustSubgraph (with proper graph reference)
    pub fn from_core_subgraph(subgraph: RustSubgraph) -> Self {
        let nodes = subgraph.node_ids();
        let edges = subgraph.edge_ids();
        let subgraph_type = subgraph.subgraph_type().to_string();
        
        PySubgraph {
            inner: Some(subgraph),
            nodes,
            edges,
            subgraph_type,
            graph: None, // Not needed when we have inner
        }
    }
    
    /// Standard PySubgraph constructor
    pub fn new(nodes: Vec<NodeId>, edges: Vec<EdgeId>, subgraph_type: String, graph: Option<Py<PyGraph>>) -> Self {
        PySubgraph {
            inner: None,
            nodes,
            edges,
            subgraph_type,
            graph,
        }
    }
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

#[pymethods]
impl PySubgraph {
    /// Get nodes as a property that supports indexing and attribute access
    #[getter]
    fn nodes(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        if let Some(graph_ref) = &self_.graph {
            Py::new(py, PyNodesAccessor {
                graph: graph_ref.clone(),
                constrained_nodes: Some(self_.nodes.clone()),
            })
        } else {
            Err(PyRuntimeError::new_err("No graph reference available"))
        }
    }
    
    /// Get edges as a property that supports indexing and attribute access
    #[getter] 
    fn edges(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        if let Some(graph_ref) = &self_.graph {
            Py::new(py, PyEdgesAccessor {
                graph: graph_ref.clone(),
                constrained_edges: Some(self_.edges.clone()),
            })
        } else {
            Err(PyRuntimeError::new_err("No graph reference available"))
        }
    }
    
    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.nodes.len()
    }
    
    /// Node count property
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Edge count property
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
    
    /// Get node IDs in this subgraph (subgraph.node_ids property)
    #[getter]
    fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.clone()
    }
    
    /// Get edge IDs in this subgraph (subgraph.edge_ids property) 
    #[getter]
    fn edge_ids(&self) -> Vec<EdgeId> {
        self.edges.clone()
    }
    
    /// Check if a node exists in this subgraph
    fn has_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains(&node_id)
    }
    
    /// Check if an edge exists in this subgraph
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains(&edge_id)
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Subgraph(nodes={}, edges={}, type={})", 
                self.nodes.len(), self.edges.len(), self.subgraph_type)
    }
    
    
    /// Filter edges within this subgraph (chainable)
    fn filter_edges(&self, _py: Python, _filter: &PyAny) -> PyResult<PySubgraph> {
        // Placeholder implementation
        Ok(PySubgraph::new(
            self.nodes.clone(),
            self.edges.clone(),
            format!("{}_edge_filtered", self.subgraph_type),
            self.graph.clone(),
        ))
    }
    
    /// Connected components within this subgraph
    fn connected_components(&self) -> PyResult<Vec<PySubgraph>> {
        // Use inner subgraph if available
        if let Some(ref inner_subgraph) = self.inner {
            let components = inner_subgraph.connected_components()
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Failed to get connected components: {}", e)))?;
            
            let mut result = Vec::new();
            for component in components {
                result.push(PySubgraph::from_core_subgraph(component));
            }
            Ok(result)
        } else {
            // Fallback - return single component for now
            Ok(vec![PySubgraph::new(
                self.nodes.clone(),
                self.edges.clone(),
                "component".to_string(),
                self.graph.clone(),
            )])
        }
    }
    
    /// Set attributes on all nodes in this subgraph (batch operation)
    #[pyo3(signature = (**kwargs))]
    fn set(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<Py<PySubgraph>> {
        // Use inner Subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            if let Some(kwargs) = kwargs {
                for (key, value) in kwargs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    
                    // Use the core Subgraph's bulk set method
                    inner_subgraph.set_node_attribute_bulk(&attr_name, attr_value)
                        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Failed to set attribute: {}", e)))?;
                }
            }
            
            // Return self for chaining  
            let new_subgraph = if let Some(ref inner) = self.inner {
                PySubgraph::from_core_subgraph(inner.clone())
            } else {
                PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(), 
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                )
            };
            Ok(Py::new(py, new_subgraph)?)
        }
        // Fallback to legacy implementation
        else if let Some(graph_ref) = &self.graph {
            if let Some(kwargs) = kwargs {
                let mut graph = graph_ref.borrow_mut(py);
                
                // Update all nodes in this subgraph
                for &node_id in &self.nodes {
                    for (key, value) in kwargs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value)?;
                        let py_attr_value = PyAttrValue { inner: attr_value };
                        
                        graph.set_node_attribute(node_id, attr_name.clone(), &py_attr_value)?;
                    }
                }
            }
            
            // Return self for chaining
            Ok(Py::new(py, PySubgraph::new(
                self.nodes.clone(),
                self.edges.clone(),
                self.subgraph_type.clone(),
                self.graph.clone(),
            ))?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot set attributes on subgraph without graph reference. Use graph.filter_nodes() or similar methods."
            ))
        }
    }
    
    /// Update attributes on all nodes in this subgraph using dict syntax
    fn update(&mut self, py: Python, data: &PyDict) -> PyResult<Py<PySubgraph>> {
        if let Some(graph_ref) = &self.graph {
            let mut graph = graph_ref.borrow_mut(py);
            
            // Update all nodes in this subgraph
            for &node_id in &self.nodes {
                for (key, value) in data.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    let py_attr_value = PyAttrValue { inner: attr_value };
                    
                    graph.set_node_attribute(node_id, attr_name.clone(), &py_attr_value)?;
                }
            }
            
            // Return self for chaining
            Ok(Py::new(py, PySubgraph::new(
                self.nodes.clone(),
                self.edges.clone(),
                self.subgraph_type.clone(),
                self.graph.clone(),
            ))?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot update attributes on subgraph without graph reference. Use graph.filter_nodes() or similar methods."
            ))
        }
    }
    
    /// Column access: get all values for a node attribute within this subgraph
    /// This enables: subgraph['component_id'] -> GraphArray with statistical methods
    fn get_node_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Py<PyGraphArray>> {
        // Use inner Subgraph if available (preferred path)
        if let Some(ref inner_subgraph) = self.inner {
            let attr_values = inner_subgraph.get_node_attribute_column(&attr_name.to_string())
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Failed to get attribute column: {}", e)))?;
            
            // Create GraphArray from the attribute values
            let graph_array = groggy::GraphArray::from_vec(attr_values);
            
            // Wrap in Python GraphArray
            let py_graph_array = PyGraphArray { inner: graph_array };
            Ok(Py::new(py, py_graph_array)?)
        }
        // Fallback to legacy implementation
        else if let Some(graph_ref) = &self.graph {
            let graph = graph_ref.borrow(py);
            let mut attr_values = Vec::new();
            
            for &node_id in &self.nodes {
                if let Ok(Some(attr_value)) = graph.inner.get_node_attr(node_id, &attr_name.to_string()) {
                    attr_values.push(attr_value);
                } else {
                    // Handle missing attributes with default value
                    attr_values.push(groggy::AttrValue::Int(0));
                }
            }
            
            // Create GraphArray from the attribute values
            let graph_array = groggy::GraphArray::from_vec(attr_values);
            
            // Wrap in Python GraphArray
            let py_graph_array = PyGraphArray { inner: graph_array };
            Ok(Py::new(py, py_graph_array)?)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot access attributes on subgraph without graph reference."
            ))
        }
    }
    
    /// Column access: get all values for an edge attribute within this subgraph
    fn get_edge_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Vec<PyObject>> {
        if let Some(graph_ref) = &self.graph {
            let graph = graph_ref.borrow(py);
            let mut values = Vec::new();
            
            for &edge_id in &self.edges {
                if let Ok(Some(attr_value)) = graph.inner.get_edge_attr(edge_id, &attr_name.to_string()) {
                    // Convert AttrValue to Python object
                    let py_value = attr_value_to_python_value(py, &attr_value)?;
                    values.push(py_value);
                } else {
                    // Handle missing attributes - use None
                    values.push(py.None());
                }
            }
            
            Ok(values)
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot access edge attributes on subgraph without graph reference."
            ))
        }
    }
    
    /// Python dict-like access with multi-column support
    /// - subgraph['attr_name'] -> single column (Vec<PyObject>)  
    /// - subgraph[['age', 'height']] -> multi-column 2D GraphArray of shape (2, n)
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try single string first (existing behavior)
        if let Ok(attr_name) = key.extract::<String>() {
            let column = self.get_node_attribute_column(py, &attr_name)?;
            return Ok(column.to_object(py));
        }
        
        // Try list of strings (multi-column access)
        if let Ok(attr_names) = key.extract::<Vec<String>>() {
            if attr_names.is_empty() {
                return Err(PyValueError::new_err("Empty attribute list"));
            }
            
            // Collect all columns
            let mut columns = Vec::new();
            for attr_name in &attr_names {
                let column = self.get_node_attribute_column(py, attr_name)?;
                columns.push(column);
            }
            
            // For multi-column, return a 2D structure
            if attr_names.len() == 1 {
                // Single column in list form: [['age']] -> same as 'age'
                return Ok(columns[0].to_object(py));
            } else {
                // Multiple columns: [['age', 'height']] -> 2D array-like structure
                // Create a list of columns (transpose-like structure)
                let result = PyList::new(py, columns);
                return Ok(result.to_object(py));
            }
        }
        
        Err(PyTypeError::new_err("Key must be a string or list of strings"))
    }
    
    /// Create GraphTable for DataFrame-like view of this subgraph nodes
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Temporarily return a simple placeholder until PyO3 trait issue is resolved
        let graph_table_module = py.import("groggy.graph_table")?;
        let graph_table_class = graph_table_module.getattr("GraphTable")?;
        
        // Simple approach: create empty GraphTable and set attributes manually  
        let empty_list = py.eval("[]", None, None)?;
        let table = graph_table_class.call1((empty_list, "nodes"))?;
        Ok(table.to_object(py))
    }
    
    /// Create GraphTable for DataFrame-like view of this subgraph edges
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        // Temporarily return a simple placeholder until PyO3 trait issue is resolved
        let graph_table_module = py.import("groggy.graph_table")?;
        let graph_table_class = graph_table_module.getattr("GraphTable")?;
        
        // Simple approach: create empty GraphTable and set attributes manually  
        let empty_list = py.eval("[]", None, None)?;
        let table = graph_table_class.call1((empty_list, "edges"))?;
        Ok(table.to_object(py))
    }
    
    /// Enhanced filter_nodes method using the existing graph's filter capabilities
    /// This enables chaining: subgraph.filter_nodes('age > 30').filter_nodes('dept == "Engineering"')
    fn filter_nodes(&self, py: Python, filter_obj: &PyAny) -> PyResult<PySubgraph> {
        if let Some(graph_ref) = &self.graph {
            let graph = graph_ref.borrow(py);
            
            // Convert the filter and apply it to this subgraph's nodes
            // For now, we'll create a basic attribute filter implementation
            if let Ok(filter_str) = filter_obj.extract::<String>() {
                // Parse the filter string and apply to our node subset
                // This is a simplified implementation - in practice, we'd use the full parser
                let filtered_nodes = self.nodes.iter()
                    .filter(|&&node_id| {
                        // For demonstration, let's support a simple "dept == 'Engineering'" pattern
                        if filter_str.contains("==") {
                            let parts: Vec<&str> = filter_str.split("==").map(|s| s.trim()).collect();
                            if parts.len() == 2 {
                                let attr_name = parts[0].trim_matches('"').trim_matches('\'');
                                let attr_value = parts[1].trim_matches('"').trim_matches('\'');
                                
                                if let Ok(Some(node_attr)) = graph.inner.get_node_attr(node_id, &attr_name.to_string()) {
                                    // Simple string comparison
                                    if let RustAttrValue::Text(text_val) = &node_attr {
                                        return text_val == attr_value;
                                    }
                                }
                            }
                        }
                        false
                    })
                    .copied()
                    .collect::<Vec<_>>();
                
                // Calculate induced edges for the filtered nodes
                let filtered_edges = self.edges.iter()
                    .filter(|&&edge_id| {
                        if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                            filtered_nodes.contains(&source) && filtered_nodes.contains(&target)
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect();
                
                Ok(PySubgraph::new(
                    filtered_nodes,
                    filtered_edges,
                    format!("{}_filtered", self.subgraph_type),
                    self.graph.clone(),
                ))
            } else {
                Err(PyErr::new::<PyValueError, _>(
                    "Filter must be a string. Example: subgraph.filter_nodes('dept == \"Engineering\"')"
                ))
            }
        } else {
            Err(PyErr::new::<PyRuntimeError, _>(
                "Cannot filter subgraph without graph reference."
            ))
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

/// Python wrapper for accessing node attributes as columns
#[pyclass(name = "NodeAttributes", unsendable)]
pub struct PyNodeAttributes {
    graph: *const RustGraph, // Pointer to avoid ownership issues
}

/// Python wrapper for accessing edge attributes as columns
#[pyclass(name = "EdgeAttributes", unsendable)]
pub struct PyEdgeAttributes {
    graph: *const RustGraph, // Pointer to avoid ownership issues
}

/// Unified attributes accessor that provides both node and edge attributes
#[pyclass(name = "Attributes", unsendable)]
pub struct PyAttributes {
    graph: *const RustGraph, // Pointer to avoid ownership issues
}

#[pymethods]
impl PyAttributes {
    /// Access node attributes - g.attributes.nodes["salary"] 
    #[getter]
    fn nodes(&self) -> PyNodeAttributes {
        PyNodeAttributes {
            graph: self.graph,
        }
    }
    
    /// Access edge attributes - g.attributes.edges["weight"]
    #[getter] 
    fn edges(&self) -> PyEdgeAttributes {
        PyEdgeAttributes {
            graph: self.graph,
        }
    }
    
    /// Legacy support: g.attributes["attr_name"] defaults to node attributes
    fn __getitem__(&self, attr_name: &str) -> PyResult<Vec<PyObject>> {
        let node_attrs = PyNodeAttributes { graph: self.graph };
        node_attrs.__getitem__(attr_name)
    }
    
    /// List available attribute names (both nodes and edges)
    fn keys(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            let node_attrs = PyNodeAttributes { graph: self.graph };
            let edge_attrs = PyEdgeAttributes { graph: self.graph };
            
            dict.set_item("nodes", node_attrs.keys()?)?;
            dict.set_item("edges", edge_attrs.keys()?)?;
            
            Ok(dict.to_object(py))
        })
    }
    
    fn __repr__(&self) -> String {
        "Attributes(nodes=NodeAttributes(...), edges=EdgeAttributes(...))".to_string()
    }
}

#[pymethods]
impl PyNodeAttributes {
    /// Access attribute column by name - g.attributes["salary"]
    fn __getitem__(&self, attr_name: &str) -> PyResult<Vec<PyObject>> {
        unsafe {
            let graph = &*self.graph;
            let node_ids = graph.node_ids();
            
            Python::with_gil(|py| {
                let mut values = Vec::new();
                
                for node_id in node_ids {
                    match graph.get_node_attr(node_id, &attr_name.to_string()) {
                        Ok(Some(attr_value)) => {
                            let py_value = match attr_value {
                                RustAttrValue::Int(i) => i.to_object(py),
                                RustAttrValue::Float(f) => f.to_object(py),
                                RustAttrValue::Text(s) => s.to_object(py),
                                RustAttrValue::Bool(b) => b.to_object(py),
                                RustAttrValue::FloatVec(v) => v.to_object(py),
                                RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                                RustAttrValue::SmallInt(i) => i.to_object(py),
                                RustAttrValue::Bytes(b) => b.to_object(py),
                                RustAttrValue::CompressedText(_) => "CompressedText".to_object(py), // TODO: decompress
                                RustAttrValue::CompressedFloatVec(_) => "CompressedFloatVec".to_object(py), // TODO: decompress
                            };
                            values.push(py_value);
                        },
                        Ok(None) => {
                            // Node doesn't have this attribute, use None
                            values.push(py.None());
                        },
                        Err(_) => {
                            // Error getting attribute, use None
                            values.push(py.None());
                        }
                    }
                }
                
                Ok(values)
            })
        }
    }
    
    /// List available attribute names
    fn keys(&self) -> PyResult<Vec<String>> {
        unsafe {
            let graph = &*self.graph;
            let node_ids = graph.node_ids();
            let mut attr_names = std::collections::HashSet::new();
            
            // Collect all unique attribute names across all nodes
            for node_id in node_ids {
                if let Ok(attrs) = graph.get_node_attrs(node_id) {
                    for (name, _) in attrs {
                        attr_names.insert(name);
                    }
                }
            }
            
            Ok(attr_names.into_iter().collect())
        }
    }
    
    /// Check if attribute exists
    fn __contains__(&self, attr_name: &str) -> PyResult<bool> {
        Ok(self.keys()?.contains(&attr_name.to_string()))
    }
    
    fn __repr__(&self) -> String {
        format!("NodeAttributes(keys={:?})", self.keys().unwrap_or_default())
    }
}

#[pymethods]
impl PyEdgeAttributes {
    /// Access edge attribute column by name - g.attributes.edges["weight"]
    fn __getitem__(&self, attr_name: &str) -> PyResult<Vec<PyObject>> {
        unsafe {
            let graph = &*self.graph;
            let edge_ids = graph.edge_ids();
            
            Python::with_gil(|py| {
                let mut values = Vec::new();
                
                for edge_id in edge_ids {
                    match graph.get_edge_attr(edge_id, &attr_name.to_string()) {
                        Ok(Some(attr_value)) => {
                            let py_value = match attr_value {
                                RustAttrValue::Int(i) => i.to_object(py),
                                RustAttrValue::Float(f) => f.to_object(py),
                                RustAttrValue::Text(s) => s.to_object(py),
                                RustAttrValue::Bool(b) => b.to_object(py),
                                RustAttrValue::FloatVec(v) => v.to_object(py),
                                RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                                RustAttrValue::SmallInt(i) => i.to_object(py),
                                RustAttrValue::Bytes(b) => b.to_object(py),
                                RustAttrValue::CompressedText(_) => "CompressedText".to_object(py),
                                RustAttrValue::CompressedFloatVec(_) => "CompressedFloatVec".to_object(py),
                            };
                            values.push(py_value);
                        },
                        Ok(None) => {
                            values.push(py.None());
                        },
                        Err(_) => {
                            values.push(py.None());
                        }
                    }
                }
                
                Ok(values)
            })
        }
    }
    
    /// List available edge attribute names
    fn keys(&self) -> PyResult<Vec<String>> {
        unsafe {
            let graph = &*self.graph;
            let edge_ids = graph.edge_ids();
            let mut attr_names = std::collections::HashSet::new();
            
            // Collect all unique attribute names across all edges
            for edge_id in edge_ids {
                if let Ok(attrs) = graph.get_edge_attrs(edge_id) {
                    for (name, _) in attrs {
                        attr_names.insert(name);
                    }
                }
            }
            
            Ok(attr_names.into_iter().collect())
        }
    }
    
    /// Check if edge attribute exists
    fn __contains__(&self, attr_name: &str) -> PyResult<bool> {
        Ok(self.keys()?.contains(&attr_name.to_string()))
    }
    
    fn __repr__(&self) -> String {
        format!("EdgeAttributes(keys={:?})", self.keys().unwrap_or_default())
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
    
    #[pyo3(signature = (**kwargs))]
    fn add_node(&mut self, kwargs: Option<&PyDict>) -> PyResult<NodeId> {
        let node_id = self.inner.add_node();
        
        // Fast path: if no kwargs, just return the node_id
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                // Only do attribute setting if we actually have attributes
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    let attr_value = python_value_to_attr_value(value)?;
                    
                    self.inner.set_node_attr(node_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }
        
        Ok(node_id)
    }
    
    #[pyo3(signature = (data, uid_key = None))]
    fn add_nodes(&mut self, data: &PyAny, uid_key: Option<String>) -> PyResult<PyObject> {
        // Fast path optimization: Check for integer first (most common case)
        if let Ok(count) = data.extract::<usize>() {
            // Old API: add_nodes(5) -> [0, 1, 2, 3, 4] - fastest path
            let node_ids = self.inner.add_nodes(count);
            return Python::with_gil(|py| Ok(node_ids.to_object(py)));
        }
        
        // Only use Python::with_gil for complex operations
        Python::with_gil(|py| {
            if let Ok(node_data_list) = data.extract::<Vec<&PyDict>>() {
                // New API: add_nodes([{"id": "alice", "age": 30}, ...], id_key="id")
                let mut id_mapping = std::collections::HashMap::new();
                
                // Create all nodes first
                let node_ids = self.inner.add_nodes(node_data_list.len());
                
                // Process each node's data
                for (i, node_dict) in node_data_list.iter().enumerate() {
                    let node_id = node_ids[i];
                    
                    // Extract the ID if id_key is provided
                    if let Some(ref key) = uid_key {
                        match node_dict.get_item(key) {
                            Ok(Some(id_value)) => {
                                if let Ok(user_id) = id_value.extract::<String>() {
                                    id_mapping.insert(user_id, node_id);
                                }
                            }
                            Ok(None) => {
                                return Err(PyErr::new::<PyKeyError, _>(format!("Missing key: {}", key)));
                            }
                            Err(e) => return Err(e),
                        }
                    }
                    
                    // Set all attributes from the dict
                    for (attr_key, attr_value) in node_dict.iter() {
                        let attr_name: String = attr_key.extract()?;
                        
                        // Store all attributes including the id_key for later uid_key lookups
                        // (Previous version skipped id_key, but we need it for uid_key resolution)
                        
                        let attr_val = python_value_to_attr_value(attr_value)?;
                        self.inner.set_node_attr(node_id, attr_name, attr_val)
                            .map_err(graph_error_to_py_err)?;
                    }
                }
                
                // Return the mapping if id_key was provided, otherwise return node IDs
                if uid_key.is_some() {
                    Ok(id_mapping.to_object(py))
                } else {
                    Ok(node_ids.to_object(py))
                }
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                    "add_nodes expects either an integer count or a list of dictionaries"
                ))
            }
        })
    }
    
    /// Helper method to resolve string ID to NodeId using uid_key attribute
    fn resolve_string_id_to_node(&self, string_id: &str, uid_key: &str) -> PyResult<NodeId> {
        let node_ids = self.inner.node_ids();
        
        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.get_node_attr(node_id, &uid_key.to_string()) {
                match attr_value {
                    RustAttrValue::Text(s) => {
                        if s == string_id {
                            return Ok(node_id);
                        }
                    },
                    RustAttrValue::CompactText(s) => {
                        if s.as_str() == string_id {
                            return Ok(node_id);
                        }
                    },
                    _ => continue, // Skip non-text attributes
                }
            }
        }
        
        Err(PyErr::new::<PyKeyError, _>(format!("No node found with {}='{}'", uid_key, string_id)))
    }
    
    #[pyo3(signature = (source, target, uid_key = None, **kwargs))]
    fn add_edge(&mut self, _py: Python, source: &PyAny, target: &PyAny, uid_key: Option<String>, kwargs: Option<&PyDict>) -> PyResult<EdgeId> {
        // Try to extract as NodeId first (most common case)
        let source_id = if let Ok(node_id) = source.extract::<NodeId>() {
            node_id
        } else if let Some(ref key) = uid_key {
            // String ID with uid_key resolution
            let source_str: String = source.extract()?;
            self.resolve_string_id_to_node(&source_str, key)?
        } else {
            return Err(PyErr::new::<PyTypeError, _>("Source must be NodeId or string with uid_key"));
        };
        
        let target_id = if let Ok(node_id) = target.extract::<NodeId>() {
            node_id
        } else if let Some(ref key) = uid_key {
            // String ID with uid_key resolution  
            let target_str: String = target.extract()?;
            self.resolve_string_id_to_node(&target_str, key)?
        } else {
            return Err(PyErr::new::<PyTypeError, _>("Target must be NodeId or string with uid_key"));
        };
        
        let edge_id = self.inner.add_edge(source_id, target_id)
            .map_err(graph_error_to_py_err)?;
        
        // Fast path: if no kwargs, just return the edge_id
        if let Some(attrs) = kwargs {
            if !attrs.is_empty() {
                // Only do attribute setting if we actually have attributes
                for (key, value) in attrs.iter() {
                    let attr_name: String = key.extract()?;
                    
                    // Skip uid_key if it's in kwargs
                    if let Some(ref uid_k) = uid_key {
                        if attr_name == *uid_k {
                            continue;
                        }
                    }
                    
                    let attr_value = python_value_to_attr_value(value)?;
                    
                    self.inner.set_edge_attr(edge_id, attr_name, attr_value)
                        .map_err(graph_error_to_py_err)?;
                }
            }
        }
        
        Ok(edge_id)
    }
    
    #[pyo3(signature = (edges, node_mapping = None, uid_key = None))]
    fn add_edges(&mut self, edges: &PyAny, node_mapping: Option<std::collections::HashMap<String, NodeId>>, uid_key: Option<String>) -> PyResult<Vec<EdgeId>> {
        // If uid_key is provided but no node_mapping, generate it automatically  
        let resolved_node_mapping = if uid_key.is_some() && node_mapping.is_none() {
            if let Some(ref key) = uid_key {
                Python::with_gil(|py| -> PyResult<std::collections::HashMap<String, NodeId>> {
                    let mapping_obj = self.get_node_mapping(key.clone())?;
                    let mapping_dict: &PyDict = mapping_obj.downcast(py)?;
                    
                    let mut string_mapping = std::collections::HashMap::new();
                    for (k, v) in mapping_dict.iter() {
                        let key_str: String = k.extract()?;
                        let node_id: NodeId = v.extract()?;
                        string_mapping.insert(key_str, node_id);
                    }
                    Ok(string_mapping)
                })?
            } else {
                std::collections::HashMap::new()
            }
        } else {
            node_mapping.unwrap_or_default()
        };
        let final_mapping = if resolved_node_mapping.is_empty() { None } else { Some(resolved_node_mapping) };
        
        // Fast path optimization: Check for simple tuple list first (most common case)
        if let Ok(edge_pairs) = edges.extract::<Vec<(NodeId, NodeId)>>() {
            // Format 1: List of (source, target) tuples - fastest path
            return Ok(self.inner.add_edges(&edge_pairs));
        }
        // Format 2: List of (source, target, attrs_dict) tuples  
        else if let Ok(edge_tuples) = edges.extract::<Vec<(&PyAny, &PyAny, Option<&PyDict>)>>() {
            let mut edge_ids = Vec::new();
            
            for (src_any, tgt_any, attrs_opt) in edge_tuples {
                let source: NodeId = src_any.extract()?;
                let target: NodeId = tgt_any.extract()?;
                
                let edge_id = self.inner.add_edge(source, target)
                    .map_err(graph_error_to_py_err)?;
                
                // Set attributes if provided
                if let Some(attrs) = attrs_opt {
                    for (key, value) in attrs.iter() {
                        let attr_name: String = key.extract()?;
                        let attr_value = python_value_to_attr_value(value)?;
                        
                        self.inner.set_edge_attr(edge_id, attr_name, attr_value)
                            .map_err(graph_error_to_py_err)?;
                    }
                }
                
                edge_ids.push(edge_id);
            }
            
            Ok(edge_ids)
        }
        // Format 3: List of dictionaries with node_mapping 
        else if let Ok(edge_dicts) = edges.extract::<Vec<&PyDict>>() {
            if let Some(mapping) = final_mapping {
                // Use the existing add_edges_from_dicts logic
                let mut edge_ids = Vec::new();
                
                for edge_dict in edge_dicts {
                    // Extract source and target IDs (support both "source"/"target" and flexible keys)
                    let source_val = edge_dict.get_item("source")
                        .or_else(|_| edge_dict.get_item("src"))
                        .or_else(|_| edge_dict.get_item("from"))?;
                    let target_val = edge_dict.get_item("target")
                        .or_else(|_| edge_dict.get_item("tgt"))
                        .or_else(|_| edge_dict.get_item("to"))?;
                    
                    if let (Some(src), Some(tgt)) = (source_val, target_val) {
                        let source_str: String = src.extract()?;
                        let target_str: String = tgt.extract()?;
                        
                        // Resolve to internal node IDs
                        let source_node = mapping.get(&source_str)
                            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("Unknown source ID: {}", source_str)))?;
                        let target_node = mapping.get(&target_str)
                            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("Unknown target ID: {}", target_str)))?;
                        
                        // Create the edge
                        let edge_id = self.inner.add_edge(*source_node, *target_node)
                            .map_err(graph_error_to_py_err)?;
                        
                        // Set attributes from dict (excluding source/target keys)
                        for (key, value) in edge_dict.iter() {
                            let attr_name: String = key.extract()?;
                            
                            // Skip source/target keys
                            if attr_name == "source" || attr_name == "target" || 
                               attr_name == "src" || attr_name == "tgt" ||
                               attr_name == "from" || attr_name == "to" {
                                continue;
                            }
                            
                            let attr_value = python_value_to_attr_value(value)?;
                            self.inner.set_edge_attr(edge_id, attr_name, attr_value)
                                .map_err(graph_error_to_py_err)?;
                        }
                        
                        edge_ids.push(edge_id);
                    } else {
                        return Err(PyErr::new::<PyKeyError, _>("Edge dict must have 'source' and 'target' keys"));
                    }
                }
                
                Ok(edge_ids)
            } else {
                Err(PyErr::new::<PyValueError, _>(
                    "Dictionary edges require node_mapping parameter"
                ))
            }
        }
        else {
            Err(PyErr::new::<PyTypeError, _>(
                "add_edges expects a list of (source, target) tuples, (source, target, attrs) tuples, or dictionaries with node_mapping"
            ))
        }
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
    
    // === BULK COLUMN ACCESS (GraphTable Optimization) ===
    
    /// Get complete attribute column for ALL nodes (optimized for table() method)
    /// 
    /// INTERNAL: This is the key optimization for GraphTable - instead of O(n*m) individual calls,
    /// we make O(m) calls to get complete columns.
    fn _get_node_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Py<PyGraphArray>> {
        match self.inner._get_node_attribute_column(&attr_name.to_string()) {
            Ok(values) => {
                // Convert Option<AttrValue> vector to AttrValue vector (convert None to appropriate AttrValue)
                let attr_values: Vec<groggy::AttrValue> = values.into_iter()
                    .map(|opt_val| opt_val.unwrap_or(groggy::AttrValue::Int(0))) // Use default for None values
                    .collect();
                
                // Create GraphArray from the attribute values
                let graph_array = groggy::GraphArray::from_vec(attr_values);
                
                // Wrap in Python GraphArray
                let py_graph_array = PyGraphArray { inner: graph_array };
                Ok(Py::new(py, py_graph_array)?)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    /// Get complete attribute column for ALL edges (optimized for edge table() method)
    fn _get_edge_attribute_column(&self, py: Python, attr_name: &str) -> PyResult<Vec<PyObject>> {
        match self.inner._get_edge_attribute_column(&attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => py_values.push(attr_value_to_python_value(py, &attr_value)?),
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    /// Get attribute column for specific nodes (optimized for subgraph tables)
    fn _get_node_attributes_for_nodes(&self, py: Python, node_ids: Vec<NodeId>, attr_name: &str) -> PyResult<Vec<PyObject>> {
        match self.inner._get_node_attributes_for_nodes(&node_ids, &attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => py_values.push(attr_value_to_python_value(py, &attr_value)?),
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }
    
    /// Get attribute column for specific edges (optimized for subgraph edge tables)
    fn _get_edge_attributes_for_edges(&self, py: Python, edge_ids: Vec<EdgeId>, attr_name: &str) -> PyResult<Vec<PyObject>> {
        match self.inner._get_edge_attributes_for_edges(&edge_ids, &attr_name.to_string()) {
            Ok(values) => {
                let mut py_values = Vec::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(attr_value) => py_values.push(attr_value_to_python_value(py, &attr_value)?),
                        None => py_values.push(py.None()),
                    }
                }
                Ok(py_values)
            }
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
    
    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.node_ids().len()
    }
    
    /// Check if a node exists in the graph
    fn has_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id)
    }
    
    /// Check if an edge exists in the graph
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id)
    }
    
    /// Get the number of nodes in the graph
    fn node_count(&self) -> usize {
        self.inner.node_ids().len()
    }
    
    /// Get the number of edges in the graph
    fn edge_count(&self) -> usize {
        self.inner.edge_ids().len()
    }
    
    /// Get all active node IDs (g.node_ids property)
    #[getter]
    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_ids()
    }
    
    /// Get all active edge IDs (g.edge_ids property)  
    #[getter]
    fn edge_ids(&self) -> Vec<EdgeId> {
        self.inner.edge_ids()
    }
    
    /// Get nodes accessor for fluent API (g.nodes property) 
    #[getter]
    fn nodes(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        let graph_ref = self_.into();
        PyGraph::create_nodes_accessor_internal(graph_ref, py)
    }
    
    /// Get edges accessor for fluent API (g.edges property)
    #[getter]
    fn edges(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        let graph_ref = self_.into();
        PyGraph::create_edges_accessor_internal(graph_ref, py)
    }
    
    /// Get unified attributes accessor for both nodes and edges (g.attributes property)
    #[getter]
    fn attributes(&self) -> PyAttributes {
        PyAttributes {
            graph: &self.inner as *const RustGraph,
        }
    }
    
    // === PHASE 3 QUERYING METHODS ===
    
    /// Phase 3.1: Advanced filtering - accepts NodeFilter or string query
    fn filter_nodes(mut self_: PyRefMut<Self>, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_node_query")?;
            let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')"
            ));
        };
        
        let filtered_nodes = self_.inner.find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;
        
        // O(k) Calculate induced edges using optimized core subgraph method
        use std::collections::HashSet;
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        
        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = self_.inner.get_columnar_topology();
        let mut induced_edges = Vec::new();
        
        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];
            
            // O(1) HashSet lookups instead of O(n) Vec::contains
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }
        
        // Get a reference to this graph to pass to the subgraph
        let graph_ref = self_.into();
        
        Ok(PySubgraph::new(
            filtered_nodes,
            induced_edges,
            "filtered_nodes".to_string(),
            Some(graph_ref),
        ))
    }
    
    /// Filter nodes within an existing subgraph (enables chaining)
    fn filter_subgraph_nodes(&mut self, py: Python, subgraph: &PySubgraph, filter: &PyAny) -> PyResult<PySubgraph> {
        // Parse the filter like in filter_nodes
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_node_query")?;
            let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query"
            ));
        };
        
        // Filter only the nodes in the current subgraph
        // Use the existing find_nodes and then intersect with subgraph nodes
        let all_matching_nodes = self.inner.find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;
        
        let mut filtered_nodes = Vec::new();
        for &node_id in &subgraph.nodes {
            if all_matching_nodes.contains(&node_id) {
                filtered_nodes.push(node_id);
            }
        }
        
        // Calculate induced edges among the filtered nodes
        let mut induced_edges = Vec::new();
        for &edge_id in &subgraph.edges {
            if let Ok((source, target)) = self.inner.edge_endpoints(edge_id) {
                if filtered_nodes.contains(&source) && filtered_nodes.contains(&target) {
                    induced_edges.push(edge_id);
                }
            }
        }
        
        Ok(PySubgraph::new(
            filtered_nodes,
            induced_edges,
            format!("{}_filtered", subgraph.subgraph_type),
            subgraph.graph.clone(),
        ))
    }
    
    /// Advanced edge filtering - accepts EdgeFilter or string query
    fn filter_edges(mut self_: PyRefMut<Self>, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for EdgeFilter object first (most common case)
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using our query parser
            let query_parser = py.import("groggy.query_parser")?;
            let parse_func = query_parser.getattr("parse_edge_query")?;
            let parsed_filter: PyEdgeFilter = parse_func.call1((query_str,))?.extract()?;
            parsed_filter.inner.clone()
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query (e.g., 'weight > 0.5')"
            ));
        };
        
        let filtered_edges = self_.inner.find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;
        
        // Collect all endpoints of the filtered edges to create the node set
        // 🚀 PERFORMANCE FIX: Use HashSet for O(1) contains/insert instead of O(n) Vec operations
        let mut endpoint_nodes_set = std::collections::HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = self_.inner.edge_endpoints(edge_id) {
                endpoint_nodes_set.insert(source);  // O(1) hash insert
                endpoint_nodes_set.insert(target);  // O(1) hash insert
            }
        }
        
        // Convert to Vec for compatibility
        let endpoint_nodes: Vec<NodeId> = endpoint_nodes_set.into_iter().collect();
        
        // Get a reference to this graph to pass to the subgraph
        let graph_ref = self_.into();
        
        Ok(PySubgraph::new(
            endpoint_nodes,
            filtered_edges,
            "filtered_edges".to_string(),
            Some(graph_ref),
        ))
    }
    
    /// Phase 3.2: Graph traversal algorithms
    
    
    
    /// Cleaner alias for traverse_bfs - shorter and more intuitive
    #[pyo3(signature = (start_node, max_depth = None, node_filter = None, edge_filter = None, inplace = false, attr_name = None))]
    fn bfs(&mut self, _py: Python, start_node: NodeId, max_depth: Option<usize>, 
           node_filter: Option<&PyNodeFilter>, edge_filter: Option<&PyEdgeFilter>,
           inplace: Option<bool>, attr_name: Option<String>) -> PyResult<PySubgraph> {
        let inplace = inplace.unwrap_or(false);
        
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
        
        // If inplace=True, set distance/order attributes on nodes
        if inplace {
            let attr_name = attr_name.unwrap_or_else(|| "bfs_distance".to_string());
            
            // Set distance attributes (distance from start_node)
            // For now, we'll set a simple order attribute
            for (order, &node_id) in result.nodes.iter().enumerate() {
                let order_value = PyAttrValue { inner: groggy::AttrValue::Int(order as i64) };
                self.set_node_attribute(node_id, attr_name.clone(), &order_value)?;
            }
        }
        
        Ok(PySubgraph::new(
            result.nodes,
            result.edges,
            "bfs_traversal".to_string(),
            None,
        ))
    }
    
    /// Cleaner alias for traverse_dfs - shorter and more intuitive  
    #[pyo3(signature = (start_node, max_depth = None, node_filter = None, edge_filter = None, inplace = false, node_attr = None, edge_attr = None))]
    fn dfs(&mut self, _py: Python, start_node: NodeId, max_depth: Option<usize>,
           node_filter: Option<&PyNodeFilter>, edge_filter: Option<&PyEdgeFilter>,
           inplace: Option<bool>, node_attr: Option<String>, edge_attr: Option<String>) -> PyResult<PySubgraph> {
        let inplace = inplace.unwrap_or(false);
        
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
        
        // If inplace=True, set attributes on nodes and edges
        if inplace {
            // Set node attributes (DFS order/distance)
            if let Some(node_attr_name) = node_attr {
                for (order, &node_id) in result.nodes.iter().enumerate() {
                    let order_value = PyAttrValue { inner: groggy::AttrValue::Int(order as i64) };
                    self.set_node_attribute(node_id, node_attr_name.clone(), &order_value)?;
                }
            } else {
                // Default node attribute
                let default_attr = "dfs_order".to_string();
                for (order, &node_id) in result.nodes.iter().enumerate() {
                    let order_value = PyAttrValue { inner: groggy::AttrValue::Int(order as i64) };
                    self.set_node_attribute(node_id, default_attr.clone(), &order_value)?;
                }
            }
            
            // Set edge attributes (tree edge or back edge)
            if let Some(edge_attr_name) = edge_attr {
                for &edge_id in &result.edges {
                    // For now, mark all edges in result as tree edges
                    let tree_edge_value = PyAttrValue { inner: groggy::AttrValue::Bool(true) };
                    self.set_edge_attribute(edge_id, edge_attr_name.clone(), &tree_edge_value)?;
                }
            }
        }
        
        Ok(PySubgraph::new(
            result.nodes,
            result.edges,
            "dfs_traversal".to_string(),
            None,
        ))
    }
    
    /// Find shortest path between two nodes with optional in-place attribute setting
    #[pyo3(signature = (source, target, weight_attribute = None, inplace = false, attr_name = None))]
    fn shortest_path(&mut self, source: NodeId, target: NodeId, weight_attribute: Option<AttrName>, 
                    inplace: Option<bool>, attr_name: Option<String>) -> PyResult<Option<PySubgraph>> {
        let inplace = inplace.unwrap_or(false);
        
        let options = groggy::core::traversal::PathFindingOptions {
            weight_attribute,
            max_path_length: None,
            heuristic: None,
        };
        
        let result = self.inner.shortest_path(source, target, options)
            .map_err(graph_error_to_py_err)?;
            
        match result {
            Some(path) => {
                if inplace {
                    if let Some(attr_name) = attr_name {
                        // Set path distance attribute on nodes
                        for (distance, &node_id) in path.nodes.iter().enumerate() {
                            let attr_value = groggy::AttrValue::Int(distance as i64);
                            self.inner.set_node_attr(node_id, attr_name.clone(), attr_value)
                                .map_err(graph_error_to_py_err)?;
                        }
                    }
                }
                
                Ok(Some(PySubgraph::new(
                    path.nodes,
                    path.edges,
                    "shortest_path".to_string(),
                    None,
                )))
            },
            None => Ok(None),
        }
    }
    
    /// Find connected components with optional in-place attribute setting
    #[pyo3(signature = (inplace = false, attr_name = None))]
    fn connected_components(&mut self, _py: Python, inplace: Option<bool>, attr_name: Option<String>) -> PyResult<Vec<PySubgraph>> {
        let inplace = inplace.unwrap_or(false);
        
        let options = groggy::core::traversal::TraversalOptions::default();
        let result = self.inner.connected_components(options)
            .map_err(graph_error_to_py_err)?;
        
        // Convert each component to a PySubgraph
        let mut subgraphs = Vec::new();
        for (i, component) in result.components.into_iter().enumerate() {
            // Create subgraph with induced edges (for now, empty - can be enhanced later)
            let subgraph = PySubgraph::new(
                component.nodes.clone(),
                Vec::new(), // TODO: Calculate induced edges between component nodes
                format!("connected_component_{}", i),
                None, // TODO: Phase 2.2 - complex graph reference sharing needed
            );
            subgraphs.push(subgraph);
            
            // If inplace=True, set component_id attribute on nodes
            if inplace {
                let attr_name = attr_name.clone().unwrap_or_else(|| "component_id".to_string());
                let component_value = PyAttrValue { inner: groggy::AttrValue::Int(i as i64) };
                
                for &node_id in &component.nodes {
                    self.set_node_attribute(node_id, attr_name.clone(), &component_value)?;
                }
            }
        }
        
        Ok(subgraphs)
    }
    
    /// Phase 3.4: Query result aggregation and analytics - UNIFIED AGGREGATE METHOD
    #[pyo3(signature = (attribute, operation, target = None, node_ids = None))]
    fn aggregate(&self, attribute: AttrName, operation: String, target: Option<String>, node_ids: Option<Vec<NodeId>>, py: Python) -> PyResult<PyObject> {
        let target = target.unwrap_or_else(|| "nodes".to_string());
        
        match target.as_str() {
            "nodes" => {
                if let Some(node_list) = node_ids {
                    // Custom node list aggregation (replaces aggregate_nodes)
                    self.aggregate_custom_nodes(node_list, attribute, py)
                } else {
                    // All nodes aggregation (replaces aggregate_node_attribute)
                    let result = self.inner.aggregate_node_attribute(&attribute, &operation)
                        .map_err(graph_error_to_py_err)?;
                    let py_result = PyAggregationResult { value: result.value };
                    Ok(Py::new(py, py_result)?.to_object(py))
                }
            },
            "edges" => {
                // Edge aggregation (replaces aggregate_edge_attribute)
                let result = self.inner.aggregate_edge_attribute(&attribute, &operation)
                    .map_err(graph_error_to_py_err)?;
                let py_result = PyAggregationResult { value: result.value };
                Ok(Py::new(py, py_result)?.to_object(py))
            },
            _ => {
                Err(PyValueError::new_err(format!("Invalid target '{}'. Use 'nodes' or 'edges'", target)))
            }
        }
    }
    
    // Helper method for custom node list aggregation (extracted from aggregate_nodes)
    fn aggregate_custom_nodes(&self, node_ids: Vec<NodeId>, attribute: AttrName, py: Python) -> PyResult<PyObject> {
        // Use bulk attribute retrieval for much better performance (10-100x faster than individual lookups)
        let bulk_attributes = self.inner._get_node_attributes_for_nodes(&node_ids, &attribute)
            .map_err(graph_error_to_py_err)?;
        let mut values = Vec::new();
        
        // Extract values from bulk result
        for attr_value in bulk_attributes {
            if let Some(value) = attr_value {
                values.push(value);
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
    
    /// Cleaner alias for group_nodes_by_attribute - shorter and more intuitive
    fn group_by(&self, attribute: AttrName, aggregation_attr: AttrName, operation: String) -> PyResult<PyGroupedAggregationResult> {
        self.group_nodes_by_attribute(attribute, aggregation_attr, operation)
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
    
    
    
    
    /// Cleaner alias for list_branches - more concise
    fn branches(&self) -> Vec<PyBranchInfo> {
        self.inner.list_branches()
            .into_iter()
            .map(|branch_info| PyBranchInfo { inner: branch_info })
            .collect()
    }
    
    /// Cleaner alias for get_commit_history - more concise
    fn commit_history(&self) -> Vec<PyCommit> {
        // Use the public commit_history method which returns CommitInfo
        // For now, return empty vector since CommitInfo != Commit
        Vec::new()
    }
    
    /// Cleaner alias for get_historical_view - more concise
    fn historical_view(&self, commit_id: StateId) -> PyResult<PyHistoricalView> {
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
    
    /// Get mapping from uid_key attribute values to internal node IDs
    /// Returns: {"alice": 0, "bob": 1, ...} for all nodes with the specified attribute
    fn get_node_mapping(&self, uid_key: String) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let node_ids = self.inner.node_ids();
            
            // Scan all nodes for the specified uid_key attribute
            for node_id in node_ids {
                if let Ok(Some(attr_value)) = self.inner.get_node_attr(node_id, &uid_key) {
                    // Convert attribute value to appropriate Python type
                    let key_value = match attr_value {
                        RustAttrValue::Text(s) => s.to_object(py),
                        RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                        RustAttrValue::Int(i) => i.to_object(py),
                        RustAttrValue::SmallInt(i) => i.to_object(py), 
                        RustAttrValue::Float(f) => f.to_object(py),
                        RustAttrValue::Bool(b) => b.to_object(py),
                        // For other types, convert to string representation
                        _ => format!("{:?}", attr_value).to_object(py),
                    };
                    
                    dict.set_item(key_value, node_id.to_object(py))?;
                }
            }
            
            Ok(dict.to_object(py))
        })
    }
    
    /// Create a GraphTable view for DataFrame-like access to node data
    /// Returns a GraphTable object with all nodes and their attributes
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Temporarily return a simple placeholder until PyO3 trait issue is resolved
        let graph_table_module = py.import("groggy.graph_table")?;
        let graph_table_class = graph_table_module.getattr("GraphTable")?;
        
        // Simple approach: create empty GraphTable and set attributes manually  
        let empty_list = py.eval("[]", None, None)?;
        let table = graph_table_class.call1((empty_list, "nodes"))?;
        Ok(table.to_object(py))
    }
    
    /// Create a GraphTable view for DataFrame-like access to edge data
    /// Returns a GraphTable object with all edges and their attributes
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        // Temporarily return a simple placeholder until PyO3 trait issue is resolved
        let graph_table_module = py.import("groggy.graph_table")?;
        let graph_table_class = graph_table_module.getattr("GraphTable")?;
        
        // Simple approach: create empty GraphTable and set attributes manually  
        let empty_list = py.eval("[]", None, None)?;
        let table = graph_table_class.call1((empty_list, "edges"))?;
        Ok(table.to_object(py))
    }
    
    // ========================================================================
    // VIEW CREATION METHODS FOR FLUENT API  
    // ========================================================================
    
    /// Create a NodeView for fluent attribute updates (internal helper)
    fn create_node_view_internal(graph_ref: Py<PyGraph>, py: Python, node_id: NodeId) -> PyResult<Py<PyNodeView>> {
        Py::new(py, PyNodeView {
            graph: graph_ref,
            node_id,
        })
    }
    
    /// Create an EdgeView for fluent attribute updates (internal helper)
    fn create_edge_view_internal(graph_ref: Py<PyGraph>, py: Python, edge_id: EdgeId) -> PyResult<Py<PyEdgeView>> {
        Py::new(py, PyEdgeView {
            graph: graph_ref,
            edge_id,
        })
    }
    
    /// Create a NodesAccessor (internal helper)
    fn create_nodes_accessor_internal(graph_ref: Py<PyGraph>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        Py::new(py, PyNodesAccessor {
            graph: graph_ref,
            constrained_nodes: None,
        })
    }
    
    /// Create an EdgesAccessor (internal helper)
    fn create_edges_accessor_internal(graph_ref: Py<PyGraph>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        Py::new(py, PyEdgesAccessor {
            graph: graph_ref,
            constrained_edges: None,
        })
    }
}

// ============================================================================
// NODE AND EDGE VIEW CLASSES FOR FLUENT API
// ============================================================================

/// Wrapper for g.nodes that supports indexing syntax: g.nodes[id] -> NodeView
#[pyclass(name = "NodesAccessor")]
pub struct PyNodesAccessor {
    graph: Py<PyGraph>,
    /// Optional constraint: if Some, only these nodes are accessible
    constrained_nodes: Option<Vec<NodeId>>,
}

#[pymethods]
impl PyNodesAccessor {
    /// Support node access: g.nodes[0] -> NodeView, g.nodes[[0,1,2]] -> Subgraph, g.nodes[0:5] -> Subgraph
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try to extract as single integer (existing behavior)
        if let Ok(node_id) = key.extract::<NodeId>() {
            // Check constraint first
            if let Some(ref constrained) = self.constrained_nodes {
                if !constrained.contains(&node_id) {
                    return Err(PyKeyError::new_err(format!("Node {} is not in this subgraph", node_id)));
                }
            }
            
            // Single node access - return NodeView
            let graph = self.graph.borrow(py);
            if !graph.has_node(node_id) {
                return Err(PyKeyError::new_err(format!("Node {} does not exist", node_id)));
            }
            
            let node_view = PyGraph::create_node_view_internal(self.graph.clone(), py, node_id)?;
            return Ok(node_view.to_object(py));
        }
        
        // Try to extract as list of integers (batch access)
        if let Ok(node_ids) = key.extract::<Vec<NodeId>>() {
            // Batch node access - return Subgraph
            let graph = self.graph.borrow(py);
            
            // Validate all nodes exist
            for &node_id in &node_ids {
                if !graph.has_node(node_id) {
                    return Err(PyKeyError::new_err(format!("Node {} does not exist", node_id)));
                }
            }
            
            // Calculate induced edges between these nodes
            // 🚀 PERFORMANCE FIX: Use HashSet for O(1) contains instead of O(n) Vec operations
            let node_set: std::collections::HashSet<NodeId> = node_ids.iter().copied().collect();
            let mut induced_edges = Vec::new();
            for edge_id in graph.edge_ids() {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    if node_set.contains(&source) && node_set.contains(&target) {  // O(1) hash lookup
                        induced_edges.push(edge_id);
                    }
                }
            }
            
            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                node_ids,
                induced_edges,
                "node_batch_selection".to_string(),
                Some(self.graph.clone()),
            );
            
            return Ok(Py::new(py, subgraph)?.to_object(py));
        }
        
        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let graph = self.graph.borrow(py);
            let all_node_ids = graph.node_ids();
            
            // Convert slice to indices
            let slice_info = slice.indices(all_node_ids.len() as i64)?;
            let start = slice_info.start as usize;
            let stop = slice_info.stop as usize;
            let step = slice_info.step as usize;
            
            // Extract nodes based on slice
            let mut selected_nodes = Vec::new();
            let mut i = start;
            while i < stop && i < all_node_ids.len() {
                selected_nodes.push(all_node_ids[i]);
                i += step;
            }
            
            // Calculate induced edges between selected nodes
            // 🚀 PERFORMANCE FIX: Use HashSet for O(1) contains instead of O(n) Vec operations  
            let selected_node_set: std::collections::HashSet<NodeId> = selected_nodes.iter().copied().collect();
            let mut induced_edges = Vec::new();
            for edge_id in graph.edge_ids() {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    if selected_node_set.contains(&source) && selected_node_set.contains(&target) {  // O(1) hash lookup
                        induced_edges.push(edge_id);
                    }
                }
            }
            
            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                selected_nodes,
                induced_edges,
                "node_slice_selection".to_string(),
                Some(self.graph.clone()),
            );
            
            return Ok(Py::new(py, subgraph)?.to_object(py));
        }
        
        // If none of the above worked, return error
        Err(PyTypeError::new_err("Node index must be int, list of ints, or slice"))
    }
    
    /// Support iteration: for node_id in g.nodes
    fn __iter__(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let node_ids = graph.node_ids();
        // Return the list directly - Python will handle iteration
        Ok(node_ids.to_object(py))
    }
    
    /// Support len(g.nodes)
    fn __len__(&self, py: Python) -> PyResult<usize> {
        if let Some(ref constrained) = self.constrained_nodes {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow(py);
            Ok(graph.node_count())
        }
    }
    
    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let count = graph.node_count();
        Ok(format!("NodesAccessor({} nodes)", count))
    }
}

/// Wrapper for subgraph.nodes that supports indexing syntax: subgraph.nodes[id] -> NodeView
/// Wrapper for g.edges that supports indexing syntax: g.edges[id] -> EdgeView  
#[pyclass(name = "EdgesAccessor")]
pub struct PyEdgesAccessor {
    graph: Py<PyGraph>,
    /// Optional constraint: if Some, only these edges are accessible
    constrained_edges: Option<Vec<EdgeId>>,
}

#[pymethods]
impl PyEdgesAccessor {
    /// Support edge access: g.edges[0] -> EdgeView, g.edges[[0,1,2]] -> Subgraph, g.edges[0:5] -> Subgraph
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try to extract as single integer (existing behavior)
        if let Ok(edge_id) = key.extract::<EdgeId>() {
            // Check constraint first
            if let Some(ref constrained) = self.constrained_edges {
                if !constrained.contains(&edge_id) {
                    return Err(PyKeyError::new_err(format!("Edge {} is not in this subgraph", edge_id)));
                }
            }
            
            // Single edge access - return EdgeView
            let graph = self.graph.borrow(py);
            if !graph.has_edge(edge_id) {
                return Err(PyKeyError::new_err(format!("Edge {} does not exist", edge_id)));
            }
            
            let edge_view = PyGraph::create_edge_view_internal(self.graph.clone(), py, edge_id)?;
            return Ok(edge_view.to_object(py));
        }
        
        // Try to extract as list of integers (batch access)
        if let Ok(edge_ids) = key.extract::<Vec<EdgeId>>() {
            // Batch edge access - return Subgraph with these edges + their endpoints
            let graph = self.graph.borrow(py);
            
            // Validate all edges exist
            for &edge_id in &edge_ids {
                if !graph.has_edge(edge_id) {
                    return Err(PyKeyError::new_err(format!("Edge {} does not exist", edge_id)));
                }
            }
            
            // Collect all endpoints of selected edges
            // 🚀 PERFORMANCE FIX: Use HashSet for O(1) contains/insert instead of O(n) Vec operations
            let mut endpoint_nodes_set = std::collections::HashSet::new();
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    endpoint_nodes_set.insert(source);  // O(1) hash insert
                    endpoint_nodes_set.insert(target);  // O(1) hash insert
                }
            }
            
            // Convert to Vec for compatibility
            let endpoint_nodes: Vec<NodeId> = endpoint_nodes_set.into_iter().collect();
            
            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                endpoint_nodes,
                edge_ids,
                "edge_batch_selection".to_string(),
                Some(self.graph.clone()),
            );
            
            return Ok(Py::new(py, subgraph)?.to_object(py));
        }
        
        // Try to extract as slice (slice access)
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let graph = self.graph.borrow(py);
            let all_edge_ids = graph.edge_ids();
            
            // Convert slice to indices
            let slice_info = slice.indices(all_edge_ids.len() as i64)?;
            let start = slice_info.start as usize;
            let stop = slice_info.stop as usize;
            let step = slice_info.step as usize;
            
            // Extract edges based on slice
            let mut selected_edges = Vec::new();
            let mut i = start;
            while i < stop && i < all_edge_ids.len() {
                selected_edges.push(all_edge_ids[i]);
                i += step;
            }
            
            // Collect all endpoints of selected edges
            // 🚀 PERFORMANCE FIX: Use HashSet for O(1) contains/insert instead of O(n) Vec operations
            let mut endpoint_nodes_set = std::collections::HashSet::new();
            for &edge_id in &selected_edges {
                if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                    endpoint_nodes_set.insert(source);  // O(1) hash insert
                    endpoint_nodes_set.insert(target);  // O(1) hash insert
                }
            }
            
            // Convert to Vec for compatibility
            let endpoint_nodes: Vec<NodeId> = endpoint_nodes_set.into_iter().collect();
            
            // Create and return Subgraph
            let subgraph = PySubgraph::new(
                endpoint_nodes,
                selected_edges,
                "edge_slice_selection".to_string(),
                Some(self.graph.clone()),
            );
            
            return Ok(Py::new(py, subgraph)?.to_object(py));
        }
        
        // If none of the above worked, return error
        Err(PyTypeError::new_err("Edge index must be int, list of ints, or slice"))
    }
    
    /// Support iteration: for edge_id in g.edges
    fn __iter__(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        let edge_ids = graph.edge_ids();
        // Return the list directly - Python will handle iteration
        Ok(edge_ids.to_object(py))
    }
    
    /// Support len(g.edges)
    fn __len__(&self, py: Python) -> PyResult<usize> {
        if let Some(ref constrained) = self.constrained_edges {
            Ok(constrained.len())
        } else {
            let graph = self.graph.borrow(py);
            Ok(graph.edge_count())
        }
    }
    
    /// String representation
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        let count = graph.edge_count();
        Ok(format!("EdgesAccessor({} edges)", count))
    }
    
    /// Get all source node IDs as a list
    #[getter]
    fn source(&self, py: Python) -> PyResult<Vec<NodeId>> {
        let graph = self.graph.borrow(py);
        let edge_ids = graph.edge_ids();
        let mut sources = Vec::new();
        
        for edge_id in edge_ids {
            if let Ok((source, _)) = graph.inner.edge_endpoints(edge_id) {
                sources.push(source);
            }
        }
        
        Ok(sources)
    }
    
    /// Get all target node IDs as a list
    #[getter]
    fn target(&self, py: Python) -> PyResult<Vec<NodeId>> {
        let graph = self.graph.borrow(py);
        let edge_ids = graph.edge_ids();
        let mut targets = Vec::new();
        
        for edge_id in edge_ids {
            if let Ok((_, target)) = graph.inner.edge_endpoints(edge_id) {
                targets.push(target);
            }
        }
        
        Ok(targets)
    }
}

/// Fluent view for a single node with chainable attribute updates
#[pyclass(name = "NodeView")]
pub struct PyNodeView {
    graph: Py<PyGraph>,
    node_id: NodeId,
}

#[pymethods]
impl PyNodeView {
    /// Set attributes using kwargs syntax: node.set(name="Alice", age=30)
    #[pyo3(signature = (**kwargs))]
    fn set(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<Py<PyNodeView>> {
        if let Some(kwargs) = kwargs {
            let mut graph = self.graph.borrow_mut(py);
            
            // Iterate through kwargs and set each attribute
            for (key, value) in kwargs.iter() {
                let attr_name: String = key.extract()?;
                let attr_value = python_value_to_attr_value(value)?;
                let py_attr_value = PyAttrValue { inner: attr_value };
                
                graph.set_node_attribute(self.node_id, attr_name, &py_attr_value)?;
            }
        }
        
        // Return self for chaining
        Ok(PyGraph::create_node_view_internal(self.graph.clone(), py, self.node_id)?)
    }
    
    /// Update attributes using dict syntax: node.update({"name": "Alice", "age": 30})
    fn update(&mut self, py: Python, data: &PyDict) -> PyResult<Py<PyNodeView>> {
        let mut graph = self.graph.borrow_mut(py);
        
        // Iterate through dict and set each attribute
        for (key, value) in data.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            let py_attr_value = PyAttrValue { inner: attr_value };
            
            graph.set_node_attribute(self.node_id, attr_name, &py_attr_value)?;
        }
        
        // Return self for chaining
        Ok(PyGraph::create_node_view_internal(self.graph.clone(), py, self.node_id)?)
    }
    
    /// Get all attributes as a dict
    fn attrs(&self, py: Python) -> PyResult<PyObject> {
        // For now, return empty dict - we'll implement this later
        let dict = pyo3::types::PyDict::new(py);
        Ok(dict.to_object(py))
    }
    
    /// Get a specific attribute value
    fn get(&self, py: Python, attr_name: String) -> PyResult<Option<PyAttrValue>> {
        let graph = self.graph.borrow(py);
        graph.get_node_attribute(self.node_id, attr_name)
    }
    
    /// Support item access: node["name"]
    fn __getitem__(&self, py: Python, attr_name: String) -> PyResult<PyAttrValue> {
        let graph = self.graph.borrow(py);
        match graph.get_node_attribute(self.node_id, attr_name.clone())? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(format!("Attribute '{}' not found", attr_name))),
        }
    }
    
    /// Support item assignment: node["name"] = "Alice"
    fn __setitem__(&mut self, py: Python, attr_name: String, value: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attr_value = python_value_to_attr_value(value)?;
        let py_attr_value = PyAttrValue { inner: attr_value };
        graph.set_node_attribute(self.node_id, attr_name, &py_attr_value)
    }
    
    /// String representation with all attributes
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        
        // Get all attributes for this node
        let mut attr_parts = Vec::new();
        
        // Try to get common attributes and add them to display
        if let Ok(Some(attr)) = graph.inner.get_node_attr(self.node_id, &"name".to_string()) {
            if let Ok(py_val) = attr_value_to_python_value(py, &attr) {
                if let Ok(name_str) = py_val.extract::<String>(py) {
                    attr_parts.push(format!("name={}", name_str));
                }
            }
        }
        
        if let Ok(Some(attr)) = graph.inner.get_node_attr(self.node_id, &"age".to_string()) {
            if let Ok(py_val) = attr_value_to_python_value(py, &attr) {
                if let Ok(age_val) = py_val.extract::<i64>(py) {
                    attr_parts.push(format!("age={}", age_val));
                }
            }
        }
        
        if let Ok(Some(attr)) = graph.inner.get_node_attr(self.node_id, &"dept".to_string()) {
            if let Ok(py_val) = attr_value_to_python_value(py, &attr) {
                if let Ok(dept_str) = py_val.extract::<String>(py) {
                    attr_parts.push(format!("dept={}", dept_str));
                }
            }
        }
        
        // Add more attributes if found (limit to first few for readability)
        let attr_display = if attr_parts.is_empty() {
            String::new()
        } else if attr_parts.len() <= 3 {
            format!(", {}", attr_parts.join(", "))
        } else {
            format!(", {}, ...", attr_parts[..3].join(", "))
        };
        
        Ok(format!("NodeView(id={}{})", self.node_id, attr_display))
    }
    
    /// Repr
    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.__str__(py)
    }
    
    /// Get the node ID
    #[getter]
    fn id(&self) -> NodeId {
        self.node_id
    }
}

/// Fluent view for a single edge with chainable attribute updates  
#[pyclass(name = "EdgeView")]
pub struct PyEdgeView {
    graph: Py<PyGraph>,
    edge_id: EdgeId,
}

#[pymethods]
impl PyEdgeView {
    /// Set attributes using kwargs syntax: edge.set(weight=0.9, type="friendship")
    #[pyo3(signature = (**kwargs))]
    fn set(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<Py<PyEdgeView>> {
        if let Some(kwargs) = kwargs {
            let mut graph = self.graph.borrow_mut(py);
            
            // Iterate through kwargs and set each attribute
            for (key, value) in kwargs.iter() {
                let attr_name: String = key.extract()?;
                let attr_value = python_value_to_attr_value(value)?;
                let py_attr_value = PyAttrValue { inner: attr_value };
                
                graph.set_edge_attribute(self.edge_id, attr_name, &py_attr_value)?;
            }
        }
        
        // Return self for chaining
        Ok(PyGraph::create_edge_view_internal(self.graph.clone(), py, self.edge_id)?)
    }
    
    /// Update attributes using dict syntax: edge.update({"weight": 0.9, "type": "friendship"})
    fn update(&mut self, py: Python, data: &PyDict) -> PyResult<Py<PyEdgeView>> {
        let mut graph = self.graph.borrow_mut(py);
        
        // Iterate through dict and set each attribute
        for (key, value) in data.iter() {
            let attr_name: String = key.extract()?;
            let attr_value = python_value_to_attr_value(value)?;
            let py_attr_value = PyAttrValue { inner: attr_value };
            
            graph.set_edge_attribute(self.edge_id, attr_name, &py_attr_value)?;
        }
        
        // Return self for chaining
        Ok(PyGraph::create_edge_view_internal(self.graph.clone(), py, self.edge_id)?)
    }
    
    /// Get all attributes as a dict (includes source/target)
    fn attrs(&self, py: Python) -> PyResult<PyObject> {
        let graph = self.graph.borrow(py);
        graph.get_edge_attributes(self.edge_id, py)
    }
    
    /// Get a specific attribute value
    fn get(&self, py: Python, attr_name: String) -> PyResult<Option<PyAttrValue>> {
        let graph = self.graph.borrow(py);
        graph.get_edge_attribute(self.edge_id, attr_name)
    }
    
    /// Support item access: edge["weight"]
    fn __getitem__(&self, py: Python, attr_name: String) -> PyResult<PyAttrValue> {
        let graph = self.graph.borrow(py);
        match graph.get_edge_attribute(self.edge_id, attr_name.clone())? {
            Some(value) => Ok(value),
            None => Err(PyKeyError::new_err(format!("Attribute '{}' not found", attr_name))),
        }
    }
    
    /// Support item assignment: edge["weight"] = 0.9
    fn __setitem__(&mut self, py: Python, attr_name: String, value: &PyAny) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attr_value = python_value_to_attr_value(value)?;
        let py_attr_value = PyAttrValue { inner: attr_value };
        graph.set_edge_attribute(self.edge_id, attr_name, &py_attr_value)
    }
    
    /// String representation with source, target, and attributes
    fn __str__(&self, py: Python) -> PyResult<String> {
        let graph = self.graph.borrow(py);
        
        // Get source and target
        let (source, target) = graph.inner.edge_endpoints(self.edge_id)
            .map_err(graph_error_to_py_err)?;
        
        // Get attributes
        let mut attr_parts = Vec::new();
        
        // Try to get common edge attributes
        if let Ok(Some(attr)) = graph.inner.get_edge_attr(self.edge_id, &"weight".to_string()) {
            if let Ok(py_val) = attr_value_to_python_value(py, &attr) {
                if let Ok(weight_val) = py_val.extract::<f64>(py) {
                    attr_parts.push(format!("weight={:.2}", weight_val));
                } else if let Ok(weight_val) = py_val.extract::<f32>(py) {
                    attr_parts.push(format!("weight={:.2}", weight_val));
                }
            }
        }
        
        if let Ok(Some(attr)) = graph.inner.get_edge_attr(self.edge_id, &"type".to_string()) {
            if let Ok(py_val) = attr_value_to_python_value(py, &attr) {
                if let Ok(type_str) = py_val.extract::<String>(py) {
                    attr_parts.push(format!("type={}", type_str));
                }
            }
        }
        
        if let Ok(Some(attr)) = graph.inner.get_edge_attr(self.edge_id, &"relationship".to_string()) {
            if let Ok(py_val) = attr_value_to_python_value(py, &attr) {
                if let Ok(rel_str) = py_val.extract::<String>(py) {
                    attr_parts.push(format!("relationship={}", rel_str));
                }
            }
        }
        
        // Format the attribute display
        let attr_display = if attr_parts.is_empty() {
            String::new()
        } else if attr_parts.len() <= 3 {
            format!(", {}", attr_parts.join(", "))
        } else {
            format!(", {}, ...", attr_parts[..3].join(", "))
        };
        
        Ok(format!("EdgeView(id={}, source={}, target={}{})", 
                  self.edge_id, source, target, attr_display))
    }
    
    /// Repr
    fn __repr__(&self, py: Python) -> PyResult<String> {
        self.__str__(py)
    }
    
    /// Get the edge ID
    #[getter]
    fn id(&self) -> EdgeId {
        self.edge_id
    }
    
    /// Get the source node ID
    #[getter]
    fn source(&self, py: Python) -> PyResult<NodeId> {
        let graph = self.graph.borrow(py);
        let (source, _) = graph.inner.edge_endpoints(self.edge_id)
            .map_err(graph_error_to_py_err)?;
        Ok(source)
    }
    
    /// Get the target node ID
    #[getter]
    fn target(&self, py: Python) -> PyResult<NodeId> {
        let graph = self.graph.borrow(py);
        let (_, target) = graph.inner.edge_endpoints(self.edge_id)
            .map_err(graph_error_to_py_err)?;
        Ok(target)
    }
    
    /// Get both endpoints as a tuple (source, target)
    #[getter]
    fn endpoints(&self, py: Python) -> PyResult<(NodeId, NodeId)> {
        let graph = self.graph.borrow(py);
        graph.inner.edge_endpoints(self.edge_id)
            .map_err(graph_error_to_py_err)
    }
}

// ================================================================================================
// PYARRAY - ENHANCED STATISTICAL ARRAYS
// ================================================================================================

/// Python wrapper for GraphArray with fast native statistical operations
#[pyclass(name = "GraphArray")]
pub struct PyGraphArray {
    inner: GraphArray,
}

#[pymethods]
impl PyGraphArray {
    /// Create a new GraphArray from a list of values
    #[new]
    fn new(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());
            
            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }
            
            Ok(PyGraphArray {
                inner: GraphArray::from_vec(attr_values),
            })
        })
    }
    
    // === LIST COMPATIBILITY ===
    
    /// Get the number of elements (len())
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Get element by index - supports arr[i] with negative indexing
    fn __getitem__(&self, py: Python, index: isize) -> PyResult<PyObject> {
        let len = self.inner.len() as isize;
        
        // Handle negative indexing (Python-style)
        let actual_index = if index < 0 {
            len + index
        } else {
            index
        };
        
        // Check bounds
        if actual_index < 0 || actual_index >= len {
            return Err(PyIndexError::new_err("Index out of range"));
        }
        
        match self.inner.get(actual_index as usize) {
            Some(attr_value) => attr_value_to_python_value(py, attr_value),
            None => Err(PyIndexError::new_err("Index out of range")),
        }
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("GraphArray(len={})", self.inner.len())
    }
    
    /// Iterator support (for value in array)
    fn __iter__(slf: PyRef<Self>) -> GraphArrayIterator {
        GraphArrayIterator {
            array: slf.inner.clone(),
            index: 0,
        }
    }
    
    /// Convert to plain Python list
    fn to_list(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut py_values = Vec::with_capacity(self.inner.len());
        
        for attr_value in self.inner.iter() {
            py_values.push(attr_value_to_python_value(py, attr_value)?);
        }
        
        Ok(py_values)
    }
    
    // === STATISTICAL OPERATIONS ===
    
    /// Calculate mean (average) of numeric values
    fn mean(&self) -> Option<f64> {
        self.inner.mean()
    }
    
    /// Calculate standard deviation of numeric values
    fn std(&self) -> Option<f64> {
        self.inner.std()
    }
    
    /// Get minimum value
    fn min(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.inner.min() {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None),
        }
    }
    
    /// Get maximum value
    fn max(&self, py: Python) -> PyResult<Option<PyObject>> {
        match self.inner.max() {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, &attr_value)?)),
            None => Ok(None),
        }
    }
    
    /// Calculate quantile (percentile)
    fn quantile(&self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }
    
    /// Calculate median (50th percentile)
    fn median(&self) -> Option<f64> {
        self.inner.median()
    }
    
    /// Get count of elements
    fn count(&self) -> usize {
        self.inner.count()
    }
    
    /// Get comprehensive statistical summary
    fn describe(&self, py: Python) -> PyResult<PyStatsSummary> {
        Ok(PyStatsSummary {
            inner: self.inner.describe(),
        })
    }
}

/// Python wrapper for StatsSummary
#[pyclass(name = "StatsSummary")]
pub struct PyStatsSummary {
    inner: StatsSummary,
}

#[pymethods]
impl PyStatsSummary {
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }
    
    #[getter]
    fn mean(&self) -> Option<f64> {
        self.inner.mean
    }
    
    #[getter]
    fn std(&self) -> Option<f64> {
        self.inner.std
    }
    
    #[getter]
    fn min(&self, py: Python) -> PyResult<Option<PyObject>> {
        match &self.inner.min {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
            None => Ok(None),
        }
    }
    
    #[getter]
    fn max(&self, py: Python) -> PyResult<Option<PyObject>> {
        match &self.inner.max {
            Some(attr_value) => Ok(Some(attr_value_to_python_value(py, attr_value)?)),
            None => Ok(None),
        }
    }
    
    #[getter]
    fn median(&self) -> Option<f64> {
        self.inner.median
    }
    
    #[getter]
    fn q25(&self) -> Option<f64> {
        self.inner.q25
    }
    
    #[getter]
    fn q75(&self) -> Option<f64> {
        self.inner.q75
    }
    
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Iterator for GraphArray
#[pyclass]
struct GraphArrayIterator {
    array: GraphArray,
    index: usize,
}

#[pymethods]
impl GraphArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<PyObject>> {
        if self.index < self.array.len() {
            let attr_value = &self.array[self.index];
            self.index += 1;
            Ok(Some(attr_value_to_python_value(py, attr_value)?))
        } else {
            Ok(None)
        }
    }
}

// Helper function to create PyGraphArray from GraphArray
impl PyGraphArray {
    pub fn from_graph_array(array: GraphArray) -> Self {
        PyGraphArray { inner: array }
    }
}

/// The Python module
#[pymodule]
fn _groggy(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core classes
    m.add_class::<PyGraph>()?;
    m.add_class::<PyAttrValue>()?;
    m.add_class::<PyNodeAttributes>()?;
    
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
    m.add_class::<PySubgraph>()?;
    m.add_class::<PyAttributeCollection>()?;
    
    // Enhanced statistical arrays
    // GraphArray is not a PyClass, it's a core Rust type
    // m.add_class::<GraphArray>()?;
    m.add_class::<PyStatsSummary>()?;
    
    // Fluent API view classes
    m.add_class::<PyNodeView>()?;
    m.add_class::<PyEdgeView>()?;
    m.add_class::<PyNodesAccessor>()?;
    m.add_class::<PyEdgesAccessor>()?;
    
    Ok(())
}
