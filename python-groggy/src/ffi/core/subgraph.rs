//! Subgraph FFI Bindings
//! 
//! Python bindings for PySubgraph - dual-mode architecture supporting both
//! core RustSubgraph integration and legacy compatibility.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyKeyError, PyIndexError, PyImportError, PyNotImplementedError};
use groggy::{NodeId, EdgeId, AttrValue};
use std::collections::HashMap;

// Import types from our FFI modules
use crate::ffi::api::graph::PyGraph;
use crate::ffi::types::PyAttrValue;
use crate::ffi::core::array::{PyGraphArray, PyGraphMatrix};
use crate::ffi::core::accessors::{PyNodesAccessor, PyEdgesAccessor};

/// Utility function to convert Python values to AttrValue
fn python_value_to_attr_value(value: &PyAny) -> PyResult<AttrValue> {
    if let Ok(int_val) = value.extract::<i64>() {
        Ok(AttrValue::Int(int_val))
    } else if let Ok(float_val) = value.extract::<f64>() {
        Ok(AttrValue::Float(float_val as f32))
    } else if let Ok(str_val) = value.extract::<String>() {
        Ok(AttrValue::Text(str_val))
    } else if let Ok(bool_val) = value.extract::<bool>() {
        Ok(AttrValue::Bool(bool_val))
    } else {
        Err(PyTypeError::new_err("Unsupported attribute value type"))
    }
}

/// Utility function to convert AttrValue to Python object
fn attr_value_to_python_value(py: Python, attr_value: &AttrValue) -> PyResult<PyObject> {
    match attr_value {
        AttrValue::Int(val) => Ok(val.to_object(py)),
        AttrValue::SmallInt(val) => Ok((*val as i64).to_object(py)),
        AttrValue::Float(val) => Ok(val.to_object(py)),
        AttrValue::Bool(val) => Ok(val.to_object(py)),
        AttrValue::Text(val) => Ok(val.to_object(py)),
        AttrValue::CompactText(val) => Ok(val.to_object(py)),
        _ => Ok(py.None()),
    }
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
    
    /// Get node IDs in this subgraph as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<groggy::AttrValue> = self.nodes.iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
    }
    
    /// Get edge IDs in this subgraph as GraphArray (lazy Rust view) - use .values for Python list
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        let attr_values: Vec<groggy::AttrValue> = self.edges.iter()
            .map(|&id| groggy::AttrValue::Int(id as i64))
            .collect();
        let graph_array = groggy::GraphArray::from_vec(attr_values);
        let py_graph_array = PyGraphArray { inner: graph_array };
        Ok(Py::new(py, py_graph_array)?)
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
            for (i, component) in components.iter().enumerate() {
                // STANDARDIZED: Use PySubgraph::new() like all other subgraph creation methods
                // This ensures consistent graph reference handling for .nodes/.edges accessors
                result.push(PySubgraph::new(
                    component.node_ids(),
                    component.edge_ids(),
                    format!("connected_component_{}", i),
                    self.graph.clone(),  // Pass the graph reference consistently
                ));
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
                        let py_attr_value = PyAttrValue::from_attr_value(attr_value);
                        
                        graph.set_node_attribute(node_id, attr_name, &py_attr_value)?;
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
                    let py_attr_value = PyAttrValue::from_attr_value(attr_value);
                    
                    graph.set_node_attribute(node_id, attr_name, &py_attr_value)?;
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
            // CRITICAL FIX: Route to edge attributes for edge subgraphs
            if self.subgraph_type == "edge_slice_selection" {
                // This is an edge subgraph - route to edge attributes
                if attr_name == "id" {
                    // Special case: edge IDs (the edges themselves)
                    let edge_ids = self.edges.iter().map(|&edge_id| {
                        groggy::AttrValue::Int(edge_id as i64)
                    }).collect();
                    let graph_array = groggy::GraphArray::from_vec(edge_ids);
                    let py_graph_array = PyGraphArray { inner: graph_array };
                    return Ok(Py::new(py, py_graph_array)?.to_object(py));
                } else {
                    // Regular edge attributes (strength, weight, etc.)
                    let edge_values = self.get_edge_attribute_column(py, &attr_name)?;
                    // Convert Vec<PyObject> to GraphArray for consistency
                    let mut attr_values = Vec::new();
                    for py_value in edge_values {
                        // Convert Python values back to AttrValue
                        if py_value.is_none(py) {
                            attr_values.push(groggy::AttrValue::Int(0)); // Default for missing
                        } else if let Ok(int_val) = py_value.extract::<i64>(py) {
                            attr_values.push(groggy::AttrValue::Int(int_val));
                        } else if let Ok(float_val) = py_value.extract::<f64>(py) {
                            attr_values.push(groggy::AttrValue::Float(float_val as f32));
                        } else if let Ok(str_val) = py_value.extract::<String>(py) {
                            attr_values.push(groggy::AttrValue::Text(str_val));
                        } else if let Ok(bool_val) = py_value.extract::<bool>(py) {
                            attr_values.push(groggy::AttrValue::Bool(bool_val));
                        } else {
                            attr_values.push(groggy::AttrValue::Int(0)); // Fallback
                        }
                    }
                    let graph_array = groggy::GraphArray::from_vec(attr_values);
                    let py_graph_array = PyGraphArray { inner: graph_array };
                    return Ok(Py::new(py, py_graph_array)?.to_object(py));
                }
            } else {
                // This is a node subgraph - route to node attributes (original behavior)
                let column = self.get_node_attribute_column(py, &attr_name)?;
                return Ok(column.to_object(py));
            }
        }
        
        // Try list of strings (multi-column access)
        if let Ok(attr_names) = key.extract::<Vec<String>>() {
            if attr_names.is_empty() {
                return Err(PyValueError::new_err("Empty attribute list"));
            }
            
            // Collect all columns as GraphArrays
            let mut columns = Vec::new();
            let mut num_rows = 0;
            
            // Type checking for mixed types
            let mut column_types = Vec::new();
            
            for attr_name in &attr_names {
                let column = self.get_node_attribute_column(py, attr_name)?;
                
                // Detect column type by borrowing temporarily
                let column_type = {
                    let graph_array = column.borrow(py);
                    
                    // Get the length and detect column type
                    num_rows = graph_array.inner.len();
                    
                    // Sample a few values to determine the predominant type
                    if num_rows > 0 {
                        let sample_size = std::cmp::min(num_rows, 3);
                        let mut type_counts = std::collections::HashMap::new();
                        
                        for i in 0..sample_size {
                            let type_name = match &graph_array.inner[i] {
                                groggy::AttrValue::Int(_) | groggy::AttrValue::SmallInt(_) => "int",
                                groggy::AttrValue::Float(_) => "float", 
                                groggy::AttrValue::Bool(_) => "bool",
                                groggy::AttrValue::Text(_) | groggy::AttrValue::CompactText(_) => "str",
                                _ => "mixed",
                            };
                            *type_counts.entry(type_name).or_insert(0) += 1;
                        }
                        
                        // Get the most common type
                        type_counts.into_iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(type_name, _)| type_name)
                            .unwrap_or("mixed")
                    } else {
                        "empty"
                    }
                }; // Borrow ends here
                
                column_types.push(column_type);
                columns.push(column);
            }
            
            // Check for mixed types (GraphMatrix constraint)  
            if attr_names.len() > 1 {
                let first_type = column_types[0];
                let has_mixed_types = column_types.iter().any(|&t| t != first_type && t != "empty");
                
                if has_mixed_types {
                    let detected_types: Vec<&str> = column_types.into_iter().collect();
                    return Err(PyTypeError::new_err(format!(
                        "Mixed types detected: [{}]. GraphMatrix requires homogeneous types.\n\
                        Use subgraph.nodes.table()[{:?}] for mixed-type data.",
                        detected_types.join(", "),
                        attr_names
                    )));
                }
            }
            
            // For single column in list form: [['age']] -> return GraphArray (same as 'age')
            if attr_names.len() == 1 {
                return Ok(columns[0].clone_ref(py).to_object(py));
            } else {
                // Multiple columns: [['age', 'height']] -> return PyGraphMatrix (structured collection)
                let matrix = PyGraphMatrix {
                    columns,
                    column_names: attr_names,
                    num_rows,
                };
                
                return Ok(Py::new(py, matrix)?.to_object(py));
            }
        }
        
        Err(PyTypeError::new_err("Key must be a string or list of strings"))
    }
    
    /// Create GraphTable for DataFrame-like view of this subgraph nodes
    fn table(&self, py: Python) -> PyResult<PyObject> {
        // Import the Python GraphTable class
        let groggy = py.import("groggy")?;
        let graph_table_class = groggy.getattr("GraphTable")?;
        
        // Create GraphTable with nodes data source
        let table = graph_table_class.call1((self.nodes.clone(), "nodes", self.graph.clone()))?;
        Ok(table.to_object(py))
    }
    
    /// Create GraphTable for DataFrame-like view of this subgraph edges
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        // Import the Python GraphTable class
        let groggy = py.import("groggy")?;
        let graph_table_class = groggy.getattr("GraphTable")?;
        
        // Create GraphTable with edges data source
        let table = graph_table_class.call1((self.edges.clone(), "edges", self.graph.clone()))?;
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
                                let attr_name = parts[0].trim();
                                let expected_value = parts[1].trim_matches('"').trim_matches('\'');
                                
                                // Check if this node has the attribute with the expected value
                                if let Ok(Some(attr_value)) = graph.inner.get_node_attr(node_id, &attr_name.to_string()) {
                                    match attr_value {
                                        AttrValue::Text(ref text) => text == expected_value,
                                        AttrValue::CompactText(ref text) => text == expected_value,
                                        _ => false,
                                    }
                                } else {
                                    false
                                }
                            } else {
                                true // Invalid filter, include all
                            }
                        } else {
                            true // Unsupported filter, include all
                        }
                    })
                    .copied()
                    .collect();
                
                // Calculate induced edges for the filtered nodes
                let filtered_node_set: std::collections::HashSet<NodeId> = filtered_nodes.iter().copied().collect();
                let induced_edges: Vec<EdgeId> = self.edges.iter()
                    .filter(|&&edge_id| {
                        if let Ok((source, target)) = graph.inner.edge_endpoints(edge_id) {
                            filtered_node_set.contains(&source) && filtered_node_set.contains(&target)
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect();
                
                Ok(PySubgraph::new(
                    filtered_nodes,
                    induced_edges,
                    format!("{}_filtered", self.subgraph_type),
                    self.graph.clone(),
                ))
            } else {
                // For other filter types, return current subgraph unchanged for now
                Ok(PySubgraph::new(
                    self.nodes.clone(),
                    self.edges.clone(),
                    self.subgraph_type.clone(),
                    self.graph.clone(),
                ))
            }
        } else {
            Err(PyRuntimeError::new_err("No graph reference available for filtering"))
        }
    }
}
