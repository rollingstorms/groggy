//! Array FFI Bindings
//!
//! Python bindings for statistical arrays and matrices.

use groggy::storage::{legacy_array::{GraphArray, StatsSummary}, array::{BaseArray, ArrayOps, ArrayIterator, NodesArray, EdgesArray, MetaNodeArray, NodeIdLike, EdgeLike, MetaNodeLike}};
use groggy::types::{AttrValue as RustAttrValue, AttrValueType, NodeId, EdgeId};
use groggy::entities::meta_node::MetaNode;
use pyo3::exceptions::{PyAttributeError, PyImportError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

// Use utility functions from utils module
use crate::ffi::utils::{attr_value_to_python_value, python_value_to_attr_value};

/// New BaseArray-powered array with chaining support
#[derive(Clone)]
#[pyclass(name = "BaseArray")]
pub struct PyBaseArray {
    pub inner: BaseArray<RustAttrValue>,
}

#[pymethods]
impl PyBaseArray {
    /// Create a new BaseArray from a list of values
    #[new]
    fn new(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mut attr_values = Vec::with_capacity(values.len());

            for value in values {
                let attr_value = python_value_to_attr_value(value.as_ref(py))?;
                attr_values.push(attr_value);
            }

            // Infer the data type from the first non-null element
            let dtype = attr_values
                .iter()
                .find(|v| !matches!(v, RustAttrValue::Null))
                .map(|v| v.dtype())
                .unwrap_or(AttrValueType::Text);

            Ok(PyBaseArray {
                inner: BaseArray::new(attr_values),
            })
        })
    }

    /// Get the number of elements (len())
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Get single element by index
    fn __getitem__(&self, py: Python, index: isize) -> PyResult<PyObject> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 { len + index } else { index };
        
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
        let dtype = self.dtype();
        format!("BaseArray[{}] (dtype: {})", self.inner.len(), dtype)
    }

    /// Get the data type of the array based on the first non-null element
    fn dtype(&self) -> String {
        let dtype = self.inner.first()
            .map(|v| v.dtype())
            .unwrap_or(AttrValueType::Null);
        format!("{:?}", dtype)
    }

    // Statistical operations
    fn head(&self, n: usize) -> Self {
        let data: Vec<RustAttrValue> = self.inner.iter().take(n).cloned().collect();
        PyBaseArray {
            inner: BaseArray::new(data),
        }
    }
    
    fn tail(&self, n: usize) -> Self {
        let len = self.inner.len();
        let start = if len > n { len - n } else { 0 };
        let data: Vec<RustAttrValue> = self.inner.iter().skip(start).cloned().collect();
        PyBaseArray {
            inner: BaseArray::new(data),
        }
    }
    
    fn unique(&self) -> Self {
        let mut unique_values: Vec<RustAttrValue> = Vec::new();
        let mut seen: std::collections::HashSet<RustAttrValue> = std::collections::HashSet::new();
        
        for value in self.inner.iter() {
            if seen.insert(value.clone()) {
                unique_values.push(value.clone());
            }
        }
        
        PyBaseArray {
            inner: BaseArray::new(unique_values),
        }
    }

    fn describe(&self, py: Python) -> PyResult<PyObject> {
        // Basic descriptive statistics
        let count = self.inner.len();
        let non_null_count = self.inner.iter().filter(|v| !matches!(v, RustAttrValue::Null)).count();
        let null_count = count - non_null_count;
        let unique_count = self.unique().inner.len();
        
        let dict = PyDict::new(py);
        dict.set_item("count", count)?;
        dict.set_item("non_null", non_null_count)?;
        dict.set_item("null", null_count)?;
        dict.set_item("unique", unique_count)?;
        dict.set_item("dtype", self.dtype())?;
        
        Ok(dict.into())
    }

    /// NEW: Enable fluent chaining with .iter() method
    fn iter(slf: PyRef<Self>) -> PyResult<PyBaseArrayIterator> {
        // Use our ArrayOps implementation to create the iterator
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyBaseArrayIterator {
            inner: array_iterator,
        })
    }
    
    /// Delegation-based method application to each element 
    /// This demonstrates the concept: apply a method to each element and return new array
    fn apply_to_each(&self, py: Python, method_name: &str, args: &PyTuple) -> PyResult<Self> {
        let mut results = Vec::new();
        
        // Apply the method to each element in the array
        for value in self.inner.data() {
            // Convert AttrValue to a Python object that might have the method
            let py_value = attr_value_to_python_value(py, value)?;
            
            // Check if this value has the requested method
            if py_value.as_ref(py).hasattr(method_name)? {
                // Get the method and call it with the provided arguments
                let method = py_value.as_ref(py).getattr(method_name)?;
                let result = method.call1(args)?;
                
                // Convert result back to AttrValue
                let attr_result = python_value_to_attr_value(result)?;
                results.push(attr_result);
            } else {
                return Err(PyAttributeError::new_err(
                    format!("Elements in array don't have method '{}'", method_name)
                ));
            }
        }
        
        // Create new BaseArray with results
        let result_array = BaseArray::from_attr_values(results);
        Ok(PyBaseArray { inner: result_array })
    }
    
    /// BaseArray __getattr__ delegation: automatically apply methods to each element
    /// This enables: array.some_method() -> applies some_method to each element
    pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        // Use apply_to_each internally to delegate to each element
        let args = pyo3::types::PyTuple::empty(py);
        let result = self.apply_to_each(py, name, args)?;
        Ok(result.into_py(py))
    }
}

/// Python wrapper for ArrayIterator<AttrValue> that supports method chaining
#[pyclass(name = "BaseArrayIterator", unsendable)]
pub struct PyBaseArrayIterator {
    inner: ArrayIterator<RustAttrValue>,
}

#[pymethods]
impl PyBaseArrayIterator {
    /// Filter elements using a Python predicate function
    fn filter(slf: PyRefMut<Self>, predicate: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = slf.inner.clone();
            let filtered = inner.filter(|attr_value| {
                // Convert AttrValue to Python and call predicate
                match attr_value_to_python_value(py, attr_value) {
                    Ok(py_value) => {
                        match predicate.call1(py, (py_value,)) {
                            Ok(result) => result.is_true(py).unwrap_or(false),
                            Err(_) => false,
                        }
                    }
                    Err(_) => false,
                }
            });
            
            Ok(Self { inner: filtered })
        })
    }
    
    /// Map elements using a Python function
    fn map(slf: PyRefMut<Self>, func: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = slf.inner.clone();
            let mapped = inner.map(|attr_value| {
                // Convert AttrValue to Python, call function, convert back
                match attr_value_to_python_value(py, &attr_value) {
                    Ok(py_value) => {
                        match func.call1(py, (py_value,)) {
                            Ok(result) => {
                                match python_value_to_attr_value(result.as_ref(py)) {
                                    Ok(new_attr_value) => new_attr_value,
                                    Err(_) => attr_value, // Keep original on conversion error
                                }
                            }
                            Err(_) => attr_value, // Keep original on call error
                        }
                    }
                    Err(_) => attr_value, // Keep original on conversion error
                }
            });
            
            Ok(Self { inner: mapped })
        })
    }
    
    /// Take first n elements
    fn take(slf: PyRefMut<Self>, n: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self { inner: inner.take(n) })
    }
    
    /// Skip first n elements
    fn skip(slf: PyRefMut<Self>, n: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self { inner: inner.skip(n) })
    }
    
    /// Collect back into a BaseArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyBaseArray> {
        let inner = slf.inner.clone();
        
        // Extract the data directly
        let elements = inner.into_vec();
        
        // Infer dtype from first non-null element
        let dtype = elements
            .iter()
            .find(|v| !matches!(v, RustAttrValue::Null))
            .map(|v| v.dtype())
            .unwrap_or(AttrValueType::Text);
        
        Ok(PyBaseArray {
            inner: BaseArray::new(elements),
        })
    }
}

// =============================================================================
// Specialized typed arrays with trait-based method injection
// =============================================================================

/// Python wrapper for NodesArray - specialized array for NodeId collections
#[pyclass(name = "NodesArray", unsendable)]
pub struct PyNodesArray {
    pub inner: NodesArray,
}

#[pymethods]
impl PyNodesArray {
    /// Create a new NodesArray from node IDs
    #[new]
    fn new(node_ids: Vec<usize>) -> PyResult<Self> {
        // Convert Python usize to NodeId
        let nodes: Vec<NodeId> = node_ids.into_iter().collect();
        
        Ok(PyNodesArray {
            inner: NodesArray::new(nodes),
        })
    }
    
    /// Get the number of nodes
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Get node ID by index
    fn __getitem__(&self, index: isize) -> PyResult<usize> {
        let len = self.inner.len() as isize;
        let actual_index = if index < 0 { len + index } else { index };
        
        if actual_index < 0 || actual_index >= len {
            return Err(PyIndexError::new_err("Index out of range"));
        }
        
        match self.inner.get(actual_index as usize) {
            Some(node_id) => Ok(*node_id),
            None => Err(PyIndexError::new_err("Index out of range")),
        }
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("NodesArray[{}]", self.inner.len())
    }
    
    /// Enable fluent chaining with .iter() method for node-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyNodesArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyNodesArrayIterator {
            inner: array_iterator,
        })
    }
}

/// Python wrapper for ArrayIterator<NodeId> with node-specific chaining operations
#[pyclass(name = "NodesArrayIterator", unsendable)]
pub struct PyNodesArrayIterator {
    inner: ArrayIterator<NodeId>,
}

#[pymethods]
impl PyNodesArrayIterator {
    /// Filter nodes by minimum degree
    /// Enables: g.nodes.ids().iter().filter_by_degree(3)
    fn filter_by_degree(slf: PyRefMut<Self>, min_degree: usize) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_by_degree(min_degree),
        })
    }
    
    /// Get neighbors for each node  
    /// Enables: node_ids.iter().get_neighbors()
    fn get_neighbors(slf: PyRefMut<Self>) -> PyResult<PyNeighborsArrayIterator> {
        let inner = slf.inner.clone();
        let neighbors_iterator = inner.get_neighbors();
        
        Ok(PyNeighborsArrayIterator {
            inner: neighbors_iterator,
        })
    }
    
    /// Convert nodes to subgraphs
    /// Enables: node_ids.iter().to_subgraph()
    fn to_subgraph(slf: PyRefMut<Self>) -> PyResult<PySubgraphArrayIterator> {
        let inner = slf.inner.clone();
        let subgraph_iterator = inner.to_subgraph();
        
        Ok(PySubgraphArrayIterator {
            inner: subgraph_iterator,
        })
    }
    
    /// Collect back into a NodesArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyNodesArray> {
        let inner = slf.inner.clone();
        let node_ids = inner.into_vec();
        
        Ok(PyNodesArray {
            inner: NodesArray::new(node_ids),
        })
    }
}

/// Python wrapper for EdgesArray - specialized array for EdgeId collections
#[pyclass(name = "EdgesArray", unsendable)]
pub struct PyEdgesArray {
    pub inner: EdgesArray,
}

#[pymethods]
impl PyEdgesArray {
    /// Create a new EdgesArray from edge IDs
    #[new]
    fn new(edge_ids: Vec<usize>) -> PyResult<Self> {
        // Convert Python usize to EdgeId
        let edges: Vec<EdgeId> = edge_ids.into_iter().collect();
        
        Ok(PyEdgesArray {
            inner: EdgesArray::new(edges),
        })
    }
    
    /// Get the number of edges
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("EdgesArray[{}]", self.inner.len())
    }
    
    /// Enable fluent chaining with .iter() method for edge-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyEdgesArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyEdgesArrayIterator {
            inner: array_iterator,
        })
    }
}

/// Python wrapper for ArrayIterator<EdgeId> with edge-specific chaining operations
#[pyclass(name = "EdgesArrayIterator", unsendable)]
pub struct PyEdgesArrayIterator {
    inner: ArrayIterator<EdgeId>,
}

#[pymethods]
impl PyEdgesArrayIterator {
    /// Filter edges by minimum weight
    /// Enables: edges.iter().filter_by_weight(0.5)
    fn filter_by_weight(slf: PyRefMut<Self>, min_weight: f64) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_by_weight(min_weight),
        })
    }
    
    /// Group edges by source node
    /// Enables: edges.iter().group_by_source()
    fn group_by_source(slf: PyRefMut<Self>) -> PyResult<PyEdgeGroupsIterator> {
        let inner = slf.inner.clone();
        let groups_iterator = inner.group_by_source();
        
        Ok(PyEdgeGroupsIterator {
            inner: groups_iterator,
        })
    }
    
    /// Collect back into an EdgesArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyEdgesArray> {
        let inner = slf.inner.clone();
        let edge_ids = inner.into_vec();
        
        Ok(PyEdgesArray {
            inner: EdgesArray::new(edge_ids),
        })
    }
}

/// Python wrapper for MetaNodeArray - specialized array for MetaNode collections
#[pyclass(name = "MetaNodeArray", unsendable)]
pub struct PyMetaNodeArray {
    pub inner: MetaNodeArray,
}

#[pymethods]
impl PyMetaNodeArray {
    /// Get the number of meta-nodes
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("MetaNodeArray[{}]", self.inner.len())
    }
    
    /// Enable fluent chaining with .iter() method for meta-node-specific operations
    fn iter(slf: PyRef<Self>) -> PyResult<PyMetaNodeArrayIterator> {
        let array_iterator = ArrayOps::iter(&slf.inner);
        
        Ok(PyMetaNodeArrayIterator {
            inner: array_iterator,
        })
    }
}

/// Python wrapper for ArrayIterator<MetaNode> with meta-node-specific chaining operations
#[pyclass(name = "MetaNodeArrayIterator", unsendable)]
pub struct PyMetaNodeArrayIterator {
    inner: ArrayIterator<MetaNode>,
}

#[pymethods]
impl PyMetaNodeArrayIterator {
    /// Expand meta-nodes back into subgraphs
    /// Enables: meta_nodes.iter().expand()
    fn expand(slf: PyRefMut<Self>) -> PyResult<PySubgraphArrayIterator> {
        let inner = slf.inner.clone();
        let subgraph_iterator = inner.expand();
        
        Ok(PySubgraphArrayIterator {
            inner: subgraph_iterator,
        })
    }
    
    /// Re-aggregate meta-nodes with new aggregation functions
    fn re_aggregate(slf: PyRefMut<Self>, aggs: std::collections::HashMap<String, String>) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.re_aggregate(aggs),
        })
    }
    
    /// Collect back into a MetaNodeArray
    fn collect(slf: PyRefMut<Self>) -> PyResult<PyMetaNodeArray> {
        let inner = slf.inner.clone();
        let meta_nodes = inner.into_vec();
        
        Ok(PyMetaNodeArray {
            inner: MetaNodeArray::new(meta_nodes),
        })
    }
}

// =============================================================================
// Supporting iterator types for complex return values
// =============================================================================

#[pyclass(name = "NeighborsArrayIterator", unsendable)]
pub struct PyNeighborsArrayIterator {
    inner: ArrayIterator<Vec<NodeId>>,
}

#[pymethods]
impl PyNeighborsArrayIterator {
    /// Flatten neighbor lists into a single NodesArray
    fn flatten(slf: PyRefMut<Self>) -> PyResult<PyNodesArray> {
        let inner = slf.inner.clone();
        let neighbor_lists = inner.into_vec();
        let flattened: Vec<NodeId> = neighbor_lists.into_iter().flatten().collect();
        
        Ok(PyNodesArray {
            inner: NodesArray::new(flattened),
        })
    }
    
    /// Collect into a list of lists
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<Vec<usize>>> {
        let inner = slf.inner.clone();
        let neighbor_lists = inner.into_vec();
        let py_lists: Vec<Vec<usize>> = neighbor_lists
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect())
            .collect();
        
        Ok(py_lists)
    }
}

#[pyclass(name = "SubgraphArrayIterator", unsendable)]
pub struct PySubgraphArrayIterator {
    inner: ArrayIterator<groggy::subgraphs::subgraph::Subgraph>,
}

#[pymethods]
impl PySubgraphArrayIterator {
    /// Filter nodes within subgraphs (inherits from SubgraphLike)
    fn filter_nodes(slf: PyRefMut<Self>, query: &str) -> PyResult<Self> {
        let inner = slf.inner.clone();
        Ok(Self {
            inner: inner.filter_nodes(query),
        })
    }
    
    /// Collect into a list of subgraphs
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<String>> {
        let inner = slf.inner.clone();
        let _subgraphs = inner.into_vec();
        
        // Placeholder: return string representations
        Ok(vec!["placeholder_subgraph".to_string()])
    }
}

#[pyclass(name = "EdgeGroupsIterator", unsendable)]
pub struct PyEdgeGroupsIterator {
    inner: ArrayIterator<Vec<EdgeId>>,
}

#[pymethods]
impl PyEdgeGroupsIterator {
    /// Collect into a list of edge groups
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<Vec<usize>>> {
        let inner = slf.inner.clone();
        let edge_groups = inner.into_vec();
        let py_groups: Vec<Vec<usize>> = edge_groups
            .into_iter()
            .map(|group| group.into_iter().collect())
            .collect();
        
        Ok(py_groups)
    }
}

/// Native performance-oriented GraphArray for statistical operations
#[pyclass(name = "GraphArray")]
#[derive(Clone)]
pub struct PyGraphArray {
    pub inner: GraphArray,
}

#[pymethods]
impl PyGraphArray {
    /// Create a new GraphArray from a list of values
    #[new]
    fn new(values: Vec<PyObject>) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Emit deprecation warning
            let _ = pyo3::PyErr::warn(
                py,
                pyo3::exceptions::PyDeprecationWarning::py_type(py),
                "GraphArray is deprecated. Use BaseArray for generic data or StatsArray for numeric arrays.",
                1,
            );
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

    /// Advanced indexing support: arr[i], arr[start:end], arr[start:end:step], arr[mask]
    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Single integer indexing: arr[5]
        if let Ok(index) = key.extract::<isize>() {
            let len = self.inner.len() as isize;

            // Handle negative indexing (Python-style)
            let actual_index = if index < 0 { len + index } else { index };

            // Check bounds
            if actual_index < 0 || actual_index >= len {
                return Err(PyIndexError::new_err("Index out of range"));
            }

            match self.inner.get(actual_index as usize) {
                Some(attr_value) => attr_value_to_python_value(py, attr_value),
                None => Err(PyIndexError::new_err("Index out of range")),
            }
        }
        // Slice indexing: arr[start:end], arr[start:end:step]
        else if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            let len = self.inner.len();
            let indices = slice.indices(
                len.try_into()
                    .map_err(|_| PyValueError::new_err("Array too large for slice"))?,
            )?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step;

            let mut result_values = Vec::new();

            if step == 1 {
                // Simple slice [start:stop]
                for i in start..stop.min(len) {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                }
            } else if step > 1 {
                // Step slice [start:stop:step]
                let mut i = start;
                while i < stop.min(len) {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                    i += step as usize;
                }
            } else {
                return Err(PyValueError::new_err("Negative step not supported"));
            }

            // Return new GraphArray with sliced data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        }
        // List of indices: arr[[1, 3, 5]]
        else if let Ok(indices) = key.extract::<Vec<isize>>() {
            let len = self.inner.len() as isize;
            let mut result_values = Vec::new();

            for &index in &indices {
                let actual_index = if index < 0 { len + index } else { index };

                if actual_index >= 0 && actual_index < len {
                    if let Some(attr_value) = self.inner.get(actual_index as usize) {
                        result_values.push(attr_value.clone());
                    }
                } else {
                    return Err(PyIndexError::new_err("Index out of range"));
                }
            }

            // Return new GraphArray with selected data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        }
        // Boolean mask indexing: arr[mask] where mask is a list of booleans
        else if let Ok(mask) = key.extract::<Vec<bool>>() {
            let len = self.inner.len();
            if mask.len() != len {
                return Err(PyValueError::new_err(
                    "Boolean mask length must match array length",
                ));
            }

            let mut result_values = Vec::new();
            for (i, &include) in mask.iter().enumerate() {
                if include {
                    if let Some(attr_value) = self.inner.get(i) {
                        result_values.push(attr_value.clone());
                    }
                }
            }

            // Return new GraphArray with masked data
            let result_array = groggy::GraphArray::from_vec(result_values);
            let py_array = PyGraphArray {
                inner: result_array,
            };
            Ok(Py::new(py, py_array)?.to_object(py))
        } else {
            Err(PyTypeError::new_err(
                "Index must be int, slice, list of ints, or list of bools",
            ))
        }
    }

    /// String representation
    fn __repr__(&self, py: Python) -> PyResult<String> {
        // Also warn on repr to surface deprecation in interactive usage
        let _ = pyo3::PyErr::warn(
            py,
            pyo3::exceptions::PyDeprecationWarning::py_type(py),
            "GraphArray is deprecated. Use BaseArray or StatsArray.",
            1,
        );
        let len = self.inner.len();
        let dtype = self._get_dtype();
        Ok(format!("GraphArray(len={}, dtype={})", len, dtype))
    }

    /// String representation (same as __repr__ for consistency)
    fn __str__(&self, py: Python) -> PyResult<String> {
        self.__repr__(py)
    }

    /// Get rich display representation using Rust formatter
    pub fn rich_display(&self, config: Option<&crate::ffi::display::PyDisplayConfig>) -> PyResult<String> {
        let array_len = self.inner.len();
        let dtype = self._get_dtype();
        let sample_size = std::cmp::min(10, array_len);
        
        // Calculate minimum column widths for index and value columns
        let mut index_width = 1; // minimum width for index
        let mut value_width = 5; // minimum width for value
        
        // Check header widths
        index_width = std::cmp::max(index_width, "#".len());
        value_width = std::cmp::max(value_width, "value".len());
        
        // Check index width (need to fit the largest index)
        if sample_size > 0 {
            index_width = std::cmp::max(index_width, (sample_size - 1).to_string().len());
        }
        
        // Check data widths
        for i in 0..sample_size {
            if let Some(value) = self.inner.get(i) {
                let value_str = match value {
                    groggy::AttrValue::Text(s) => s.clone(),
                    groggy::AttrValue::CompactText(s) => format!("{:?}", s),
                    groggy::AttrValue::Int(n) => format!("{}", n),
                    groggy::AttrValue::SmallInt(n) => format!("{}", n),
                    groggy::AttrValue::Float(f) => format!("{}", f),
                    groggy::AttrValue::Bool(b) => format!("{}", b),
                    groggy::AttrValue::Null => "null".to_string(),
                    _ => format!("{:?}", value),
                };
                value_width = std::cmp::max(value_width, value_str.len());
            }
        }
        
        // Create array display with index column
        let mut lines = Vec::new();
        
        // Top border
        lines.push(format!("╭{}┬{}╮", 
            "─".repeat(index_width + 2), 
            "─".repeat(value_width + 2)));
        
        // Header
        lines.push(format!("│ {:<width1$} │ {:<width2$} │", "#", "value", width1 = index_width, width2 = value_width));
        lines.push(format!("│ {:<width1$} │ {:<width2$} │", "", "obj", width1 = index_width, width2 = value_width));
        
        // Header separator  
        lines.push(format!("├{}┼{}┤", 
            "─".repeat(index_width + 2), 
            "─".repeat(value_width + 2)));
        
        // Data rows
        for i in 0..sample_size {
            if let Some(value) = self.inner.get(i) {
                let value_str = match value {
                    groggy::AttrValue::Text(s) => s.clone(),
                    groggy::AttrValue::CompactText(s) => format!("{:?}", s),
                    groggy::AttrValue::Int(n) => format!("{}", n),
                    groggy::AttrValue::SmallInt(n) => format!("{}", n),
                    groggy::AttrValue::Float(f) => format!("{}", f),
                    groggy::AttrValue::Bool(b) => format!("{}", b),
                    groggy::AttrValue::Null => "null".to_string(),
                    _ => format!("{:?}", value),
                };
                lines.push(format!("│ {:<width1$} │ {:<width2$} │", 
                    i, value_str,
                    width1 = index_width, width2 = value_width));
            }
        }
        
        // Bottom border
        lines.push(format!("╰{}┴{}╯", 
            "─".repeat(index_width + 2), 
            "─".repeat(value_width + 2)));
        
        // Footer with array info
        let footer = format!("rows: {} • type: GraphArray • dtype: {}", array_len, dtype);
        lines.push(footer);
        
        Ok(lines.join("\n"))
    }
    
    /// Rich HTML representation for Jupyter notebooks
    fn _repr_html_(&self, _py: Python) -> PyResult<String> {
        let formatted = self.rich_display(None)?;
        // Convert to HTML format for Jupyter
        Ok(format!("<pre>{}</pre>", html_escape::encode_text(&formatted)))
    }

    /// Iterator support (for value in array)
    fn __iter__(slf: PyRef<Self>) -> GraphArrayIterator {
        GraphArrayIterator {
            array: slf.inner.clone(),
            index: 0,
        }
    }

    /// Convert to plain Python list
    pub fn to_list(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut py_values = Vec::with_capacity(self.inner.len());

        for attr_value in self.inner.iter() {
            py_values.push(attr_value_to_python_value(py, attr_value)?);
        }

        Ok(py_values)
    }

    /// Return iterator over array items (compatibility with pandas/numpy)
    /// Returns iterator of (index, value) tuples
    fn items(&self, py: Python) -> PyResult<PyObject> {
        let items: PyResult<Vec<(usize, PyObject)>> = self
            .inner
            .iter()
            .enumerate()
            .map(|(i, attr_value)| Ok((i, attr_value_to_python_value(py, attr_value)?)))
            .collect();

        Ok(items?.to_object(py))
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

    /// Count non-null values
    fn count(&self) -> usize {
        let values = self.inner.materialize();
        values
            .iter()
            .filter(|value| !matches!(value, groggy::AttrValue::Null))
            .count()
    }

    /// Check if array contains any null values
    fn has_null(&self) -> bool {
        let values = self.inner.materialize();
        values
            .iter()
            .any(|value| matches!(value, groggy::AttrValue::Null))
    }

    /// Count null values
    fn null_count(&self) -> usize {
        let values = self.inner.materialize();
        values
            .iter()
            .filter(|value| matches!(value, groggy::AttrValue::Null))
            .count()
    }

    /// Drop null values, returning a new array
    fn drop_na(&self, py: Python) -> PyResult<PyObject> {
        let values = self.inner.materialize();
        let non_null_values: Vec<groggy::AttrValue> = values
            .iter()
            .filter(|value| !matches!(value, groggy::AttrValue::Null))
            .cloned()
            .collect();

        let new_array = groggy::storage::GraphArray::from_vec(non_null_values);
        let py_array = PyGraphArray::from_graph_array(new_array);
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Fill null values with a specified value, returning a new array
    fn fill_na(&self, py: Python, fill_value: &PyAny) -> PyResult<PyObject> {
        let fill_attr_value = crate::ffi::utils::python_value_to_attr_value(fill_value)?;
        let values = self.inner.materialize();

        let filled_values: Vec<groggy::AttrValue> = values
            .iter()
            .map(|value| {
                if matches!(value, groggy::AttrValue::Null) {
                    fill_attr_value.clone()
                } else {
                    value.clone()
                }
            })
            .collect();

        let new_array = groggy::storage::GraphArray::from_vec(filled_values);
        let py_array = PyGraphArray::from_graph_array(new_array);
        Ok(Py::new(py, py_array)?.to_object(py))
    }

    /// Calculate quantile (percentile)
    fn quantile(&self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }

    /// Calculate percentile (user-friendly wrapper for quantile)
    /// percentile: 0-100 (e.g., 25 for 25th percentile, 90 for 90th percentile)
    fn percentile(&self, p: f64) -> Option<f64> {
        if !(0.0..=100.0).contains(&p) {
            return None;
        }
        self.inner.quantile(p / 100.0)
    }

    /// Calculate median (50th percentile)
    fn median(&self) -> Option<f64> {
        self.inner.median()
    }

    /// Get unique values as a new GraphArray
    fn unique(&self, py: Python) -> PyResult<Py<PyGraphArray>> {
        use std::collections::HashSet;

        // Use HashSet to find unique values
        let mut unique_set = HashSet::new();
        let mut unique_values = Vec::new();

        for attr_value in self.inner.iter() {
            // Create a simple hash key based on the value
            let key = match attr_value {
                RustAttrValue::Int(i) => format!("i:{}", i),
                RustAttrValue::SmallInt(i) => format!("si:{}", i),
                RustAttrValue::Float(f) => format!("f:{}", f),
                RustAttrValue::Text(s) => format!("t:{}", s),
                RustAttrValue::CompactText(s) => format!("ct:{}", s.as_str()),
                RustAttrValue::Bool(b) => format!("b:{}", b),
                RustAttrValue::Bytes(b) => format!("bytes:{}", b.len()), // Simple hash for bytes
                _ => format!("other:{:?}", attr_value),                  // Fallback for other types
            };

            if unique_set.insert(key) {
                // This is a new unique value
                unique_values.push(attr_value.clone());
            }
        }

        // Create new GraphArray with unique values
        let unique_array = GraphArray::from_vec(unique_values);
        let py_unique = PyGraphArray {
            inner: unique_array,
        };

        Py::new(py, py_unique)
    }

    /// Get value counts (frequency of each unique value) as Python dict
    fn value_counts(&self, py: Python) -> PyResult<PyObject> {
        use std::collections::HashMap;

        let mut counts: HashMap<String, (i32, RustAttrValue)> = HashMap::new();

        for attr_value in self.inner.iter() {
            // Create a string representation for HashMap key
            let key_str = match attr_value {
                RustAttrValue::Int(i) => format!("i:{}", i),
                RustAttrValue::SmallInt(i) => format!("si:{}", i),
                RustAttrValue::Float(f) => format!("f:{}", f),
                RustAttrValue::Text(s) => format!("t:{}", s),
                RustAttrValue::CompactText(s) => format!("ct:{}", s.as_str()),
                RustAttrValue::Bool(b) => format!("b:{}", b),
                RustAttrValue::Bytes(b) => format!("bytes:{}", b.len()),
                _ => format!("other:{:?}", attr_value),
            };

            match counts.get_mut(&key_str) {
                Some((count, _)) => *count += 1,
                None => {
                    counts.insert(key_str, (1, attr_value.clone()));
                }
            }
        }

        // Convert to Python dict
        let dict = pyo3::types::PyDict::new(py);
        for (_, (count, attr_value)) in counts {
            let py_key = attr_value_to_python_value(py, &attr_value)?;
            dict.set_item(py_key, count)?;
        }

        Ok(dict.to_object(py))
    }

    /// Get comprehensive statistical summary
    fn describe(&self, _py: Python) -> PyResult<PyStatsSummary> {
        Ok(PyStatsSummary {
            inner: self.inner.describe(),
        })
    }

    // ========================================================================
    // LAZY EVALUATION & MATERIALIZATION
    // ========================================================================

    /// Get array values (materializes data to Python objects)
    /// This is the primary materialization method - use sparingly for large arrays
    #[getter]
    fn values(&self, py: Python) -> PyResult<PyObject> {
        let materialized = self.inner.materialize();
        let py_values: PyResult<Vec<PyObject>> = materialized
            .iter()
            .map(|val| attr_value_to_python_value(py, val))
            .collect();

        Ok(py_values?.to_object(py))
    }

    /// Get preview of array for display (first 10 elements by default)
    fn preview(&self, py: Python, limit: Option<usize>) -> PyResult<PyObject> {
        let limit = limit.unwrap_or(10);
        let preview_values = self.inner.preview(limit);
        let py_values: PyResult<Vec<PyObject>> = preview_values
            .iter()
            .map(|val| attr_value_to_python_value(py, val))
            .collect();

        Ok(py_values?.to_object(py))
    }

    /// Check if array is sparse (has many default values)
    #[getter]
    fn is_sparse(&self) -> bool {
        self.inner.is_sparse()
    }

    /// Get summary information without materializing data
    fn summary(&self) -> String {
        self.inner.summary_info()
    }

    // ========================================================================
    // SCIENTIFIC COMPUTING CONVERSIONS
    // ========================================================================

    /// Convert to NumPy array (when numpy available)
    /// Uses .values property to materialize data
    fn to_numpy(&self, py: Python) -> PyResult<PyObject> {
        // Try to import numpy
        let numpy = py.import("numpy").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "numpy is required for to_numpy(). Install with: pip install numpy",
            )
        })?;

        // Get materialized data using .values property
        let values = self.values(py)?;

        // Convert to numpy array
        let array = numpy.call_method1("array", (values,))?;
        Ok(array.to_object(py))
    }

    /// Convert to Pandas Series (when pandas available)
    fn to_pandas(&self, py: Python) -> PyResult<PyObject> {
        // Try to import pandas
        let pandas = py.import("pandas").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "pandas is required for to_pandas(). Install with: pip install pandas",
            )
        })?;

        // Get data as Python list
        let values = self.values(py)?;

        // Create Series
        let series = pandas.call_method1("Series", (values,))?;
        Ok(series.to_object(py))
    }

    /// Convert to SciPy sparse array (for compatibility - GraphArray is dense by nature)
    fn to_scipy_sparse(&self, py: Python) -> PyResult<PyObject> {
        // Try to import scipy.sparse
        let scipy_sparse = py.import("scipy.sparse").map_err(|_| {
            PyErr::new::<PyImportError, _>(
                "scipy is required for to_scipy_sparse(). Install with: pip install scipy",
            )
        })?;

        // Get data as Python list
        let values = self.values(py)?;

        // Convert to numpy first, then to sparse
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (values,))?;

        // Create CSR matrix (compressed sparse row) from the dense array
        let sparse_matrix = scipy_sparse.call_method1("csr_matrix", (array,))?;
        Ok(sparse_matrix.to_object(py))
    }

    // ========================================================================
    // DISPLAY INTEGRATION METHODS
    // ========================================================================

    /// Extract display data for Python display formatters
    /// Returns a dictionary with the structure expected by array_display.py
    fn _get_display_data(&self, py: Python) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);

        // Extract array data - convert to Python list
        let data = self.to_list(py)?;
        dict.set_item("data", data)?;

        // Get array metadata
        dict.set_item("shape", (self.inner.len(),))?;
        dict.set_item("dtype", self._get_dtype())?;
        dict.set_item(
            "name",
            self._get_name().unwrap_or_else(|| "array".to_string()),
        )?;

        Ok(dict.to_object(py))
    }

    /// Get data type string for display
    fn _get_dtype(&self) -> String {
        // Sample first few elements to determine predominant type
        let sample_size = std::cmp::min(self.inner.len(), 5);
        if sample_size == 0 {
            return "object".to_string();
        }

        let mut type_counts = std::collections::HashMap::new();

        for i in 0..sample_size {
            let type_name = match &self.inner[i] {
                RustAttrValue::Int(_) | RustAttrValue::SmallInt(_) => "int64",
                RustAttrValue::Float(_) => "f32",
                RustAttrValue::Bool(_) => "bool",
                RustAttrValue::Text(_) | RustAttrValue::CompactText(_) => "str",
                _ => "object",
            };
            *type_counts.entry(type_name).or_insert(0) += 1;
        }

        // Return the most common type
        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(type_name, _)| type_name.to_string())
            .unwrap_or_else(|| "object".to_string())
    }

    /// Get name for display (optional)
    fn _get_name(&self) -> Option<String> {
        // For now, we don't store names in GraphArray
        // This can be enhanced later when we add named arrays
        None
    }

    // === COMPARISON OPERATORS FOR BOOLEAN INDEXING ===

    /// Greater than comparison - returns boolean array
    fn __gt__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a > b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a > b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a > b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a > (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) > *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) > (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) > (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() > b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() > b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() > b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() > b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a > b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Less than comparison - returns boolean array
    fn __lt__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a < b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a < b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a < b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a < (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) < *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) < (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) < (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => a.as_str() < b.as_str(),
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() < b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() < b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() < b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a < b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Greater than or equal comparison - returns boolean array
    fn __ge__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a >= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a >= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a >= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a >= (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) >= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) >= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) >= (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() >= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() >= b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a >= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => false,
                (_, groggy::AttrValue::Null) => true,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Comparison not supported for these types",
                    ))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Less than or equal comparison - returns boolean array
    fn __le__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = match (value, &other_value) {
                // Integer comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::Int(b)) => a <= b,
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::SmallInt(b)) => a <= b,
                (groggy::AttrValue::Float(a), groggy::AttrValue::Float(b)) => a <= b,

                // Mixed numeric type comparisons
                (groggy::AttrValue::Int(a), groggy::AttrValue::SmallInt(b)) => *a <= (*b as i64),
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Int(b)) => (*a as i64) <= *b,
                (groggy::AttrValue::Int(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::Int(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::SmallInt(a), groggy::AttrValue::Float(b)) => {
                    (*a as f64) <= (*b as f64)
                }
                (groggy::AttrValue::Float(a), groggy::AttrValue::SmallInt(b)) => {
                    (*a as f64) <= (*b as f64)
                }

                // String comparisons
                (groggy::AttrValue::Text(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::Text(a), groggy::AttrValue::CompactText(b)) => {
                    a.as_str() <= b.as_str()
                }
                (groggy::AttrValue::CompactText(a), groggy::AttrValue::Text(b)) => {
                    a.as_str() <= b.as_str()
                }

                // Boolean comparisons (false < true)
                (groggy::AttrValue::Bool(a), groggy::AttrValue::Bool(b)) => a <= b,

                // Handle nulls - null is less than everything
                (groggy::AttrValue::Null, _) => true,
                (_, groggy::AttrValue::Null) => false,

                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Comparison not supported between {:?} and {:?}",
                        value, other_value
                    )))
                }
            };
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Equality comparison - returns boolean array
    fn __eq__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = value == &other_value;
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Not equal comparison - returns boolean array
    fn __ne__(&self, _py: Python, other: &PyAny) -> PyResult<PyGraphArray> {
        let other_value = crate::ffi::utils::python_value_to_attr_value(other)?;

        let mut result = Vec::new();
        for value in self.inner.iter() {
            let comparison_result = value != &other_value;
            result.push(groggy::AttrValue::Bool(comparison_result));
        }

        Ok(PyGraphArray {
            inner: groggy::GraphArray::from_vec(result),
        })
    }

    /// Extract indices where boolean array is True
    /// This is used for efficient boolean indexing with node/edge accessors
    fn true_indices(&self) -> PyResult<Vec<usize>> {
        let mut indices = Vec::new();

        for (i, value) in self.inner.iter().enumerate() {
            if let groggy::AttrValue::Bool(true) = value {
                indices.push(i);
            }
        }

        Ok(indices)
    }
    
    /// Delegation-based method application to each element 
    /// This demonstrates the concept: apply a method to each element and return new array
    fn apply_to_each(&self, py: Python, method_name: &str, args: &PyTuple) -> PyResult<Self> {
        let mut results = Vec::new();
        
        // Apply the method to each element in the array
        for value in self.inner.iter() {
            // Convert AttrValue to a Python object that might have the method
            let py_value = attr_value_to_python_value(py, value)?;
            
            // Check if this value has the requested method
            if py_value.as_ref(py).hasattr(method_name)? {
                // Get the method and call it with the provided arguments
                let method = py_value.as_ref(py).getattr(method_name)?;
                let result = method.call1(args)?;
                
                // Convert result back to AttrValue
                let attr_result = python_value_to_attr_value(result)?;
                results.push(attr_result);
            } else {
                return Err(PyAttributeError::new_err(
                    format!("Elements in array don't have method '{}'", method_name)
                ));
            }
        }
        
        // Create new PyGraphArray with results
        let result_array = GraphArray::from_vec(results);
        Ok(PyGraphArray::from_graph_array(result_array))
    }
}

// Implement display data conversion for PyGraphArray
impl PyGraphArray {
    /// Convert array to display data format expected by Rust formatter
    fn to_display_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        
        // Format as single-column table with just values
        let array_len = self.inner.len();
        let sample_size = std::cmp::min(10, array_len);
        
        // Table shape (show total length, not just sample)
        data.insert("shape".to_string(), serde_json::Value::Array(vec![
            serde_json::Value::Number(serde_json::Number::from(array_len)), 
            serde_json::Value::Number(serde_json::Number::from(1))
        ]));
        
        // Column names - just "value"
        data.insert("columns".to_string(), serde_json::Value::Array(vec![
            serde_json::Value::String("value".to_string())
        ]));
        
        // Create rows data
        let mut rows = Vec::new();
        for i in 0..sample_size {
            let mut row = Vec::new();
            
            // Value column
            if let Some(value) = self.inner.get(i) {
                let json_value = match value {
                    groggy::AttrValue::Text(s) => serde_json::Value::String(s.clone()),
                    groggy::AttrValue::CompactText(s) => serde_json::Value::String(format!("{:?}", s)),
                    groggy::AttrValue::Int(n) => serde_json::Value::Number(serde_json::Number::from(*n)),
                    groggy::AttrValue::SmallInt(n) => serde_json::Value::Number(serde_json::Number::from(*n)),
                    groggy::AttrValue::Float(f) => serde_json::Value::Number(serde_json::Number::from_f64(*f as f64).unwrap_or(serde_json::Number::from(0))),
                    groggy::AttrValue::Bool(b) => serde_json::Value::Bool(*b),
                    groggy::AttrValue::Null => serde_json::Value::Null,
                    // Handle other variants by converting to string representation
                    _ => serde_json::Value::String(format!("{:?}", value)),
                };
                row.push(json_value);
            } else {
                row.push(serde_json::Value::Null);
            }
            
            rows.push(serde_json::Value::Array(row));
        }
        data.insert("data".to_string(), serde_json::Value::Array(rows));
        
        data
    }
}

/// Python wrapper for StatsSummary
#[pyclass(name = "StatsSummary")]
pub struct PyStatsSummary {
    pub inner: StatsSummary,
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
pub struct GraphArrayIterator {
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

    /// Public constructor for use by builder functions
    pub fn from_py_objects(values: Vec<PyObject>) -> PyResult<Self> {
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
}
