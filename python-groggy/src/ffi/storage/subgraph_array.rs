//! PySubgraphArray - Specialized array for PySubgraph objects
//!
//! Provides a typed container for collections of PySubgraph objects with full ArrayOps support

use crate::ffi::subgraphs::subgraph::PySubgraph;
use groggy::storage::array::{ArrayOps, ArrayIterator};
use pyo3::prelude::*;
use std::sync::Arc;

/// Specialized array for PySubgraph objects
#[pyclass(name = "SubgraphArray", unsendable)]
#[derive(Clone)]
pub struct PySubgraphArray {
    /// Internal storage of subgraphs
    inner: Arc<Vec<PySubgraph>>,
}

impl PySubgraphArray {
    /// Create new PySubgraphArray from vector of subgraphs
    pub fn new(subgraphs: Vec<PySubgraph>) -> Self {
        Self {
            inner: Arc::new(subgraphs),
        }
    }
    
    /// Create from Arc<Vec<PySubgraph>> for zero-copy sharing
    pub fn from_arc(subgraphs: Arc<Vec<PySubgraph>>) -> Self {
        Self { inner: subgraphs }
    }
}

#[pymethods]
impl PySubgraphArray {
    /// Get the number of subgraphs
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if the array is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Get subgraph at index with bounds checking
    fn __getitem__(&self, index: isize) -> PyResult<PySubgraph> {
        let len = self.inner.len() as isize;
        
        // Handle negative indexing
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };
        
        if actual_index >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Subgraph index {} out of range (0-{})",
                index,
                self.inner.len() - 1
            )));
        }
        
        Ok(self.inner[actual_index].clone())
    }
    
    /// Iterate over subgraphs
    fn __iter__(slf: PyRef<Self>) -> PySubgraphArrayIterator {
        PySubgraphArrayIterator {
            array: slf.into(),
            index: 0,
        }
    }
    
    /// Convert to Python list
    fn to_list(&self) -> Vec<PySubgraph> {
        self.inner.as_ref().clone()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("SubgraphArray({} subgraphs)", self.inner.len())
    }
    
    /// Collect all subgraphs into a Python list (for compatibility with iterator patterns)
    fn collect(&self) -> Vec<PySubgraph> {
        self.to_list()
    }
    
    /// Create iterator for method chaining
    fn iter(&self) -> PySubgraphArrayChainIterator {
        PySubgraphArrayChainIterator {
            inner: ArrayIterator::new(self.inner.as_ref().clone()),
        }
    }
    
    /// Apply table() to all subgraphs and return PyTableArray
    fn table(&self) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let mut tables = Vec::new();
        
        Python::with_gil(|py| {
            for subgraph in self.inner.iter() {
                match subgraph.table(py) {
                    Ok(table) => tables.push(table),
                    Err(_) => continue, // Skip failed subgraphs
                }
            }
        });
        
        Ok(crate::ffi::storage::table_array::PyTableArray::new(tables))
    }
    
    /// Apply sample(k) to all subgraphs
    fn sample(&self, k: usize) -> PyResult<PySubgraphArray> {
        let mut sampled = Vec::new();
        
        for subgraph in self.inner.iter() {
            match subgraph.sample(k) {
                Ok(sampled_subgraph) => sampled.push(sampled_subgraph),
                Err(_) => continue, // Skip failed subgraphs
            }
        }
        
        Ok(PySubgraphArray::new(sampled))
    }

    /// Apply group_by to all subgraphs and flatten results
    ///
    /// Args:
    ///     attr_name: Name of the attribute to group by
    ///     element_type: Either 'nodes' or 'edges' to specify what to group
    ///
    /// Returns:
    ///     SubgraphArray: Flattened array of all grouped subgraphs
    ///
    /// Example:
    ///     nested_groups = subgraph_array.group_by('department', 'nodes')
    ///     # Returns all department groups from all input subgraphs
    pub fn group_by(&self, attr_name: String, element_type: String) -> PyResult<PySubgraphArray> {
        let mut all_groups = Vec::new();
        
        // Apply group_by to each subgraph and flatten results
        for subgraph in self.inner.iter() {
            match subgraph.group_by(attr_name.clone(), element_type.clone()) {
                Ok(groups) => {
                    // Flatten the groups into our result vector
                    for i in 0..groups.len() {
                        if let Some(group_subgraph) = groups.get(i) {
                            all_groups.push(group_subgraph.clone());
                        }
                    }
                },
                Err(_) => continue, // Skip failed subgraphs
            }
        }
        
        Ok(PySubgraphArray::new(all_groups))
    }
}

// Implement ArrayOps for integration with core array system
impl ArrayOps<PySubgraph> for PySubgraphArray {
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn get(&self, index: usize) -> Option<&PySubgraph> {
        self.inner.get(index)
    }
    
    fn iter(&self) -> ArrayIterator<PySubgraph> 
    where 
        PySubgraph: Clone + 'static 
    {
        ArrayIterator::new(self.inner.as_ref().clone())
    }
}

// From implementations for easy conversion
impl From<Vec<PySubgraph>> for PySubgraphArray {
    fn from(subgraphs: Vec<PySubgraph>) -> Self {
        Self::new(subgraphs)
    }
}

impl From<PySubgraphArray> for Vec<PySubgraph> {
    fn from(array: PySubgraphArray) -> Self {
        Arc::try_unwrap(array.inner).unwrap_or_else(|arc| arc.as_ref().clone())
    }
}

/// Python iterator for PySubgraphArray
#[pyclass]
pub struct PySubgraphArrayIterator {
    array: Py<PySubgraphArray>,
    index: usize,
}

#[pymethods]
impl PySubgraphArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<PySubgraph>> {
        let array = self.array.borrow(py);
        if self.index < array.inner.len() {
            let result = array.inner[self.index].clone();
            self.index += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}

/// Chainable iterator for PySubgraphArray that supports method forwarding
#[pyclass(name = "SubgraphArrayIterator", unsendable)]
pub struct PySubgraphArrayChainIterator {
    inner: ArrayIterator<PySubgraph>,
}

#[pymethods]
impl PySubgraphArrayChainIterator {
    /// Apply table() to each subgraph and return list of tables
    fn table(&mut self) -> PyResult<Vec<PyObject>> {
        let subgraphs = self.inner.clone().into_vec();
        let mut tables = Vec::new();
        
        Python::with_gil(|py| {
            for subgraph in subgraphs {
                match subgraph.table(py) {
                    Ok(table) => tables.push(table),
                    Err(_) => continue, // Skip failed subgraphs
                }
            }
        });
        
        Ok(tables)
    }
    
    /// Apply sample(k) to each subgraph
    fn sample(&mut self, k: usize) -> PyResult<PySubgraphArray> {
        let subgraphs = self.inner.clone().into_vec();
        let mut sampled = Vec::new();
        
        for subgraph in subgraphs {
            match subgraph.sample(k) {
                Ok(sampled_subgraph) => sampled.push(sampled_subgraph),
                Err(_) => continue, // Skip failed subgraphs
            }
        }
        
        Ok(PySubgraphArray::new(sampled))
    }
    
    /// Materialize iterator back into PySubgraphArray
    fn collect(&mut self) -> PyResult<PySubgraphArray> {
        let subgraphs = self.inner.clone().into_vec();
        Ok(PySubgraphArray::new(subgraphs))
    }
    
    /// Apply filter to subgraphs and return filtered iterator
    fn filter(&mut self, py: Python, predicate: PyObject) -> PyResult<Self> {
        let subgraphs = self.inner.clone().into_vec();
        let filtered: Vec<PySubgraph> = subgraphs.into_iter().filter_map(|subgraph| {
            // Apply Python predicate function to each subgraph
            match predicate.call1(py, (subgraph.clone(),)) {
                Ok(result) => {
                    if result.is_true(py).unwrap_or(false) {
                        Some(subgraph)
                    } else {
                        None
                    }
                },
                Err(_) => None, // Skip on error
            }
        }).collect();
        
        Ok(Self {
            inner: ArrayIterator::new(filtered),
        })
    }
    
    /// Take first n elements
    fn take(&mut self, n: usize) -> PyResult<Self> {
        let taken_elements = self.inner.clone().take(n);
        Ok(Self { inner: taken_elements })
    }
    
    /// Skip first n elements  
    fn skip(&mut self, n: usize) -> PyResult<Self> {
        let skipped_elements = self.inner.clone().skip(n);
        Ok(Self { inner: skipped_elements })
    }
}