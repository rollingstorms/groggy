use crate::ffi::storage::accessors::PyEdgesAccessor;
use crate::ffi::storage::nodes_array::PyNodesArray;
use crate::ffi::storage::table_array::PyTableArray;
use crate::ffi::viz_accessor::VizAccessor;
use groggy::storage::array::{ArrayIterator, BaseArray};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// EdgesArray: Collection of EdgesAccessor objects with delegation to BaseArray
/// Provides basic array operations and graph-specific transformations
#[pyclass(name = "EdgesArray", unsendable)]
pub struct PyEdgesArray {
    // Delegates to BaseArray for basic operations
    base: BaseArray<PyEdgesAccessor>,
}

#[pymethods]
impl PyEdgesArray {
    /// Create a new EdgesArray from a vector of EdgesAccessor objects
    #[new]
    pub fn new(edges: Vec<PyEdgesAccessor>) -> Self {
        Self {
            base: BaseArray::new(edges),
        }
    }

    // BaseArray delegation - basic operations

    /// Get the number of EdgesAccessor objects
    fn __len__(&self) -> usize {
        self.base.len()
    }

    /// Check if the array is empty
    fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Get an EdgesAccessor by index
    fn __getitem__(&self, index: isize) -> PyResult<PyEdgesAccessor> {
        let len = self.base.len();
        if len == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "index out of range: array is empty",
            ));
        }

        let actual_index = if index < 0 {
            let positive_index = len as isize + index;
            if positive_index < 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "index {} is out of range for array of length {}",
                    index, len
                )));
            }
            positive_index as usize
        } else {
            index as usize
        };

        self.base.get(actual_index).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "index {} is out of range for array of length {}",
                index, len
            ))
        })
    }

    /// Get the first EdgesAccessor
    fn first(&self) -> Option<PyEdgesAccessor> {
        self.base.first().cloned()
    }

    /// Get the last EdgesAccessor
    fn last(&self) -> Option<PyEdgesAccessor> {
        self.base.last().cloned()
    }

    /// Check if the array contains a specific EdgesAccessor
    fn contains(&self, item: &PyEdgesAccessor) -> bool {
        // For simplicity, compare by edge count since PyEdgesAccessor doesn't implement PartialEq
        let target_count = item.edge_count();
        self.base
            .iter()
            .any(|accessor| accessor.edge_count() == target_count)
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<PyEdgesAccessor> {
        self.base.clone_vec()
    }

    // Domain-specific operations (apply-on-each pattern)

    /// Apply table() to each EdgesAccessor in the array
    fn table(&self) -> PyResult<PyTableArray> {
        let mut tables = Vec::new();

        // Iterate through EdgesAccessor objects and apply table()
        for accessor in self.base.iter() {
            let edge_table = accessor.table()?;
            Python::with_gil(|py| {
                tables.push(edge_table.into_py(py));
            });
        }

        Ok(PyTableArray::new(tables))
    }

    /// Get source and target nodes from all EdgesAccessor objects
    fn nodes(&self) -> PyResult<PyNodesArray> {
        let mut nodes_accessors = Vec::new();

        for accessor in self.base.iter() {
            match accessor.nodes() {
                Ok(nodes_accessor) => nodes_accessors.push(nodes_accessor),
                Err(_) => continue, // Skip failed accessors
            }
        }

        Ok(PyNodesArray::new(nodes_accessors))
    }

    /// Filter EdgesAccessor objects using a Python predicate function
    fn filter(&self, predicate: &PyAny) -> PyResult<Self> {
        let mut filtered = Vec::new();

        Python::with_gil(|_py| {
            for accessor in self.base.iter() {
                match predicate.call1((accessor.clone(),)) {
                    Ok(result) => {
                        if result.is_true().unwrap_or(false) {
                            filtered.push(accessor.clone());
                        }
                    }
                    Err(_) => continue, // Skip on error
                }
            }
        });

        Ok(Self::new(filtered))
    }

    /// Filter EdgesAccessor objects by edge count threshold
    fn filter_by_size(&self, min_size: usize) -> Self {
        let filtered: Vec<PyEdgesAccessor> = self
            .base
            .iter()
            .filter(|accessor| accessor.edge_count() >= min_size)
            .cloned()
            .collect();

        Self::new(filtered)
    }

    /// Filter EdgesAccessor objects by weight threshold (if edges have weights)
    fn filter_by_weight(&self, _min_weight: f64) -> PyResult<Self> {
        let mut filtered = Vec::new();

        // For now, just return all edges as placeholder
        // In full implementation, would check edge weights
        for accessor in self.base.iter() {
            filtered.push(accessor.clone());
        }

        Ok(Self::new(filtered))
    }

    // Reduce operations (many-to-one)

    /// Combine all EdgesAccessor objects into a single unified EdgesAccessor
    fn union(&self) -> PyResult<PyEdgesAccessor> {
        if self.base.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot create union of empty EdgesArray",
            ));
        }

        // For now, return the first accessor as a placeholder
        // In full implementation, would properly merge all edge sets
        Ok(self.base.first().unwrap().clone())
    }

    /// Get total edge count across all EdgesAccessor objects
    fn total_edge_count(&self) -> usize {
        self.base.iter().map(|accessor| accessor.edge_count()).sum()
    }

    /// Get statistics about the EdgesAccessor collection
    fn stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let edge_counts: Vec<usize> = self
                .base
                .iter()
                .map(|accessor| accessor.edge_count())
                .collect();

            let total = edge_counts.iter().sum::<usize>();
            let count = edge_counts.len();
            let min = edge_counts.iter().min().copied().unwrap_or(0);
            let max = edge_counts.iter().max().copied().unwrap_or(0);
            let mean = if count > 0 {
                total as f64 / count as f64
            } else {
                0.0
            };

            let stats_dict = pyo3::types::PyDict::new(py);
            stats_dict.set_item("count", count)?;
            stats_dict.set_item("total_edges", total)?;
            stats_dict.set_item("min_edges", min)?;
            stats_dict.set_item("max_edges", max)?;
            stats_dict.set_item("mean_edges", mean)?;

            Ok(stats_dict.into())
        })
    }

    /// Create an iterator for method chaining
    fn iter(&self) -> PyEdgesArrayIterator {
        PyEdgesArrayIterator::new(self.base.clone())
    }

    // Python dunder methods

    fn __repr__(&self) -> String {
        format!("EdgesArray(length={})", self.base.len())
    }

    fn __str__(&self) -> String {
        if self.base.is_empty() {
            "EdgesArray[]".to_string()
        } else {
            let total_edges: usize = self.base.iter().map(|accessor| accessor.edge_count()).sum();
            format!(
                "EdgesArray[{} accessors, {} total edges]",
                self.base.len(),
                total_edges
            )
        }
    }

    /// Launch interactive visualization for this EdgesArray
    ///
    /// Converts the EdgesArray to a table view and launches visualization
    ///
    /// # Arguments
    /// * `port` - Optional port number (0 for auto-assign)
    /// * `layout` - Layout algorithm: \"force-directed\", \"circular\", \"grid\", \"hierarchical\"
    /// * `theme` - Visual theme: \"light\", \"dark\", \"publication\", \"minimal\"
    /// * `width` - Canvas width in pixels
    /// * `height` - Canvas height in pixels
    ///
    /// # Returns
    /// VizAccessor for launching interactive visualization
    pub fn interactive(
        &self,
        _port: Option<u16>,
        _layout: Option<String>,
        _theme: Option<String>,
        _width: Option<u32>,
        _height: Option<u32>,
    ) -> PyResult<VizAccessor> {
        // Convert EdgesArray to table for visualization via delegation
        // For now, delegate through the base interactive method
        // TODO: Implement proper table conversion once table array structure is clarified
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "EdgesArray.interactive() not yet implemented - use .table().interactive_viz() instead",
        ))
    }
}

/// Iterator for EdgesArray that enables method chaining
#[pyclass(name = "EdgesArrayIterator", unsendable)]
pub struct PyEdgesArrayIterator {
    inner: ArrayIterator<PyEdgesAccessor>,
}

impl PyEdgesArrayIterator {
    pub fn new(base: BaseArray<PyEdgesAccessor>) -> Self {
        Self {
            inner: ArrayIterator::new(base.clone_vec()),
        }
    }
}

#[pymethods]
impl PyEdgesArrayIterator {
    /// Apply table() to each EdgesAccessor in the iterator
    fn table(&mut self) -> PyResult<PyTableArray> {
        let mut tables = Vec::new();

        // Convert iterator to vector and iterate through EdgesAccessor objects
        let accessors = self.inner.clone().into_vec();
        for accessor in accessors {
            let edge_table = accessor.table()?;
            Python::with_gil(|py| {
                tables.push(edge_table.into_py(py));
            });
        }

        Ok(PyTableArray::new(tables))
    }

    /// Get nodes from each EdgesAccessor in the iterator
    fn nodes(&mut self) -> PyResult<PyNodesArray> {
        let elements = self.inner.clone().into_vec();
        let mut nodes_accessors = Vec::new();

        for accessor in elements {
            match accessor.nodes() {
                Ok(nodes_accessor) => nodes_accessors.push(nodes_accessor),
                Err(_) => continue,
            }
        }

        Ok(PyNodesArray::new(nodes_accessors))
    }

    /// Filter the iterator using a Python predicate function
    fn filter(&mut self, predicate: &PyAny) -> PyResult<Self> {
        let elements = self.inner.clone().into_vec();
        let mut filtered = Vec::new();

        Python::with_gil(|_py| {
            for accessor in elements {
                match predicate.call1((accessor.clone(),)) {
                    Ok(result) => {
                        if result.is_true().unwrap_or(false) {
                            filtered.push(accessor);
                        }
                    }
                    Err(_) => continue,
                }
            }
        });

        Ok(Self::new(BaseArray::new(filtered)))
    }

    /// Filter by weight threshold
    fn filter_by_weight(&mut self, _min_weight: f64) -> PyResult<Self> {
        let elements = self.inner.clone().into_vec();
        let mut filtered = Vec::new();

        // For now, just return all edges as placeholder
        // In full implementation, would check edge weights
        for accessor in elements {
            filtered.push(accessor);
        }

        Ok(Self::new(BaseArray::new(filtered)))
    }

    /// Collect the iterator back into an EdgesArray
    fn collect(&mut self) -> PyEdgesArray {
        let elements = self.inner.clone().into_vec();
        PyEdgesArray::new(elements)
    }

    /// Get the current length of the iterator
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
