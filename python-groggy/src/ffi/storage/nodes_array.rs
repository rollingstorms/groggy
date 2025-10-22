use crate::ffi::storage::accessors::PyNodesAccessor;
use crate::ffi::storage::table_array::PyTableArray;
use crate::ffi::viz_accessor::VizAccessor;
use groggy::storage::array::{ArrayIterator, BaseArray};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// NodesArray: Collection of NodesAccessor objects with delegation to BaseArray
/// Provides basic array operations and graph-specific transformations
#[pyclass(name = "NodesArray", unsendable)]
pub struct PyNodesArray {
    // Delegates to BaseArray for basic operations
    base: BaseArray<PyNodesAccessor>,
}

#[pymethods]
impl PyNodesArray {
    /// Create a new NodesArray from a vector of NodesAccessor objects
    #[new]
    pub fn new(nodes: Vec<PyNodesAccessor>) -> Self {
        Self {
            base: BaseArray::new(nodes),
        }
    }

    // BaseArray delegation - basic operations

    /// Get the number of NodesAccessor objects
    fn __len__(&self) -> usize {
        self.base.len()
    }

    /// Check if the array is empty
    fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Get a NodesAccessor by index
    fn __getitem__(&self, index: isize) -> PyResult<PyNodesAccessor> {
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

    /// Get the first NodesAccessor
    fn first(&self) -> Option<PyNodesAccessor> {
        self.base.first().cloned()
    }

    /// Get the last NodesAccessor  
    fn last(&self) -> Option<PyNodesAccessor> {
        self.base.last().cloned()
    }

    /// Check if the array contains a specific NodesAccessor
    fn contains(&self, item: &PyNodesAccessor) -> bool {
        // For simplicity, compare by node count since PyNodesAccessor doesn't implement PartialEq
        let target_count = item.node_count();
        self.base
            .iter()
            .any(|accessor| accessor.node_count() == target_count)
    }

    /// Convert to Python list
    fn to_list(&self) -> Vec<PyNodesAccessor> {
        self.base.clone_vec()
    }

    // Domain-specific operations (apply-on-each pattern)

    /// Apply table() to each NodesAccessor and return TableArray
    fn table(&self) -> PyResult<PyTableArray> {
        let mut tables = Vec::new();

        // Apply table() to each NodesAccessor in the array
        for accessor in self.base.iter() {
            let node_table = accessor.table()?;
            Python::with_gil(|py| {
                tables.push(node_table.into_py(py));
            });
        }

        Ok(PyTableArray::new(tables))
    }

    /// Filter NodesAccessor objects using a Python predicate function
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

    /// Filter NodesAccessor objects by node count threshold
    fn filter_by_size(&self, min_size: usize) -> Self {
        let filtered: Vec<PyNodesAccessor> = self
            .base
            .iter()
            .filter(|accessor| accessor.node_count() >= min_size)
            .cloned()
            .collect();

        Self::new(filtered)
    }

    // Reduce operations (many-to-one)

    /// Combine all NodesAccessor objects into a single unified NodesAccessor
    fn union(&self) -> PyResult<PyNodesAccessor> {
        if self.base.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot create union of empty NodesArray",
            ));
        }

        // For now, return the first accessor as a placeholder
        // In full implementation, would properly merge all node sets
        Ok(self.base.first().unwrap().clone())
    }

    /// Get total node count across all NodesAccessor objects
    fn total_node_count(&self) -> usize {
        self.base.iter().map(|accessor| accessor.node_count()).sum()
    }

    /// Get statistics about the NodesAccessor collection
    fn stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let node_counts: Vec<usize> = self
                .base
                .iter()
                .map(|accessor| accessor.node_count())
                .collect();

            let total = node_counts.iter().sum::<usize>();
            let count = node_counts.len();
            let min = node_counts.iter().min().copied().unwrap_or(0);
            let max = node_counts.iter().max().copied().unwrap_or(0);
            let mean = if count > 0 {
                total as f64 / count as f64
            } else {
                0.0
            };

            let stats_dict = pyo3::types::PyDict::new(py);
            stats_dict.set_item("count", count)?;
            stats_dict.set_item("total_nodes", total)?;
            stats_dict.set_item("min_nodes", min)?;
            stats_dict.set_item("max_nodes", max)?;
            stats_dict.set_item("mean_nodes", mean)?;

            Ok(stats_dict.into())
        })
    }

    /// Create an iterator for method chaining
    fn iter(&self) -> PyNodesArrayIterator {
        PyNodesArrayIterator::new(self.base.clone())
    }

    // Python dunder methods

    fn __repr__(&self) -> String {
        format!("NodesArray(length={})", self.base.len())
    }

    fn __str__(&self) -> String {
        if self.base.is_empty() {
            "NodesArray[]".to_string()
        } else {
            let total_nodes: usize = self.base.iter().map(|accessor| accessor.node_count()).sum();
            format!(
                "NodesArray[{} accessors, {} total nodes]",
                self.base.len(),
                total_nodes
            )
        }
    }

    /// Launch interactive visualization for this NodesArray
    ///
    /// Converts the NodesArray to a table view and launches visualization
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
        // Convert NodesArray to table for visualization via delegation
        // For now, delegate through the base interactive method
        // TODO: Implement proper table conversion once table array structure is clarified
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "NodesArray.interactive() not yet implemented - use .table().interactive_viz() instead",
        ))
    }
}

/// Iterator for NodesArray that enables method chaining
#[pyclass(name = "NodesArrayIterator", unsendable)]
pub struct PyNodesArrayIterator {
    inner: ArrayIterator<PyNodesAccessor>,
}

impl PyNodesArrayIterator {
    pub fn new(base: BaseArray<PyNodesAccessor>) -> Self {
        Self {
            inner: ArrayIterator::new(base.clone_vec()),
        }
    }
}

#[pymethods]
impl PyNodesArrayIterator {
    /// Apply table() to each NodesAccessor in the iterator
    fn table(&mut self) -> PyResult<PyTableArray> {
        let mut tables = Vec::new();

        // Convert iterator to vector and iterate through NodesAccessor objects
        let accessors = self.inner.clone().into_vec();
        for accessor in accessors {
            let node_table = accessor.table()?;
            Python::with_gil(|py| {
                tables.push(node_table.into_py(py));
            });
        }

        Ok(PyTableArray::new(tables))
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

    /// Collect the iterator back into a NodesArray
    fn collect(&mut self) -> PyNodesArray {
        let elements = self.inner.clone().into_vec();
        PyNodesArray::new(elements)
    }

    /// Get the current length of the iterator
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}
