//! Neighborhood sampling FFI bindings
//!
//! Python bindings for neighborhood subgraph generation functionality.

use crate::ffi::subgraphs::subgraph::PySubgraph;
use groggy::subgraphs::{NeighborhoodResult, NeighborhoodStats, NeighborhoodSubgraph};
use groggy::traits::{GraphEntity, NeighborhoodOperations, SubgraphOperations};
use groggy::NodeId;
use pyo3::prelude::*;

/// Python wrapper for NeighborhoodSubgraph
#[pyclass(name = "NeighborhoodSubgraph", unsendable)]
#[derive(Clone)]
pub struct PyNeighborhoodSubgraph {
    pub inner: NeighborhoodSubgraph,
}

impl PyNeighborhoodSubgraph {
    /// Get the subgraph object (public method for Rust code)
    pub fn subgraph(&self, _py: Python) -> PyResult<PySubgraph> {
        // Create PySubgraph using the same pattern as connected components
        // Create core Subgraph first, then wrap in PySubgraph
        let core_subgraph = groggy::subgraphs::Subgraph::new(
            self.inner.graph_ref().clone(),
            self.inner.node_set().clone(),
            self.inner.edge_set().clone(),
            format!("neighborhood_hops_{}", self.inner.hops()),
        );
        PySubgraph::from_core_subgraph(core_subgraph)
    }
}

#[pymethods]
impl PyNeighborhoodSubgraph {
    // === NeighborhoodOperations - Specialized methods ===

    #[getter]
    fn central_nodes(&self) -> Vec<NodeId> {
        self.inner.central_nodes().to_vec()
    }

    #[getter]
    fn hops(&self) -> usize {
        self.inner.hops()
    }

    /// Check if a node is a central node
    fn is_central_node(&self, node_id: NodeId) -> bool {
        self.inner.is_central_node(node_id)
    }

    // === SubgraphOperations - Inherited methods ===

    /// Get the subgraph object using the same pattern as connected components
    /// I would rather that the NeighborhoodSubgraph object be the subgraph object itself
    #[pyo3(name = "subgraph")]
    fn subgraph_py(&self, py: Python) -> PyResult<PySubgraph> {
        self.subgraph(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborhoodSubgraph(central_nodes={:?}, hops={}, nodes={}, edges={})",
            self.inner.central_nodes(),
            self.inner.hops(),
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }

    fn __str__(&self) -> String {
        if self.inner.central_nodes().len() == 1 {
            format!(
                "Neighborhood of node {} ({}-hop, {} nodes, {} edges)",
                self.inner.central_nodes()[0],
                self.inner.hops(),
                self.inner.node_count(),
                self.inner.edge_count()
            )
        } else {
            format!(
                "Neighborhood of {} nodes ({}-hop, {} nodes, {} edges)",
                self.inner.central_nodes().len(),
                self.inner.hops(),
                self.inner.node_count(),
                self.inner.edge_count()
            )
        }
    }

    /// Delegate unknown attribute access to the subgraph, so methods like .table() work directly.
    fn __getattr__(&self, name: &str, py: Python) -> PyResult<PyObject> {
        let subgraph = self.subgraph(py)?;
        let subgraph_obj = Py::new(py, subgraph)?;
        subgraph_obj.getattr(py, name)
    }
}

/// Python wrapper for NeighborhoodArray - specialized SubgraphArray with metadata
///
/// This replaces NeighborhoodResult and provides both SubgraphArray operations
/// and neighborhood-specific metadata methods.
#[pyclass(name = "NeighborhoodArray", unsendable)]
#[derive(Clone)]
pub struct PyNeighborhoodArray {
    /// Internal subgraph array
    pub subgraph_array: crate::ffi::storage::subgraph_array::PySubgraphArray,
    /// Metadata from the neighborhood operation
    pub total_neighborhoods: usize,
    pub largest_neighborhood_size: usize,
    pub execution_time_ms: f64,
}

impl PyNeighborhoodArray {
    /// Create a new NeighborhoodArray from NeighborhoodResult
    pub fn from_result(result: NeighborhoodResult) -> PyResult<Self> {
        // Convert NeighborhoodSubgraph to PySubgraph
        let mut subgraphs = Vec::with_capacity(result.neighborhoods.len());
        for neighborhood in result.neighborhoods.iter() {
            let core_subgraph = groggy::subgraphs::Subgraph::new(
                neighborhood.graph_ref(),
                neighborhood.node_set().clone(),
                neighborhood.edge_set().clone(),
                format!("neighborhood_hops_{}", neighborhood.hops()),
            );
            subgraphs.push(crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(core_subgraph)?);
        }

        Ok(Self {
            subgraph_array: crate::ffi::storage::subgraph_array::PySubgraphArray::new(subgraphs),
            total_neighborhoods: result.total_neighborhoods,
            largest_neighborhood_size: result.largest_neighborhood_size,
            execution_time_ms: result.execution_time.as_secs_f64() * 1000.0,
        })
    }

    /// Get the inner subgraph array
    pub fn inner(&self) -> &crate::ffi::storage::subgraph_array::PySubgraphArray {
        &self.subgraph_array
    }
}

#[pymethods]
impl PyNeighborhoodArray {
    // === Metadata methods (specialized for neighborhoods) ===
    
    #[getter]
    fn total_neighborhoods(&self) -> usize {
        self.total_neighborhoods
    }

    #[getter]
    fn largest_neighborhood_size(&self) -> usize {
        self.largest_neighborhood_size
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.execution_time_ms
    }

    // === Delegate to SubgraphArray for standard array operations ===

    fn __len__(&self) -> usize {
        self.subgraph_array.len()
    }

    fn __getitem__(&self, py: Python, key: &PyAny) -> PyResult<PyObject> {
        // Try list of strings (multiple column extraction -> TableArray)
        if let Ok(columns) = key.extract::<Vec<String>>() {
            return self.extract_columns_as_tables(py, &columns);
        }

        // Try string key (single attribute extraction -> ArrayArray)
        if let Ok(attr_name) = key.extract::<String>() {
            return self.extract_node_attribute(py, &attr_name);
        }

        // Try integer index
        if let Ok(index) = key.extract::<isize>() {
            let len = self.subgraph_array.len() as isize;

            // Handle negative indexing
            let actual_index = if index < 0 {
                (len + index) as usize
            } else {
                index as usize
            };

            if actual_index >= self.subgraph_array.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Index {} out of range for neighborhoods with length {}",
                    index,
                    self.subgraph_array.len()
                )));
            }

            return Ok(self.subgraph_array.as_vec()[actual_index].clone().into_py(py));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "NeighborhoodArray indices must be integers, strings, or lists of strings",
        ))
    }

    fn __iter__(slf: PyRef<Self>) -> PyNeighborhoodArrayIterator {
        PyNeighborhoodArrayIterator {
            array: slf.subgraph_array.clone(),
            index: 0,
        }
    }

    fn to_list(&self) -> Vec<crate::ffi::subgraphs::subgraph::PySubgraph> {
        self.subgraph_array.as_vec().clone()
    }

    fn collect(&self) -> Vec<crate::ffi::subgraphs::subgraph::PySubgraph> {
        self.subgraph_array.as_vec().clone()
    }

    fn is_empty(&self) -> bool {
        self.subgraph_array.is_empty_pub()
    }

    // Forward table/nodes_table/edges_table/summary via Python method calls
    fn table(&self, py: Python) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method0(py, "table")?;
        result.extract::<crate::ffi::storage::table_array::PyTableArray>(py)
    }

    fn sample(&self, py: Python, k: usize) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method1(py, "sample", (k,))?;
        result.extract::<crate::ffi::storage::subgraph_array::PySubgraphArray>(py)
    }

    fn group_by(&self, py: Python, attr_name: String, element_type: String) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method1(py, "group_by", (attr_name, element_type))?;
        result.extract::<crate::ffi::storage::subgraph_array::PySubgraphArray>(py)
    }

    fn nodes_table(&self, py: Python) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method0(py, "nodes_table")?;
        result.extract::<crate::ffi::storage::table_array::PyTableArray>(py)
    }

    fn edges_table(&self, py: Python) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method0(py, "edges_table")?;
        result.extract::<crate::ffi::storage::table_array::PyTableArray>(py)
    }

    fn summary(&self, py: Python) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method0(py, "summary")?;
        result.extract::<crate::ffi::storage::table::PyBaseTable>(py)
    }

    #[getter]
    fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let viz_obj = array_obj.getattr(py, "viz")?;
        viz_obj.extract::<Py<crate::ffi::viz_accessor::VizAccessor>>(py)
    }

    fn map(&self, py: Python, func: PyObject) -> PyResult<crate::ffi::storage::array::PyBaseArray> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method1(py, "map", (func,))?;
        result.extract::<crate::ffi::storage::array::PyBaseArray>(py)
    }

    fn merge(&self, py: Python) -> PyResult<crate::ffi::api::graph::PyGraph> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let result = array_obj.call_method0(py, "merge")?;
        result.extract::<crate::ffi::api::graph::PyGraph>(py)
    }

    #[pyo3(signature = (
        node_aggs = None,
        edge_aggs = None,
        edge_strategy = "aggregate",
        node_strategy = "extract",
        preset = None,
        include_edge_count = true,
        mark_entity_type = true,
        entity_type = "meta",
        allow_missing_attributes = true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn collapse(
        &self,
        py: Python,
        node_aggs: Option<&PyAny>,
        edge_aggs: Option<&PyAny>,
        edge_strategy: &str,
        node_strategy: &str,
        preset: Option<String>,
        include_edge_count: bool,
        mark_entity_type: bool,
        entity_type: &str,
        allow_missing_attributes: bool,
    ) -> PyResult<Vec<Py<crate::ffi::entities::meta_node::PyMetaNode>>> {
        let array_obj = Py::new(py, self.subgraph_array.clone())?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(na) = node_aggs {
            kwargs.set_item("node_aggs", na)?;
        }
        if let Some(ea) = edge_aggs {
            kwargs.set_item("edge_aggs", ea)?;
        }
        kwargs.set_item("edge_strategy", edge_strategy)?;
        kwargs.set_item("node_strategy", node_strategy)?;
        if let Some(p) = preset {
            kwargs.set_item("preset", p)?;
        }
        kwargs.set_item("include_edge_count", include_edge_count)?;
        kwargs.set_item("mark_entity_type", mark_entity_type)?;
        kwargs.set_item("entity_type", entity_type)?;
        kwargs.set_item("allow_missing_attributes", allow_missing_attributes)?;
        
        let result = array_obj.call_method(py, "collapse", (), Some(kwargs))?;
        result.extract::<Vec<Py<crate::ffi::entities::meta_node::PyMetaNode>>>(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborhoodArray({} neighborhoods, largest_size={}, time={:.2}ms)",
            self.total_neighborhoods,
            self.largest_neighborhood_size,
            self.execution_time_ms
        )
    }
}

// Helper methods for PyNeighborhoodArray (not exposed to Python)
impl PyNeighborhoodArray {
    // Helper methods borrowed from SubgraphArray
    fn extract_columns_as_tables(&self, py: Python, columns: &[String]) -> PyResult<PyObject> {
        use groggy::storage::array::BaseArray;
        use groggy::storage::table::BaseTable;
        use std::collections::HashMap;

        let mut tables = Vec::new();

        for subgraph in self.subgraph_array.as_vec().iter() {
            let graph_ref = subgraph.inner.graph();
            let graph_borrow = graph_ref.borrow();

            let mut column_map = HashMap::new();

            for col_name in columns {
                let mut col_values = Vec::new();

                for &node_id in subgraph.inner.nodes() {
                    if let Ok(Some(attr_value)) = graph_borrow.get_node_attr(node_id, col_name) {
                        col_values.push(attr_value);
                    } else {
                        col_values.push(groggy::types::AttrValue::Null);
                    }
                }

                column_map.insert(col_name.clone(), BaseArray::from_attr_values(col_values));
            }

            match BaseTable::from_columns(column_map) {
                Ok(table) => {
                    let py_table = crate::ffi::storage::table::PyBaseTable::from_table(table);
                    tables.push(py_table.into_py(py));
                }
                Err(_) => continue,
            }
        }

        let table_array = crate::ffi::storage::table_array::PyTableArray::new(tables);
        Ok(table_array.into_py(py))
    }

    fn extract_node_attribute(&self, py: Python, attr_name: &str) -> PyResult<PyObject> {
        use groggy::storage::array::{ArrayArray, BaseArray};
        use groggy::types::AttrValue;

        let mut arrays = Vec::new();
        let mut keys = Vec::new();
        let attr_name_string = attr_name.to_string();

        for (idx, subgraph) in self.subgraph_array.as_vec().iter().enumerate() {
            let graph_ref = subgraph.inner.graph();
            let graph_borrow = graph_ref.borrow();

            let node_values: Vec<AttrValue> = subgraph
                .inner
                .nodes()
                .iter()
                .filter_map(|&node_id| {
                    graph_borrow
                        .get_node_attr(node_id, &attr_name_string)
                        .ok()
                        .flatten()
                })
                .collect();

            if node_values.is_empty() {
                let edge_values: Vec<AttrValue> = subgraph
                    .inner
                    .edges()
                    .iter()
                    .filter_map(|&edge_id| {
                        graph_borrow
                            .get_edge_attr(edge_id, &attr_name_string)
                            .ok()
                            .flatten()
                    })
                    .collect();

                if edge_values.is_empty() {
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "Attribute '{}' not found in neighborhood {}",
                        attr_name, idx
                    )));
                }

                arrays.push(BaseArray::from_attr_values(edge_values));
                keys.push(format!("neighborhood_{}", idx));
                continue;
            }

            arrays.push(BaseArray::from_attr_values(node_values));
            keys.push(format!("neighborhood_{}", idx));
        }

        let array_array = ArrayArray::with_keys(arrays, keys)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let py_array_array = crate::PyArrayArray::from_array_array(array_array);
        Ok(py_array_array.into_py(py))
    }
}

/// Iterator for PyNeighborhoodArray
#[pyclass(unsendable)]
pub struct PyNeighborhoodArrayIterator {
    array: crate::ffi::storage::subgraph_array::PySubgraphArray,
    index: usize,
}

#[pymethods]
impl PyNeighborhoodArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> Option<crate::ffi::subgraphs::subgraph::PySubgraph> {
        if self.index < self.array.len() {
            let result = self.array.as_vec()[self.index].clone();
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Python wrapper for NeighborhoodResult (backward compatibility)
///
/// This is kept for backward compatibility but now wraps PyNeighborhoodArray
#[pyclass(name = "NeighborhoodResult", unsendable)]
#[derive(Clone)]
pub struct PyNeighborhoodResult {
    pub inner: NeighborhoodResult,
}

#[pymethods]
impl PyNeighborhoodResult {
    #[getter]
    fn neighborhoods(&self) -> Vec<PyNeighborhoodSubgraph> {
        self.inner
            .neighborhoods
            .iter()
            .map(|n| PyNeighborhoodSubgraph { inner: n.clone() })
            .collect()
    }

    #[getter]
    fn total_neighborhoods(&self) -> usize {
        self.inner.total_neighborhoods
    }

    #[getter]
    fn largest_neighborhood_size(&self) -> usize {
        self.inner.largest_neighborhood_size
    }

    #[getter]
    fn execution_time_ms(&self) -> f64 {
        self.inner.execution_time.as_secs_f64() * 1000.0
    }

    fn __len__(&self) -> usize {
        self.inner.neighborhoods.len()
    }

    fn __getitem__(&self, index: usize) -> PyResult<PyNeighborhoodSubgraph> {
        self.inner
            .neighborhoods
            .get(index)
            .map(|n| PyNeighborhoodSubgraph { inner: n.clone() })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "Index {} out of range for neighborhoods with length {}",
                    index,
                    self.inner.neighborhoods.len()
                ))
            })
    }

    fn __iter__(slf: PyRef<Self>) -> PyNeighborhoodResultIterator {
        PyNeighborhoodResultIterator {
            inner: slf.inner.clone(),
            index: 0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborhoodResult({} neighborhoods, largest_size={}, time={:.2}ms)",
            self.inner.total_neighborhoods,
            self.inner.largest_neighborhood_size,
            self.execution_time_ms()
        )
    }
}

/// Iterator for NeighborhoodResult
#[pyclass(unsendable)]
pub struct PyNeighborhoodResultIterator {
    inner: NeighborhoodResult,
    index: usize,
}

#[pymethods]
impl PyNeighborhoodResultIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<PyNeighborhoodSubgraph> {
        if slf.index < slf.inner.neighborhoods.len() {
            let result = PyNeighborhoodSubgraph {
                inner: slf.inner.neighborhoods[slf.index].clone(),
            };
            slf.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Python wrapper for NeighborhoodStats
#[pyclass(name = "NeighborhoodStats")]
#[derive(Clone)]
pub struct PyNeighborhoodStats {
    pub inner: NeighborhoodStats,
}

#[pymethods]
impl PyNeighborhoodStats {
    #[getter]
    fn total_neighborhoods(&self) -> usize {
        self.inner.total_neighborhoods
    }

    #[getter]
    fn total_nodes_sampled(&self) -> usize {
        self.inner.total_nodes_sampled
    }

    #[getter]
    fn total_time_ms(&self) -> f64 {
        self.inner.total_time.as_secs_f64() * 1000.0
    }

    #[getter]
    fn operation_counts(&self) -> std::collections::HashMap<String, usize> {
        self.inner.operation_counts.clone()
    }

    /// Get average nodes per neighborhood
    fn avg_nodes_per_neighborhood(&self) -> f64 {
        if self.inner.total_neighborhoods > 0 {
            self.inner.total_nodes_sampled as f64 / self.inner.total_neighborhoods as f64
        } else {
            0.0
        }
    }

    /// Get average time per neighborhood in milliseconds
    fn avg_time_per_neighborhood_ms(&self) -> f64 {
        if self.inner.total_neighborhoods > 0 {
            self.total_time_ms() / self.inner.total_neighborhoods as f64
        } else {
            0.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborhoodStats(neighborhoods={}, nodes={}, time={:.2}ms, avg={:.1} nodes/nbh)",
            self.inner.total_neighborhoods,
            self.inner.total_nodes_sampled,
            self.total_time_ms(),
            self.avg_nodes_per_neighborhood()
        )
    }
}
