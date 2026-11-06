//! PySubgraphArray - Specialized array for PySubgraph objects
//!
//! Provides a typed container for collections of PySubgraph objects with full ArrayOps support

use crate::ffi::entities::meta_node::PyMetaNode;
use crate::ffi::subgraphs::subgraph::PySubgraph;
use groggy::storage::array::{ArrayIterator, ArrayOps};
use pyo3::prelude::*;
use std::sync::Arc;

/// Specialized array for PySubgraph objects
///
/// Note: Uses Arc<Vec<PySubgraph>> for zero-copy sharing. PySubgraph is marked
/// as `unsendable` so this won't be used across threads. Arc is used here for
/// efficient cloning/sharing within a single thread.
#[allow(clippy::arc_with_non_send_sync)]
#[pyclass(name = "SubgraphArray", unsendable)]
#[derive(Clone)]
pub struct PySubgraphArray {
    /// Internal storage of subgraphs
    inner: Arc<Vec<PySubgraph>>,
}

impl PySubgraphArray {
    /// Create new PySubgraphArray from vector of subgraphs
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(subgraphs: Vec<PySubgraph>) -> Self {
        Self {
            inner: Arc::new(subgraphs),
        }
    }

    /// Create from Arc<Vec<PySubgraph>> for zero-copy sharing
    pub fn from_arc(subgraphs: Arc<Vec<PySubgraph>>) -> Self {
        Self { inner: subgraphs }
    }

    /// Get length of array (public accessor)
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty (public accessor)
    pub fn is_empty_pub(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get reference to inner vec (public accessor)
    pub fn as_vec(&self) -> &Vec<PySubgraph> {
        &self.inner
    }

    /// Marshal source-bound selections to one canonical immutable union.
    fn merged_source(&self, py: Python) -> PyResult<Option<groggy::GraphView>> {
        let views = self
            .inner
            .iter()
            .map(|selection| selection.inner.pin_view())
            .collect::<groggy::GraphResult<Vec<_>>>()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        py.allow_threads(|| groggy::GraphView::merge_all(&views))
            .map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    /// Helper: Extract multiple columns from each subgraph's nodes and return as TableArray
    fn extract_columns_as_tables(&self, py: Python, columns: &[String]) -> PyResult<PyObject> {
        use groggy::storage::table::BaseTable;
        use std::collections::HashMap;
        let mut tables = Vec::with_capacity(self.inner.len());
        for subgraph in self.inner.iter() {
            let view = subgraph
                .inner
                .pin_view()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let table = py
                .allow_threads(|| {
                    let nodes = view.node_accessor();
                    BaseTable::from_columns(
                        columns
                            .iter()
                            .map(|name| (name.clone(), nodes.column(name)))
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            tables.push(crate::ffi::storage::table::PyBaseTable::from_table(table).into_py(py));
        }
        Ok(crate::ffi::storage::table_array::PyTableArray::new(tables).into_py(py))
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

    /// Get subgraph at index or extract node attributes
    ///
    /// Supports three modes:
    /// - Integer index: returns the subgraph at that position
    /// - String key: extracts single node attribute from all subgraphs, returns ArrayArray
    /// - List of strings: extracts multiple attributes as tables from all subgraphs, returns TableArray
    fn __getitem__(&self, key: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
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

                return Ok(self.inner[actual_index].clone().into_py(py));
            }

            Err(pyo3::exceptions::PyTypeError::new_err(
                "SubgraphArray indices must be integers, strings, or lists of strings",
            ))
        })
    }

    /// Extract a node attribute from all subgraphs and return as ArrayArray
    fn extract_node_attribute(&self, py: Python, attr_name: &str) -> PyResult<PyObject> {
        use groggy::storage::array::ArrayArray;
        let mut arrays = Vec::with_capacity(self.inner.len());
        let mut keys = Vec::with_capacity(self.inner.len());
        let name = attr_name.to_owned();
        for (index, subgraph) in self.inner.iter().enumerate() {
            let view = subgraph
                .inner
                .pin_view()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            // Preserve the legacy node-first/edge-fallback selector. Native readers
            // own presence semantics and ordered projection from the captured source.
            let values = py
                .allow_threads(|| {
                    let nodes = view.node_accessor();
                    if nodes.has_column(&name) {
                        return Some(nodes.present_column(&name));
                    }
                    let edges = view.edge_accessor();
                    edges.has_column(&name).then(|| edges.present_column(&name))
                })
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!(
                        "Attribute '{attr_name}' not found in subgraph {index}"
                    ))
                })?;
            arrays.push(values);
            keys.push(format!("subgraph_{index}"));
        }
        let array = ArrayArray::with_keys(arrays, keys)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(crate::PyArrayArray::from_array_array(array).into_py(py))
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

    /// Collapse each subgraph using the Subgraph.collapse API and materialize meta-nodes.
    #[pyo3(signature = (
        node_aggs = None,
        edge_aggs = None,
        edge_strategy = None,
        node_strategy = "extract",
        preset = None,
        include_edge_count = None,
        mark_entity_type = None,
        entity_type = None,
        allow_missing_attributes = true,
        meta_targets = None
    ))]
    // TODO: Refactor to use config/builder pattern
    #[allow(clippy::too_many_arguments)]
    fn collapse(
        &self,
        py: Python,
        node_aggs: Option<&PyAny>,
        edge_aggs: Option<&PyAny>,
        edge_strategy: Option<&str>,
        node_strategy: &str,
        preset: Option<String>,
        include_edge_count: Option<bool>,
        mark_entity_type: Option<bool>,
        entity_type: Option<&str>,
        allow_missing_attributes: bool,
        meta_targets: Option<Vec<usize>>,
    ) -> PyResult<Vec<Py<PyMetaNode>>> {
        use crate::ffi::subgraphs::composer::{collapse_many, parse_composer_options};
        let options = parse_composer_options(
            node_aggs,
            edge_aggs,
            edge_strategy,
            node_strategy,
            preset,
            include_edge_count,
            mark_entity_type,
            entity_type,
        )?;
        collapse_many(
            py,
            self.inner.iter().map(|selection| selection.inner.clone()),
            options,
            allow_missing_attributes,
            meta_targets.as_deref(),
        )
    }

    /// Apply table() to all subgraphs and return PyTableArray
    fn table(&self) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        Python::with_gil(|py| {
            let tables = self
                .inner
                .iter()
                .map(|subgraph| subgraph.table(py))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(crate::ffi::storage::table_array::PyTableArray::new(tables))
        })
    }

    /// Apply sample(k) to all subgraphs
    fn sample(&self, k: usize) -> PyResult<PySubgraphArray> {
        let mut sampled = Vec::new();

        for subgraph in self.inner.iter() {
            sampled.push(subgraph.sample_k(k)?);
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
        if !matches!(element_type.as_str(), "nodes" | "edges") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "element_type must be either 'nodes' or 'edges'",
            ));
        }
        let mut all_groups = Vec::new();
        for subgraph in self.inner.iter() {
            let groups = subgraph.group_by(attr_name.clone(), element_type.clone())?;
            all_groups.extend(groups.as_vec().iter().cloned());
        }
        Ok(PySubgraphArray::new(all_groups))
    }

    /// Get nodes tables from all subgraphs as TableArray
    ///
    /// Returns a TableArray where each table contains the nodes from one subgraph
    /// with all their attributes
    ///
    /// Example:
    ///     components = g.connected_components()
    ///     node_tables = components.nodes_table()
    ///     # Returns TableArray with one table per component
    fn nodes_table(&self, py: Python) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let mut tables = Vec::with_capacity(self.inner.len());
        for selection in self.inner.iter() {
            let view = selection
                .inner
                .pin_view()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let table = py
                .allow_threads(|| view.nodes_table())
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            tables.push(crate::ffi::storage::table::PyNodesTable { table }.into_py(py));
        }
        Ok(crate::ffi::storage::table_array::PyTableArray::new(tables))
    }

    /// Get edges tables from all subgraphs as TableArray
    ///
    /// Returns a TableArray where each table contains the edges from one subgraph
    /// with all their attributes
    ///
    /// Example:
    ///     components = g.connected_components()
    ///     edge_tables = components.edges_table()
    ///     # Returns TableArray with one table per component
    fn edges_table(&self, py: Python) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let mut tables = Vec::with_capacity(self.inner.len());
        for selection in self.inner.iter() {
            let view = selection
                .inner
                .pin_view()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            let table = py
                .allow_threads(|| view.edges_table())
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            tables.push(crate::ffi::storage::table::PyEdgesTable { table }.into_py(py));
        }
        Ok(crate::ffi::storage::table_array::PyTableArray::new(tables))
    }

    /// Get summary statistics for all subgraphs as a BaseTable
    ///
    /// Returns a table with one row per subgraph containing:
    /// - subgraph_id: Index of the subgraph
    /// - node_count: Number of nodes in the subgraph
    /// - edge_count: Number of edges in the subgraph
    /// - density: Edge density of the subgraph
    ///
    /// Example:
    ///     components = g.connected_components()
    ///     summary = components.summary()
    ///     print(summary)
    fn summary(&self, _py: Python) -> PyResult<crate::ffi::storage::table::PyBaseTable> {
        use groggy::storage::array::BaseArray;
        use groggy::storage::table::BaseTable;
        use groggy::types::AttrValue;
        use std::collections::HashMap;

        let mut ids = Vec::new();
        let mut node_counts = Vec::new();
        let mut edge_counts = Vec::new();
        let mut densities = Vec::new();

        for (idx, subgraph) in self.inner.iter().enumerate() {
            let node_count = subgraph.inner.node_count();
            let edge_count = subgraph.inner.edge_count();

            let density = subgraph.inner.density();

            ids.push(AttrValue::Int(idx as i64));
            node_counts.push(AttrValue::Int(node_count as i64));
            edge_counts.push(AttrValue::Int(edge_count as i64));
            densities.push(AttrValue::Float64(density));
        }

        // Create columns
        let mut columns = HashMap::new();
        columns.insert("subgraph_id".to_string(), BaseArray::from_attr_values(ids));
        columns.insert(
            "node_count".to_string(),
            BaseArray::from_attr_values(node_counts),
        );
        columns.insert(
            "edge_count".to_string(),
            BaseArray::from_attr_values(edge_counts),
        );
        columns.insert(
            "density".to_string(),
            BaseArray::from_attr_values(densities),
        );

        // Create BaseTable
        let table = BaseTable::from_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(crate::ffi::storage::table::PyBaseTable::from_table(table))
    }

    /// Get viz accessor for visualization operations
    #[getter]
    fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        use groggy::viz::streaming::GraphDataSource;
        let view = match self.merged_source(py)? {
            Some(view) => view,
            None => groggy::Graph::new()
                .read_view()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        };
        let data = py
            .allow_threads(|| GraphDataSource::from_view(&view))
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Py::new(
            py,
            crate::ffi::viz_accessor::VizAccessor::with_data_source(data, "SubgraphArray".into()),
        )
    }

    /// Map a function over all subgraphs and return a BaseArray
    ///
    /// Args:
    ///     func: Python callable that takes a Subgraph and returns a numeric value
    ///
    /// Returns:
    ///     BaseArray containing the results
    ///
    /// Example:
    /// ```python
    /// # Get node count for each subgraph
    /// node_counts = subgraph_array.map(lambda sg: sg.node_count())
    ///
    /// # Get average degree
    /// avg_degrees = subgraph_array.map(lambda sg: sum(sg.degrees()) / sg.node_count())
    /// ```
    fn map(&self, py: Python, func: PyObject) -> PyResult<crate::ffi::storage::array::PyBaseArray> {
        use groggy::storage::array::BaseArray;

        let mut results = Vec::new();

        for subgraph in self.inner.iter() {
            // Call function with subgraph
            let result = func.call1(py, (subgraph.clone(),))?;

            // Convert result to AttrValue
            let attr_value = crate::ffi::utils::python_value_to_attr_value(result.as_ref(py))?;

            results.push(attr_value);
        }

        // Create BaseArray from results and wrap in PyBaseArray
        let base_array = BaseArray::from_attr_values(results);
        Ok(crate::ffi::storage::array::PyBaseArray { inner: base_array })
    }

    /// Merge all subgraphs into a single Graph
    ///
    /// Combines all nodes and edges from all subgraphs into one unified graph.
    /// Node and edge IDs are preserved from the original graph.
    /// Overlapping instances are included once. Inputs must share one capture;
    /// current owner changes are never consulted and no extra edges are induced.
    ///
    /// Returns:
    ///     PyGraph containing all merged subgraphs
    ///
    /// Example:
    /// ```python
    /// # Group by department, then merge back
    /// dept_groups = g.nodes.group_by('department')
    /// merged_graph = dept_groups.merge()
    /// ```
    fn merge(&self, py: Python) -> PyResult<crate::ffi::api::graph::PyGraph> {
        let graph = match self.merged_source(py)? {
            Some(view) => view
                .to_graph()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
            None => groggy::Graph::new(),
        };
        Ok(crate::ffi::api::graph::PyGraph {
            inner: graph.into_shared(),
            cached_view: std::cell::RefCell::new(None),
            cached_live_view: std::cell::RefCell::new(None),
        })
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
        PySubgraph: Clone + 'static,
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
