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

    /// Helper: Extract multiple columns from each subgraph's nodes and return as TableArray
    fn extract_columns_as_tables(&self, py: Python, columns: &[String]) -> PyResult<PyObject> {
        use groggy::storage::array::BaseArray;
        use groggy::storage::table::BaseTable;
        use std::collections::HashMap;

        let mut tables = Vec::new();

        for subgraph in self.inner.iter() {
            let graph_ref = subgraph.inner.graph();
            let graph_borrow = graph_ref.borrow();

            // Build columns map for this subgraph
            let mut column_map = HashMap::new();

            for col_name in columns {
                let mut col_values = Vec::new();

                // Extract attribute values from all nodes in this subgraph
                for &node_id in subgraph.inner.nodes() {
                    if let Ok(Some(attr_value)) = graph_borrow.get_node_attr(node_id, col_name) {
                        col_values.push(attr_value);
                    } else {
                        // If any node is missing this attribute, use Null
                        col_values.push(groggy::types::AttrValue::Null);
                    }
                }

                column_map.insert(col_name.clone(), BaseArray::from_attr_values(col_values));
            }

            // Create BaseTable for this subgraph
            match BaseTable::from_columns(column_map) {
                Ok(table) => {
                    // Wrap in PyBaseTable and convert to PyObject
                    let py_table = crate::ffi::storage::table::PyBaseTable::from_table(table);
                    tables.push(py_table.into_py(py));
                }
                Err(_) => continue, // Skip failed tables
            }
        }

        // Wrap in PyTableArray
        let table_array = crate::ffi::storage::table_array::PyTableArray::new(tables);
        Ok(table_array.into_py(py))
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
        use groggy::storage::array::{ArrayArray, BaseArray};
        use groggy::types::AttrValue;

        let mut arrays = Vec::new();
        let mut keys = Vec::new();
        let attr_name_string = attr_name.to_string();

        for (idx, subgraph) in self.inner.iter().enumerate() {
            // Get graph reference and extract node attribute values
            let graph_ref = subgraph.inner.graph();
            let graph_borrow = graph_ref.borrow();

            let node_values: Vec<AttrValue> = subgraph
                .inner
                .nodes()
                .iter() // Add .iter() to iterate over HashSet
                .filter_map(|&node_id| {
                    graph_borrow
                        .get_node_attr(node_id, &attr_name_string)
                        .ok()
                        .flatten()
                })
                .collect();

            // If no node values found, try extracting from edges
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
                        "Attribute '{}' not found in subgraph {}",
                        attr_name, idx
                    )));
                }

                arrays.push(BaseArray::from_attr_values(edge_values));
                keys.push(format!("subgraph_{}", idx));
                continue;
            }

            arrays.push(BaseArray::from_attr_values(node_values));
            keys.push(format!("subgraph_{}", idx));
        }

        // Create ArrayArray with keys
        let array_array = ArrayArray::with_keys(arrays, keys)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Wrap in PyArrayArray
        let py_array_array = crate::PyArrayArray::from_array_array(array_array);
        Ok(py_array_array.into_py(py))
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
        edge_strategy = "aggregate",
        node_strategy = "extract",
        preset = None,
        include_edge_count = true,
        mark_entity_type = true,
        entity_type = "meta",
        allow_missing_attributes = true
    ))]
    // TODO: Refactor to use config/builder pattern
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
    ) -> PyResult<Vec<Py<PyMetaNode>>> {
        let mut results = Vec::with_capacity(self.inner.len());

        for subgraph in self.inner.iter() {
            let meta_node = subgraph.collapse(
                py,
                node_aggs,
                edge_aggs,
                edge_strategy,
                node_strategy,
                preset.clone(),
                include_edge_count,
                mark_entity_type,
                entity_type,
                allow_missing_attributes,
            )?;
            let meta_node = meta_node.extract::<Py<PyMetaNode>>(py)?;
            results.push(meta_node);
        }

        Ok(results)
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
                }
                Err(_) => continue, // Skip failed subgraphs
            }
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
        let mut tables = Vec::new();

        for subgraph in self.inner.iter() {
            match subgraph.inner.nodes_table() {
                Ok(nodes_table) => {
                    // Wrap NodesTable in PyNodesTable
                    let py_table = crate::ffi::storage::table::PyNodesTable { table: nodes_table };
                    tables.push(py_table.into_py(py));
                }
                Err(_) => continue, // Skip failed subgraphs
            }
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
        let mut tables = Vec::new();

        for subgraph in self.inner.iter() {
            match subgraph.inner.edges_table() {
                Ok(edges_table) => {
                    // Wrap EdgesTable in PyEdgesTable
                    let py_table = crate::ffi::storage::table::PyEdgesTable { table: edges_table };
                    tables.push(py_table.into_py(py));
                }
                Err(_) => continue, // Skip failed subgraphs
            }
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

            // Calculate density: 2 * edges / (nodes * (nodes - 1)) for undirected
            let density = if node_count < 2 {
                0.0
            } else {
                let max_edges = (node_count * (node_count - 1)) / 2;
                edge_count as f64 / max_edges as f64
            };

            ids.push(AttrValue::Int(idx as i64));
            node_counts.push(AttrValue::Int(node_count as i64));
            edge_counts.push(AttrValue::Int(edge_count as i64));
            densities.push(AttrValue::Float(density as f32));
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

        // For SubgraphArray, create a graph that contains all subgraphs
        let mut viz_graph = groggy::api::graph::Graph::new();

        for subgraph in self.inner.iter() {
            let node_ids = subgraph.inner.node_ids();
            let edge_ids = subgraph.inner.edge_ids();

            // Copy nodes with their attributes
            for &node_id in &node_ids {
                // Check if node already exists to avoid duplicates
                if !viz_graph.node_ids().contains(&node_id) {
                    viz_graph.add_node();
                    if let Ok(attrs) = subgraph.inner.graph().borrow().get_node_attrs(node_id) {
                        for (attr_name, attr_value) in attrs {
                            let _ = viz_graph.set_node_attr(node_id, attr_name, attr_value);
                        }
                    }
                }
            }

            // Copy edges with their attributes
            let graph_ref = subgraph.inner.graph();
            for &edge_id in &edge_ids {
                if let Ok((source, target)) = graph_ref.borrow().edge_endpoints(edge_id) {
                    if let Ok(new_edge_id) = viz_graph.add_edge(source, target) {
                        if let Ok(attrs) = graph_ref.borrow().get_edge_attrs(edge_id) {
                            for (attr_name, attr_value) in attrs {
                                let _ = viz_graph.set_edge_attr(new_edge_id, attr_name, attr_value);
                            }
                        }
                    }
                }
            }
        }

        let graph_data_source = GraphDataSource::new(&viz_graph);
        let viz_accessor = crate::ffi::viz_accessor::VizAccessor::with_data_source(
            graph_data_source,
            "SubgraphArray".to_string(),
        );

        Py::new(py, viz_accessor)
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
        use groggy::types::AttrValue;

        let mut results = Vec::new();

        for subgraph in self.inner.iter() {
            // Call function with subgraph
            let result = func.call1(py, (subgraph.clone(),))?;

            // Convert result to AttrValue
            let attr_value = if let Ok(f) = result.extract::<f64>(py) {
                AttrValue::Float(f as f32)
            } else if let Ok(i) = result.extract::<i64>(py) {
                AttrValue::Int(i)
            } else if let Ok(s) = result.extract::<String>(py) {
                AttrValue::Text(s)
            } else if let Ok(b) = result.extract::<bool>(py) {
                AttrValue::Bool(b)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Map function must return int, float, str, or bool",
                ));
            };

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
    /// Duplicate nodes/edges are handled by taking the first occurrence.
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
    fn merge(&self, _py: Python) -> PyResult<crate::ffi::api::graph::PyGraph> {
        use std::cell::RefCell;
        use std::collections::HashSet;
        use std::rc::Rc;

        // Create a new graph
        let mut merged_graph = groggy::api::graph::Graph::new();
        let mut seen_nodes = HashSet::new();
        let mut seen_edges = HashSet::new();

        // Iterate over all subgraphs
        for subgraph in self.inner.iter() {
            let graph_ref = subgraph.inner.graph();
            let graph_borrow = graph_ref.borrow();

            // Copy nodes with their attributes
            for &node_id in subgraph.inner.nodes() {
                if !seen_nodes.contains(&node_id) {
                    seen_nodes.insert(node_id);

                    // Create enough nodes to include this node_id
                    while merged_graph.node_ids().len() <= node_id {
                        merged_graph.add_node();
                    }

                    // Copy attributes
                    if let Ok(attrs) = graph_borrow.get_node_attrs(node_id) {
                        for (attr_name, attr_value) in attrs {
                            let _ = merged_graph.set_node_attr(node_id, attr_name, attr_value);
                        }
                    }
                }
            }

            // Copy edges with their attributes
            for &edge_id in subgraph.inner.edges() {
                if !seen_edges.contains(&edge_id) {
                    seen_edges.insert(edge_id);

                    // Get edge endpoints
                    if let Ok((source, target)) = graph_borrow.edge_endpoints(edge_id) {
                        // Add edge to merged graph
                        if let Ok(new_edge_id) = merged_graph.add_edge(source, target) {
                            // Copy attributes
                            if let Ok(attrs) = graph_borrow.get_edge_attrs(edge_id) {
                                for (attr_name, attr_value) in attrs {
                                    let _ = merged_graph.set_edge_attr(
                                        new_edge_id,
                                        attr_name,
                                        attr_value,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Wrap in PyGraph by manually constructing the struct
        Ok(crate::ffi::api::graph::PyGraph {
            inner: Rc::new(RefCell::new(merged_graph)),
            cached_view: RefCell::new(None),
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
