//! Thin adapters for immutable native read selections; no writable graph ownership.
use crate::ffi::utils::{attr_value_to_python_value, graph_error_to_py_err};
use groggy::{EdgeId, GraphView, NodeId};
use pyo3::prelude::*;
#[pyclass(name = "GraphView")]
#[derive(Clone)]
pub struct PyGraphView {
    pub inner: GraphView,
}
#[pymethods]
impl PyGraphView {
    #[getter]
    fn node_ids(&self, py: Python) -> Vec<NodeId> {
        py.allow_threads(|| self.inner.node_ids())
    }
    #[getter]
    fn edge_ids(&self, py: Python) -> Vec<EdgeId> {
        py.allow_threads(|| self.inner.edge_ids())
    }
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }
    fn contains_node(&self, id: NodeId) -> bool {
        self.inner.contains_node(id)
    }
    fn contains_edge(&self, id: EdgeId) -> bool {
        self.inner.contains_edge(id)
    }
    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.is_directed()
    }
    fn node_attr(&self, py: Python, id: NodeId, name: &str) -> PyResult<PyObject> {
        match self
            .inner
            .node_attr(id, &name.into())
            .map_err(graph_error_to_py_err)?
        {
            Some(value) => attr_value_to_python_value(py, &value),
            None => Ok(py.None()),
        }
    }
    fn edge_attr(&self, py: Python, id: EdgeId, name: &str) -> PyResult<PyObject> {
        match self
            .inner
            .edge_attr(id, &name.into())
            .map_err(graph_error_to_py_err)?
        {
            Some(value) => attr_value_to_python_value(py, &value),
            None => Ok(py.None()),
        }
    }
    fn has_node_attribute(&self, id: NodeId, name: &str) -> PyResult<bool> {
        self.inner
            .has_node_attribute(id, &name.into())
            .map_err(graph_error_to_py_err)
    }
    fn has_edge_attribute(&self, id: EdgeId, name: &str) -> PyResult<bool> {
        self.inner
            .has_edge_attribute(id, &name.into())
            .map_err(graph_error_to_py_err)
    }
    fn node_attribute_keys(&self, id: NodeId) -> PyResult<Vec<String>> {
        self.inner
            .node_attribute_keys(id)
            .map_err(graph_error_to_py_err)
    }
    fn edge_attribute_keys(&self, id: EdgeId) -> PyResult<Vec<String>> {
        self.inner
            .edge_attribute_keys(id)
            .map_err(graph_error_to_py_err)
    }
    fn edge_endpoints(&self, id: EdgeId) -> PyResult<(NodeId, NodeId)> {
        self.inner.edge_endpoints(id).map_err(graph_error_to_py_err)
    }
    fn neighbors(&self, id: NodeId) -> PyResult<Vec<NodeId>> {
        self.inner.neighbors(id).map_err(graph_error_to_py_err)
    }
    fn incident_edges(&self, id: NodeId) -> PyResult<Vec<EdgeId>> {
        self.inner.incident_edges(id).map_err(graph_error_to_py_err)
    }
    fn degree(&self, id: NodeId) -> PyResult<usize> {
        self.inner.degree(id).map_err(graph_error_to_py_err)
    }
    fn out_degree(&self, id: NodeId) -> PyResult<usize> {
        self.inner.out_degree(id).map_err(graph_error_to_py_err)
    }
    fn in_degree(&self, id: NodeId) -> PyResult<usize> {
        self.inner.in_degree(id).map_err(graph_error_to_py_err)
    }
    fn select_nodes(&self, py: Python, ids: Vec<NodeId>) -> PyResult<Self> {
        py.allow_threads(|| self.inner.select_nodes(&ids))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    fn select_edges(&self, py: Python, ids: Vec<EdgeId>) -> PyResult<Self> {
        py.allow_threads(|| self.inner.select_edges(&ids))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Filter captured selected nodes using a native filter or the existing query syntax.
    fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<Self> {
        let filter = if let Ok(filter) = filter.extract::<crate::ffi::query::query::PyNodeFilter>()
        {
            filter.inner
        } else if let Ok(query) = filter.extract::<String>() {
            groggy::query::QueryParser::new()
                .parse_node_query(&query)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "filter must be a NodeFilter or query string",
            ));
        };
        py.allow_threads(|| self.inner.filter_nodes(&filter))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Filter captured selected edge instances using the shared native executor.
    fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<Self> {
        let filter = if let Ok(filter) = filter.extract::<crate::ffi::query::query::PyEdgeFilter>()
        {
            filter.inner
        } else if let Ok(query) = filter.extract::<String>() {
            groggy::query::QueryParser::new()
                .parse_edge_query(&query)
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "filter must be an EdgeFilter or query string",
            ));
        };
        py.allow_threads(|| self.inner.filter_edges(&filter))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Frozen hop distances as (node ID, distance) rows; writes no properties.
    fn bfs_distances(&self, py: Python, start: NodeId) -> PyResult<Vec<(NodeId, i64)>> {
        py.allow_threads(|| self.inner.bfs_distances(start))
            .map_err(graph_error_to_py_err)
    }
    /// Frozen ascending-neighbor DFS preorder as (node ID, order) rows.
    fn dfs_order(&self, py: Python, start: NodeId) -> PyResult<Vec<(NodeId, i64)>> {
        py.allow_threads(|| self.inner.dfs_order(start))
            .map_err(graph_error_to_py_err)
    }
    /// Owned adjacency operator from this selection's captured weights/topology.
    #[pyo3(signature = (sparse=true, weight_attr=None))]
    fn adjacency_matrix(
        &self,
        py: Python,
        sparse: bool,
        weight_attr: Option<String>,
    ) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        owned_matrix(py, &self.inner, move |view| {
            view.adjacency_matrix::<f64>(sparse, weight_attr.as_deref())
        })
    }
    /// Owned scalar feature matrix using the graph API's shared conversion policy.
    fn to_matrix(&self, py: Python) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        owned_matrix(py, &self.inner, |view| view.to_matrix::<f64>())
    }
    /// Owned node table from captured values; no working-graph writeback.
    fn nodes_table(&self, py: Python) -> PyResult<crate::ffi::storage::table::PyNodesTable> {
        let data = py
            .allow_threads(|| self.inner.node_table_data())
            .map_err(graph_error_to_py_err)?;
        Ok(crate::ffi::storage::table::PyNodesTable {
            table: data.into_nodes_table().map_err(graph_error_to_py_err)?,
        })
    }
    /// Owned edge table from captured values and endpoints.
    fn edges_table(&self, py: Python) -> PyResult<crate::ffi::storage::table::PyEdgesTable> {
        let data = py
            .allow_threads(|| self.inner.edge_table_data())
            .map_err(graph_error_to_py_err)?;
        Ok(crate::ffi::storage::table::PyEdgesTable {
            table: data.into_edges_table().map_err(graph_error_to_py_err)?,
        })
    }
    /// Owned graph table; both components use this immutable state.
    fn table(&self, py: Python) -> PyResult<crate::ffi::storage::table::PyGraphTable> {
        let (nodes, edges) = py
            .allow_threads(|| -> groggy::GraphResult<_> {
                Ok((self.inner.node_table_data()?, self.inner.edge_table_data()?))
            })
            .map_err(graph_error_to_py_err)?;
        let table = groggy::storage::table::GraphTable::new(
            nodes.into_nodes_table().map_err(graph_error_to_py_err)?,
            edges.into_edges_table().map_err(graph_error_to_py_err)?,
        );
        Ok(crate::ffi::storage::table::PyGraphTable::from_table(table))
    }
    /// Cheap ownership copy; the frozen state and selected membership remain shared.
    fn copy(&self) -> Self {
        self.clone()
    }
}

// Only immutable input and owned Send numerical data cross the GIL boundary.
fn owned_matrix(
    py: Python,
    view: &GraphView,
    operation: impl FnOnce(&GraphView) -> groggy::GraphResult<groggy::storage::GraphMatrix> + Send,
) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
    let data = py
        .allow_threads(move || {
            operation(view)
                .and_then(|matrix| matrix.numerical_data())
                .map_err(|error| error.to_string())
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let matrix = data.into_matrix().map_err(graph_error_to_py_err)?;
    Ok(crate::ffi::storage::matrix::PyGraphMatrix::from_graph_matrix(matrix))
}
