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
    #[pyo3(signature=(filter, *, order_by=None, descending=false, offset=None, limit=None))]
    fn query_nodes(
        &self,
        py: Python<'_>,
        filter: &PyAny,
        order_by: Option<String>,
        descending: bool,
        offset: Option<&PyAny>,
        limit: Option<&PyAny>,
    ) -> PyResult<Vec<usize>> {
        let filter = crate::ffi::query::query::node_filter_from_python(filter)?;
        let options = crate::ffi::query::query::query_options(order_by, descending, offset, limit)?;
        py.allow_threads(|| self.inner.query_nodes(&filter, &options))
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(filter, *, order_by=None, descending=false, offset=None, limit=None))]
    fn query_edges(
        &self,
        py: Python<'_>,
        filter: &PyAny,
        order_by: Option<String>,
        descending: bool,
        offset: Option<&PyAny>,
        limit: Option<&PyAny>,
    ) -> PyResult<Vec<usize>> {
        let filter = crate::ffi::query::query::edge_filter_from_python(filter)?;
        let options = crate::ffi::query::query::query_options(order_by, descending, offset, limit)?;
        py.allow_threads(|| self.inner.query_edges(&filter, &options))
            .map_err(graph_error_to_py_err)
    }
    fn build_group_index(
        &self,
        py: Python<'_>,
    ) -> PyResult<crate::ffi::query::groups::PyGroupMembershipIndex> {
        py.allow_threads(|| self.inner.build_group_index())
            .map(|inner| crate::ffi::query::groups::PyGroupMembershipIndex { inner })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(seeds, max_hops, *, direction="outgoing"))]
    fn within_hops(
        &self,
        py: Python<'_>,
        seeds: &PyAny,
        max_hops: &PyAny,
        direction: &str,
    ) -> PyResult<Self> {
        use crate::ffi::query::query::nonnegative_count;
        let seeds = seeds
            .iter()?
            .map(|v| v.and_then(|v| nonnegative_count(v, "seed ID")))
            .collect::<PyResult<Vec<_>>>()?;
        let hops = nonnegative_count(max_hops, "max_hops")?;
        let direction = direction.parse().map_err(graph_error_to_py_err)?;
        py.allow_threads(|| self.inner.within_hops(&seeds, hops, direction))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(*, node_properties=Vec::new(), edge_properties=Vec::new(), topology=true))]
    fn build_index(
        &self,
        py: Python<'_>,
        node_properties: Vec<String>,
        edge_properties: Vec<String>,
        topology: bool,
    ) -> PyResult<crate::ffi::query::index::PyGraphQueryIndex> {
        py.allow_threads(|| {
            self.inner
                .build_index(&node_properties, &edge_properties, topology)
        })
        .map(|inner| crate::ffi::query::index::PyGraphQueryIndex { inner })
        .map_err(graph_error_to_py_err)
    }
    /// Retrieve the frozen group revision referenced by this captured node.
    fn meta_subgraph(&self, py: Python<'_>, node: usize) -> PyResult<Self> {
        py.allow_threads(|| self.inner.meta_subgraph(node))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Clocks and branch/head of an effective-time capture, or None for other views.
    #[getter]
    fn read_time(&self, py: Python<'_>) -> PyResult<PyObject> {
        let Some(time) = self.inner.read_state().read_time() else {
            return Ok(py.None());
        };
        let result = pyo3::types::PyDict::new(py);
        result.set_item("branch", &time.branch)?;
        result.set_item("published_head", time.published_head)?;
        result.set_item("effective_at", time.effective_at)?;
        result.set_item("recorded_by", time.recorded_by)?;
        if !time.group_sources.is_empty() {
            result.set_item("group_sources", &time.group_sources)?;
        }
        if !time.legacy_clocks.is_empty() {
            result.set_item("legacy_times", &time.legacy_clocks)?;
        }
        Ok(result.to_object(py))
    }
    /// Accepted source state for checked writeback; None for dirty/standalone captures.
    #[getter]
    fn write_state_id(&self) -> Option<groggy::StateId> {
        self.inner.write_state_id()
    }
    #[getter]
    fn nodes(&self) -> super::view_entities::PyGraphViewEntities {
        super::view_entities::PyGraphViewEntities {
            inner: self.inner.node_accessor(),
        }
    }
    #[getter]
    fn edges(&self) -> super::view_entities::PyGraphViewEntities {
        super::view_entities::PyGraphViewEntities {
            inner: self.inner.edge_accessor(),
        }
    }
    /// Frozen node column aligned with node_ids; absent cells project to Null.
    fn __getitem__(&self, py: Python<'_>, name: &str) -> crate::ffi::storage::array::PyBaseArray {
        crate::ffi::storage::array::PyBaseArray {
            inner: py.allow_threads(|| self.inner.node_accessor().column(&name.to_owned())),
        }
    }
    /// Read-only captured entity adapters, sharing this selection and source pin.
    fn to_nodes(&self) -> super::view_entities::PyGraphViewEntities { self.nodes() }
    fn to_edges(&self) -> super::view_entities::PyGraphViewEntities { self.edges() }
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
    /// Frozen selected outgoing rows, including isolates and parallel occurrences.
    fn adjacency_list(&self, py: Python<'_>) -> PyResult<std::collections::BTreeMap<NodeId, Vec<NodeId>>> {
        py.allow_threads(|| self.inner.adjacency_list()).map_err(graph_error_to_py_err)
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
    /// Uniform induced node sample sharing this immutable capture.
    #[pyo3(signature=(k, *, seed=None))]
    fn sample(&self, py: Python<'_>, k: &PyAny, seed: Option<&PyAny>) -> PyResult<Self> {
        let seed = crate::ffi::utils::optional_seed(seed)?;
        let k = crate::ffi::utils::sample_count(k)?;
        py.allow_threads(|| self.inner.sample(k, seed))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
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
        let filter = crate::ffi::query::query::node_filter_from_python(filter)?;
        py.allow_threads(|| self.inner.filter_nodes(&filter))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Filter captured selected edge instances using the shared native executor.
    fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<Self> {
        let filter = crate::ffi::query::query::edge_filter_from_python(filter)?;
        py.allow_threads(|| self.inner.filter_edges(&filter))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Exact selection union; both views must derive from the same capture.
    fn merge_with(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.allow_threads(|| self.inner.merge_with(&other.inner))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    fn intersect_with(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.allow_threads(|| self.inner.intersect_with(&other.inner))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    fn subtract_from(&self, py: Python<'_>, other: &Self) -> PyResult<Self> {
        py.allow_threads(|| self.inner.subtract_from(&other.inner))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(other, metric="jaccard"))]
    fn calculate_similarity(&self, py: Python<'_>, other: &Self, metric: &str) -> PyResult<f64> {
        let metric = metric.parse().map_err(graph_error_to_py_err)?;
        py.allow_threads(|| self.inner.calculate_similarity(&other.inner, metric))
            .map_err(graph_error_to_py_err)
    }
    fn group_by_nodes(&self, py: Python<'_>, attr_name: String) -> PyResult<Vec<Self>> {
        py.allow_threads(|| self.inner.group_by_nodes(&attr_name))
            .map(|views| views.into_iter().map(|inner| Self { inner }).collect())
            .map_err(graph_error_to_py_err)
    }
    fn group_by_edges(&self, py: Python<'_>, attr_name: String) -> PyResult<Vec<Self>> {
        py.allow_threads(|| self.inner.group_by_edges(&attr_name))
            .map(|views| views.into_iter().map(|inner| Self { inner }).collect())
            .map_err(graph_error_to_py_err)
    }
    /// Captured Brandes node betweenness as owned f64 rows.
    #[pyo3(signature = (normalized=true, weight_attr=None))]
    fn betweenness(
        &self,
        py: Python,
        normalized: bool,
        weight_attr: Option<String>,
    ) -> PyResult<Vec<(NodeId, f64)>> {
        py.allow_threads(|| self.inner.betweenness(normalized, weight_attr))
            .map_err(graph_error_to_py_err)
    }
    /// Captured classic or harmonic closeness, returned as owned f64 rows.
    #[pyo3(signature = (harmonic=true, weight_attr=None, direction="outgoing", normalized=false))]
    fn closeness(
        &self,
        py: Python,
        harmonic: bool,
        weight_attr: Option<String>,
        direction: &str,
        normalized: bool,
    ) -> PyResult<Vec<(NodeId, f64)>> {
        let direction = groggy::algorithms::centrality::ClosenessDirection::parse(direction)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        py.allow_threads(|| {
            self.inner
                .closeness(harmonic, weight_attr, direction, normalized)
        })
        .map_err(graph_error_to_py_err)
    }
    /// Captured unweighted PageRank, including explicit convergence status.
    #[pyo3(signature = (damping=0.85, max_iter=50, tolerance=1e-6, personalization_attr=None))]
    fn pagerank(
        &self,
        py: Python,
        damping: f64,
        max_iter: usize,
        tolerance: f64,
        personalization_attr: Option<String>,
    ) -> PyResult<PyPageRankResult> {
        py.allow_threads(|| {
            self.inner
                .pagerank(damping, max_iter, tolerance, personalization_attr)
        })
        .map(|inner| PyPageRankResult { inner })
        .map_err(graph_error_to_py_err)
    }
    /// Owned community rows from captured topology; never attaches properties.
    #[pyo3(signature=(*, max_iter=None, max_phases=None, resolution=1.0))]
    fn louvain(
        &self,
        py: Python<'_>,
        max_iter: Option<&PyAny>,
        max_phases: Option<&PyAny>,
        resolution: f64,
    ) -> PyResult<Vec<(NodeId, usize)>> {
        let iterations = community_count(max_iter, "max_iter", 20)?;
        let phases = community_count(max_phases, "max_phases", 10)?;
        py.allow_threads(|| self.inner.louvain(iterations, phases, resolution))
            .map_err(graph_error_to_py_err)
    }
    /// Bounded Leiden using captured topology and an optional reproducible seed.
    #[pyo3(signature=(*, max_iter=None, max_phases=None, resolution=1.0, seed=None, theta=0.01))]
    fn leiden(
        &self,
        py: Python<'_>,
        max_iter: Option<&PyAny>,
        max_phases: Option<&PyAny>,
        resolution: f64,
        seed: Option<&PyAny>,
        theta: f64,
    ) -> PyResult<Vec<(NodeId, usize)>> {
        let iterations = community_count(max_iter, "max_iter", 20)?;
        let phases = community_count(max_phases, "max_phases", 10)?;
        let seed = crate::ffi::utils::optional_seed(seed)?;
        py.allow_threads(|| {
            self.inner
                .leiden(iterations, phases, resolution, seed, theta)
        })
        .map_err(graph_error_to_py_err)
    }
    /// Bounded two-level Infomap; direction and edge instances are captured.
    #[pyo3(signature=(*, teleportation=0.15, num_trials=None, max_iter=None, flow_max_iter=None, seed=None))]
    fn infomap(
        &self,
        py: Python<'_>,
        teleportation: f64,
        num_trials: Option<&PyAny>,
        max_iter: Option<&PyAny>,
        flow_max_iter: Option<&PyAny>,
        seed: Option<&PyAny>,
    ) -> PyResult<Vec<(NodeId, usize)>> {
        let trials = community_count(num_trials, "num_trials", 10)?;
        let iterations = community_count(max_iter, "max_iter", 100)?;
        let flow = community_count(flow_max_iter, "flow_max_iter", 10_000)?;
        let seed = crate::ffi::utils::optional_seed(seed)?;
        py.allow_threads(|| {
            self.inner
                .infomap(teleportation, trials, iterations, flow, seed)
        })
        .map_err(graph_error_to_py_err)
    }
    /// Small-graph Girvan–Newman with frozen strictly positive edge lengths.
    #[pyo3(signature=(*, num_levels=None, modularity_threshold=None, weight_attr=None))]
    fn girvan_newman(
        &self,
        py: Python<'_>,
        num_levels: Option<&PyAny>,
        modularity_threshold: Option<f64>,
        weight_attr: Option<String>,
    ) -> PyResult<Vec<(NodeId, usize)>> {
        let levels = num_levels
            .map(|v| crate::ffi::query::query::nonnegative_count(v, "num_levels"))
            .transpose()?;
        py.allow_threads(|| {
            self.inner
                .girvan_newman(levels, modularity_threshold, weight_attr)
        })
        .map_err(graph_error_to_py_err)
    }
    /// Owned label values, including nonnumeric labels, from captured seed attributes.
    #[pyo3(signature=(*, max_iter=None, tolerance=0.0, seed_attr=None, seed=None))]
    fn label_propagation(
        &self,
        py: Python<'_>,
        max_iter: Option<&PyAny>,
        tolerance: f64,
        seed_attr: Option<String>,
        seed: Option<&PyAny>,
    ) -> PyResult<Vec<(NodeId, PyObject)>> {
        let iterations = community_count(max_iter, "max_iter", 20)?;
        let seed = crate::ffi::utils::optional_seed(seed)?;
        let rows = py
            .allow_threads(|| {
                self.inner
                    .label_propagation(iterations, tolerance, seed_attr, seed)
            })
            .map_err(graph_error_to_py_err)?;
        rows.into_iter()
            .map(|(id, value)| Ok((id, attr_value_to_python_value(py, &value)?)))
            .collect()
    }
    /// Immutable components sharing captured state; ignores orientation by default.
    #[pyo3(signature = (mode="undirected"))]
    fn connected_components(&self, py: Python, mode: &str) -> PyResult<Vec<Self>> {
        let mode = groggy::algorithms::community::ComponentMode::parse(mode)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        py.allow_threads(|| self.inner.connected_components(mode))
            .map(|views| views.into_iter().map(|inner| Self { inner }).collect())
            .map_err(graph_error_to_py_err)
    }
    /// Local or average simple undirected clustering on captured topology.
    #[pyo3(signature=(node_id=None))]
    fn clustering_coefficient(&self, py: Python<'_>, node_id: Option<&PyAny>) -> PyResult<f64> {
        let node = node_id
            .map(|v| crate::ffi::query::query::nonnegative_count(v, "node_id"))
            .transpose()?;
        py.allow_threads(|| self.inner.clustering_coefficient(node))
            .map_err(graph_error_to_py_err)
    }
    fn clustering_coefficients(&self, py: Python<'_>) -> PyResult<Vec<(NodeId, f64)>> {
        py.allow_threads(|| self.inner.clustering_coefficients())
            .map_err(graph_error_to_py_err)
    }
    fn transitivity(&self, py: Python<'_>) -> PyResult<f64> {
        py.allow_threads(|| self.inner.transitivity())
            .map_err(graph_error_to_py_err)
    }
    fn density(&self) -> f64 {
        self.inner.density()
    }
    /// Exact frozen BFS discovery-tree edges, bounded by hop depth.
    #[pyo3(signature=(start, max_depth=None))]
    fn bfs(&self, py: Python<'_>, start: NodeId, max_depth: Option<&PyAny>) -> PyResult<Self> {
        let depth = max_depth
            .map(|v| crate::ffi::query::query::nonnegative_count(v, "max_depth"))
            .transpose()?;
        py.allow_threads(|| self.inner.bfs(start, depth))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    /// Exact frozen ascending-neighbor DFS discovery tree, bounded by discovery depth.
    #[pyo3(signature=(start, max_depth=None))]
    fn dfs(&self, py: Python<'_>, start: NodeId, max_depth: Option<&PyAny>) -> PyResult<Self> {
        let depth = max_depth
            .map(|v| crate::ffi::query::query::nonnegative_count(v, "max_depth"))
            .transpose()?;
        py.allow_threads(|| self.inner.dfs(start, depth))
            .map(|inner| Self { inner })
            .map_err(graph_error_to_py_err)
    }
    fn has_path(&self, py: Python<'_>, source: NodeId, target: NodeId) -> PyResult<bool> {
        py.allow_threads(|| self.inner.has_path(source, target))
            .map_err(graph_error_to_py_err)
    }
    fn shortest_path_subgraph(
        &self,
        py: Python<'_>,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<Option<Self>> {
        py.allow_threads(|| self.inner.shortest_path_subgraph(source, target))
            .map(|view| view.map(|inner| Self { inner }))
            .map_err(graph_error_to_py_err)
    }
    #[pyo3(signature=(mode="undirected"))]
    fn is_connected(&self, py: Python<'_>, mode: &str) -> PyResult<bool> {
        let mode = groggy::algorithms::community::ComponentMode::parse(mode)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        py.allow_threads(|| self.inner.is_connected(mode))
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
    /// Nonnegative weighted distances from captured values, with no property writes.
    #[pyo3(signature = (start, weight_attr=None))]
    fn dijkstra_distances(
        &self,
        py: Python,
        start: NodeId,
        weight_attr: Option<String>,
    ) -> PyResult<Vec<(NodeId, f64)>> {
        py.allow_threads(|| self.inner.dijkstra_distances(start, weight_attr.as_deref()))
            .map_err(graph_error_to_py_err)
    }
    /// Owned captured A* path, including selected parallel-edge identity and cost.
    #[pyo3(signature = (start, goal, weight_attr=None, heuristic_attr=None))]
    fn astar_path(
        &self,
        py: Python,
        start: NodeId,
        goal: NodeId,
        weight_attr: Option<String>,
        heuristic_attr: Option<String>,
    ) -> PyResult<PyGraphPath> {
        py.allow_threads(|| {
            self.inner.astar_path(
                start,
                goal,
                weight_attr.as_deref(),
                heuristic_attr.as_deref(),
            )
        })
        .map(|inner| PyGraphPath { inner })
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

fn community_count(value: Option<&PyAny>, name: &str, default: usize) -> PyResult<usize> {
    value
        .map(|v| crate::ffi::query::query::nonnegative_count(v, name))
        .transpose()
        .map(|v| v.unwrap_or(default))
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

/// Owned native path result; IDs and weight remain independent of working graphs.
#[pyclass(name = "GraphPath", frozen)]
#[derive(Clone)]
pub struct PyGraphPath {
    inner: groggy::query::Path,
}
#[pymethods]
impl PyGraphPath {
    #[getter]
    fn nodes(&self) -> Vec<NodeId> {
        self.inner.nodes.clone()
    }
    #[getter]
    fn edges(&self) -> Vec<EdgeId> {
        self.inner.edges.clone()
    }
    #[getter]
    fn total_weight(&self) -> f64 {
        self.inner.total_weight
    }
    #[getter]
    fn length(&self) -> usize {
        self.inner.metadata.length
    }
    #[getter]
    fn algorithm(&self) -> String {
        self.inner.metadata.algorithm.clone()
    }
}

/// Owned scores and convergence information from the native PageRank kernel.
#[pyclass(name = "PageRankResult", frozen)]
pub struct PyPageRankResult {
    inner: groggy::algorithms::centrality::PageRankResult,
}
#[pymethods]
impl PyPageRankResult {
    #[getter]
    fn scores(&self) -> Vec<(NodeId, f64)> {
        self.inner.scores.clone()
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }
    #[getter]
    fn max_update(&self) -> f64 {
        self.inner.max_update
    }
}
