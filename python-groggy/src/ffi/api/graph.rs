use crate::ffi::api::graph_attributes::{PyGraphAttr, PyGraphAttrMut};
use crate::ffi::api::graph_batch::parse_batch;
// Graph FFI Coordinator
//
// Main Python bindings for the Graph API.

use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::{
    AttrName, AttrValue as RustAttrValue, EdgeId, EdgeUpsert, ExternalEntityRef,
    Graph as RustGraph, NodeId, NodeUpsert, StateId,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// Import all graph modules
use crate::ffi::storage::accessors::{PyEdgesAccessor, PyNodesAccessor}; // Essential FFI - re-enabled
use crate::ffi::storage::array::PyBaseArray;
use crate::ffi::storage::views::{PyEdgeView, PyNodeView}; // Essential FFI - re-enabled
use crate::PyNumArray;
// use crate::ffi::core::path_result::PyPathResult; // Unused
use crate::ffi::query::query::{PyEdgeFilter, PyNodeFilter};
use crate::ffi::query::traversal::PyGroupedAggregationResult;
use crate::ffi::subgraphs::subgraph::PySubgraph;
use crate::ffi::temporal::PyTemporalSnapshot;
use crate::ffi::types::PyAttrValue;
use crate::ffi::utils::{graph_error_to_py_err, parse_creation_attrs};

// Import version control types
use crate::ffi::api::graph_version::{PyBranchInfo, PyCommit, PyHistoricalView};
// Import internal helper classes
use crate::ffi::api::graph_analysis::PyGraphAnalysis;
use crate::ffi::api::graph_matrix::PyGraphMatrixHelper;

/// Decode configuration only; defaults and validation belong to the core type.
pub(crate) fn parse_config(
    config: Option<&PyDict>,
    directed: Option<bool>,
) -> PyResult<groggy::GraphConfig> {
    let mut config = match config {
        Some(config) => {
            let RustAttrValue::Json(json) = crate::ffi::utils::python_value_to_attr_value(config)?
            else {
                return Err(PyTypeError::new_err("config must be a dictionary"));
            };
            serde_json::from_str::<groggy::GraphConfig>(&json)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
        }
        None => groggy::GraphConfig::new(),
    };
    if let Some(directed) = directed {
        config.graph_type = if directed {
            groggy::types::GraphType::Directed
        } else {
            groggy::types::GraphType::Undirected
        };
    }
    Ok(config)
}

/// Translate a schema dictionary; all rule semantics remain in the Rust core.
pub(crate) fn parse_schema(schema: &PyDict) -> PyResult<groggy::GraphSchema> {
    let value = crate::ffi::utils::python_value_to_attr_value(schema)?;
    let RustAttrValue::Json(json) = value else {
        return Err(PyTypeError::new_err("Schema must be a dictionary"));
    };
    groggy::GraphSchema::from_json(&json).map_err(graph_error_to_py_err)
}

/// Decode all rows before a core upsert is allowed to mutate the graph.
fn parse_identity_rows<T>(
    data: &PyAny,
    mut parse: impl FnMut(&PyDict) -> PyResult<T>,
) -> PyResult<(Vec<T>, Vec<usize>, Vec<String>)> {
    let mut rows = Vec::new();
    let mut positions = Vec::new();
    let mut errors = Vec::new();
    for (position, row) in data.iter()?.enumerate() {
        let decoded = row
            .and_then(|row| row.downcast::<PyDict>().map_err(Into::into))
            .and_then(&mut parse);
        match decoded {
            Ok(row) => {
                rows.push(row);
                positions.push(position);
            }
            Err(error) => errors.push(format!("{position}: {error}")),
        }
    }
    Ok((rows, positions, errors))
}

fn required_identity_field(row: &PyDict, key: &str) -> PyResult<String> {
    row.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("Missing key: {key}")))?
        .extract()
}

/// Specification for neighborhood hop parameters
#[derive(Debug, Clone)]

/// Python wrapper for AggregationResult
#[pyclass(name = "AggregationResult")]
pub struct PyAggregationResult {
    pub value: f64,
}

pub(crate) fn parse_node_upserts(
    graph: &RustGraph,
    rows: &PyAny,
    uid_key: &str,
) -> PyResult<Vec<NodeUpsert>> {
    let (rows, positions, mut errors) = parse_identity_rows(rows, |row| {
        let id = required_identity_field(row, uid_key);
        let (attrs, mut errors) = parse_creation_attrs(Some(row), &[uid_key]);
        if let Err(error) = &id {
            errors.push(error.to_string());
        }
        if !errors.is_empty() {
            return Err(PyValueError::new_err(errors.join("; ")));
        }
        Ok(NodeUpsert {
            external_id: id?,
            attrs,
        })
    })?;
    if !errors.is_empty() {
        if let Err(error) = graph.validate_node_upserts(&rows, Some(&positions)) {
            errors.push(graph_error_to_py_err(error).to_string());
        }
        return Err(PyValueError::new_err(format!(
            "Invalid upsert rows: {}",
            errors.join("; ")
        )));
    }
    Ok(rows)
}
pub(crate) fn parse_edge_upserts(
    graph: &RustGraph,
    rows: &PyAny,
    uid_key: &str,
) -> PyResult<Vec<EdgeUpsert>> {
    if uid_key == "source" || uid_key == "target" {
        return Err(PyValueError::new_err(
            "The external-ID key must differ from source and target",
        ));
    }
    let (rows, positions, mut errors) = parse_identity_rows(rows, |row| {
        let id = required_identity_field(row, uid_key);
        let endpoint = |key: &str| -> PyResult<NodeId> {
            let value = row
                .get_item(key)?
                .ok_or_else(|| PyKeyError::new_err(format!("Missing key: {key}")))?;
            if let Ok(id) = value.extract::<NodeId>() {
                return Ok(id);
            }
            let name: String = value.extract()?;
            graph.node_by_external_id(&name, None).ok_or_else(|| {
                PyKeyError::new_err(format!("No active node with external ID {name:?}"))
            })
        };
        let source = endpoint("source");
        let target = endpoint("target");
        let (attrs, mut errors) = parse_creation_attrs(Some(row), &[uid_key, "source", "target"]);
        if let Err(error) = &id {
            errors.push(error.to_string());
        }
        if let Err(error) = &source {
            errors.push(format!("source: {error}"));
        }
        if let Err(error) = &target {
            errors.push(format!("target: {error}"));
        }
        if !errors.is_empty() {
            return Err(PyValueError::new_err(errors.join("; ")));
        }
        Ok(EdgeUpsert {
            external_id: id?,
            source: source?,
            target: target?,
            attrs,
        })
    })?;
    if !errors.is_empty() {
        if let Err(error) = graph.validate_edge_upserts(&rows, Some(&positions)) {
            errors.push(graph_error_to_py_err(error).to_string());
        }
        return Err(PyValueError::new_err(format!(
            "Invalid upsert rows: {}",
            errors.join("; ")
        )));
    }
    Ok(rows)
}

#[pymethods]
impl PyAggregationResult {
    #[getter]
    fn value(&self) -> f64 {
        self.value
    }

    fn __repr__(&self) -> String {
        format!("AggregationResult({})", self.value)
    }
}

/// Python wrapper for the main Graph
#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    pub inner: Rc<RefCell<RustGraph>>,
    // Cache for the full graph view to avoid expensive recreation
    pub cached_view: RefCell<Option<(u64, Py<PySubgraph>)>>, // (version, cached_subgraph)
}

impl Clone for PyGraph {
    fn clone(&self) -> Self {
        PyGraph {
            inner: self.inner.clone(),
            cached_view: RefCell::new(None), // Don't clone cache - let each instance build its own
        }
    }
}

impl PyGraph {
    /// Convert this graph to a SubgraphOperations trait object containing all nodes and edges
    ///
    /// This enables all SubgraphOperations methods to work on the full graph:
    /// - connected_components(), bfs(), dfs()
    /// - node_count(), edge_count(), neighbors(), degree()  
    /// - nodes_table(), edges_table()
    ///
    /// # Performance
    /// Creates lightweight wrapper with references to existing graph data
    pub fn as_subgraph(&self) -> PyResult<Box<dyn SubgraphOperations>> {
        let graph = self.inner.borrow();

        // Get all nodes and edges from the graph
        let all_nodes = graph.node_ids().into_iter().collect();
        let all_edges = graph.edge_ids().into_iter().collect();
        drop(graph); // Release borrow

        // Create subgraph containing the entire graph
        let full_subgraph = Subgraph::new(
            self.inner.clone(),
            all_nodes,
            all_edges,
            "full_graph".to_string(),
        );

        Ok(Box::new(full_subgraph))
    }

    /// Helper to execute operations on the cached full-graph view.
    ///
    /// This helper uses the cached `view()` method to avoid repeatedly creating
    /// subgraphs for delegation. The view is cached and reused across calls when
    /// the graph hasn't been modified.
    pub(crate) fn with_full_view<'py, R, F>(
        graph_ref: PyRef<'py, Self>,
        py: Python<'py>,
        f: F,
    ) -> PyResult<R>
    where
        F: for<'a> FnOnce(PyRef<'a, PySubgraph>, Python<'a>) -> PyResult<R>,
    {
        let view = Self::view(graph_ref, py)?;
        f(view.borrow(py), py)
    }

    /// Helper to call a Python method on the cached full-graph view.
    ///
    /// This is useful for methods that are defined in PySubgraph's #[pymethods]
    /// but not exposed as pub fn (Python-only visibility).
    pub(crate) fn call_on_view<'py>(
        graph_ref: PyRef<'py, Self>,
        py: Python<'py>,
        method_name: &str,
        args: &pyo3::types::PyTuple,
    ) -> PyResult<PyObject> {
        let view = Self::view(graph_ref, py)?;
        view.call_method1(py, method_name, args)
    }

    /// Helper to call a Python method with keyword args on the cached full-graph view.
    pub(crate) fn call_on_view_kwargs<'py>(
        graph_ref: PyRef<'py, Self>,
        py: Python<'py>,
        method_name: &str,
        args: &pyo3::types::PyTuple,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let view = Self::view(graph_ref, py)?;
        view.call_method(py, method_name, args, kwargs)
    }
}

#[pymethods]
impl PyGraph {
    /// Combine embedding matrices by checked node identity and column strategy.
    #[pyo3(signature = (matrices, strategy="interleave", chunk_size=1, weights=None))]
    fn combine_embeddings(
        &self,
        matrices: Vec<PyRef<'_, crate::ffi::storage::matrix::PyGraphMatrix>>,
        strategy: &str,
        chunk_size: usize,
        weights: Option<Vec<f64>>,
    ) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        use groggy::viz::embeddings::{combine_embedding_matrices, CombinationStrategy};
        let strategy = CombinationStrategy::from_name(strategy, chunk_size)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let matrices: Vec<_> = matrices.iter().map(|matrix| matrix.inner.clone()).collect();
        let inner = combine_embedding_matrices(
            &self.inner.borrow(),
            &matrices,
            weights.as_deref(),
            &strategy,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(crate::ffi::storage::matrix::PyGraphMatrix { inner })
    }

    /// Compute an owned bounded graph layout aligned with embedding row identity.
    #[pyo3(signature = (embedding, iterations=800, learning_rate=0.03, edge_cohesion_weight=1.0, repulsion_weight=0.1, repulsion_power=1.5, spread_weight=0.05, radius_clip=0.98, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn flat_embedding(
        &self,
        embedding: &crate::ffi::storage::matrix::PyGraphMatrix,
        iterations: usize,
        learning_rate: f64,
        edge_cohesion_weight: f64,
        repulsion_weight: f64,
        repulsion_power: f64,
        spread_weight: f64,
        radius_clip: f64,
        seed: u64,
    ) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        use groggy::viz::embeddings::flat_embedding::{
            compute_flat_embedding_matrix, FlatEmbedConfig,
        };
        let config = FlatEmbedConfig {
            iterations,
            learning_rate,
            edge_cohesion_weight,
            repulsion_weight,
            repulsion_power,
            spread_weight,
            radius_clip,
            seed,
        };
        let inner = compute_flat_embedding_matrix(&embedding.inner, &self.inner.borrow(), &config)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(crate::ffi::storage::matrix::PyGraphMatrix { inner })
    }

    /// Compute owned undirected spectral coordinates with sorted node row labels.
    #[pyo3(signature = (dimensions, normalized=true, eigenvalue_threshold=1e-8, skip_nullspace=true))]
    fn spectral_embedding(
        &self,
        dimensions: usize,
        normalized: bool,
        eigenvalue_threshold: f64,
        skip_nullspace: bool,
    ) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        use groggy::viz::embeddings::{spectral::SpectralEmbeddingBuilder, EmbeddingEngine};
        let engine = SpectralEmbeddingBuilder::new()
            .normalized(normalized)
            .eigenvalue_threshold(eigenvalue_threshold)
            .skip_constant_eigenvector(skip_nullspace)
            .build();
        let inner = engine
            .compute_embedding(&self.inner.borrow(), dimensions)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(crate::ffi::storage::matrix::PyGraphMatrix { inner })
    }

    #[new]
    #[pyo3(signature = (directed = None, _config = None, *, manual = false, config = None))]
    fn new(
        directed: Option<bool>,
        _config: Option<&PyDict>,
        manual: bool,
        config: Option<&PyDict>,
    ) -> PyResult<Self> {
        if _config.is_some() && config.is_some() {
            return Err(PyValueError::new_err(
                "Supply config or legacy _config, not both",
            ));
        }
        let config = parse_config(config.or(_config), directed)?;
        let mut rust_graph = RustGraph::with_config(config).map_err(graph_error_to_py_err)?;
        rust_graph
            .set_automatic_publication(!manual)
            .map_err(graph_error_to_py_err)?;
        let inner = rust_graph.into_shared();
        Ok(Self {
            inner,
            cached_view: RefCell::new(None),
        })
    }

    /// Export active topology and typed properties; history/pool recovery is separate.
    fn save_to_path(&self, path: std::path::PathBuf) -> PyResult<()> {
        self.inner
            .borrow()
            .save_to_path(&path)
            .map_err(graph_error_to_py_err)
    }

    /// Load a typed active graph export or a legacy unversioned JSON graph.
    #[staticmethod]
    #[pyo3(signature=(path,*,manual=false))]
    fn load_from_path(path: std::path::PathBuf, manual: bool) -> PyResult<Self> {
        let graph = if manual {
            RustGraph::load_from_path(&path)
        } else {
            RustGraph::load_publishing_from_path(&path)
        }
        .map_err(graph_error_to_py_err)?;
        Ok(Self {
            inner: graph.into_shared(),
            cached_view: RefCell::new(None),
        })
    }

    /// Verify all retained graph states without publishing or rebuilding indexes.
    fn check_integrity(&self, py: Python<'_>) -> PyResult<PyObject> {
        let plan = self
            .inner
            .borrow()
            .prepare_integrity_check()
            .map_err(graph_error_to_py_err)?;
        let report = py
            .allow_threads(move || plan.execute())
            .map_err(graph_error_to_py_err)?;
        integrity_report_to_py(py, report)
    }

    /// Save all retained graph state without publishing pending manual changes.
    fn save_archive(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        let prepared = self
            .inner
            .borrow()
            .prepare_archive()
            .map_err(graph_error_to_py_err)?;
        py.allow_threads(move || prepared.save(&path))
            .map_err(graph_error_to_py_err)
    }

    /// Restore an independent complete graph, retaining its publication policy.
    #[staticmethod]
    fn load_archive(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let prepared = py
            .allow_threads(move || groggy::api::graph::PreparedArchive::read(&path))
            .map_err(graph_error_to_py_err)?;
        let graph = prepared.restore().map_err(graph_error_to_py_err)?;
        Ok(Self {
            inner: graph.into_shared(),
            cached_view: RefCell::new(None),
        })
    }

    /// Replace a node's semantic labels; independent of base/meta entity_type.
    fn set_node_labels(&mut self, node: NodeId, labels: Vec<String>) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_node_labels(node, labels)
            .map_err(graph_error_to_py_err)
    }

    fn node_labels(&self, node: NodeId) -> PyResult<Vec<String>> {
        self.inner
            .borrow()
            .node_labels(node)
            .map_err(graph_error_to_py_err)
    }

    #[pyo3(signature = (edge, kind=None))]
    fn set_edge_relationship_type(&mut self, edge: EdgeId, kind: Option<String>) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_edge_relationship_type(edge, kind)
            .map_err(graph_error_to_py_err)
    }

    fn edge_relationship_type(&self, edge: EdgeId) -> PyResult<Option<String>> {
        self.inner
            .borrow()
            .edge_relationship_type(edge)
            .map_err(graph_error_to_py_err)
    }

    fn nodes_with_label(&self, label: &str) -> PyResult<Vec<NodeId>> {
        self.inner
            .borrow()
            .nodes_with_label(label)
            .map_err(graph_error_to_py_err)
    }

    fn edges_with_relationship_type(&self, kind: &str) -> PyResult<Vec<EdgeId>> {
        self.inner
            .borrow()
            .edges_with_relationship_type(kind)
            .map_err(graph_error_to_py_err)
    }

    /// Audit an explicit schema; this does not install a write policy.
    fn validate_schema(&self, py: Python, schema: &PyDict) -> PyResult<PyObject> {
        let schema = parse_schema(schema)?;
        let violations = self
            .inner
            .borrow()
            .validate_schema(&schema)
            .map_err(graph_error_to_py_err)?;
        let output = PyList::empty(py);
        for violation in violations {
            let row = PyDict::new(py);
            let (kind, id) = match violation.entity {
                groggy::ExternalEntityRef::Node(id) => ("node", id),
                groggy::ExternalEntityRef::Edge(id) => ("edge", id),
            };
            row.set_item("entity_type", kind)?;
            row.set_item("entity_id", id)?;
            row.set_item("selector", violation.selector)?;
            row.set_item("property", violation.property)?;
            row.set_item("message", violation.message)?;
            output.append(row)?;
        }
        Ok(output.to_object(py))
    }

    /// Start an optimistic direct-write session; each call publishes one batch.
    #[pyo3(signature = (*, author="writer", schema=None))]
    fn writer(
        slf: PyRef<'_, Self>,
        py: Python,
        author: &str,
        schema: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        if let Some(schema) = schema {
            parse_schema(schema)?;
        }
        let class = py.import("groggy.writer")?.getattr("GraphWriter")?;
        Ok(class
            .call1((slf.into_py(py), author, schema))?
            .to_object(py))
    }

    #[getter]
    fn automatic_publication(&self) -> bool {
        self.inner.borrow().automatic_publication()
    }

    /// Check if this graph is directed
    #[getter]
    fn is_directed(&self) -> bool {
        self.inner.borrow().is_directed()
    }

    /// Check if this graph is undirected
    #[getter]
    fn is_undirected(&self) -> bool {
        self.inner.borrow().is_undirected()
    }

    // === CORE GRAPH OPERATIONS ===

    #[pyo3(signature = (node_id = None, **kwargs))]
    fn add_node(&mut self, node_id: Option<NodeId>, kwargs: Option<&PyDict>) -> PyResult<NodeId> {
        if node_id.is_some() {
            return Err(PyValueError::new_err(
                "Custom internal node IDs are unsupported; use upsert_nodes with external_id",
            ));
        }
        let (attrs, mut errors) = parse_creation_attrs(kwargs, &[]);
        if !errors.is_empty() {
            if let Err(error) = self
                .inner
                .borrow()
                .validate_new_node_attrs(std::slice::from_ref(&attrs))
            {
                errors.push(error.to_string());
            }
            return Err(PyValueError::new_err(errors.join("; ")));
        }
        let actual_node_id = self
            .inner
            .borrow_mut()
            .add_nodes_with_attrs(vec![attrs])
            .map_err(graph_error_to_py_err)?[0];

        Ok(actual_node_id)
    }

    #[pyo3(signature = (data, uid_key = None))]
    fn add_nodes(&mut self, data: &PyAny, uid_key: Option<String>) -> PyResult<PyObject> {
        // Fast path optimization: Check for integer first (most common case)
        if let Ok(count) = data.extract::<usize>() {
            // Old API: add_nodes(5) -> [0, 1, 2, 3, 4] - fastest path
            let node_ids = self
                .inner
                .borrow_mut()
                .insert_nodes(count)
                .map_err(graph_error_to_py_err)?;
            return Python::with_gil(|py| Ok(node_ids.to_object(py)));
        }

        // Only use Python::with_gil for complex operations
        Python::with_gil(|py| {
            if let Ok(node_data_list) = data.extract::<Vec<&PyAny>>() {
                let mut rows = Vec::with_capacity(node_data_list.len());
                let mut user_ids = Vec::with_capacity(node_data_list.len());
                let mut errors = Vec::new();
                for (position, node_data) in node_data_list.iter().enumerate() {
                    let node_dict = match node_data.extract::<&PyDict>() {
                        Ok(dict) => dict,
                        Err(error) => {
                            errors.push(format!("{position}: expected a node dictionary: {error}"));
                            rows.push(HashMap::new());
                            continue;
                        }
                    };
                    if let Some(key) = &uid_key {
                        let user_id = (|| -> PyResult<String> {
                            node_dict
                                .get_item(key)?
                                .ok_or_else(|| PyKeyError::new_err(format!("Missing key: {key}")))?
                                .extract::<String>()
                        })();
                        match user_id {
                            Ok(id) => user_ids.push(id),
                            Err(error) => errors.push(format!("{position}.{key}: {error}")),
                        }
                    }
                    let (attrs, conversion_errors) = parse_creation_attrs(Some(node_dict), &[]);
                    errors.extend(
                        conversion_errors
                            .into_iter()
                            .map(|error| format!("{position}: {error}")),
                    );
                    rows.push(attrs);
                }
                if !errors.is_empty() {
                    if let Err(error) = self.inner.borrow().validate_new_node_attrs(&rows) {
                        errors.push(error.to_string());
                    }
                    return Err(PyValueError::new_err(format!(
                        "Invalid nodes at batch positions {}",
                        errors.join("; ")
                    )));
                }
                if let Some(key) = &uid_key {
                    self.inner
                        .borrow()
                        .validate_new_text_ids(key, &user_ids)
                        .map_err(graph_error_to_py_err)?;
                }
                let node_ids = self
                    .inner
                    .borrow_mut()
                    .add_nodes_with_attrs(rows)
                    .map_err(graph_error_to_py_err)?;
                let id_mapping: HashMap<_, _> =
                    user_ids.into_iter().zip(node_ids.iter().copied()).collect();

                // Return the mapping if id_key was provided, otherwise return node IDs
                if uid_key.is_some() {
                    Ok(id_mapping.to_object(py))
                } else {
                    Ok(node_ids.to_object(py))
                }
            } else {
                Err(PyErr::new::<PyTypeError, _>(
                    "add_nodes expects either an integer count or a list of dictionaries",
                ))
            }
        })
    }

    /// Upsert rows by a managed external ID, preserving unspecified properties.
    #[pyo3(signature = (rows, uid_key = "external_id"))]
    fn upsert_nodes(&mut self, rows: &PyAny, uid_key: &str) -> PyResult<Vec<NodeId>> {
        let rows = parse_node_upserts(&self.inner.borrow(), rows, uid_key)?;
        self.inner
            .borrow_mut()
            .upsert_nodes(rows)
            .map_err(graph_error_to_py_err)
    }

    /// Publish managed node upserts and identity-index updates through one checked batch.
    #[pyo3(signature = (rows, expected_state, *, author="upsert", uid_key="external_id", schema=None, effective_at=None))]
    fn publish_upsert_nodes(
        &mut self,
        py: Python,
        rows: &PyAny,
        expected_state: StateId,
        author: &str,
        uid_key: &str,
        schema: Option<&PyDict>,
        effective_at: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let rows = parse_node_upserts(&self.inner.borrow(), rows, uid_key)?;
        let schema = schema.map(parse_schema).transpose()?;
        let at = effective_at
            .map(crate::ffi::api::graph_batch::timestamp)
            .transpose()?;
        let mut graph = self.inner.borrow_mut();
        let mut writer = graph
            .writer(author)
            .expect_state(expected_state)
            .with_effective_at(at);
        if let Some(schema) = &schema {
            writer = writer.with_schema(schema);
        }
        let ids = writer.upsert_nodes(rows).map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("node_ids", ids)?;
        output.set_item("state_id", writer.state_id())?;
        Ok(output.to_object(py))
    }

    /// Publish managed edge upserts and identity-index updates through one checked batch.
    #[pyo3(signature = (rows, expected_state, *, author="upsert", uid_key="external_id", schema=None, effective_at=None))]
    fn publish_upsert_edges(
        &mut self,
        py: Python,
        rows: &PyAny,
        expected_state: StateId,
        author: &str,
        uid_key: &str,
        schema: Option<&PyDict>,
        effective_at: Option<&PyAny>,
    ) -> PyResult<PyObject> {
        let rows = parse_edge_upserts(&self.inner.borrow(), rows, uid_key)?;
        let schema = schema.map(parse_schema).transpose()?;
        let at = effective_at
            .map(crate::ffi::api::graph_batch::timestamp)
            .transpose()?;
        let mut graph = self.inner.borrow_mut();
        let mut writer = graph
            .writer(author)
            .expect_state(expected_state)
            .with_effective_at(at);
        if let Some(schema) = &schema {
            writer = writer.with_schema(schema);
        }
        let ids = writer.upsert_edges(rows).map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("edge_ids", ids)?;
        output.set_item("state_id", writer.state_id())?;
        Ok(output.to_object(py))
    }

    /// Edge endpoints can be internal numeric IDs or managed external string IDs.
    #[pyo3(signature = (rows, uid_key = "external_id"))]
    fn upsert_edges(&mut self, rows: &PyAny, uid_key: &str) -> PyResult<Vec<EdgeId>> {
        let rows = parse_edge_upserts(&self.inner.borrow(), rows, uid_key)?;
        self.inner
            .borrow_mut()
            .upsert_edges(rows)
            .map_err(graph_error_to_py_err)
    }

    #[pyo3(signature = (external_id, group_id = None))]
    fn node_by_external_id(&self, external_id: &str, group_id: Option<&str>) -> Option<NodeId> {
        self.inner
            .borrow()
            .node_by_external_id(external_id, group_id)
    }

    #[pyo3(signature = (external_id, group_id = None))]
    fn edge_by_external_id(&self, external_id: &str, group_id: Option<&str>) -> Option<EdgeId> {
        self.inner
            .borrow()
            .edge_by_external_id(external_id, group_id)
    }

    fn set_node_external_id(&mut self, node: NodeId, external_id: String) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_node_external_id(node, external_id)
            .map_err(graph_error_to_py_err)
    }

    fn set_edge_external_id(&mut self, edge: EdgeId, external_id: String) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .set_edge_external_id(edge, external_id)
            .map_err(graph_error_to_py_err)
    }

    fn allocated_entity_for_external_id(&self, external_id: &str) -> Option<(&'static str, usize)> {
        self.inner
            .borrow()
            .allocated_entity_for_external_id(external_id)
            .map(|entity| match entity {
                ExternalEntityRef::Node(id) => ("node", id),
                ExternalEntityRef::Edge(id) => ("edge", id),
            })
    }

    /// Helper method to resolve string ID to NodeId using uid_key attribute
    fn resolve_string_id_to_node(&self, string_id: &str, uid_key: &str) -> PyResult<NodeId> {
        self.inner
            .borrow()
            .node_by_unique_text_attr(uid_key, string_id)
            .map_err(graph_error_to_py_err)?
            .ok_or_else(|| {
                PyKeyError::new_err(format!("No node found with {uid_key}={string_id:?}"))
            })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={})",
            self.node_count(),
            self.edge_count()
        )
    }

    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.borrow_mut().node_ids().len()
    }

    /// Check if a node exists in the graph - DELEGATED
    fn has_node(&self, node_id: NodeId) -> bool {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.contains_node(node_id),
            Err(_) => false,
        }
    }

    /// Check if an edge exists in the graph - DELEGATED  
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.contains_edge(edge_id),
            Err(_) => false,
        }
    }

    /// Get the number of nodes in the graph - DELEGATED
    fn node_count(&self) -> usize {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.node_count(),
            Err(_) => 0,
        }
    }

    /// Get the number of edges in the graph - DELEGATED
    fn edge_count(&self) -> usize {
        // Pure delegation to SubgraphOperations
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.edge_count(),
            Err(_) => 0,
        }
    }

    /// Calculate graph density (number of edges / number of possible edges) - PURE DELEGATION
    fn density(&self) -> f64 {
        // Pure delegation to SubgraphOperations trait
        match self.as_subgraph() {
            Ok(subgraph) => subgraph.density(),
            Err(_) => 0.0,
        }
    }

    // === TOPOLOGY OPERATIONS ===

    /// Check if node exists - DELEGATED (same as has_node)
    fn contains_node(&self, node: NodeId) -> bool {
        // Pure delegation to SubgraphOperations (same logic as has_node)
        self.has_node(node)
    }

    /// Check if edge exists - DELEGATED (same as has_edge)  
    fn contains_edge(&self, edge: EdgeId) -> bool {
        // Pure delegation to SubgraphOperations (same logic as has_edge)
        self.has_edge(edge)
    }

    fn edge_endpoints(&self, edge: EdgeId) -> PyResult<(NodeId, NodeId)> {
        self.inner
            .borrow_mut()
            .edge_endpoints(edge)
            .map_err(graph_error_to_py_err)
    }

    /// Get all active node IDs as IntArray
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<crate::ffi::storage::num_array::PyIntArray>> {
        let node_ids = self.inner.borrow_mut().node_ids();
        let py_int_array = crate::ffi::storage::num_array::PyIntArray::from_node_ids(node_ids)?;
        Py::new(py, py_int_array)
    }

    /// Check whether the graph contains any nodes or edges
    fn is_empty(&self) -> bool {
        let graph_ref = self.inner.borrow();
        graph_ref.is_empty()
    }

    /// Get all active edge IDs as IntArray  
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<crate::ffi::storage::num_array::PyIntArray>> {
        let edge_ids = self.inner.borrow_mut().edge_ids();
        let py_int_array = crate::ffi::storage::num_array::PyIntArray::from_node_ids(edge_ids)?;
        Py::new(py, py_int_array)
    }

    /// Get nodes accessor for fluent API (g.nodes property)
    #[getter]
    fn nodes(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        let graph_ref = self_.into();
        PyGraph::create_nodes_accessor_internal(graph_ref, py)
    }

    /// Get edges accessor for fluent API (g.edges property)
    #[getter]
    fn edges(self_: PyRef<Self>, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        let graph_ref = self_.into();
        PyGraph::create_edges_accessor_internal(graph_ref, py)
    }

    /// Delegate item access to the nodes accessor so g[key] mirrors g.nodes[key]
    #[pyo3(name = "__getitem__")]
    fn getitem(self_: PyRef<Self>, py: Python, key: &PyAny) -> PyResult<PyObject> {
        let graph_ref: Py<PyGraph> = self_.into();
        let nodes_accessor = PyGraph::create_nodes_accessor_internal(graph_ref, py)?;
        let key_obj = key.to_object(py);
        nodes_accessor
            .as_ref(py)
            .call_method1("__getitem__", (key_obj,))
            .map(|obj| obj.to_object(py))
    }

    /// Add multiple edges at once
    #[pyo3(signature = (edges, node_mapping = None, uid_key = None, source = None, target = None, warm_cache = None))]
    // TODO: Refactor to use config/builder pattern
    #[allow(clippy::too_many_arguments)]
    fn add_edges(
        &mut self,
        py: Python,
        edges: &PyAny,
        node_mapping: Option<std::collections::HashMap<String, NodeId>>,
        uid_key: Option<String>,
        source: Option<String>,
        target: Option<String>,
        warm_cache: Option<bool>,
    ) -> PyResult<Vec<EdgeId>> {
        // Parse and validate every input before mutating the graph. This also ensures
        // conversion errors in a later row cannot leave earlier edges inserted.
        let source_field = source.as_deref().unwrap_or("source");
        let target_field = target.as_deref().unwrap_or("target");
        let mut parsed: Vec<(
            NodeId,
            NodeId,
            std::collections::HashMap<String, RustAttrValue>,
        )> = Vec::new();
        let mut errors = Vec::new();
        let rows = edges.iter().map_err(|_| {
            PyTypeError::new_err("add_edges expects an iterable of edge tuples or dictionaries")
        })?;

        for (position, row) in rows.enumerate() {
            let mut conversion_errors = Vec::new();
            let result = (|| -> PyResult<_> {
                let row = row?;
                let (source_value, target_value, attributes) =
                    if let Ok(edge_tuple) = row.extract::<&PyTuple>() {
                        if edge_tuple.len() != 2 && edge_tuple.len() != 3 {
                            return Err(PyTypeError::new_err(
                                "edge tuples must have two endpoints and optional attributes",
                            ));
                        }
                        let attrs = if edge_tuple.len() == 3 {
                            Some(edge_tuple.get_item(2)?)
                        } else {
                            None
                        };
                        (edge_tuple.get_item(0)?, edge_tuple.get_item(1)?, attrs)
                    } else if let Ok(edge_list) = row.extract::<&PyList>() {
                        if edge_list.len() != 2 && edge_list.len() != 3 {
                            return Err(PyTypeError::new_err(
                                "edge lists must have two endpoints and optional attributes",
                            ));
                        }
                        let attrs = if edge_list.len() == 3 {
                            Some(edge_list.get_item(2)?)
                        } else {
                            None
                        };
                        (edge_list.get_item(0)?, edge_list.get_item(1)?, attrs)
                    } else if let Ok(edge_dict) = row.extract::<&PyDict>() {
                        let source_value = edge_dict.get_item(source_field)?.ok_or_else(|| {
                            PyKeyError::new_err(format!("Missing source field: {source_field}"))
                        })?;
                        let target_value = edge_dict.get_item(target_field)?.ok_or_else(|| {
                            PyKeyError::new_err(format!("Missing target field: {target_field}"))
                        })?;
                        (source_value, target_value, Some(row))
                    } else {
                        return Err(PyTypeError::new_err(
                            "each edge must be a tuple or dictionary",
                        ));
                    };

                let resolve = |value: &PyAny| -> PyResult<NodeId> {
                    if let Ok(id) = value.extract::<NodeId>() {
                        return Ok(id);
                    }
                    let external_id: String = value.extract().map_err(|_| {
                        PyTypeError::new_err("edge endpoints must be numeric or string IDs")
                    })?;
                    if let Some(mapping) = &node_mapping {
                        return mapping.get(&external_id).copied().ok_or_else(|| {
                            PyKeyError::new_err(format!("Node {external_id} not found in mapping"))
                        });
                    }
                    if let Some(key) = &uid_key {
                        return self.resolve_string_id_to_node(&external_id, key);
                    }
                    Err(PyTypeError::new_err(
                        "String node IDs require node_mapping or uid_key",
                    ))
                };
                let source_id = resolve(source_value)?;
                let target_id = resolve(target_value)?;

                let mut attrs = std::collections::HashMap::new();
                if let Some(attributes) = attributes {
                    if !attributes.is_none() {
                        let attrs_dict: &PyDict = attributes.extract().map_err(|_| {
                            PyTypeError::new_err("edge attributes must be a dictionary or None")
                        })?;
                        let excluded = if row.extract::<&PyDict>().is_ok() {
                            vec![source_field, target_field]
                        } else {
                            Vec::new()
                        };
                        let (values, errors) = parse_creation_attrs(Some(attrs_dict), &excluded);
                        attrs = values;
                        conversion_errors = errors;
                    }
                }
                Ok((source_id, target_id, attrs))
            })();
            match result {
                Ok(edge) => {
                    if let Err(error) = self
                        .inner
                        .borrow()
                        .validate_new_edge_attrs(std::slice::from_ref(&edge.2))
                    {
                        errors.push(format!("{position}: {error}"));
                    }
                    let graph = self.inner.borrow();
                    if !graph.has_node(edge.0) || !graph.has_node(edge.1) {
                        errors.push(format!(
                            "{position}: invalid endpoints ({}, {})",
                            edge.0, edge.1
                        ));
                    } else {
                        parsed.push(edge);
                    }
                }
                Err(error) => errors.push(format!("{position}: {error}")),
            }
            errors.extend(
                conversion_errors
                    .into_iter()
                    .map(|error| format!("{position}: {error}")),
            );
        }
        if !errors.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Invalid edges at batch positions {}",
                errors.join(", ")
            )));
        }

        let edge_ids = self
            .inner
            .borrow_mut()
            .add_edges_with_attrs(parsed)
            .map_err(graph_error_to_py_err)?;
        if warm_cache.unwrap_or(false) {
            self.warm_caches_after_bulk_operation(py)?;
        }
        Ok(edge_ids)
    }

    /// Retain a frozen source revision; preparation runs without the GIL.
    #[pyo3(signature=(source, kind="group"))]
    fn retain_group(
        &self,
        py: Python<'_>,
        source: &crate::ffi::subgraphs::view::PyGraphView,
        kind: &str,
    ) -> PyResult<usize> {
        let revision = py
            .allow_threads(|| {
                groggy::storage::retained_group::RetainedGroup::new(
                    source.inner.clone(),
                    kind.to_owned(),
                )
            })
            .map_err(graph_error_to_py_err)?;
        self.inner
            .borrow_mut()
            .retain_group_revision(revision)
            .map_err(graph_error_to_py_err)
    }
    /// Create a meta-node referencing an explicitly retained source revision.
    #[pyo3(signature=(group, *, effective_at=None))]
    fn create_meta_node(&self, group: usize, effective_at: Option<&PyAny>) -> PyResult<usize> {
        let at = effective_at
            .map(crate::ffi::api::graph_batch::timestamp)
            .transpose()?;
        self.inner
            .borrow_mut()
            .create_meta_node_at(group, at)
            .map_err(graph_error_to_py_err)
    }
    /// Pin legacy membership using an explicitly chosen retained source state.
    fn pin_legacy_group(
        &self,
        group: usize,
        source: &crate::ffi::subgraphs::view::PyGraphView,
    ) -> PyResult<usize> {
        self.inner
            .borrow_mut()
            .pin_legacy_group(group, &source.inner)
            .map_err(graph_error_to_py_err)
    }
    /// Reconstruct immutable effective-time state from a pinned published head.
    #[pyo3(signature=(effective_at, *, recorded_by=None, legacy_times=None, group_sources=None))]
    fn read_view_at(
        &self,
        py: Python<'_>,
        effective_at: &PyAny,
        recorded_by: Option<&PyAny>,
        legacy_times: Option<&PyDict>,
        group_sources: Option<&PyDict>,
    ) -> PyResult<crate::ffi::subgraphs::view::PyGraphView> {
        let effective_at = crate::ffi::api::graph_batch::timestamp(effective_at)?;
        let recorded_by = recorded_by
            .map(crate::ffi::api::graph_batch::timestamp)
            .transpose()?;
        let legacy = crate::ffi::api::graph_batch::parse_legacy_times(legacy_times)?;
        let sources = crate::ffi::api::graph_batch::parse_group_sources(group_sources)?;
        let options = groggy::state::effective_read::EffectiveReadOptions {
            recorded_by,
            legacy_publications: legacy,
            group_sources: sources,
        };
        let plan = self
            .inner
            .borrow()
            .prepare_effective_read(effective_at, options)
            .map_err(graph_error_to_py_err)?;
        py.allow_threads(move || plan.read_view())
            .map(|inner| crate::ffi::subgraphs::view::PyGraphView { inner })
            .map_err(graph_error_to_py_err)
    }

    /// Capture immutable current state; later working-graph writes do not alter this view.
    fn read_view(&self) -> PyResult<crate::ffi::subgraphs::view::PyGraphView> {
        self.inner
            .borrow()
            .read_view()
            .map(|inner| crate::ffi::subgraphs::view::PyGraphView { inner })
            .map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    /// Filter nodes using NodeFilter object or string query
    ///
    /// OPTIMIZED: Direct implementation avoiding expensive view() creation overhead
    fn filter_nodes(slf: PyRefMut<Self>, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for NodeFilter object first (most common case)
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            // Direct NodeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser
            let mut parser = groggy::query::QueryParser::new();
            parser.parse_node_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) =
            Self::validate_node_filter_attributes(&slf.inner.borrow(), &node_filter)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Attribute '{}' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes.", attr_name)
            ));
        }

        let filtered_nodes = slf
            .inner
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;

        // OPTIMIZED: Pre-allocate HashSet capacity for results only
        use std::collections::HashSet;
        let node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();

        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = slf.inner.borrow().get_columnar_topology();
        let mut induced_edges = Vec::new();

        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];

            // O(1) HashSet lookups instead of O(n) Vec::contains
            if node_set.contains(&source) && node_set.contains(&target) {
                induced_edges.push(edge_id);
            }
        }

        // Create edge HashSet - only for the filtered results
        let edge_set: HashSet<EdgeId> = induced_edges.iter().copied().collect();

        // Use optimized fast Subgraph constructor
        let subgraph = Subgraph::new(
            slf.inner.clone(),
            node_set,
            edge_set,
            "filtered_nodes".to_string(),
        );
        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Filter edges using EdgeFilter object or string query
    ///
    /// OPTIMIZED: Direct implementation avoiding expensive view() creation overhead
    fn filter_edges(slf: PyRefMut<Self>, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Fast path optimization: Check for EdgeFilter object first (most common case)
        let edge_filter = if let Ok(filter_obj) = filter.extract::<PyEdgeFilter>() {
            // Direct EdgeFilter object - fastest path
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser
            let mut parser = groggy::query::QueryParser::new();
            parser.parse_edge_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query",
            ));
        };

        // Validate that any referenced attributes exist in the graph
        if let Err(attr_name) =
            Self::validate_edge_filter_attributes(&slf.inner.borrow(), &edge_filter)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                format!("Attribute '{}' does not exist on any edges in the graph. Use graph.edges.table().columns to see available attributes.", attr_name)
            ));
        }

        let filtered_edges = slf
            .inner
            .borrow_mut()
            .find_edges(edge_filter)
            .map_err(graph_error_to_py_err)?;

        // Calculate nodes that are connected by the filtered edges
        use std::collections::HashSet;
        let mut nodes = HashSet::new();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = slf.inner.borrow().edge_endpoints(edge_id) {
                nodes.insert(source);
                nodes.insert(target);
            }
        }

        // Create edge HashSet - only for the filtered results
        let edge_set: HashSet<EdgeId> = filtered_edges.iter().copied().collect();

        // Use optimized fast Subgraph constructor
        let subgraph = Subgraph::new(
            slf.inner.clone(),
            nodes,
            edge_set,
            "filtered_edges".to_string(),
        );
        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Get analytics module for this graph
    // analytics() method deleted - functionality moved to direct PyGraph delegation methods
    /// Group nodes by attribute value and compute aggregates for each group
    pub fn group_nodes_by_attribute(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult> {
        let results = self
            .inner
            .borrow_mut()
            .group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;

        Python::with_gil(|py| {
            // Convert HashMap to Python dict
            let dict = PyDict::new(py);
            for (attr_value, agg_result) in results {
                let py_attr_value = PyAttrValue { inner: attr_value };
                let py_agg_result = PyAggregationResult {
                    value: agg_result.value,
                };
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
    }

    /// Group nodes by attribute value and compute aggregates (public method for benchmarks)
    pub fn group_by(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult> {
        self.group_nodes_by_attribute_internal(attribute, aggregation_attr, operation)
    }

    // === REMOVAL OPERATIONS ===

    /// Remove a single node from the graph
    fn remove_node(&mut self, node: NodeId) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .remove_node(node)
            .map_err(graph_error_to_py_err)
    }

    /// Remove a single edge from the graph
    fn remove_edge(&mut self, edge: EdgeId) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .remove_edge(edge)
            .map_err(graph_error_to_py_err)
    }

    /// Remove multiple nodes from the graph
    fn remove_nodes(&mut self, nodes: Vec<NodeId>) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .remove_nodes(&nodes)
            .map_err(graph_error_to_py_err)
    }

    /// Remove multiple edges from the graph
    fn remove_edges(&mut self, edges: Vec<EdgeId>) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .remove_edges(&edges)
            .map_err(graph_error_to_py_err)
    }

    // === SINGLE EDGE ADDITION ===

    /// Add a single edge to the graph with support for string IDs and attributes
    #[pyo3(signature = (source, target, uid_key = None, **kwargs))]
    fn add_edge(
        &mut self,
        _py: Python,
        source: &PyAny,
        target: &PyAny,
        uid_key: Option<String>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<EdgeId> {
        // Try to extract as NodeId first (most common case)
        let source_id = if let Ok(node_id) = source.extract::<NodeId>() {
            node_id
        } else if let Ok(string_id) = source.extract::<String>() {
            if let Some(ref key) = uid_key {
                self.resolve_string_id_to_node(&string_id, key)?
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "String node IDs require uid_key parameter",
                ));
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Source must be NodeId or string",
            ));
        };

        let target_id = if let Ok(node_id) = target.extract::<NodeId>() {
            node_id
        } else if let Ok(string_id) = target.extract::<String>() {
            if let Some(ref key) = uid_key {
                self.resolve_string_id_to_node(&string_id, key)?
            } else {
                return Err(PyErr::new::<PyTypeError, _>(
                    "String node IDs require uid_key parameter",
                ));
            }
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "Target must be NodeId or string",
            ));
        };

        let (attrs, mut errors) = parse_creation_attrs(kwargs, &[]);
        if let Err(error) = self
            .inner
            .borrow()
            .validate_new_edge_attrs(std::slice::from_ref(&attrs))
        {
            errors.push(error.to_string());
        }
        if !errors.is_empty() {
            let graph = self.inner.borrow();
            if !graph.has_node(source_id) {
                errors.push(format!("source: node {source_id} is not active"));
            }
            if !graph.has_node(target_id) {
                errors.push(format!("target: node {target_id} is not active"));
            }
            return Err(PyValueError::new_err(errors.join("; ")));
        }
        let edge_id = self
            .inner
            .borrow_mut()
            .add_edges_with_attrs(vec![(source_id, target_id, attrs)])
            .map_err(graph_error_to_py_err)?[0];

        Ok(edge_id)
    }

    // === ATTRIBUTE OPERATIONS (direct core delegation) ===

    /// Set single node attribute - delegates to PyGraphAttrMut
    #[pyo3(signature = (node, attr, value))]
    fn set_node_attr(
        &mut self,
        node: NodeId,
        attr: String,
        value: &PyAny,
        py: Python,
    ) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_node_attr(py, node, attr, value)
    }

    /// Get single node attribute - delegates to PyGraphAttr
    #[pyo3(signature = (node, attr, default = None))]
    fn get_node_attr(
        &self,
        node: NodeId,
        attr: String,
        default: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_node_attr(py, node, attr, default)
    }

    /// Set single edge attribute - delegates to PyGraphAttrMut
    #[pyo3(signature = (edge, attr, value))]
    fn set_edge_attr(
        &mut self,
        edge: EdgeId,
        attr: String,
        value: &PyAny,
        py: Python,
    ) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_edge_attr(py, edge, attr, value)
    }

    /// Get single edge attribute - delegates to PyGraphAttr
    #[pyo3(signature = (edge, attr, default = None))]
    fn get_edge_attr(
        &self,
        edge: EdgeId,
        attr: String,
        default: Option<&PyAny>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_edge_attr(py, edge, attr, default)
    }

    /// Set bulk node attributes - delegates to PyGraphAttrMut
    #[pyo3(signature = (attrs_dict))]
    fn set_node_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_node_attrs(py, attrs_dict)
    }

    /// Get bulk node attributes - delegates to PyGraphAttr
    #[pyo3(signature = (nodes, attrs))]
    fn get_node_attrs(
        &self,
        nodes: Vec<NodeId>,
        attrs: Vec<AttrName>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_node_attrs(py, nodes, attrs)
    }

    /// Set bulk edge attributes - delegates to PyGraphAttrMut
    #[pyo3(signature = (attrs_dict))]
    fn set_edge_attrs(&mut self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let mut attr_handler = PyGraphAttrMut::new(self.inner.clone());
        attr_handler.set_edge_attrs(py, attrs_dict)
    }

    /// Get bulk edge attributes - delegates to PyGraphAttr
    #[pyo3(signature = (edges, attrs))]
    fn get_edge_attrs(
        &self,
        edges: Vec<EdgeId>,
        attrs: Vec<String>,
        py: Python,
    ) -> PyResult<PyObject> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        attr_handler.get_edge_attrs(py, edges, attrs)
    }

    /// Check if node has specific attribute - delegates to PyGraphAttr
    #[pyo3(signature = (node_id, attr_name))]
    fn has_node_attribute(&self, node_id: NodeId, attr_name: &str, py: Python) -> PyResult<bool> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.has_node_attribute(py, node_id, attr_name))
    }

    /// Check if edge has specific attribute - delegates to PyGraphAttr
    #[pyo3(signature = (edge_id, attr_name))]
    fn has_edge_attribute(&self, edge_id: EdgeId, attr_name: &str, py: Python) -> PyResult<bool> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.has_edge_attribute(py, edge_id, attr_name))
    }

    /// Get all attribute keys for a node - delegates to PyGraphAttr
    #[pyo3(signature = (node_id))]
    fn node_attribute_keys(&self, node_id: NodeId, py: Python) -> PyResult<Vec<String>> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.node_attribute_keys(py, node_id))
    }

    /// Get all attribute keys for an edge - delegates to PyGraphAttr
    #[pyo3(signature = (edge_id))]
    fn edge_attribute_keys(&self, edge_id: EdgeId, py: Python) -> PyResult<Vec<String>> {
        let attr_handler = PyGraphAttr::new(self.inner.clone());
        Ok(attr_handler.edge_attribute_keys(py, edge_id))
    }

    /// Explicit absence; use a None assignment for an explicit null instead.
    fn unset_node_attr(&self, node_id: usize, name: String) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .unset_node_attr(node_id, name)
            .map_err(graph_error_to_py_err)
    }

    fn unset_node_attrs(
        &self,
        columns: std::collections::HashMap<String, Vec<usize>>,
    ) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .unset_node_attrs(columns)
            .map_err(graph_error_to_py_err)
    }

    /// Explicit absence; use a None assignment for an explicit null instead.
    fn unset_edge_attr(&self, edge_id: usize, name: String) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .unset_edge_attr(edge_id, name)
            .map_err(graph_error_to_py_err)
    }

    fn unset_edge_attrs(
        &self,
        columns: std::collections::HashMap<String, Vec<usize>>,
    ) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .unset_edge_attrs(columns)
            .map_err(graph_error_to_py_err)
    }

    /// Get all unique node attribute names across the entire graph
    fn all_node_attribute_names(&self) -> Vec<String> {
        use std::collections::HashSet;
        let graph_ref = self.inner.borrow();
        let mut all_attrs = HashSet::new();

        for node_id in graph_ref.node_ids() {
            if let Ok(attrs) = graph_ref.get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        result
    }

    /// Get all unique edge attribute names across the entire graph
    fn all_edge_attribute_names(&self) -> Vec<String> {
        use std::collections::HashSet;
        let graph_ref = self.inner.borrow();
        let mut all_attrs = HashSet::new();

        for edge_id in graph_ref.edge_ids() {
            if let Ok(attrs) = graph_ref.get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        let mut result: Vec<String> = all_attrs.into_iter().collect();
        result.sort();
        result
    }

    // === ALGORITHM OPERATIONS (delegate to PyGraphAnalysis helper) ===

    /// Get neighbors of nodes - delegates to PyGraphAnalysis helper
    fn neighbors(&mut self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.neighbors(py, nodes)
    }

    /// Get neighborhood sampling - delegates to PyGraphAnalysis helper
    fn neighborhood(
        &mut self,
        py: Python,
        center_nodes: &PyAny,
        radius: Option<usize>,
        max_nodes: Option<usize>,
    ) -> PyResult<crate::ffi::subgraphs::neighborhood::PyNeighborhoodArray> {
        // Convert center_nodes to Vec<NodeId> - accept both single node and sequence
        let center_nodes_vec = if let Ok(single_node) = center_nodes.extract::<NodeId>() {
            // Single node passed
            vec![single_node]
        } else if let Ok(nodes_vec) = center_nodes.extract::<Vec<NodeId>>() {
            // Vector of nodes passed
            nodes_vec
        } else {
            // Try to extract as sequence
            let seq = center_nodes.iter()?;
            let mut nodes = Vec::new();
            for item in seq {
                nodes.push(item?.extract::<NodeId>()?);
            }
            nodes
        };

        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.neighborhood(py, center_nodes_vec, radius, max_nodes)
    }

    /// Get shortest path - delegates to PyGraphAnalysis helper
    #[pyo3(signature = (source, target, weight_attribute = None, inplace = None, attr_name = None))]
    fn shortest_path(
        &self,
        py: Python,
        source: NodeId,
        target: NodeId,
        weight_attribute: Option<AttrName>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        let analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.shortest_path(py, source, target, weight_attribute, inplace, attr_name)
    }

    /// BFS traversal - delegates to PyGraphAnalysis helper
    #[pyo3(signature = (start, max_depth = None, inplace = None, attr_name = None))]
    fn bfs(
        &mut self,
        py: Python,
        start: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.bfs(py, start, max_depth, inplace, attr_name)
    }

    /// DFS traversal - delegates to PyGraphAnalysis helper  
    #[pyo3(signature = (start, max_depth = None, inplace = None, attr_name = None))]
    fn dfs(
        &mut self,
        py: Python,
        start: NodeId,
        max_depth: Option<usize>,
        inplace: Option<bool>,
        attr_name: Option<String>,
    ) -> PyResult<PyObject> {
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, self.clone())?)?;
        analysis_handler.dfs(py, start, max_depth, inplace, attr_name)
    }

    // === MATRIX OPERATIONS (delegate to PyGraphMatrixHelper) ===

    // Adjacency methods removed - these should only be available on Subgraph/view()
    // Use g.view().adjacency_matrix(), g.view().adj(), etc. instead

    /// Get Laplacian matrix - delegates to PyGraphMatrixHelper
    fn laplacian_matrix(
        &mut self,
        py: Python,
        normalized: Option<bool>,
    ) -> PyResult<Py<crate::ffi::storage::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.laplacian_matrix(py, normalized)
    }

    /// Generate transition matrix - delegates to PyGraphMatrixHelper
    fn transition_matrix(
        &mut self,
        py: Python,
    ) -> PyResult<Py<crate::ffi::storage::matrix::PyGraphMatrix>> {
        let mut matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.transition_matrix(py)
    }

    /// Convert graph to attribute matrix - delegates to PyGraphMatrixHelper
    fn to_matrix(&self, py: Python) -> PyResult<Py<crate::ffi::storage::matrix::PyGraphMatrix>> {
        let matrix_handler = PyGraphMatrixHelper::new(Py::new(py, self.clone())?)?;
        matrix_handler.to_matrix(py)
    }

    /// Aggregate attribute values across nodes or edges
    #[pyo3(signature = (attribute, operation, target = None, _node_ids = None))]
    fn aggregate(
        &self,
        py: Python,
        attribute: AttrName,
        operation: String,
        target: Option<String>,
        _node_ids: Option<Vec<NodeId>>,
    ) -> PyResult<PyObject> {
        let target = target.unwrap_or_else(|| "nodes".to_string());

        match target.as_str() {
            "nodes" => {
                // TODO: Core doesn't have aggregate_nodes_custom, implement if needed
                let result = self
                    .inner
                    .borrow_mut()
                    .aggregate_node_attribute(&attribute, &operation);
                match result {
                    Ok(agg_result) => {
                        let py_result = PyAggregationResult {
                            value: agg_result.value,
                        };
                        Ok(Py::new(py, py_result)?.to_object(py))
                    }
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            }
            "edges" => {
                let result = self
                    .inner
                    .borrow_mut()
                    .aggregate_edge_attribute(&attribute, &operation);
                match result {
                    Ok(agg_result) => {
                        let py_result = PyAggregationResult {
                            value: agg_result.value,
                        };
                        Ok(Py::new(py, py_result)?.to_object(py))
                    }
                    Err(e) => Err(graph_error_to_py_err(e)),
                }
            }
            _ => Err(PyErr::new::<PyTypeError, _>(
                "Target must be 'nodes' or 'edges'",
            )),
        }
    }

    // === VERSION CONTROL OPERATIONS ===

    /// Apply one validated mixed write against an expected published state.
    /// Optional schema checks the entire proposed result before any mutation.
    /// This call does not install a schema policy on subsequent writes.
    #[pyo3(signature = (batch, expected_state, message, author, *, schema=None))]
    fn apply_batch(
        &mut self,
        py: Python,
        batch: &PyAny,
        expected_state: StateId,
        message: String,
        author: String,
        schema: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let batch = parse_batch(batch)?;
        let schema = schema.map(parse_schema).transpose()?;
        let result = if let Some(schema) = &schema {
            self.inner.borrow_mut().apply_batch_with_schema(
                batch,
                expected_state,
                message,
                author,
                schema,
            )
        } else {
            self.inner
                .borrow_mut()
                .apply_batch(batch, expected_state, message, author)
        }
        .map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("state_id", result.state_id)?;
        output.set_item("added_nodes", result.added_nodes)?;
        output.set_item("added_edges", result.added_edges)?;
        Ok(output.to_object(py))
    }

    /// Apply a checked batch within an immutable view and return its refreshed selection.
    #[pyo3(signature = (view, batch, message, author, *, schema=None))]
    fn apply_view_batch(
        &mut self,
        py: Python,
        view: &crate::ffi::subgraphs::view::PyGraphView,
        batch: &PyAny,
        message: String,
        author: String,
        schema: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let batch = parse_batch(batch)?;
        let schema = schema.map(parse_schema).transpose()?;
        let result = if let Some(schema) = schema.as_ref() {
            self.inner.borrow_mut().apply_view_batch_with_schema(
                &view.inner,
                batch,
                message,
                author,
                schema,
            )
        } else {
            self.inner
                .borrow_mut()
                .apply_view_batch(&view.inner, batch, message, author)
        }
        .map_err(graph_error_to_py_err)?;
        crate::ffi::api::graph_batch::view_result_to_python(py, result)
    }

    /// Get immutable effective/recorded times for an automatic publication.
    fn publication_metadata(&self, py: Python, state_id: StateId) -> PyResult<PyObject> {
        match self
            .inner
            .borrow()
            .publication_metadata(state_id)
            .map_err(graph_error_to_py_err)?
        {
            Some(metadata) => crate::ffi::api::graph_batch::publication_to_python(py, &metadata),
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn current_state_id(&self) -> StateId {
        self.inner.borrow().current_state_id()
    }

    /// Compare the effective contents of two published states.
    fn diff_commits(&self, py: Python, from: StateId, to: StateId) -> PyResult<PyObject> {
        let diff = self
            .inner
            .borrow()
            .diff_commits(from, to)
            .map_err(graph_error_to_py_err)?;
        let output = PyDict::new(py);
        output.set_item("from_state", diff.from_commit)?;
        output.set_item("to_state", diff.to_commit)?;
        output.set_item("nodes_added", diff.nodes_added)?;
        output.set_item("nodes_removed", diff.nodes_removed)?;
        output.set_item("edges_added", diff.edges_added)?;
        output.set_item("edges_removed", diff.edges_removed)?;
        let changes = PyList::empty(py);
        for change in diff.attr_changes {
            let item = PyDict::new(py);
            let kind = match change.entity_type {
                groggy::state::change::EntityType::Node => "node",
                groggy::state::change::EntityType::Edge => "edge",
            };
            item.set_item("entity_type", kind)?;
            item.set_item("entity_id", change.entity_id)?;
            item.set_item("name", change.attr_name)?;
            item.set_item(
                "old",
                change
                    .old_value
                    .as_ref()
                    .map(|value| crate::ffi::utils::attr_value_to_python_value(py, value))
                    .transpose()?,
            )?;
            item.set_item(
                "new",
                change
                    .new_value
                    .as_ref()
                    .map(|value| crate::ffi::utils::attr_value_to_python_value(py, value))
                    .transpose()?,
            )?;
            changes.append(item)?;
        }
        output.set_item("attribute_changes", changes)?;
        Ok(output.to_object(py))
    }

    /// Commit current state of the graph (FFI wrapper around core history system)
    fn commit(&mut self, message: String, author: String) -> PyResult<StateId> {
        self.inner
            .borrow_mut()
            .commit(message, author)
            .map_err(graph_error_to_py_err)
    }

    /// Explicitly checkpoint a clean published state, preserving its ID.
    #[pyo3(signature=(message,author,*,expected_state=None,schema=None))]
    fn checkpoint(
        &mut self,
        message: String,
        author: String,
        expected_state: Option<StateId>,
        schema: Option<&PyDict>,
    ) -> PyResult<StateId> {
        let mut graph = self.inner.borrow_mut();
        let schema = schema.map(parse_schema).transpose()?;
        let expected = expected_state.unwrap_or_else(|| graph.current_state_id());
        graph
            .checkpoint_with_schema(expected, message, author, schema.as_ref())
            .map_err(graph_error_to_py_err)
    }
    fn checkpoints(&self, py: Python) -> PyResult<PyObject> {
        super::graph_version::checkpoints_to_python(py, self.inner.borrow().checkpoints())
    }

    /// Create a new branch (FFI wrapper around core history system)
    fn create_branch(&mut self, branch_name: String) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .create_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }

    /// Checkout a branch (FFI wrapper around core history system)
    fn checkout_branch(&mut self, branch_name: String) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .checkout_branch(branch_name)
            .map_err(graph_error_to_py_err)
    }

    /// List all branches (FFI wrapper around core history system)
    fn branches(&self) -> Vec<PyBranchInfo> {
        self.inner
            .borrow_mut()
            .list_branches()
            .into_iter()
            .map(PyBranchInfo::new)
            .collect()
    }

    /// Get commit history (FFI wrapper around core history system)
    fn commit_history(&self) -> Vec<PyCommit> {
        // Delegate to core history system
        self.inner
            .borrow_mut()
            .history_records()
            .into_iter()
            .map(PyCommit::from_core_commit)
            .collect()
    }

    /// Get historical view at specific commit (FFI wrapper around core history system)
    fn historical_view(&self, commit_id: StateId) -> PyResult<PyHistoricalView> {
        // Delegate to core history system
        match self.inner.borrow_mut().view_at_commit(commit_id) {
            Ok(_historical_view) => Ok(PyHistoricalView {
                state_id: commit_id,
            }),
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Capture a recorded snapshot; owned=True explicitly copies backing storage/history.
    #[pyo3(signature=(commit_id,*,owned=false))]
    fn snapshot_at_commit(&self, commit_id: StateId, owned: bool) -> PyResult<PyTemporalSnapshot> {
        let graph = self.inner.borrow();
        let snapshot = if owned {
            graph.owned_snapshot_at_commit(commit_id)
        } else {
            graph.snapshot_at_commit(commit_id)
        }
        .map_err(graph_error_to_py_err)?;
        Ok(PyTemporalSnapshot { inner: snapshot })
    }

    #[pyo3(signature=(timestamp,*,owned=false))]
    fn snapshot_at_timestamp(&self, timestamp: u64, owned: bool) -> PyResult<PyTemporalSnapshot> {
        let graph = self.inner.borrow();
        let snapshot = if owned {
            graph.owned_snapshot_at_timestamp(timestamp)
        } else {
            graph.snapshot_at_timestamp(timestamp)
        }
        .map_err(graph_error_to_py_err)?;
        Ok(PyTemporalSnapshot { inner: snapshot })
    }

    fn configuration(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json = serde_json::to_string(self.inner.borrow().configuration())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(py
            .import("json")?
            .call_method1("loads", (json,))?
            .to_object(py))
    }

    /// Discard pending manual edits and restore the selected published state.
    /// Allocated IDs and retained backing values remain reserved.
    fn reset_hard(&self) -> PyResult<()> {
        self.inner
            .borrow_mut()
            .reset_hard()
            .map_err(graph_error_to_py_err)
    }

    /// Explicit history-only pruning; backing values and identity reservations remain.
    fn prune_branch(&self, name: &str) -> PyResult<usize> {
        self.inner
            .borrow_mut()
            .prune_branch(name)
            .map_err(graph_error_to_py_err)
    }
    fn gc_history(&self) -> PyResult<usize> {
        self.inner
            .borrow_mut()
            .gc_history()
            .map_err(graph_error_to_py_err)
    }

    /// Explicitly deduplicate raw columns without deleting distinct values/history.
    fn compact_pool(&self, py: Python<'_>) -> PyResult<PyObject> {
        let plan = self
            .inner
            .borrow()
            .prepare_pool_compaction()
            .map_err(graph_error_to_py_err)?;
        let outcome = py
            .allow_threads(move || plan.execute())
            .map_err(graph_error_to_py_err)?;
        let report = self
            .inner
            .borrow_mut()
            .finish_pool_compaction(outcome)
            .map_err(graph_error_to_py_err)?;
        pool_compaction_report_to_py(py, &report)
    }

    fn optimize(&self, py: Python<'_>) -> PyResult<()> {
        self.compact_pool(py).map(|_| ())
    }

    /// Build a temporal index from the graph history for efficient temporal queries.
    ///
    /// The temporal index provides O(log n) or better lookups for:
    /// - When nodes/edges were created/deleted
    /// - Which nodes/edges existed at any commit
    /// - Attribute value timelines
    /// - Neighbors at specific commits or within time windows
    fn build_temporal_index(&self) -> PyResult<crate::ffi::temporal::PyTemporalIndex> {
        let index = self
            .inner
            .borrow()
            .build_temporal_index()
            .map_err(graph_error_to_py_err)?;
        Ok(crate::ffi::temporal::PyTemporalIndex { inner: index })
    }

    /// Get neighbors of nodes as they existed at a specific commit.
    fn neighbors_at_commit(
        &self,
        node_ids: Vec<NodeId>,
        commit_id: StateId,
    ) -> PyResult<HashMap<NodeId, Vec<NodeId>>> {
        self.inner
            .borrow()
            .neighbors_at_commit(&node_ids, commit_id)
            .map_err(graph_error_to_py_err)
    }

    /// Get neighbors that existed at any point within a commit range.
    fn neighbors_in_window(
        &self,
        node_id: NodeId,
        start_commit: StateId,
        end_commit: StateId,
    ) -> PyResult<Vec<NodeId>> {
        self.inner
            .borrow()
            .neighbors_in_window(node_id, start_commit, end_commit)
            .map_err(graph_error_to_py_err)
    }

    /// Get attribute value history for a node within a commit range.
    fn node_attr_history(
        &self,
        node_id: NodeId,
        attr: &str,
        from_commit: StateId,
        to_commit: StateId,
    ) -> PyResult<Vec<(StateId, Option<PyAttrValue>)>> {
        let history = self
            .inner
            .borrow()
            .node_attr_history(node_id, &attr.to_string(), from_commit, to_commit)
            .map_err(graph_error_to_py_err)?;
        Ok(history
            .into_iter()
            .map(|(cid, val)| (cid, val.map(PyAttrValue::new)))
            .collect())
    }

    // === STATE METHODS ===

    /// Check if there are uncommitted changes (FFI wrapper around core history system)
    fn has_uncommitted_changes(&self) -> bool {
        self.inner.borrow_mut().has_uncommitted_changes()
    }

    /// Get node mapping for a specific attribute (FFI wrapper around core operations)
    #[pyo3(signature = (uid_key, return_inverse = false))]
    fn get_node_mapping(
        &self,
        py: Python,
        uid_key: String,
        return_inverse: bool,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Delegate to core for node IDs and attributes
        let node_ids = self.inner.borrow_mut().node_ids();

        // Use core attribute access for each node
        for node_id in node_ids {
            if let Ok(Some(attr_value)) = self.inner.borrow_mut().get_node_attr(node_id, &uid_key) {
                // Convert attribute value to appropriate Python type
                let attr_value_py = match attr_value {
                    RustAttrValue::Text(s) => s.to_object(py),
                    RustAttrValue::CompactText(s) => s.as_str().to_object(py),
                    RustAttrValue::Int(i) => i.to_object(py),
                    RustAttrValue::SmallInt(i) => i.to_object(py),
                    RustAttrValue::Float(f) => f.to_object(py),
                    RustAttrValue::Float64(f) => f.to_object(py),
                    RustAttrValue::Bool(b) => b.to_object(py),
                    _ => continue, // Skip unsupported types
                };

                if return_inverse {
                    // Return node_id -> attribute_value mapping
                    dict.set_item(node_id, attr_value_py)?;
                } else {
                    // Return attribute_value -> node_id mapping (default behavior)
                    if dict.contains(attr_value_py.clone_ref(py))? {
                        return Err(PyValueError::new_err(format!(
                            "Ambiguous mapping for {uid_key}: duplicate property value"
                        )));
                    }
                    dict.set_item(attr_value_py, node_id)?;
                }
            }
        }

        Ok(dict.to_object(py))
    }

    // === GRAPH MERGING OPERATIONS ===

    /// Add another graph to this graph (merge nodes and edges)
    ///
    /// All nodes and edges from the other graph will be added to this graph.
    /// Node and edge IDs may be remapped to avoid conflicts.
    /// Attributes are preserved during the merge.
    pub fn add_graph(&mut self, _py: Python, other: &PyGraph) -> PyResult<()> {
        if Rc::ptr_eq(&self.inner, &other.inner) {
            return Err(PyValueError::new_err("Cannot add a graph to itself"));
        }
        self.inner
            .borrow_mut()
            .add_graph(&other.inner.borrow())
            .map_err(graph_error_to_py_err)
    }

    /// Return a full-view Subgraph (whole graph as a subgraph).
    /// Downstream code can always resolve parent graph from this object.
    pub fn view(self_: PyRef<Self>, py: Python<'_>) -> PyResult<Py<PySubgraph>> {
        // Check cache first - avoid expensive view creation if version hasn't changed
        let current_version = {
            let borrowed = self_.inner.borrow();
            // Use node count + edge count as a simple version indicator
            (borrowed.node_ids().len() + borrowed.edge_ids().len()) as u64
        };

        // Check if we have a cached view for this version
        {
            let cache = self_.cached_view.borrow();
            if let Some((cached_version, cached_subgraph)) = cache.as_ref() {
                if *cached_version == current_version {
                    // Cache hit - return cached result
                    return Ok(cached_subgraph.clone());
                }
            }
        }
        // Cache miss - need to rebuild

        // OPTIMIZATION: Get HashSets directly (avoids Vec intermediate)
        let graph_ref = self_.inner.borrow();
        let node_set = graph_ref.node_ids_set();
        let edge_set = graph_ref.edge_ids_set();
        drop(graph_ref);

        // OPTIMIZATION: Use reserved ID 1 for cached views (avoids atomic counter overhead)
        let subgraph = groggy::subgraphs::Subgraph::new_with_simple_id(
            self_.inner.clone(),
            node_set,
            edge_set,
            "cached_full_view".to_string(),
            1, // Reserved ID for cached views
        );

        let py_subgraph = PySubgraph::from_core_subgraph(subgraph)?;
        let py_result = Py::new(py, py_subgraph)?;

        // Cache the result for future calls
        {
            let mut cache = self_.cached_view.borrow_mut();
            *cache = Some((current_version, py_result.clone()));
        }

        Ok(py_result)
    }

    /// Check if the graph is connected (delegates to subgraph implementation)
    pub fn is_connected(self_: PyRef<Self>, _py: Python<'_>) -> PyResult<bool> {
        // Create a subgraph containing all nodes and edges, then check connectivity
        let all_nodes: Vec<NodeId> = self_.inner.borrow().node_ids();
        let all_edges: Vec<EdgeId> = self_.inner.borrow().edge_ids();

        let mut node_set = std::collections::HashSet::with_capacity(all_nodes.len());
        let mut edge_set = std::collections::HashSet::with_capacity(all_edges.len());
        node_set.extend(all_nodes);
        edge_set.extend(all_edges);

        let subgraph = groggy::subgraphs::Subgraph::new(
            self_.inner.clone(),
            node_set,
            edge_set,
            "connectivity_check".to_string(),
        );

        let result = subgraph.is_connected().map_err(graph_error_to_py_err)?;
        Ok(result)
    }

    // === EXPLICIT DELEGATION METHODS ===
    // These methods avoid expensive dynamic __getattr__ delegation by explicitly
    // forwarding to the cached full-graph subgraph view.

    /// Find connected components in the graph.
    pub fn connected_components(
        slf: PyRef<Self>,
        py: Python,
    ) -> PyResult<crate::ffi::storage::components::PyComponentsArray> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            let components = subgraph
                .inner
                .connected_components()
                .map_err(graph_error_to_py_err)?;
            Ok(
                crate::ffi::storage::components::PyComponentsArray::from_components(
                    components,
                    subgraph.inner.graph().clone(),
                ),
            )
        })
    }

    /// Calculate clustering coefficient (not yet implemented in core).
    #[pyo3(signature = (_node_id = None))]
    pub fn clustering_coefficient(
        slf: PyRef<Self>,
        py: Python,
        _node_id: Option<NodeId>,
    ) -> PyResult<f64> {
        Self::with_full_view(slf, py, |_subgraph, _py| {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Clustering coefficient not yet implemented - coming in future version",
            ))
        })
    }

    /// Calculate transitivity (not yet implemented in core).
    pub fn transitivity(slf: PyRef<Self>, py: Python) -> PyResult<f64> {
        Self::with_full_view(slf, py, |_subgraph, _py| {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Transitivity not yet implemented - coming in future version",
            ))
        })
    }

    /// Check if path exists between two nodes.
    pub fn has_path(
        slf: PyRef<Self>,
        py: Python,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<bool> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            subgraph
                .inner
                .bfs(source, None)
                .map(|bfs_subgraph| bfs_subgraph.contains_node(target))
                .map_err(graph_error_to_py_err)
        })
    }

    /// Sample k random nodes from the graph.
    pub fn sample(slf: PyRef<Self>, py: Python, k: usize) -> PyResult<PySubgraph> {
        Self::with_full_view(slf, py, |subgraph, _py| subgraph.sample_k(k))
    }

    /// Create induced subgraph from list of nodes.
    pub fn induced_subgraph(
        slf: PyRef<Self>,
        py: Python,
        nodes: Vec<NodeId>,
    ) -> PyResult<PySubgraph> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            let trait_subgraph = subgraph
                .inner
                .induced_subgraph(&nodes)
                .map_err(graph_error_to_py_err)?;
            let node_set = trait_subgraph.node_set().clone();
            let edge_set = trait_subgraph.edge_set().clone();
            let graph_ref = subgraph.inner.graph().clone();
            let concrete_subgraph = Subgraph::new(
                graph_ref,
                node_set,
                edge_set,
                "induced_subgraph".to_string(),
            );
            PySubgraph::from_core_subgraph(concrete_subgraph)
        })
    }

    /// Create subgraph from list of edges.
    pub fn subgraph_from_edges(
        slf: PyRef<Self>,
        py: Python,
        edges: Vec<EdgeId>,
    ) -> PyResult<PySubgraph> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            let trait_subgraph = subgraph
                .inner
                .subgraph_from_edges(&edges)
                .map_err(graph_error_to_py_err)?;
            let node_set = trait_subgraph.node_set().clone();
            let edge_set = trait_subgraph.edge_set().clone();
            let graph_ref = subgraph.inner.graph().clone();
            let concrete_subgraph = Subgraph::new(
                graph_ref,
                node_set,
                edge_set,
                "subgraph_from_edges".to_string(),
            );
            PySubgraph::from_core_subgraph(concrete_subgraph)
        })
    }

    /// Get summary string describing the graph.
    pub fn summary(slf: PyRef<Self>, py: Python) -> PyResult<String> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            let node_count = subgraph.inner.node_count();
            let edge_count = subgraph.inner.edge_count();
            let density = subgraph.inner.density();
            Ok(format!(
                "Graph: {} nodes, {} edges, density={:.3}",
                node_count, edge_count, density
            ))
        })
    }

    // === DEGREE METHODS (via Python method calls) ===

    /// Get degree for nodes - delegates via Python method call.
    #[pyo3(signature = (nodes = None, *, full_graph = false))]
    pub fn degree(
        slf: PyRef<Self>,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
    ) -> PyResult<PyObject> {
        use pyo3::types::PyTuple;
        let args = PyTuple::new(
            py,
            &[nodes.map(|n| n.to_object(py)).unwrap_or_else(|| py.None())],
        );
        let kwargs = PyDict::new(py);
        kwargs.set_item("full_graph", full_graph)?;
        Self::call_on_view_kwargs(slf, py, "degree", args, Some(kwargs))
    }

    /// Get in-degree for nodes - delegates via Python method call.
    #[pyo3(signature = (nodes = None, *, full_graph = false))]
    pub fn in_degree(
        slf: PyRef<Self>,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
    ) -> PyResult<PyObject> {
        use pyo3::types::PyTuple;
        let args = PyTuple::new(
            py,
            &[nodes.map(|n| n.to_object(py)).unwrap_or_else(|| py.None())],
        );
        let kwargs = PyDict::new(py);
        kwargs.set_item("full_graph", full_graph)?;
        Self::call_on_view_kwargs(slf, py, "in_degree", args, Some(kwargs))
    }

    /// Get out-degree for nodes - delegates via Python method call.
    #[pyo3(signature = (nodes = None, *, full_graph = false))]
    pub fn out_degree(
        slf: PyRef<Self>,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
    ) -> PyResult<PyObject> {
        use pyo3::types::PyTuple;
        let args = PyTuple::new(
            py,
            &[nodes.map(|n| n.to_object(py)).unwrap_or_else(|| py.None())],
        );
        let kwargs = PyDict::new(py);
        kwargs.set_item("full_graph", full_graph)?;
        Self::call_on_view_kwargs(slf, py, "out_degree", args, Some(kwargs))
    }

    // === CONVERSION METHODS (via Python) ===

    /// Convert to NodesAccessor - delegates via Python.
    pub fn to_nodes(
        slf: PyRef<Self>,
        py: Python,
    ) -> PyResult<crate::ffi::storage::accessors::PyNodesAccessor> {
        Self::with_full_view(slf, py, |subgraph, _py| subgraph.to_nodes())
    }

    /// Convert to EdgesAccessor - delegates via Python.
    pub fn to_edges(
        slf: PyRef<Self>,
        py: Python,
    ) -> PyResult<crate::ffi::storage::accessors::PyEdgesAccessor> {
        Self::with_full_view(slf, py, |subgraph, _py| subgraph.to_edges())
    }

    /// Get edges table - delegates via Python.
    pub fn edges_table(slf: PyRef<Self>, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "edges_table", PyTuple::empty(py))
    }

    /// Convert to independent Graph - delegates via Python.
    pub fn to_graph(slf: PyRef<Self>, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "to_graph", PyTuple::empty(py))
    }

    // === TOPOLOGY/ANALYSIS (via Python) ===

    /// Check if edge exists between nodes - delegates via Python.
    pub fn has_edge_between(
        slf: PyRef<Self>,
        py: Python,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<bool> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "has_edge_between",
            PyTuple::new(py, [source, target]),
        )
        .and_then(|obj| obj.extract(py))
    }

    /// Calculate similarity - delegates via Python.
    #[pyo3(signature = (other, metric = "jaccard"))]
    pub fn calculate_similarity(
        slf: PyRef<Self>,
        py: Python,
        other: &PySubgraph,
        metric: &str,
    ) -> PyResult<f64> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "calculate_similarity",
            PyTuple::new(
                py,
                &[
                    Py::new(py, other.clone())?.to_object(py),
                    metric.to_object(py),
                ],
            ),
        )
        .and_then(|obj| obj.extract(py))
    }

    /// Get shortest path as subgraph - delegates via Python.
    pub fn shortest_path_subgraph(
        slf: PyRef<Self>,
        py: Python,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<Option<PySubgraph>> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "shortest_path_subgraph",
            PyTuple::new(py, [source, target]),
        )
        .and_then(|obj| obj.extract(py))
    }

    // === ATTRIBUTE ACCESS (via Python) ===

    /// Get node attribute - delegates via Python.
    pub fn get_node_attribute(
        slf: PyRef<Self>,
        py: Python,
        node_id: NodeId,
        attr_name: String,
    ) -> PyResult<Option<PyObject>> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "get_node_attribute",
            PyTuple::new(py, &[node_id.to_object(py), attr_name.to_object(py)]),
        )
        .and_then(|obj| obj.extract(py))
    }

    /// Get edge attribute - delegates via Python.
    pub fn get_edge_attribute(
        slf: PyRef<Self>,
        py: Python,
        edge_id: EdgeId,
        attr_name: String,
    ) -> PyResult<Option<PyObject>> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "get_edge_attribute",
            PyTuple::new(py, &[edge_id.to_object(py), attr_name.to_object(py)]),
        )
        .and_then(|obj| obj.extract(py))
    }

    // === UTILITY METHODS (via Python) ===

    /// Get entity type - delegates via Python.
    pub fn entity_type(slf: PyRef<Self>, py: Python) -> PyResult<String> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "entity_type", PyTuple::empty(py))
            .and_then(|obj| obj.extract(py))
    }

    /// Visualization accessor matching the subgraph API.
    #[getter]
    pub fn viz(
        slf: PyRef<Self>,
        py: Python,
    ) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        Self::with_full_view(slf, py, |subgraph, py_ctx| subgraph.viz(py_ctx))
    }

    // === SET OPERATIONS (Placeholders) ===

    /// Merge with another subgraph (not yet implemented).
    pub fn merge_with(slf: PyRef<Self>, py: Python, other: &PySubgraph) -> PyResult<PySubgraph> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "merge_with",
            PyTuple::new(py, &[Py::new(py, other.clone())?.to_object(py)]),
        )
        .and_then(|obj| obj.extract(py))
    }

    /// Intersect with another subgraph (not yet implemented).
    pub fn intersect_with(
        slf: PyRef<Self>,
        py: Python,
        other: &PySubgraph,
    ) -> PyResult<PySubgraph> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "intersect_with",
            PyTuple::new(py, &[Py::new(py, other.clone())?.to_object(py)]),
        )
        .and_then(|obj| obj.extract(py))
    }

    /// Subtract another subgraph (not yet implemented).
    pub fn subtract_from(slf: PyRef<Self>, py: Python, other: &PySubgraph) -> PyResult<PySubgraph> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "subtract_from",
            PyTuple::new(py, &[Py::new(py, other.clone())?.to_object(py)]),
        )
        .and_then(|obj| obj.extract(py))
    }

    // === HIERARCHICAL METHODS (via Python) ===

    /// Get parent meta-node - delegates via Python.
    pub fn parent_meta_node(slf: PyRef<Self>, py: Python) -> PyResult<Option<PyObject>> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "parent_meta_node", PyTuple::empty(py))
            .and_then(|obj| obj.extract(py))
    }

    /// Get child meta-nodes - delegates via Python.
    pub fn child_meta_nodes(slf: PyRef<Self>, py: Python) -> PyResult<Vec<PyObject>> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "child_meta_nodes", PyTuple::empty(py))
            .and_then(|obj| obj.extract(py))
    }

    /// Check if has meta-nodes - delegates via Python.
    pub fn has_meta_nodes(slf: PyRef<Self>, py: Python) -> PyResult<bool> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "has_meta_nodes", PyTuple::empty(py))
            .and_then(|obj| obj.extract(py))
    }

    /// Get all meta-nodes - delegates via Python.
    pub fn meta_nodes(slf: PyRef<Self>, py: Python) -> PyResult<Vec<PyObject>> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "meta_nodes", PyTuple::empty(py))
            .and_then(|obj| obj.extract(py))
    }

    /// Get hierarchy level - delegates via Python.
    pub fn hierarchy_level(slf: PyRef<Self>, py: Python) -> PyResult<usize> {
        use pyo3::types::PyTuple;
        Self::call_on_view(slf, py, "hierarchy_level", PyTuple::empty(py))
            .and_then(|obj| obj.extract(py))
    }

    /// Collapse subgraph - delegates via Python.
    pub fn collapse(slf: PyRef<Self>, py: Python, aggregations: &PyDict) -> PyResult<PyObject> {
        use pyo3::types::PyTuple;
        Self::call_on_view(
            slf,
            py,
            "collapse",
            PyTuple::new(py, &[aggregations.to_object(py)]),
        )
    }

    /// Get adjacency list (not yet implemented).
    pub fn adjacency_list(slf: PyRef<Self>, py: Python) -> PyResult<PyObject> {
        Self::with_full_view(slf, py, |_subgraph, _py| {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "adjacency_list not yet implemented",
            ))
        })
    }

    /// Convert this graph to a NetworkX graph
    ///
    /// Returns a NetworkX Graph or DiGraph (depending on the graph type)
    /// with all nodes, edges, and their attributes preserved.
    ///
    /// # Returns
    /// * `PyObject` - A NetworkX graph object
    ///
    /// # Raises
    /// * `ImportError` - If NetworkX is not installed
    /// * `RuntimeError` - If conversion fails
    ///
    /// # Examples
    /// ```python
    /// import groggy
    /// import networkx as nx
    ///
    /// g = groggy.Graph()
    /// # ... add nodes and edges ...
    /// nx_graph = g.to_networkx()
    /// ```
    fn to_networkx(&self, py: Python) -> PyResult<PyObject> {
        // Convert to our internal NetworkX representation
        let nx_graph = self.inner.borrow().to_networkx().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert to NetworkX: {}", e))
        })?;

        // Convert to actual Python NetworkX graph
        crate::ffi::utils::convert::networkx_graph_to_python(py, &nx_graph)
    }

    /// Enable property-style attribute access (g.age instead of g.nodes['age'])
    /// This method is called when accessing attributes that don't exist as methods
    /// Call experimental prototype methods.
    ///
    /// **Experimental API**: This method provides access to prototype graph operations
    /// that are under development and not yet part of the stable API. Methods available
    /// here depend on the `experimental-delegation` feature flag.
    ///
    /// # Usage
    ///
    /// ```python
    /// # Enable experimental features:
    /// # - At build time: maturin develop --features experimental-delegation
    /// # - At runtime: GROGGY_EXPERIMENTAL=1 python script.py
    ///
    /// # Call experimental method
    /// result = graph.experimental("pagerank", damping=0.85)
    ///
    /// # List available experimental methods
    /// methods = graph.experimental("list")
    ///
    /// # Get method description
    /// info = graph.experimental("describe", "pagerank")
    /// ```
    ///
    /// # Arguments
    ///
    /// * `method_name` - Name of the experimental method to call, or special commands:
    ///   - "list" - Returns list of available experimental methods
    ///   - "describe" - Requires second arg with method name, returns description
    /// * `args` - Positional arguments to pass to the method
    /// * `kwargs` - Keyword arguments to pass to the method
    ///
    /// # Returns
    ///
    /// Result of the experimental method call, or metadata for list/describe commands.
    ///
    /// # Errors
    ///
    /// - `AttributeError` if method not found or experimental features not enabled
    /// - Method-specific errors depending on the called function
    ///
    /// # Stability
    ///
    /// **No stability guarantees**. Experimental methods may:
    /// - Change signature without notice
    /// - Be renamed or removed
    /// - Move to stable API with different names
    /// - Have incomplete implementations
    ///
    /// Use only for prototyping and research. Production code should use stable API.
    #[pyo3(signature = (method_name, *args, **_kwargs))]
    pub fn experimental(
        _slf: PyRef<Self>,
        py: Python,
        method_name: &str,
        args: &PyTuple,
        _kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        use pyo3::types::PyList;

        let registry = crate::ffi::experimental::get_registry();

        // Handle special commands
        match method_name {
            "list" => {
                let methods = registry.list_methods();
                let py_list = PyList::new(py, methods);
                return Ok(py_list.into());
            }
            "describe" => {
                if args.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "describe requires method name as argument",
                    ));
                }
                let target_method: &str = args.get_item(0)?.extract()?;
                match registry.describe(target_method) {
                    Some(desc) => return Ok(desc.to_object(py)),
                    None => {
                        return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                            "Method '{}' not found in experimental registry",
                            target_method
                        )))
                    }
                }
            }
            _ => {}
        }

        // Check if experimental features are enabled
        #[cfg(not(feature = "experimental-delegation"))]
        {
            Err(pyo3::exceptions::PyAttributeError::new_err(
                format!(
                    "Experimental method '{}' not available. Rebuild with --features experimental-delegation or set GROGGY_EXPERIMENTAL=1",
                    method_name
                )
            ))
        }

        // Call the experimental method
        #[cfg(feature = "experimental-delegation")]
        {
            let obj = slf.into_py(py);
            registry.call(method_name, obj, py, args, kwargs)
        }
    }

    /// Dynamic attribute access for node and edge attribute dictionaries.
    ///
    /// **Intentional dynamic pattern**: This method remains dynamic to support runtime
    /// attribute dictionary projection. When accessing `graph.age`, it returns a dict of
    /// `{node_id: age_value}` for all nodes with that attribute.
    ///
    /// This is inherently data-dependent and cannot be made static since:
    /// - Attribute names are defined by users at runtime (e.g., "age", "salary", "title")
    /// - Different graphs have different attribute schemas
    /// - Attributes can be added/removed dynamically during graph operations
    ///
    /// All **method** calls have been moved to explicit implementations above to provide
    /// IDE discoverability and avoid expensive delegation. Only **attribute data** access
    /// remains dynamic here.
    fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
        use pyo3::exceptions::PyAttributeError;
        use pyo3::types::PyDict;

        // Early filtering: Skip expensive operations for internal Python attributes
        if name.starts_with("_") || name.starts_with("__") {
            return Err(PyAttributeError::new_err(format!(
                "'Graph' object has no attribute '{}'",
                name
            )));
        }

        // Skip common internal attributes that Python's display/introspection system uses
        match name.as_str() {
            "shape"
            | "dtype"
            | "ndim"
            | "size"
            | "T"
            | "real"
            | "imag"
            | "__array_interface__"
            | "__array__"
            | "__array_struct__"
            | "_repr_html_"
            | "_repr_png_"
            | "_repr_svg_"
            | "_repr_latex_"
            | "__class__"
            | "__dict__"
            | "__module__"
            | "__doc__"
            | "__weakref__"
            | "__slots__" => {
                return Err(PyAttributeError::new_err(format!(
                    "'Graph' object has no attribute '{}'",
                    name
                )));
            }
            _ => {}
        }

        // INTENTIONAL DYNAMIC PATTERN: Node attribute dictionary projection
        // Check if this is a node attribute name that exists in the graph
        // Returns {node_id: value} dict for data analysis workflows
        let all_node_attrs = self.all_node_attribute_names();
        if all_node_attrs.contains(&name) {
            // Return a dictionary mapping node IDs to their attribute values
            let result_dict = PyDict::new(py);
            let graph_ref = self.inner.borrow();

            for node_id in graph_ref.node_ids() {
                match graph_ref.get_node_attr(node_id, &name) {
                    Ok(Some(attr_value)) => {
                        let py_value =
                            crate::ffi::utils::attr_value_to_python_value(py, &attr_value)?;
                        result_dict.set_item(node_id, py_value)?;
                    }
                    Ok(None) => {
                        // Node doesn't have this attribute, skip or set to None
                        result_dict.set_item(node_id, py.None())?;
                    }
                    Err(_) => {
                        // Error accessing attribute, skip this node
                        continue;
                    }
                }
            }

            return Ok(result_dict.to_object(py));
        }

        // INTENTIONAL DYNAMIC PATTERN: Edge attribute dictionary projection
        // Check if this is an edge attribute name that exists in the graph
        // Returns {edge_id: value} dict for edge analysis workflows
        let all_edge_attrs = self.all_edge_attribute_names();
        if all_edge_attrs.contains(&name) {
            // Return a dictionary mapping edge IDs to their attribute values
            let result_dict = PyDict::new(py);
            let graph_ref = self.inner.borrow();

            for edge_id in graph_ref.edge_ids() {
                match graph_ref.get_edge_attr(edge_id, &name) {
                    Ok(Some(attr_value)) => {
                        let py_value =
                            crate::ffi::utils::attr_value_to_python_value(py, &attr_value)?;
                        result_dict.set_item(edge_id, py_value)?;
                    }
                    Ok(None) => {
                        // Edge doesn't have this attribute, skip or set to None
                        result_dict.set_item(edge_id, py.None())?;
                    }
                    Err(_) => {
                        // Error accessing attribute, skip this edge
                        continue;
                    }
                }
            }

            return Ok(result_dict.to_object(py));
        }

        // Try delegating to subgraph methods (like degree, in_degree, out_degree)
        // Create a full subgraph containing all nodes and edges
        // PERFORMANCE: Use reserved ID 2 for delegation views (fastest path)

        use std::time::Instant;
        let delegation_start = Instant::now();

        // OPTIMIZATION: Get HashSets directly (avoids Vec→HashSet conversion)
        let graph_ref = self.inner.borrow();
        let all_nodes = graph_ref.node_ids_set();
        let all_edges = graph_ref.edge_ids_set();
        drop(graph_ref); // Release borrow

        let collect_elapsed = delegation_start.elapsed();

        let subgraph_create_start = Instant::now();
        let concrete_subgraph = groggy::subgraphs::Subgraph::new_with_simple_id(
            self.inner.clone(),
            all_nodes,
            all_edges,
            "full_graph_delegation".to_string(),
            2, // Reserved ID for delegation views
        );
        let subgraph_create_elapsed = subgraph_create_start.elapsed();

        let py_wrap_start = Instant::now();
        match PySubgraph::from_core_subgraph(concrete_subgraph) {
            Ok(py_subgraph) => {
                let py_wrap_elapsed = py_wrap_start.elapsed();

                let py_new_start = Instant::now();
                let subgraph_obj = Py::new(py, py_subgraph)?;
                let py_new_elapsed = py_new_start.elapsed();

                let getattr_start = Instant::now();
                match subgraph_obj.getattr(py, name.as_str()) {
                    Ok(result) => {
                        let getattr_elapsed = getattr_start.elapsed();
                        let total_elapsed = delegation_start.elapsed();

                        // Log delegation timing for debugging
                        if std::env::var("GROGGY_DEBUG_DELEGATION").is_ok() {
                            eprintln!("__getattr__ delegation timing for '{}':", name);
                            eprintln!("  collect_ids: {:?}", collect_elapsed);
                            eprintln!("  create_subgraph: {:?}", subgraph_create_elapsed);
                            eprintln!("  py_wrap: {:?}", py_wrap_elapsed);
                            eprintln!("  py_new: {:?}", py_new_elapsed);
                            eprintln!("  getattr: {:?}", getattr_elapsed);
                            eprintln!("  TOTAL: {:?}", total_elapsed);
                        }

                        return Ok(result);
                    }
                    Err(_) => {
                        // Subgraph doesn't have this method either, fall through to error
                    }
                }
            }
            Err(_) => {
                // Failed to create subgraph, fall through to error
            }
        }

        // Attribute not found
        Err(PyAttributeError::new_err(format!(
            "Attribute '{}' not found. Available node attributes: {:?}, Available edge attributes: {:?}",
            name, all_node_attrs, all_edge_attrs
        )))
    }

    /// Get a GraphTable representation of this graph
    /// Implements: g.table()
    pub fn table(&self, py: Python<'_>) -> PyResult<crate::ffi::storage::table::PyGraphTable> {
        let view = self
            .inner
            .borrow()
            .read_view()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let table = py
            .allow_threads(|| view.table())
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(crate::ffi::storage::table::PyGraphTable { table })
    }

    /// Create a full-graph subgraph view efficiently (for apply method)
    /// This is used by Graph.apply() to avoid __getattr__ delegation overhead
    pub fn to_subgraph(&self) -> PyResult<crate::ffi::subgraphs::subgraph::PySubgraph> {
        let graph_ref = self.inner.borrow();
        // OPTIMIZATION: Use _set() methods to clone HashSets directly (avoids Vec intermediate)
        let all_nodes = graph_ref.node_ids_set();
        let all_edges = graph_ref.edge_ids_set();
        drop(graph_ref);

        // OPTIMIZATION: Use reserved ID 0 for apply views (avoids atomic counter overhead)
        let concrete_subgraph = groggy::subgraphs::Subgraph::new_with_simple_id(
            self.inner.clone(),
            all_nodes,
            all_edges,
            "full_graph_view".to_string(),
            0, // Reserved ID for apply views
        );

        crate::ffi::subgraphs::subgraph::PySubgraph::from_core_subgraph(concrete_subgraph)
    }
}

// Internal methods for FFI integration (not exposed to Python)
impl PyGraph {
    /// Internal helper methods for accessors/views
    pub fn create_node_view_internal(
        graph: Py<PyGraph>,
        py: Python,
        node_id: NodeId,
    ) -> PyResult<Py<PyNodeView>> {
        let core_graph = graph.borrow(py).inner.clone();
        Py::new(
            py,
            PyNodeView {
                graph: core_graph,
                node_id,
            },
        )
    }

    pub fn create_edge_view_internal(
        graph: Py<PyGraph>,
        py: Python,
        edge_id: EdgeId,
    ) -> PyResult<Py<PyEdgeView>> {
        let core_graph = graph.borrow(py).inner.clone();
        Py::new(
            py,
            PyEdgeView {
                graph: core_graph,
                edge_id,
            },
        )
    }

    /// Create a NodesAccessor (internal helper)
    fn create_nodes_accessor_internal(
        graph_ref: Py<PyGraph>,
        py: Python,
    ) -> PyResult<Py<PyNodesAccessor>> {
        let core_graph = graph_ref.borrow(py).inner.clone();
        Py::new(
            py,
            PyNodesAccessor {
                graph: core_graph,
                constrained_nodes: None,
            },
        )
    }

    /// Create an EdgesAccessor (internal helper)
    fn create_edges_accessor_internal(
        graph_ref: Py<PyGraph>,
        py: Python,
    ) -> PyResult<Py<PyEdgesAccessor>> {
        let core_graph = graph_ref.borrow(py).inner.clone();
        Py::new(
            py,
            PyEdgesAccessor {
                graph: core_graph,
                constrained_edges: None,
            },
        )
    }

    // === CACHE OPTIMIZATION METHODS ===

    /// Warm various caches after bulk operations to improve subsequent algorithm performance
    ///
    /// This method proactively builds caches that are commonly needed after bulk data insertion:
    /// - View cache: Pre-builds the full graph view for quick access
    /// - Adjacency snapshots: Triggers traversal engine cache building
    ///
    /// Use this after large bulk operations (add_edges, add_nodes) before running algorithms
    /// like BFS, DFS, connected_components, etc. for optimal performance.
    /// Future cache optimization
    #[allow(dead_code)]
    fn warm_caches_after_bulk_operation(&mut self, py: Python) -> PyResult<()> {
        // 1. Warm the view cache by creating a full graph view
        // This is the most commonly accessed cache for algorithms
        // Note: We can't directly call view() from &mut self, but the warming effect
        // is mainly from the traversal operations below

        // 2. Warm the traversal engine snapshot cache by triggering a minimal operation
        // This builds the adjacency map cache that BFS/DFS/shortest_path algorithms use
        let node_ids = self.inner.borrow().node_ids();
        if !node_ids.is_empty() {
            // Trigger a minimal BFS to warm the traversal snapshot cache
            let _ = self.bfs(py, node_ids[0], Some(1), None, None);
        }

        Ok(())
    }

    /// Advanced cache warming with fine-grained control
    ///
    /// Parameters:
    /// - warm_view: Whether to warm the view cache (default: true)
    /// - warm_traversal: Whether to warm traversal caches (default: true)
    /// - sample_node: Optional specific node to use for traversal warming (default: first node)
    ///
    /// Future cache optimization
    #[allow(dead_code)]
    fn warm_caches(
        &mut self,
        py: Python,
        warm_view: bool,
        warm_traversal: bool,
        sample_node: Option<NodeId>,
    ) -> PyResult<()> {
        if warm_view {
            // Note: View cache warming would require PyRef access
            // For now, the main warming effect comes from traversal operations
        }

        if warm_traversal {
            // Warm traversal engine caches
            let node_ids = self.inner.borrow().node_ids();
            if !node_ids.is_empty() {
                let start_node = sample_node.unwrap_or(node_ids[0]);
                // Minimal BFS to warm adjacency cache
                let _ = self.bfs(py, start_node, Some(1), None, None);
            }
        }

        Ok(())
    }

    // === HELPER METHODS FOR OTHER MODULES ===

    pub fn get_edge_endpoints(&self, edge_id: EdgeId) -> Result<(NodeId, NodeId), String> {
        self.inner
            .borrow_mut()
            .edge_endpoints(edge_id)
            .map_err(|e| e.to_string())
    }

    pub fn get_node_ids(&self) -> PyResult<Vec<NodeId>> {
        Ok(self.inner.borrow_mut().node_ids())
    }

    pub fn get_edge_ids(&self) -> PyResult<Vec<EdgeId>> {
        Ok(self.inner.borrow_mut().edge_ids())
    }

    // Additional public methods for internal module access

    /// Internal helper - DELEGATED
    pub fn has_node_internal(&self, node_id: NodeId) -> bool {
        // Pure delegation (same as has_node/contains_node)
        self.has_node(node_id)
    }

    /// Internal helper - DELEGATED
    pub fn has_edge_internal(&self, edge_id: EdgeId) -> bool {
        // Pure delegation (same as has_edge/contains_edge)
        self.has_edge(edge_id)
    }

    /// Internal helper - DELEGATED
    pub fn get_node_count(&self) -> usize {
        // Pure delegation (same as node_count)
        self.node_count()
    }

    /// Internal helper - DELEGATED
    pub fn get_edge_count(&self) -> usize {
        // Pure delegation (same as edge_count)
        self.edge_count()
    }

    pub fn node_ids_vec(&self) -> Vec<NodeId> {
        self.inner.borrow_mut().node_ids()
    }

    pub fn edge_ids_vec(&self) -> Vec<EdgeId> {
        self.inner.borrow_mut().edge_ids()
    }

    pub fn get_node_ids_array(&self, py: Python) -> PyResult<Py<PyNumArray>> {
        let node_ids = self.inner.borrow_mut().node_ids();
        let values: Vec<f64> = node_ids.into_iter().map(|id| id as f64).collect();
        let py_num_array = PyNumArray::new(values);
        Py::new(py, py_num_array)
    }

    pub fn get_edge_ids_array(&self, py: Python) -> PyResult<Py<PyNumArray>> {
        let edge_ids = self.inner.borrow_mut().edge_ids();
        let values: Vec<f64> = edge_ids.into_iter().map(|id| id as f64).collect();
        let py_num_array = PyNumArray::new(values);
        Py::new(py, py_num_array)
    }

    /// Group nodes by attribute value and compute aggregates for each group (internal method)
    pub fn group_nodes_by_attribute_internal(
        &self,
        attribute: AttrName,
        aggregation_attr: AttrName,
        operation: String,
    ) -> PyResult<PyGroupedAggregationResult> {
        let results = self
            .inner
            .borrow_mut()
            .group_nodes_by_attribute(&attribute, &aggregation_attr, &operation)
            .map_err(graph_error_to_py_err)?;

        Python::with_gil(|py| {
            // Convert HashMap to Python dict
            let dict = PyDict::new(py);
            for (attr_value, agg_result) in results {
                let py_attr_value = PyAttrValue { inner: attr_value };
                let py_agg_result = PyAggregationResult {
                    value: agg_result.value,
                };
                dict.set_item(Py::new(py, py_attr_value)?, Py::new(py, py_agg_result)?)?;
            }

            Ok(PyGroupedAggregationResult {
                groups: dict.to_object(py),
                operation: operation.clone(),
                attribute: attribute.clone(),
            })
        })
    }

    // === PROPERTY GETTERS FOR ATTRIBUTE ACCESS ===

    /// Get complete attribute column for ALL nodes (optimized for table() method)
    /// Returns BaseArray for attribute columns with proper integration with table columns
    fn _get_node_attr_column(&self, py: Python, attr_name: &str) -> PyResult<Py<PyBaseArray>> {
        match self
            .inner
            .borrow_mut()
            ._get_node_attribute_column(&attr_name.to_string())
        {
            Ok(values) => {
                // Convert Option<AttrValue> vector to AttrValue vector (convert None to appropriate AttrValue)
                let attr_values: Vec<groggy::AttrValue> = values
                    .into_iter()
                    .map(|opt_val| opt_val.unwrap_or(groggy::AttrValue::Null)) // Use Null for missing values
                    .collect();

                // Create BaseArray directly from attribute values
                let py_base_array = PyBaseArray {
                    inner: groggy::storage::array::BaseArray::new(attr_values),
                };

                Py::new(py, py_base_array)
            }
            Err(e) => Err(graph_error_to_py_err(e)),
        }
    }

    /// Validate that all attributes referenced in a NodeFilter exist in the graph
    /// Returns Ok(()) if valid, Err(attr_name) if an attribute doesn't exist
    fn validate_node_filter_attributes(
        graph: &groggy::api::graph::Graph,
        filter: &groggy::query::NodeFilter,
    ) -> Result<(), String> {
        use groggy::query::NodeFilter;

        match filter {
            NodeFilter::AttributeFilter { name, .. }
            | NodeFilter::AttributeEquals { name, .. }
            | NodeFilter::HasAttribute { name } => {
                // Check if this attribute exists on any nodes
                if !Self::attribute_exists_on_nodes(graph, name) {
                    return Err(name.clone());
                }
            }
            NodeFilter::And(filters) => {
                for f in filters {
                    let _ = Self::validate_node_filter_attributes(graph, f);
                }
            }
            NodeFilter::Or(filters) => {
                for f in filters {
                    let _ = Self::validate_node_filter_attributes(graph, f);
                }
            }
            NodeFilter::Not(filter) => {
                let _ = Self::validate_node_filter_attributes(graph, filter);
            }
            // Other filter types don't reference attributes
            _ => {}
        }

        Ok(())
    }

    /// Validate that all attributes referenced in an EdgeFilter exist in the graph
    /// Returns Ok(()) if valid, Err(attr_name) if an attribute doesn't exist
    fn validate_edge_filter_attributes(
        graph: &groggy::api::graph::Graph,
        filter: &groggy::query::EdgeFilter,
    ) -> Result<(), String> {
        use groggy::query::EdgeFilter;

        match filter {
            EdgeFilter::AttributeFilter { name, .. }
            | EdgeFilter::AttributeEquals { name, .. }
            | EdgeFilter::HasAttribute { name } => {
                // Check if this attribute exists on any edges
                if !Self::attribute_exists_on_edges(graph, name) {
                    return Err(name.clone());
                }
            }
            EdgeFilter::And(filters) => {
                for f in filters {
                    let _ = Self::validate_edge_filter_attributes(graph, f);
                }
            }
            EdgeFilter::Or(filters) => {
                for f in filters {
                    let _ = Self::validate_edge_filter_attributes(graph, f);
                }
            }
            EdgeFilter::Not(filter) => {
                let _ = Self::validate_edge_filter_attributes(graph, filter);
            }
            // Other filter types don't reference attributes
            _ => {}
        }

        Ok(())
    }

    /// Check if an attribute exists on any nodes in the graph
    fn attribute_exists_on_nodes(graph: &groggy::api::graph::Graph, attr_name: &str) -> bool {
        let node_ids = graph.node_ids();
        for node_id in node_ids.iter().take(100) {
            // Sample first 100 nodes for performance
            if let Ok(Some(_)) = graph.get_node_attr(*node_id, &attr_name.to_string()) {
                return true;
            }
        }
        false
    }

    /// Check if an attribute exists on any edges in the graph
    fn attribute_exists_on_edges(graph: &groggy::api::graph::Graph, attr_name: &str) -> bool {
        let edge_ids = graph.edge_ids();
        for edge_id in edge_ids.iter().take(100) {
            // Sample first 100 edges for performance
            if let Ok(Some(_)) = graph.get_edge_attr(*edge_id, &attr_name.to_string()) {
                return true;
            }
        }
        false
    }

    // === DELEGATED METHODS - Pure SubgraphOperations delegation ===

    // === HIERARCHICAL METHODS ===

    /// Add multiple subgraphs to the graph as meta-nodes (batch operation)
    ///
    /// Args:
    ///     subgraphs: List of Subgraph objects to add as meta-nodes
    ///     agg_functions: Optional dict mapping attribute names to aggregation functions
    ///
    /// Returns:
    ///     List[MetaNode]: The created meta-nodes
    ///
    /// Example:
    ///     communities = g.connected_components()
    ///     meta_nodes = g.add_subgraphs(communities, {
    ///         "total_weight": "sum",
    ///         "avg_degree": "mean"
    ///     })
    /// Future cache optimization
    #[allow(dead_code)]
    fn add_subgraphs(
        &self,
        py: Python,
        plans: Vec<Py<crate::ffi::subgraphs::composer::PyMetaNodePlan>>,
    ) -> PyResult<Vec<PyObject>> {
        let mut meta_nodes = Vec::new();

        for py_plan in plans {
            // Extract the plan and execute it
            let plan = py_plan.borrow(py);
            let meta_node = plan.add_to_graph(py)?;
            meta_nodes.push(meta_node);
        }

        Ok(meta_nodes)
    }
}

/// Translate a native verification receipt without interpreting engine policy.
pub(super) fn integrity_report_to_py(
    py: Python<'_>,
    report: groggy::IntegrityReport,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("selected_state", report.selected_state)?;
    result.set_item("selected_branch", report.selected_branch)?;
    result.set_item("schema_revision", report.schema_revision)?;
    result.set_item("active_nodes", report.active_nodes)?;
    result.set_item("active_edges", report.active_edges)?;
    result.set_item("allocated_nodes", report.allocated_nodes)?;
    result.set_item("allocated_edges", report.allocated_edges)?;
    result.set_item("node_columns", report.node_columns)?;
    result.set_item("edge_columns", report.edge_columns)?;
    result.set_item("node_values", report.node_values)?;
    result.set_item("edge_values", report.edge_values)?;
    result.set_item("retained_versions", report.retained_versions)?;
    result.set_item("verified_states", report.verified_states)?;
    result.set_item("legacy_versions", report.legacy_versions)?;
    result.set_item("branches", report.branches)?;
    result.set_item(
        "retained_group_references",
        report.retained_group_references,
    )?;
    result.set_item("external_id_reservations", report.external_id_reservations)?;
    result.set_item("maximum_active_nodes", report.maximum_active_nodes)?;
    result.set_item("maximum_active_edges", report.maximum_active_edges)?;
    result.set_item("current_schema_checked", report.current_schema_checked)?;
    Ok(result.into())
}

/// Pure report marshalling shared by ordinary and durable maintenance.
pub(super) fn pool_compaction_report_to_py(
    py: Python<'_>,
    report: &groggy::PoolCompactionReport,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    result.set_item("selected_state", report.selected_state)?;
    result.set_item("previous_node_values", report.previous_node_values)?;
    result.set_item("current_node_values", report.current_node_values)?;
    result.set_item("previous_edge_values", report.previous_edge_values)?;
    result.set_item("current_edge_values", report.current_edge_values)?;
    Ok(result.into())
}
