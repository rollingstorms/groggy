//! Simplified Subgraph FFI Bindings - Complete Replacement
//!
//! Pure delegation to core Subgraph with ALL the same methods as the current PySubgraph.
//! This replaces the 800+ line complex version with pure delegation to existing trait methods.

use crate::ffi::api::pipeline::{
    py_build_pipeline, py_build_sample_pipeline, py_drop_pipeline, py_drop_sample_pipeline,
    py_run_pipeline, py_run_sample_pipeline, PyPipelineHandle, PySamplePipelineHandle,
};
use crate::ffi::storage::subgraph_array::PySubgraphArray;
// use crate::ffi::core::path_result::PyPathResult; // Unused
use groggy::storage::array::BaseArray;
use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::{AttrValue, EdgeId, NodeId, SimilarityMetric};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{Py, PyCell, PyObject};
use std::time::Instant;

// Import FFI types we need to preserve compatibility
use crate::ffi::api::graph::PyGraph;
use crate::ffi::storage::accessors::{PyEdgesAccessor, PyNodesAccessor}; // Essential FFI - re-enabled
use crate::ffi::storage::array::PyBaseArray;
use crate::ffi::storage::components::PyComponentsArray;
use crate::PyNumArray;
// use crate::ffi::storage::table::PyBaseTable; // Temporarily disabled

/// Python wrapper for core Subgraph - Pure delegation to existing trait methods
///
/// This completely replaces the complex dual-mode PySubgraph with simple delegation
/// to the existing SubgraphOperations trait methods. Same API, much simpler implementation.
#[pyclass(name = "Subgraph", weakref)]
#[derive(Clone)]
pub struct PySubgraph {
    pub inner: Subgraph,
}

impl PySubgraph {
    /// Create from Rust Subgraph
    pub fn from_core_subgraph(subgraph: Subgraph) -> PyResult<Self> {
        Ok(Self {
            inner: subgraph
                .freeze()
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        })
    }

    /// Explicit live compatibility scope, used only by owner execution adapters.
    pub fn from_live_subgraph(inner: Subgraph) -> Self {
        Self { inner }
    }

    /// Preserve a derived trait object's original immutable source.
    pub fn from_trait_object(
        subgraph: Box<dyn groggy::traits::SubgraphOperations>,
    ) -> PyResult<Self> {
        let graph = subgraph.graph_ref();
        let inner = if let Some(view) = subgraph.frozen_view() {
            Subgraph::from_frozen_view(graph, view.clone(), "derived".into())
        } else {
            Subgraph::new(
                graph,
                subgraph.node_set().clone(),
                subgraph.edge_set().clone(),
                "derived".into(),
            )
        };
        Self::from_core_subgraph(inner)
    }

    fn degree_result(
        &self,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
        mode: groggy::subgraphs::DegreeMode,
    ) -> PyResult<PyObject> {
        let scalar = nodes.and_then(|value| value.extract::<NodeId>().ok());
        let ids = match (scalar, nodes) {
            (Some(node), _) => vec![node],
            (None, Some(value)) => value.extract::<Vec<NodeId>>()?,
            (None, None) => self.inner.node_ids(),
        };
        let read = self
            .inner
            .degree_read(mode, full_graph)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let degrees = py
            .allow_threads(|| read.counts(&ids))
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        if scalar.is_some() {
            Ok(degrees[0].to_object(py))
        } else {
            let values = degrees
                .into_iter()
                .map(|value| {
                    i64::try_from(value).map(AttrValue::Int).map_err(|_| {
                        pyo3::exceptions::PyOverflowError::new_err(
                            "Degree exceeds signed 64-bit range",
                        )
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Py::new(py, PyNumArray::from_attr_values(values)?)?.to_object(py))
        }
    }

    fn try_apply_pipeline_object(
        &self,
        py: Python,
        candidate: &PyAny,
        persist: bool,
        atomic: bool,
    ) -> PyResult<Option<(PySubgraph, PyObject)>> {
        if !(candidate.hasattr("_ensure_built")? && candidate.hasattr("_handle")?) {
            return Ok(None);
        }

        candidate.call_method0("_ensure_built")?;
        let handle_obj = candidate.getattr("_handle")?;
        if handle_obj.is_none() {
            return Err(PyRuntimeError::new_err(
                "Pipeline could not be built for apply(); _handle is None",
            ));
        }

        let result = {
            let handle_cell: &PyCell<PyPipelineHandle> = handle_obj.downcast().map_err(|_| {
                PyTypeError::new_err("apply() expected pipeline._handle to be a PyPipelineHandle")
            })?;
            let handle_ref = handle_cell.borrow();
            py_run_pipeline(py, &handle_ref, self, persist, atomic)?
        };

        Ok(Some(result))
    }

    fn algorithm_to_spec(py: Python, algo: &PyAny) -> PyResult<PyObject> {
        if let Ok(dict) = algo.downcast::<PyDict>() {
            return Ok(dict.into());
        }

        if algo.hasattr("to_spec")? {
            let spec_any = algo.call_method0("to_spec")?;
            if !spec_any.is_instance_of::<PyDict>() {
                return Err(PyTypeError::new_err(
                    "AlgorithmHandle.to_spec() must return a dict",
                ));
            }
            return Ok(spec_any.into_py(py));
        }

        Err(PyTypeError::new_err(
            "apply() expected an AlgorithmHandle or spec dict",
        ))
    }

    fn collect_algorithm_specs(py: Python, candidate: &PyAny) -> PyResult<Vec<PyObject>> {
        let mut specs = Vec::new();

        if let Ok(list) = candidate.downcast::<PyList>() {
            for item in list.iter() {
                specs.push(Self::algorithm_to_spec(py, item)?);
            }
        } else if let Ok(tuple) = candidate.downcast::<PyTuple>() {
            for item in tuple.iter() {
                specs.push(Self::algorithm_to_spec(py, item)?);
            }
        } else {
            specs.push(Self::algorithm_to_spec(py, candidate)?);
        }

        if specs.is_empty() {
            return Err(PyTypeError::new_err(
                "apply() requires at least one algorithm to execute",
            ));
        }

        Ok(specs)
    }
}

#[pymethods]
impl PySubgraph {
    // === Basic Properties - delegate to SubgraphOperations ===

    fn query(&self) -> PyResult<crate::ffi::subgraphs::view::PyGraphView> {
        self.inner
            .query()
            .map(|inner| crate::ffi::subgraphs::view::PyGraphView { inner })
            .map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    fn build_group_index(
        &self,
        py: Python<'_>,
    ) -> PyResult<crate::ffi::query::groups::PyGroupMembershipIndex> {
        let view = self
            .inner
            .pin_view()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        py.allow_threads(|| view.build_group_index())
            .map(|inner| crate::ffi::query::groups::PyGroupMembershipIndex { inner })
            .map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    #[pyo3(signature=(*, node_properties=Vec::new(), edge_properties=Vec::new(), topology=true))]
    fn build_index(
        &self,
        py: Python<'_>,
        node_properties: Vec<String>,
        edge_properties: Vec<String>,
        topology: bool,
    ) -> PyResult<crate::ffi::query::index::PyGraphQueryIndex> {
        let view = self
            .inner
            .pin_view()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        py.allow_threads(|| view.build_index(&node_properties, &edge_properties, topology))
            .map(|inner| crate::ffi::query::index::PyGraphQueryIndex { inner })
            .map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    /// Get nodes as a property that supports indexing and attribute access
    #[getter]
    fn nodes(&self, py: Python) -> PyResult<PyObject> {
        if let Some(view) = self.inner.frozen_view() {
            return Ok(Py::new(
                py,
                super::view_entities::PyGraphViewEntities {
                    inner: view.node_accessor(),
                },
            )?
            .into_py(py));
        }
        Ok(Py::new(
            py,
            PyNodesAccessor {
                graph: self.inner.graph(),
                constrained_nodes: Some(self.inner.node_ids()),
            },
        )?
        .into_py(py))
    }

    #[getter]
    fn edges(&self, py: Python) -> PyResult<PyObject> {
        if let Some(view) = self.inner.frozen_view() {
            return Ok(Py::new(
                py,
                super::view_entities::PyGraphViewEntities {
                    inner: view.edge_accessor(),
                },
            )?
            .into_py(py));
        }
        Ok(Py::new(
            py,
            PyEdgesAccessor {
                graph: self.inner.graph(),
                constrained_edges: Some(self.inner.edge_ids()),
            },
        )?
        .into_py(py))
    }

    /// Get viz accessor for visualization operations
    #[getter]
    pub(crate) fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        let view = self.inner.pin_view().map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let source = py.allow_threads(|| groggy::viz::streaming::GraphDataSource::from_view(&view))
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let viz_accessor = crate::ffi::viz_accessor::VizAccessor::with_data_source(source, "Subgraph".into());
        Py::new(py, viz_accessor)
    }

    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.node_count() // SubgraphOperations::node_count()
    }

    /// Explicit frozen read selection of the current working state.
    fn pin_view(&self) -> PyResult<crate::ffi::subgraphs::view::PyGraphView> {
        self.inner
            .pin_view()
            .map(|inner| crate::ffi::subgraphs::view::PyGraphView { inner })
            .map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    /// Node count property
    fn node_count(&self) -> usize {
        self.inner.node_count() // SubgraphOperations::node_count()
    }

    /// Edge count property
    fn edge_count(&self) -> usize {
        self.inner.edge_count() // SubgraphOperations::edge_count()
    }

    /// Get node IDs as PyIntArray
    #[getter]
    fn node_ids(&self, py: Python) -> PyResult<Py<crate::ffi::storage::num_array::PyIntArray>> {
        let node_ids = self.inner.node_ids();
        let py_int_array = crate::ffi::storage::num_array::PyIntArray::from_node_ids(node_ids)?;
        Py::new(py, py_int_array)
    }

    /// Get edge IDs as PyIntArray
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<crate::ffi::storage::num_array::PyIntArray>> {
        let edge_ids = self.inner.edge_ids();
        let py_int_array = crate::ffi::storage::num_array::PyIntArray::from_node_ids(edge_ids)?;
        Py::new(py, py_int_array)
    }

    /// Check if a node exists in this subgraph
    fn has_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id) // SubgraphOperations::contains_node()
    }

    /// Check if an edge exists in this subgraph
    fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id) // SubgraphOperations::contains_edge()
    }

    /// Apply algorithms or pipelines to produce a new subgraph with computed attributes.
    ///
    /// Supports three usage forms:
    /// 1. `subgraph.apply(algorithm_handle)`
    /// 2. `subgraph.apply([algo1, algo2, ...])`
    /// 3. `subgraph.apply(pipeline.Pipeline([...]))`
    ///
    /// Use `return_profile=True` to receive `(subgraph, profile)`; otherwise returns the subgraph alone.
    #[pyo3(signature = (algorithm_or_pipeline, persist = true, return_profile = false, *, atomic = false))]
    pub fn apply(
        &self,
        py: Python,
        algorithm_or_pipeline: &PyAny,
        persist: bool,
        return_profile: bool,
        atomic: bool,
    ) -> PyResult<PyObject> {
        let apply_total_start = Instant::now();

        if let Some((result, profile)) =
            self.try_apply_pipeline_object(py, algorithm_or_pipeline, persist, atomic)?
        {
            let subgraph_obj = Py::new(py, result)?.into_py(py);
            if return_profile {
                let tuple = PyTuple::new(py, &[subgraph_obj.clone_ref(py), profile]);
                return Ok(tuple.into());
            }
            return Ok(subgraph_obj);
        }

        let collect_start = Instant::now();
        let specs = Self::collect_algorithm_specs(py, algorithm_or_pipeline)?;
        let collect_elapsed = collect_start.elapsed();
        let spec_list = PyList::empty(py);
        for spec in specs {
            spec_list.append(spec)?;
        }

        let build_start = Instant::now();
        let handle = py_build_pipeline(py, spec_list.as_ref())?;
        let build_elapsed = build_start.elapsed();

        let run_call_start = Instant::now();
        let run_result = py_run_pipeline(py, &handle, self, persist, atomic);
        let run_call_elapsed = run_call_start.elapsed();

        let drop_start = Instant::now();
        py_drop_pipeline(&handle);
        let drop_elapsed = drop_start.elapsed();
        let (result, profile) = run_result?;

        let create_subgraph_obj_start = Instant::now();
        let subgraph_obj = Py::new(py, result)?.into_py(py);
        let create_subgraph_obj_elapsed = create_subgraph_obj_start.elapsed();

        if let Ok(profile_dict) = profile.as_ref(py).downcast::<PyDict>() {
            let python_apply = PyDict::new(py);
            python_apply.set_item("collect_spec", collect_elapsed.as_secs_f64())?;
            python_apply.set_item("build_pipeline_py", build_elapsed.as_secs_f64())?;
            python_apply.set_item("run_pipeline_py", run_call_elapsed.as_secs_f64())?;
            python_apply.set_item("drop_pipeline_py", drop_elapsed.as_secs_f64())?;
            python_apply.set_item(
                "create_subgraph_obj_py",
                create_subgraph_obj_elapsed.as_secs_f64(),
            )?;

            if let Ok(Some(run_time_obj)) = profile_dict.get_item("run_time") {
                if let Ok(run_time) = run_time_obj.extract::<f64>() {
                    let overhead = run_call_elapsed.as_secs_f64() - run_time;
                    python_apply.set_item("run_pipeline_overhead", overhead)?;
                }
            }

            let accounted = collect_elapsed
                + build_elapsed
                + run_call_elapsed
                + drop_elapsed
                + create_subgraph_obj_elapsed;
            let remaining = apply_total_start
                .elapsed()
                .checked_sub(accounted)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            python_apply.set_item("apply_remaining_py", remaining)?;

            profile_dict.set_item("python_apply", python_apply)?;
        }
        if return_profile {
            let tuple = PyTuple::new(py, &[subgraph_obj.clone_ref(py), profile]);
            Ok(tuple.into())
        } else {
            Ok(subgraph_obj)
        }
    }

    // === Analysis Methods - delegate to SubgraphOperations ===

    /// Edge-instance density using the native directed/undirected definition.
    fn density(&self) -> f64 { self.inner.density() }

    /// Get connected components within this subgraph (lazy array)
    fn connected_components(&self) -> PyResult<PyComponentsArray> {
        let components = self
            .inner
            .connected_components()
            .map_err(|e| PyRuntimeError::new_err(format!("Connected components error: {}", e)))?;

        // Create lazy ComponentsArray - no immediate PySubgraph materialization!
        let components_array = PyComponentsArray::from_components(
            components,
            self.inner.graph().clone(),
            self.inner
                .clone()
                .freeze()
                .and_then(|scope| scope.pin_view())
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        )?;

        Ok(components_array)
    }

    /// Empty selections are not connected; nonempty selections use weak connectivity.
    fn is_connected(&self) -> PyResult<bool> {
        self.inner.is_connected().map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    /// Check if there is a path between two nodes within this subgraph
    ///
    /// This is more efficient than `shortest_path_subgraph` when you only need
    /// to know if a path exists, not the actual path.
    ///
    /// # Arguments
    /// * `node1_id` - The starting node ID
    /// * `node2_id` - The destination node ID
    ///
    /// # Returns
    /// * `True` if a path exists between the nodes within this subgraph
    /// * `False` if no path exists or either node is not in this subgraph
    ///
    /// # Example
    /// ```python
    /// # Check if there's a path between node 1 and node 5 in the subgraph
    /// path_exists = subgraph.has_path(1, 5)
    /// ```
    fn has_path(&self, node1_id: NodeId, node2_id: NodeId) -> PyResult<bool> {
        self.inner
            .has_path(node1_id, node2_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Error checking path: {}", e)))
    }

    // === Visualization Methods ===

    // === Data Export Methods ===

    /// Convert subgraph to a GraphTable containing both nodes and edges
    /// Pure delegation to core GraphTable
    pub fn table(&self, py: Python) -> PyResult<PyObject> {
        let view = self
            .inner
            .pin_view()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let table = py
            .allow_threads(|| view.table())
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(crate::ffi::storage::table::PyGraphTable::from_table(table).into_py(py))
    }

    /// Export edge columns from one pinned current selection.
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        let view = self
            .inner
            .pin_view()
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let table = py
            .allow_threads(|| view.edges_table())
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(Py::new(py, crate::ffi::storage::table::PyEdgesTable { table })?.into_py(py))
    }

    // === Filtering Methods - delegate to SubgraphOperations ===

    /// Filter nodes and return new subgraph  
    fn filter_nodes(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Extract the filter from Python object - support both NodeFilter objects and string queries
        let node_filter = if let Ok(filter_obj) =
            filter.extract::<crate::ffi::query::query::PyNodeFilter>()
        {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser
            let mut parser = groggy::query::QueryParser::new();
            parser.parse_node_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "filter must be a NodeFilter object or a string query (e.g., 'salary > 120000')",
            ));
        };

        if let Some(view) = self.inner.frozen_view() {
            let child = py
                .allow_threads(|| view.filter_nodes(&node_filter))
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            return Self::from_core_subgraph(Subgraph::from_frozen_view(
                self.inner.graph(),
                child,
                "filtered".into(),
            ));
        }
        let subgraph = self
            .inner
            .filter_nodes(&node_filter)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        PySubgraph::from_core_subgraph(subgraph)
    }

    /// Filter edges and return new subgraph
    fn filter_edges(&self, py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
        // Extract the filter from Python object - support both EdgeFilter objects and string queries
        let edge_filter = if let Ok(filter_obj) =
            filter.extract::<crate::ffi::query::query::PyEdgeFilter>()
        {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // String query - parse it using Rust core query parser
            let mut parser = groggy::query::QueryParser::new();
            parser.parse_edge_query(&query_str).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Query parse error: {}", e))
            })?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "filter must be an EdgeFilter object or a string query (e.g., 'weight > 0.5')",
            ));
        };

        if let Some(view) = self.inner.frozen_view() {
            let child = py
                .allow_threads(|| view.filter_edges(&edge_filter))
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            return Self::from_core_subgraph(Subgraph::from_frozen_view(
                self.inner.graph(),
                child,
                "filtered".into(),
            ));
        }
        let subgraph = self
            .inner
            .filter_edges(&edge_filter)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        PySubgraph::from_core_subgraph(subgraph)
    }

    // === Graph Conversion Methods ===

    /// Materialize the validated selected state into an independent publishing graph.
    fn to_graph(&self) -> PyResult<PyGraph> {
        let graph = self
            .inner
            .pin_view()
            .and_then(|view| view.to_graph())
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(PyGraph {
            inner: graph.into_shared(),
            cached_view: groggy::ownership::OwnerCell::new(None),
            cached_live_view: groggy::ownership::OwnerCell::new(None),
        })
    }

    /// Convert to NetworkX graph (if available)
    ///
    /// Returns a NetworkX Graph or DiGraph (depending on the parent graph type)
    /// containing only the nodes and edges from this subgraph, with all attributes preserved.
    ///
    /// # Returns
    /// * `PyObject` - A NetworkX graph object containing only this subgraph
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
    /// subgraph = g.filter_nodes("age > 25")
    /// nx_subgraph = subgraph.to_networkx()
    /// ```
    fn to_networkx(&self, py: Python) -> PyResult<PyObject> {
        // Convert to our internal NetworkX representation
        let nx_graph = self.inner.to_networkx().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert subgraph to NetworkX: {}", e))
        })?;

        // Convert to actual Python NetworkX graph
        crate::ffi::utils::convert::networkx_graph_to_python(py, &nx_graph)
    }

    /// Count incident edge instances (loops once), in caller order or ascending IDs.
    /// Invalid or stale selections fail instead of silently dropping result rows.
    #[pyo3(signature = (nodes = None, *, full_graph = false))]
    fn degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        self.degree_result(
            py,
            nodes,
            full_graph,
            groggy::subgraphs::DegreeMode::Incident,
        )
    }

    /// Incoming instance counts; undirected inputs use incident degree.
    #[pyo3(signature = (nodes = None, full_graph = false))]
    fn in_degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        self.degree_result(
            py,
            nodes,
            full_graph,
            groggy::subgraphs::DegreeMode::Incoming,
        )
    }

    /// Outgoing instance counts; undirected inputs use incident degree.
    #[pyo3(signature = (nodes = None, full_graph = false))]
    fn out_degree(
        &self,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
    ) -> PyResult<PyObject> {
        self.degree_result(
            py,
            nodes,
            full_graph,
            groggy::subgraphs::DegreeMode::Outgoing,
        )
    }

    /// Calculate similarity between subgraphs using various metrics
    #[pyo3(signature = (other, metric = "jaccard"))]
    fn calculate_similarity(&self, other: &PySubgraph, metric: &str, _py: Python) -> PyResult<f64> {
        let similarity_metric = match metric {
            "jaccard" => SimilarityMetric::Jaccard,
            "dice" => SimilarityMetric::Dice,
            "cosine" => SimilarityMetric::Cosine,
            "overlap" => SimilarityMetric::Overlap,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown similarity metric: '{}'. Valid options: 'jaccard', 'dice', 'cosine', 'overlap'", metric)
            ))
        };

        self.inner
            .calculate_similarity(&other.inner, similarity_metric)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Similarity calculation error: {}",
                    e
                ))
            })
    }

    /// Attribute values aligned with ascending selected node IDs. Missing values
    /// project to Null; invalid live IDs and corrupt columns remain errors.
    fn __getitem__(&self, key: &str) -> PyResult<PyBaseArray> {
        let values = self.inner.get_node_attribute_column(&key.to_owned())
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(PyBaseArray { inner: BaseArray::new(values) })
    }

    /// Compute neighborhoods from this subgraph and return them as a SubgraphArray
    #[pyo3(signature = (center_nodes = None, hops = 1))]
    pub fn neighborhood(
        &self,
        py: Python,
        center_nodes: Option<&PyAny>,
        hops: usize,
    ) -> PyResult<PySubgraphArray> {
        // Just wrap the graph_analysis version - create a temporary PyGraph from our core graph
        use crate::ffi::api::graph::PyGraph;
        use crate::ffi::api::graph_analysis::PyGraphAnalysis;

        // Normalize central node identifiers into a Vec<NodeId>
        let central_nodes_vec = match center_nodes {
            Some(arg) => {
                if let Ok(single) = arg.extract::<NodeId>() {
                    vec![single]
                } else if let Ok(many) = arg.extract::<Vec<NodeId>>() {
                    many
                } else {
                    let seq = arg.iter()?;
                    let mut nodes = Vec::new();
                    for item in seq {
                        nodes.push(item?.extract::<NodeId>()?);
                    }
                    nodes
                }
            }
            None => self.inner.node_set().iter().copied().collect(),
        };

        if central_nodes_vec.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "neighborhood() requires at least one center node",
            ));
        }

        if let Some(view) = self.inner.frozen_view() {
            let mut subgraphs = Vec::with_capacity(central_nodes_vec.len());
            for center in central_nodes_vec {
                let child = view
                    .within_hops(
                        &[center],
                        hops,
                        groggy::query::selection::TraversalDirection::Both,
                    )
                    .map_err(crate::ffi::utils::graph_error_to_py_err)?;
                subgraphs.push(Self::from_core_subgraph(Subgraph::from_frozen_view(
                    self.inner.graph(),
                    child,
                    "neighborhood".into(),
                ))?);
            }
            return Ok(PySubgraphArray::new(subgraphs));
        }

        // Create a temporary PyGraph wrapper
        let py_graph = PyGraph {
            inner: self.inner.graph(),
            cached_view: groggy::ownership::OwnerCell::new(None),
            cached_live_view: groggy::ownership::OwnerCell::new(None),
        };

        // Create PyGraphAnalysis and delegate to it
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, py_graph)?)?;
        let result = analysis_handler.neighborhood(py, central_nodes_vec, Some(hops), None)?;

        // Convert the NeighborhoodArray into a SubgraphArray for Python consumption
        Ok(result.inner().clone())
    }

    /// Sample using either a sampler pipeline or a simple k value.
    #[pyo3(signature = (sampler_or_k, *, seed=None))]
    pub fn sample(&self, py: Python, sampler_or_k: &PyAny, seed: Option<&PyAny>) -> PyResult<PyObject> {
        let seed = crate::ffi::utils::optional_seed(seed)?;
        if let Ok(k) = crate::ffi::utils::sample_count(sampler_or_k) {
            return Ok(PySubgraph::from_core_subgraph(
                self.inner.sample(k, seed).map_err(crate::ffi::utils::graph_error_to_py_err)?
            )?.into_py(py));
        }

        if seed.is_some() {
            return Err(PyTypeError::new_err("seed is only supported for integer sampling; set sampler seeds in its specification"));
        }
        if sampler_or_k.hasattr("to_spec")? {
            let spec_any = sampler_or_k.call_method0("to_spec")?;
            let handle = py_build_sample_pipeline(py, spec_any)?;
            let result = py_run_sample_pipeline(py, &handle, self);
            py_drop_sample_pipeline(&handle);
            return Ok(result?.into_py(py));
        }

        Err(PyTypeError::new_err(
            "sample() expects an int or a sampler with to_spec()",
        ))
    }

    pub(crate) fn sample_k(&self, k: usize) -> PyResult<PySubgraph> {
        PySubgraph::from_core_subgraph(
            self.inner.sample(k, None).map_err(crate::ffi::utils::graph_error_to_py_err)?
        )
    }

    // === String representations ===

    fn __repr__(&self) -> String {
        format!(
            "Subgraph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }

    // === MISSING BASIC OPERATIONS ===

    /// Check if subgraph is empty
    fn is_empty(&self) -> bool {
        self.inner.node_count() == 0
    }

    /// Get text summary of subgraph
    fn summary(&self) -> String {
        format!(
            "Subgraph: {} nodes, {} edges, density: {:.3}",
            self.inner.node_count(),
            self.inner.edge_count(),
            self.inner.density()
        )
    }

    /// Check if subgraph contains a specific node (alias for has_node)
    fn contains_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id)
    }

    /// Check if subgraph contains a specific edge (alias for has_edge)  
    fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id)
    }

    /// Get neighbors of a node within the subgraph
    fn neighbors(&self, py: Python, node_id: NodeId) -> PyResult<Py<PyNumArray>> {
        let ids = if let Some(view) = self.inner.frozen_view() {
            py.allow_threads(|| view.neighbors(node_id))
        } else {
            self.inner.neighbors(node_id)
        }
        .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Py::new(py, PyNumArray::from_node_ids(ids)?)
    }

    /// Get edge endpoints (source, target)
    fn edge_endpoints(&self, _py: Python, edge_id: EdgeId) -> PyResult<(NodeId, NodeId)> {
        match self.inner.edge_endpoints(edge_id) {
            Ok(endpoints) => Ok(endpoints),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Edge endpoints error: {}",
                e
            ))),
        }
    }

    /// Check if edge exists between two nodes
    fn has_edge_between(&self, _py: Python, source: NodeId, target: NodeId) -> PyResult<bool> {
        match self.inner.has_edge_between(source, target) {
            Ok(exists) => Ok(exists),
            Err(e) => Err(PyRuntimeError::new_err(format!("Edge check error: {}", e))),
        }
    }

    // === MISSING ATTRIBUTE ACCESS METHODS ===

    /// Get a single node attribute value
    fn get_node_attribute(
        &self,
        py: Python,
        node_id: NodeId,
        attr_name: String,
    ) -> PyResult<Option<PyObject>> {
        use crate::ffi::utils::attr_value_to_python_value;
        match self.inner.get_node_attribute(node_id, &attr_name) {
            Ok(Some(attr_value)) => {
                let py_value = attr_value_to_python_value(py, &attr_value)?;
                Ok(Some(py_value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Get node attribute error: {}",
                e
            ))),
        }
    }

    /// Get a single edge attribute value
    fn get_edge_attribute(
        &self,
        py: Python,
        edge_id: EdgeId,
        attr_name: String,
    ) -> PyResult<Option<PyObject>> {
        use crate::ffi::utils::attr_value_to_python_value;
        match self.inner.get_edge_attribute(edge_id, &attr_name) {
            Ok(Some(attr_value)) => {
                let py_value = attr_value_to_python_value(py, &attr_value)?;
                Ok(Some(py_value))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Get edge attribute error: {}",
                e
            ))),
        }
    }

    /// Set multiple node attributes (bulk operation) - delegates to accessor
    fn set_node_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let scope = self
            .inner
            .execution_scope(true)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        PyNodesAccessor {
            graph: scope.graph(),
            constrained_nodes: Some(scope.node_ids()),
        }
        .set_attrs_internal(py, attrs_dict)
    }

    /// Publish an explicit checked bulk edge update; this read stays unchanged.
    fn set_edge_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        let scope = self
            .inner
            .execution_scope(true)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        PyEdgesAccessor {
            graph: scope.graph(),
            constrained_edges: Some(scope.edge_ids()),
        }
        .set_attrs_internal(py, attrs_dict)
    }

    // === MISSING GRAPH METRICS ===

    /// Local or average simple undirected clustering; prepared topology crosses the GIL boundary.
    #[pyo3(signature=(node_id=None))]
    fn clustering_coefficient(&self, py: Python<'_>, node_id: Option<&PyAny>) -> PyResult<f64> {
        let node = node_id.map(|v| crate::ffi::query::query::nonnegative_count(v, "node_id")).transpose()?;
        let input = self.inner.structural_read().map_err(crate::ffi::utils::graph_error_to_py_err)?;
        py.allow_threads(|| input.clustering_coefficient(node)).map_err(crate::ffi::utils::graph_error_to_py_err)
    }
    fn transitivity(&self, py: Python<'_>) -> PyResult<f64> {
        let input = self.inner.structural_read().map_err(crate::ffi::utils::graph_error_to_py_err)?;
        py.allow_threads(|| input.transitivity()).map_err(crate::ffi::utils::graph_error_to_py_err)
    }

    // === ENTITY TYPE METHOD ===

    /// Return the entity type string
    fn entity_type(&self) -> PyResult<String> {
        Ok("Subgraph".to_string())
    }

    // === MISSING SUBGRAPH OPERATIONS ===

    /// Traverse the captured selection and preserve its source.
    fn bfs(&self, _py: Python, start: NodeId, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        Self::from_trait_object(
            self.inner
                .bfs(start, max_depth)
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        )
    }
    fn dfs(&self, _py: Python, start: NodeId, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        Self::from_trait_object(
            self.inner
                .dfs(start, max_depth)
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        )
    }
    fn shortest_path_subgraph(
        &self,
        _py: Python,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<Option<PySubgraph>> {
        self.inner
            .shortest_path_subgraph(source, target)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?
            .map(Self::from_trait_object)
            .transpose()
    }
    fn induced_subgraph(&self, _py: Python, nodes: Vec<NodeId>) -> PyResult<PySubgraph> {
        Self::from_trait_object(
            self.inner
                .induced_subgraph(&nodes)
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        )
    }
    fn subgraph_from_edges(&self, _py: Python, edges: Vec<EdgeId>) -> PyResult<PySubgraph> {
        Self::from_trait_object(
            self.inner
                .subgraph_from_edges(&edges)
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
        )
    }

    /// Exact membership algebra over a common live owner.
    fn merge_with(&self, _py: Python, other: &PySubgraph) -> PyResult<PySubgraph> {
        self.inner
            .merge_with(&other.inner)
            .map_err(crate::ffi::utils::graph_error_to_py_err)
            .and_then(Self::from_core_subgraph)
    }

    fn intersect_with(&self, _py: Python, other: &PySubgraph) -> PyResult<PySubgraph> {
        self.inner
            .intersect_with(&other.inner)
            .map_err(crate::ffi::utils::graph_error_to_py_err)
            .and_then(Self::from_core_subgraph)
    }

    fn subtract_from(&self, _py: Python, other: &PySubgraph) -> PyResult<PySubgraph> {
        self.inner
            .subtract_from(&other.inner)
            .map_err(crate::ffi::utils::graph_error_to_py_err)
            .and_then(Self::from_core_subgraph)
    }

    /// Enhanced collapse supporting three syntax forms for flexible aggregation
    ///
    /// # Supported Syntax Forms:
    ///
    /// ## Form 1: Simple (backward compatible)
    /// ```python
    /// subgraph.add_to_graph({"age": "mean", "salary": "sum"})
    /// ```
    ///
    /// ## Form 2: Tuple (custom attribute names)
    /// ```python
    /// subgraph.add_to_graph({
    ///     "avg_age": ("mean", "age"),
    ///     "total_salary": ("sum", "salary"),
    ///     "person_count": ("count", None)
    /// })
    /// ```
    ///
    /// ## Form 3: Dict-of-dicts (advanced with defaults)
    /// ```python
    /// subgraph.add_to_graph({
    ///     "avg_age": {"func": "mean", "source": "age"},
    ///     "total_salary": {"func": "sum", "source": "salary", "default": 0}
    /// })
    /// ```
    // === HIERARCHICAL OPERATIONS ===
    /// Get parent meta-node if this subgraph is contained within one
    /// NOTE: This feature is not yet implemented - always returns None
    fn parent_meta_node(&self, _py: Python) -> PyResult<Option<PyObject>> {
        // TODO: Implement hierarchical navigation in future version
        // The current HierarchicalOperations trait methods are stubs that return None
        Ok(None)
    }

    /// Get child meta-nodes if this subgraph contains them
    /// NOTE: This feature is not yet implemented - always returns empty list
    fn child_meta_nodes(&self, _py: Python) -> PyResult<Vec<PyObject>> {
        // TODO: Implement hierarchical navigation in future version
        // The current HierarchicalOperations trait methods are stubs
        Ok(Vec::new())
    }

    /// Get hierarchy level of this subgraph (0 = root level)
    #[getter]
    fn hierarchy_level(&self) -> PyResult<usize> {
        use groggy::subgraphs::HierarchicalOperations;

        self.inner.hierarchy_level().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to get hierarchy level: {}",
                e
            ))
        })
    }

    /// Check if this subgraph contains nodes that are meta-nodes
    fn has_meta_nodes(&self) -> bool {
        use groggy::subgraphs::HierarchicalOperations;

        // Check if any child meta-nodes exist
        !self.inner.child_meta_nodes().unwrap_or_default().is_empty()
    }

    /// Get all meta-nodes within this subgraph
    fn meta_nodes(&self, py: Python) -> PyResult<Vec<PyObject>> {
        self.child_meta_nodes(py)
    }

    /// Modern MetaGraph Composer API - Clean interface for meta-node creation
    ///
    /// This is the new, intuitive way to create meta-nodes with flexible configuration.
    /// Returns a MetaNodePlan that can be previewed, modified, and executed.
    ///
    /// # Arguments
    /// * `node_aggs` - Node aggregation specifications (dict or list format)
    /// * `edge_aggs` - Edge aggregation specifications (dict format)
    /// * `edge_strategy` - Edge handling strategy ("aggregate", "keep_external", "drop_all", "contract_all")
    /// * `node_strategy` - Node handling strategy ("extract", "collapse")
    /// * `preset` - Optional preset name ("social_network", "org_hierarchy", "flow_network")
    /// * `include_edge_count` - Include edge_count attribute in meta-edges
    /// * `mark_entity_type` - Mark meta-nodes/edges with entity_type
    /// * `entity_type` - Entity type for marking
    ///
    /// # Examples
    /// ```python
    /// # Dict format for node aggregations
    /// plan = subgraph.collapse(
    ///     node_aggs={"avg_salary": ("mean", "salary"), "size": "count"},
    ///     edge_aggs={"weight": "mean"},
    ///     edge_strategy="aggregate",
    ///     node_strategy="extract"
    /// )
    /// meta_node = plan.add_to_graph()
    ///
    /// # With preset
    /// plan = subgraph.collapse(preset="social_network")
    /// meta_node = plan.add_to_graph()
    /// ```
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
    pub fn collapse(
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
    ) -> PyResult<PyObject> {
        let options = crate::ffi::subgraphs::composer::parse_composer_options(
            node_aggs,
            edge_aggs,
            edge_strategy,
            node_strategy,
            preset,
            include_edge_count,
            mark_entity_type,
            entity_type,
        )?;
        let plan = self.inner.collapse_with_options(options).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create plan: {e}"))
        })?;
        let meta_node = plan
            .execute_configured(
                &self.inner,
                allow_missing_attributes,
                meta_targets.as_deref(),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create meta-node: {e}"
                ))
            })?;

        // Return the MetaNode directly using the new trait-based entity
        use crate::ffi::entities::PyMetaNode;
        let py_meta_node = PyMetaNode::from_meta_node(meta_node);
        Ok(Py::new(py, py_meta_node)?.to_object(py))
    }

    fn __str__(&self) -> String {
        let mut result = format!(
            "Subgraph with {} nodes and {} edges",
            self.inner.node_count(),
            self.inner.edge_count()
        );

        // Add edge table if there are edges
        if self.inner.edge_count() > 0 {
            result.push_str("\n\nEdges:");
            result.push_str("\n  ID    Source → Target");
            result.push_str("\n  ----  ---------------");

            // Iterate through edges in the subgraph
            for &edge_id in self.inner.edge_set() {
                if let Ok((source, target)) = self.inner.edge_endpoints(edge_id) {
                    result.push_str(&format!("\n  {:4}  {:6} → {}", edge_id, source, target));
                }
            }
        }

        result
    }

    // ========================================================================
    // PHASE 3: Cross-Type Conversions - Enable unified delegation architecture
    // ========================================================================

    /// Return immutable node accessors sharing the captured source.
    pub fn to_nodes(&self, py: Python) -> PyResult<PyObject> {
        self.nodes(py)
    }
    /// Return immutable edge accessors sharing the captured source.
    pub fn to_edges(&self, py: Python) -> PyResult<PyObject> {
        self.edges(py)
    }

    /// Convert this subgraph to its adjacency matrix representation
    /// Enables chaining like: subgraph.to_matrix().eigen().stats()
    pub fn to_matrix(&self, py: Python) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        self.adjacency_matrix(py, false, None)
    }

    // === ADJACENCY METHODS (moved from Graph to Subgraph) ===

    /// Convert this subgraph to its adjacency matrix representation
    /// Returns a GraphMatrix with rows/columns scoped to this subgraph
    #[pyo3(signature = (*, sparse=false, weight_attr=None))]
    fn adjacency_matrix(
        &self,
        py: Python,
        sparse: bool,
        weight_attr: Option<&str>,
    ) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        if let Some(view) = self.inner.frozen_view() {
            return super::view::owned_matrix(py, view, |view| {
                view.adjacency_matrix::<f64>(sparse, weight_attr)
            });
        }
        let matrix = self
            .inner
            .adjacency_matrix(sparse, weight_attr)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(crate::ffi::storage::matrix::PyGraphMatrix::from_graph_matrix(matrix))
    }

    /// Get adjacency matrix (alias for adjacency_matrix).
    #[pyo3(signature = (*, sparse=false, weight_attr=None))]
    fn adj(
        &self,
        py: Python,
        sparse: bool,
        weight_attr: Option<&str>,
    ) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        self.adjacency_matrix(py, sparse, weight_attr)
    }

    /// Selected outgoing adjacency from a detached topology-only read.
    fn adjacency_list(&self, py: Python<'_>) -> PyResult<std::collections::BTreeMap<NodeId, Vec<NodeId>>> {
        let input = self.inner.adjacency_read().map_err(crate::ffi::utils::graph_error_to_py_err)?;
        Ok(py.allow_threads(|| input.adjacency_list()))
    }

    /// Group subgraph by attribute value
    ///
    /// Args:
    ///     attr_name: Name of the attribute to group by
    ///     element_type: Either 'nodes' or 'edges' to specify what to group
    ///
    /// Returns:
    ///     SubgraphArray: Array of subgraphs, one for each unique attribute value
    ///
    /// Example:
    ///     dept_groups = subgraph.group_by('department', 'nodes')
    ///     type_groups = subgraph.group_by('interaction_type', 'edges')
    pub fn group_by(
        &self,
        attr_name: String,
        element_type: String,
    ) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let attr_name = groggy::types::AttrName::from(attr_name);

        let subgraphs = match element_type.as_str() {
            "nodes" => self
                .inner
                .group_by_nodes(&attr_name)
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
            "edges" => self
                .inner
                .group_by_edges(&attr_name)
                .map_err(crate::ffi::utils::graph_error_to_py_err)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "element_type must be either 'nodes' or 'edges'",
                ));
            }
        };

        // Convert to PySubgraph objects
        let py_subgraphs: Result<Vec<_>, _> = subgraphs
            .into_iter()
            .map(PySubgraph::from_core_subgraph)
            .collect();

        let py_subgraphs = py_subgraphs?;
        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            py_subgraphs,
        ))
    }

    /// Dynamic attribute access for node and edge attribute dictionaries within subgraph.
    ///
    /// **Intentional dynamic pattern**: Enables property-style attribute access for runtime
    /// data analysis. When accessing `subgraph.age`, returns a dict of `{node_id: age_value}`
    /// for nodes in this subgraph.
    ///
    /// This pattern remains dynamic because:
    /// - Attribute names are user-defined and vary per graph (e.g., "age", "weight", "label")
    /// - Subgraphs inherit attributes from parent graph, making schema data-dependent
    /// - Common in data science workflows: `sg.age.mean()`, `sg.salary.sum()`
    ///
    /// All subgraph **methods** (sample, filter, bfs, etc.) are explicitly defined above.
    /// Only **attribute data** projections remain dynamic for ergonomic data access.
    fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
        use pyo3::exceptions::PyAttributeError;
        use pyo3::types::PyDict;
        use std::collections::HashSet;

        // Prevent access to special Python attributes
        match name.as_str() {
            "__dict__" | "__class__" | "__module__" | "__doc__" | "__weakref__" | "__slots__" => {
                return Err(PyAttributeError::new_err(format!(
                    "'Subgraph' object has no attribute '{}'",
                    name
                )));
            }
            _ => {}
        }

        if let Some(view) = self.inner.frozen_view() {
            let nodes = view.node_accessor();
            let edges = view.edge_accessor();
            let project = |accessor: groggy::subgraphs::GraphViewEntities| -> PyResult<PyObject> {
                let result = PyDict::new(py);
                for id in accessor.ids() {
                    let value = accessor
                        .by_id(*id)
                        .map_err(crate::ffi::utils::graph_error_to_py_err)?
                        .get(&name);
                    result.set_item(
                        id,
                        match value {
                            Some(value) => {
                                crate::ffi::utils::attr_value_to_python_value(py, &value)?
                            }
                            None => py.None(),
                        },
                    )?;
                }
                Ok(result.into())
            };
            if nodes.attribute_names().contains(&name) {
                return project(nodes);
            }
            if edges.attribute_names().contains(&name) {
                return project(edges);
            }
            return Err(PyAttributeError::new_err(format!(
                "'Subgraph' object has no attribute '{}'",
                name
            )));
        }

        // Get all available node attribute names in this subgraph
        let graph_rc = self.inner.graph().clone();
        let graph_ref = graph_rc.borrow();
        let mut all_node_attrs = HashSet::new();
        for node_id in self.inner.node_set() {
            if let Ok(attrs) = graph_ref.get_node_attrs(*node_id) {
                for attr_name in attrs.keys() {
                    all_node_attrs.insert(attr_name.clone());
                }
            }
        }

        // Get all available edge attribute names in this subgraph
        let mut all_edge_attrs = HashSet::new();
        for edge_id in self.inner.edge_set() {
            if let Ok(attrs) = graph_ref.get_edge_attrs(*edge_id) {
                for attr_name in attrs.keys() {
                    all_edge_attrs.insert(attr_name.clone());
                }
            }
        }

        // INTENTIONAL DYNAMIC PATTERN: Node attribute dictionary projection
        // Check if this is a node attribute name within this subgraph's nodes
        // Returns {node_id: value} dict scoped to subgraph's node set
        if all_node_attrs.contains(&name) {
            let result_dict = PyDict::new(py);

            for node_id in self.inner.node_set() {
                match graph_ref.get_node_attr(*node_id, &name) {
                    Ok(Some(attr_value)) => {
                        let py_value =
                            crate::ffi::utils::attr_value_to_python_value(py, &attr_value)?;
                        result_dict.set_item(node_id, py_value)?;
                    }
                    Ok(None) => {
                        result_dict.set_item(node_id, py.None())?;
                    }
                    Err(_) => continue,
                }
            }

            return Ok(result_dict.to_object(py));
        }

        // INTENTIONAL DYNAMIC PATTERN: Edge attribute dictionary projection
        // Check if this is an edge attribute name within this subgraph's edges
        // Returns {edge_id: value} dict scoped to subgraph's edge set
        if all_edge_attrs.contains(&name) {
            let result_dict = PyDict::new(py);

            for edge_id in self.inner.edge_set() {
                match graph_ref.get_edge_attr(*edge_id, &name) {
                    Ok(Some(attr_value)) => {
                        let py_value =
                            crate::ffi::utils::attr_value_to_python_value(py, &attr_value)?;
                        result_dict.set_item(edge_id, py_value)?;
                    }
                    Ok(None) => {
                        result_dict.set_item(edge_id, py.None())?;
                    }
                    Err(_) => continue,
                }
            }

            return Ok(result_dict.to_object(py));
        }

        // Convert to Vec for error message
        let node_attrs_vec: Vec<String> = all_node_attrs.into_iter().collect();
        let edge_attrs_vec: Vec<String> = all_edge_attrs.into_iter().collect();

        // Attribute not found
        Err(PyAttributeError::new_err(format!(
            "'Subgraph' object has no attribute '{}'. Available node attributes: {:?}, Available edge attributes: {:?}",
            name, node_attrs_vec, edge_attrs_vec
        )))
    }
}

/// Parse enhanced aggregation specification from Python dict supporting three syntax forms
/// Future feature for enhanced aggregation
#[allow(dead_code)]
fn parse_enhanced_aggregation_spec(
    py_dict: &pyo3::types::PyDict,
) -> PyResult<Vec<groggy::traits::subgraph_operations::AggregationSpec>> {
    let mut specs = Vec::new();

    for (key, value) in py_dict {
        let target_attr = key.extract::<String>()?;

        // Parse the value based on its type
        let agg_spec = if let Ok(func_str) = value.extract::<String>() {
            // FORM 1: Simple - {"age": "mean"}
            groggy::traits::subgraph_operations::AggregationSpec {
                target_attr: target_attr.clone(),
                function: func_str,
                source_attr: Some(target_attr), // source = target for simple form
                default_value: None,
            }
        } else if let Ok(tuple) = value.extract::<(&str, Option<&str>)>() {
            // FORM 2: Tuple - {"avg_age": ("mean", "age")}
            let (func_str, source_str) = tuple;
            groggy::traits::subgraph_operations::AggregationSpec {
                target_attr,
                function: func_str.to_string(),
                source_attr: source_str.map(|s| s.to_string()),
                default_value: None,
            }
        } else if let Ok(dict) = value.extract::<&pyo3::types::PyDict>() {
            // FORM 3: Dict - {"avg_age": {"func": "mean", "source": "age", "default": 0}}
            let func_str = dict
                .get_item("func")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Missing 'func' in aggregation spec")
                })?
                .extract::<String>()?;

            let source_attr = dict
                .get_item("source")?
                .map(|s| s.extract::<String>())
                .transpose()?;

            let default_value = if let Some(default_item) = dict.get_item("default")? {
                let attr_value = crate::ffi::utils::python_value_to_attr_value(default_item)?;
                Some(attr_value)
            } else {
                None
            };

            groggy::traits::subgraph_operations::AggregationSpec {
                target_attr,
                function: func_str,
                source_attr,
                default_value,
            }
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Invalid aggregation specification for '{}'. Expected string, tuple, or dict.",
                target_attr
            )));
        };

        specs.push(agg_spec);
    }

    Ok(specs)
}

// ============================================================================
// TRAIT IMPLEMENTATION - Core delegation pattern
// ============================================================================

// Shadow trait implementation removed - PySubgraph now uses direct delegation only
