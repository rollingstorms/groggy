//! Simplified Subgraph FFI Bindings - Complete Replacement
//!
//! Pure delegation to core Subgraph with ALL the same methods as the current PySubgraph.
//! This replaces the 800+ line complex version with pure delegation to existing trait methods.

use crate::ffi::api::pipeline::{
    py_build_pipeline, py_drop_pipeline, py_run_pipeline, PyPipelineHandle,
};
use crate::ffi::storage::subgraph_array::PySubgraphArray;
// use crate::ffi::core::path_result::PyPathResult; // Unused
use groggy::storage::array::BaseArray;
use groggy::subgraphs::Subgraph;
use groggy::traits::{GraphEntity, SubgraphOperations};
use groggy::{AttrValue, EdgeId, NodeId, SimilarityMetric};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{Py, PyCell, PyObject};
use std::collections::HashSet;
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
#[pyclass(name = "Subgraph", unsendable)]
#[derive(Clone)]
pub struct PySubgraph {
    pub inner: Subgraph,
}

impl PySubgraph {
    /// Create from Rust Subgraph
    pub fn from_core_subgraph(subgraph: Subgraph) -> PyResult<Self> {
        Ok(Self { inner: subgraph })
    }

    /// Create from trait object (used by trait delegation)
    pub fn from_trait_object(
        _subgraph: Box<dyn groggy::traits::SubgraphOperations>,
    ) -> PyResult<Self> {
        // For now, we'll use a simpler approach - assume we can only handle concrete Subgraph types
        // In the future, we might need better trait object handling with proper Any downcasting
        Err(PyRuntimeError::new_err(
            "from_trait_object not yet implemented - use concrete Subgraph types",
        ))
    }

    fn try_apply_pipeline_object(
        &self,
        py: Python,
        candidate: &PyAny,
        persist: bool,
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
            py_run_pipeline(py, &handle_ref, self, persist)?
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

    /// Get nodes as a property that supports indexing and attribute access
    #[getter]
    fn nodes(&self, py: Python) -> PyResult<Py<PyNodesAccessor>> {
        let node_collection = self.inner.node_set().iter().copied().collect();

        Py::new(
            py,
            PyNodesAccessor {
                graph: self.inner.graph(),
                constrained_nodes: Some(node_collection),
            },
        )
    }

    /// Get edges as a property that supports indexing and attribute access
    #[getter]
    fn edges(&self, py: Python) -> PyResult<Py<PyEdgesAccessor>> {
        let edge_collection = self.inner.edge_set().iter().copied().collect();

        Py::new(
            py,
            PyEdgesAccessor {
                graph: self.inner.graph(),
                constrained_edges: Some(edge_collection),
            },
        )
    }

    /// Get viz accessor for visualization operations
    #[getter]
    pub(crate) fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        use groggy::viz::streaming::GraphDataSource;

        // Create GraphDataSource directly from the subgraph's underlying graph
        // The subgraph already filters to the right nodes/edges
        let graph_ref = self.inner.graph();
        let graph_data_source = GraphDataSource::new(&graph_ref.borrow());

        let viz_accessor = crate::ffi::viz_accessor::VizAccessor::with_data_source(
            graph_data_source,
            "Subgraph".to_string(),
        );

        Py::new(py, viz_accessor)
    }

    /// Python len() support - returns number of nodes
    fn __len__(&self) -> usize {
        self.inner.node_count() // SubgraphOperations::node_count()
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
        let node_ids: Vec<usize> = self.inner.node_set().iter().copied().collect();
        let py_int_array = crate::ffi::storage::num_array::PyIntArray::from_node_ids(node_ids);
        Py::new(py, py_int_array)
    }

    /// Get edge IDs as PyIntArray
    #[getter]
    fn edge_ids(&self, py: Python) -> PyResult<Py<crate::ffi::storage::num_array::PyIntArray>> {
        let edge_ids: Vec<usize> = self.inner.edge_set().iter().copied().collect();
        let py_int_array = crate::ffi::storage::num_array::PyIntArray::from_node_ids(edge_ids);
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
    #[pyo3(signature = (algorithm_or_pipeline, persist = true, return_profile = false))]
    pub fn apply(
        &self,
        py: Python,
        algorithm_or_pipeline: &PyAny,
        persist: bool,
        return_profile: bool,
    ) -> PyResult<PyObject> {
        let apply_total_start = Instant::now();
        
        if let Some((result, profile)) =
            self.try_apply_pipeline_object(py, algorithm_or_pipeline, persist)?
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
        let run_result = py_run_pipeline(py, &handle, self, persist);
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
            python_apply.set_item("create_subgraph_obj_py", create_subgraph_obj_elapsed.as_secs_f64())?;

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
            let remaining = apply_total_start.elapsed().checked_sub(accounted).map(|d| d.as_secs_f64()).unwrap_or(0.0);
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

    /// Calculate density of this subgraph
    fn density(&self) -> f64 {
        // Use same calculation as original but with trait data
        let num_nodes = self.inner.node_count();
        let num_edges = self.inner.edge_count();

        if num_nodes <= 1 {
            return 0.0;
        }

        // For undirected graph: max edges = n(n-1)/2
        let max_possible_edges = (num_nodes * (num_nodes - 1)) / 2;

        if max_possible_edges > 0 {
            num_edges as f64 / max_possible_edges as f64
        } else {
            0.0
        }
    }

    /// Get connected components within this subgraph (lazy array)
    fn connected_components(&self) -> PyResult<PyComponentsArray> {
        let components = self
            .inner
            .connected_components()
            .map_err(|e| PyRuntimeError::new_err(format!("Connected components error: {}", e)))?;

        // Create lazy ComponentsArray - no immediate PySubgraph materialization!
        let components_array =
            PyComponentsArray::from_components(components, self.inner.graph().clone());

        Ok(components_array)
    }

    /// Check if this subgraph is connected
    fn is_connected(&self) -> PyResult<bool> {
        // Use connected_components to check - if only 1 component, it's connected
        let components = self.connected_components()?;
        Ok(components.len() <= 1)
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
        let core_table = self
            .inner
            .table()
            .map_err(|e| PyRuntimeError::new_err(format!("Table creation error: {}", e)))?;

        // Return as PyGraphTable
        let py_table = crate::ffi::storage::table::PyGraphTable::from_table(core_table);
        Ok(py_table.into_py(py))
    }

    /// Convert subgraph edges to a table - pure delegation to core GraphTable
    fn edges_table(&self, py: Python) -> PyResult<PyObject> {
        let core_table = self
            .inner
            .edges_table()
            .map_err(|e| PyRuntimeError::new_err(format!("Edges table creation error: {}", e)))?;

        // Return as PyEdgesTable
        let py_table = crate::ffi::storage::table::PyEdgesTable { table: core_table };
        Ok(Py::new(py, py_table)?.into_py(py))
    }

    // === Filtering Methods - delegate to SubgraphOperations ===

    /// Filter nodes and return new subgraph  
    fn filter_nodes(&self, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
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

        // Delegate to core Graph.find_nodes method
        let graph_ref = self.inner.graph();
        let filtered_nodes = graph_ref
            .borrow_mut()
            .find_nodes(node_filter)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        // Create induced subgraph using core Subgraph
        let filtered_node_set: HashSet<NodeId> = filtered_nodes.iter().copied().collect();
        let induced_edges = Subgraph::calculate_induced_edges(&graph_ref, &filtered_node_set)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        let new_subgraph = Subgraph::new(
            graph_ref.clone(),
            filtered_node_set,
            induced_edges,
            format!("{}_filtered_nodes", self.inner.subgraph_type()),
        );

        PySubgraph::from_core_subgraph(new_subgraph)
    }

    /// Filter edges and return new subgraph
    fn filter_edges(&self, _py: Python, filter: &PyAny) -> PyResult<PySubgraph> {
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

        // Delegate to core Graph.find_edges method
        let graph_ref = self.inner.graph();
        let filtered_edges = graph_ref
            .borrow_mut()
            .find_edges(edge_filter)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("{:?}", e)))?;

        // Create subgraph with filtered edges and their incident nodes
        let filtered_edge_set: HashSet<EdgeId> = filtered_edges.iter().copied().collect();
        let mut incident_nodes = HashSet::new();

        // Collect all nodes incident to the filtered edges
        for &edge_id in &filtered_edge_set {
            if let Ok((source, target)) = graph_ref.borrow().edge_endpoints(edge_id) {
                incident_nodes.insert(source);
                incident_nodes.insert(target);
            }
        }

        let new_subgraph = Subgraph::new(
            graph_ref.clone(),
            incident_nodes,
            filtered_edge_set,
            format!("{}_filtered_edges", self.inner.subgraph_type()),
        );

        PySubgraph::from_core_subgraph(new_subgraph)
    }

    // === Graph Conversion Methods ===

    /// Convert to a new independent graph
    fn to_graph(&self, py: Python) -> PyResult<PyObject> {
        // Create new PyGraph with only this subgraph's nodes and edges
        let graph_type = py.get_type::<PyGraph>();
        let new_graph = graph_type.call0()?;

        // Add nodes and edges from this subgraph to the new graph
        // This would require copying data from inner subgraph

        Ok(new_graph.to_object(py))
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

    /// Get degree of nodes in subgraph as GraphArray
    ///
    /// Usage:
    /// - degree(node_id, full_graph=False) -> int: degree of single node (local or full graph)
    /// - degree(node_ids, full_graph=False) -> GraphArray: degrees for list of nodes
    /// - degree(full_graph=False) -> GraphArray: degrees for all nodes in subgraph
    ///
    /// Parameters:
    /// - nodes: Optional node ID, list of node IDs, or None for all nodes
    /// - full_graph: If False (default), compute degrees within subgraph only.
    ///               If True, compute degrees from the original full graph.
    #[pyo3(signature = (nodes = None, *, full_graph = false))]
    fn degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        let graph_ref = self.inner.graph();

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.inner.node_set().contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let deg = if full_graph {
                    // Get degree from full graph
                    let graph = graph_ref.borrow();
                    graph.degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local degree within subgraph
                    self.inner
                        .edge_set()
                        .iter()
                        .filter(|&&edge_id| {
                            let graph = graph_ref.borrow();
                            if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                src == node_id || tgt == node_id
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(deg.to_object(py))
            }

            // List of nodes case
            Some(node_arg) if node_arg.extract::<Vec<NodeId>>().is_ok() => {
                let node_ids = node_arg.extract::<Vec<NodeId>>()?;
                let mut degrees = Vec::new();

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.inner.node_set().contains(&node_id) {
                        continue; // Skip nodes not in subgraph
                    }

                    let deg = if full_graph {
                        // Get degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                    src == node_id || tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                // Convert to NumArray for numerical operations and rich display
                let py_num_array = PyNumArray::from_attr_values(degrees)?;
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut degrees = Vec::new();

                for &node_id in self.inner.node_set() {
                    let deg = if full_graph {
                        // Get degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                    src == node_id || tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    degrees.push(groggy::AttrValue::Int(deg as i64));
                }

                // Convert to NumArray for numerical operations and rich display
                let py_num_array = PyNumArray::from_attr_values(degrees)?;
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }

            // Invalid argument type
            Some(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "degree() nodes argument must be a NodeId, list of NodeIds, or None",
            )),
        }
    }

    /// Get in-degree of nodes within subgraph
    #[pyo3(signature = (nodes = None, full_graph = false))]
    fn in_degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        let graph_ref = self.inner.graph();

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.inner.node_set().contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let in_deg = if full_graph {
                    // Get in-degree from full graph
                    let graph = graph_ref.borrow();
                    graph.in_degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local in-degree within subgraph
                    self.inner
                        .edge_set()
                        .iter()
                        .filter(|&&edge_id| {
                            let graph = graph_ref.borrow();
                            if let Ok((_, tgt)) = graph.edge_endpoints(edge_id) {
                                tgt == node_id
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(in_deg.to_object(py))
            }

            // Multiple nodes case
            Some(nodes_arg) => {
                let node_ids: Vec<NodeId> = nodes_arg.extract()?;
                let mut in_degrees = Vec::new();

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.inner.node_set().contains(&node_id) {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Node {} is not in this subgraph",
                            node_id
                        )));
                    }

                    let in_deg = if full_graph {
                        // Get in-degree from full graph
                        let graph = graph_ref.borrow();
                        match graph.in_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local in-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((_, tgt)) = graph.edge_endpoints(edge_id) {
                                    tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    in_degrees.push(groggy::AttrValue::Int(in_deg as i64));
                }

                // Convert to NumArray for rich display and comparison operations
                let py_num_array = PyNumArray::from_attr_values(in_degrees)?;
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut in_degrees = Vec::new();

                for &node_id in self.inner.node_set() {
                    let in_deg = if full_graph {
                        // Get in-degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.in_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local in-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((_, tgt)) = graph.edge_endpoints(edge_id) {
                                    tgt == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    in_degrees.push(groggy::AttrValue::Int(in_deg as i64));
                }

                // Convert to NumArray for rich display and comparison operations
                let py_num_array = PyNumArray::from_attr_values(in_degrees)?;
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
        }
    }

    /// Get out-degree of nodes within subgraph
    #[pyo3(signature = (nodes = None, full_graph = false))]
    fn out_degree(
        &self,
        py: Python,
        nodes: Option<&PyAny>,
        full_graph: bool,
    ) -> PyResult<PyObject> {
        let graph_ref = self.inner.graph();

        match nodes {
            // Single node case
            Some(node_arg) if node_arg.extract::<NodeId>().is_ok() => {
                let node_id = node_arg.extract::<NodeId>()?;

                // Verify node is in subgraph
                if !self.inner.node_set().contains(&node_id) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Node {} is not in this subgraph",
                        node_id
                    )));
                }

                let out_deg = if full_graph {
                    // Get out-degree from full graph
                    let graph = graph_ref.borrow();
                    graph.out_degree(node_id).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
                    })?
                } else {
                    // Calculate local out-degree within subgraph
                    self.inner
                        .edge_set()
                        .iter()
                        .filter(|&&edge_id| {
                            let graph = graph_ref.borrow();
                            if let Ok((src, _)) = graph.edge_endpoints(edge_id) {
                                src == node_id
                            } else {
                                false
                            }
                        })
                        .count()
                };

                Ok(out_deg.to_object(py))
            }

            // Multiple nodes case
            Some(nodes_arg) => {
                let node_ids: Vec<NodeId> = nodes_arg.extract()?;
                let mut out_degrees = Vec::new();

                for node_id in node_ids {
                    // Verify node is in subgraph
                    if !self.inner.node_set().contains(&node_id) {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Node {} is not in this subgraph",
                            node_id
                        )));
                    }

                    let out_deg = if full_graph {
                        // Get out-degree from full graph
                        let graph = graph_ref.borrow();
                        match graph.out_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local out-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, _)) = graph.edge_endpoints(edge_id) {
                                    src == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    out_degrees.push(groggy::AttrValue::Int(out_deg as i64));
                }

                // Convert to NumArray for rich display and comparison operations
                let py_num_array = PyNumArray::from_attr_values(out_degrees)?;
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }

            // All nodes case (or None)
            None => {
                let mut out_degrees = Vec::new();

                for &node_id in self.inner.node_set() {
                    let out_deg = if full_graph {
                        // Get out-degree from main graph
                        let graph = graph_ref.borrow();
                        match graph.out_degree(node_id) {
                            Ok(d) => d,
                            Err(_) => continue, // Skip invalid nodes
                        }
                    } else {
                        // Calculate local out-degree within subgraph
                        self.inner
                            .edge_set()
                            .iter()
                            .filter(|&&edge_id| {
                                let graph = graph_ref.borrow();
                                if let Ok((src, _)) = graph.edge_endpoints(edge_id) {
                                    src == node_id
                                } else {
                                    false
                                }
                            })
                            .count()
                    };

                    out_degrees.push(groggy::AttrValue::Int(out_deg as i64));
                }

                // Convert to NumArray for rich display and comparison operations
                let py_num_array = PyNumArray::from_attr_values(out_degrees)?;
                Ok(Py::new(py, py_num_array)?.to_object(py))
            }
        }
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

    /// Support attribute access via indexing: subgraph['attr_name'] -> BaseArray
    fn __getitem__(&self, key: &PyAny, py: Python) -> PyResult<PyObject> {
        // Only support string keys (attribute names) for now
        if let Ok(attr_name) = key.extract::<String>() {
            // Return BaseArray of attribute values for all nodes in the subgraph
            let graph_ref = self.inner.graph();
            let mut attr_values = Vec::new();

            for &node_id in self.inner.node_set() {
                let graph = graph_ref.borrow();
                match graph.get_node_attr(node_id, &attr_name) {
                    Ok(Some(attr_value)) => {
                        attr_values.push(attr_value);
                    }
                    Ok(None) | Err(_) => {
                        // Use null for missing attributes
                        attr_values.push(AttrValue::Null);
                    }
                }
            }

            let py_base_array = PyBaseArray {
                inner: BaseArray::new(attr_values),
            };
            return Ok(Py::new(py, py_base_array)?.to_object(py));
        }

        // For now, only support string attribute access
        Err(PyTypeError::new_err(
            "Subgraph indexing only supports string attribute names. \
             Example: subgraph['community']",
        ))
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

        // Create a temporary PyGraph wrapper
        let py_graph = PyGraph {
            inner: self.inner.graph(),
            cached_view: std::cell::RefCell::new(None),
        };

        // Create PyGraphAnalysis and delegate to it
        let mut analysis_handler = PyGraphAnalysis::new(Py::new(py, py_graph)?)?;
        let result = analysis_handler.neighborhood(py, central_nodes_vec, Some(hops), None)?;
        
        // Convert the NeighborhoodArray into a SubgraphArray for Python consumption
        Ok(result.inner().clone())
    }

    /// Sample k nodes from this subgraph randomly
    pub fn sample(&self, k: usize) -> PyResult<PySubgraph> {
        let node_ids: Vec<NodeId> = self.inner.node_set().iter().copied().collect();

        if k >= node_ids.len() {
            // Return the same subgraph if k is larger than available nodes
            return Ok(self.clone());
        }

        // Simple sampling: take first k nodes (for now - would use proper random sampling in production)
        let _sampled_nodes: Vec<NodeId> = node_ids.into_iter().take(k).collect();

        // For now, just return a clone of the original subgraph as a placeholder
        // In a full implementation, we would properly create an induced subgraph
        // with the sampled nodes using the core graph algorithms
        Ok(self.clone())
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
        match self.inner.neighbors(node_id) {
            Ok(neighbor_ids) => {
                let values: Vec<f64> = neighbor_ids.into_iter().map(|id| id as f64).collect();
                let py_array = PyNumArray::new(values);
                Py::new(py, py_array)
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Neighbors error: {}", e))),
        }
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
        // Get the nodes accessor and delegate to its internal method
        let nodes_accessor = self.nodes(py)?;
        let nodes_accessor_ref: PyRef<PyNodesAccessor> = nodes_accessor.extract(py)?;
        nodes_accessor_ref.set_attrs_internal(py, attrs_dict)
    }

    /// Set multiple edge attributes (bulk operation) - delegates to accessor  
    fn set_edge_attrs(&self, py: Python, attrs_dict: &PyDict) -> PyResult<()> {
        // Get the edges accessor and delegate to its internal method
        let edges_accessor = self.edges(py)?;
        let edges_accessor_ref: PyRef<PyEdgesAccessor> = edges_accessor.extract(py)?;
        edges_accessor_ref.set_attrs_internal(py, attrs_dict)
    }

    // === MISSING GRAPH METRICS ===

    /// Calculate clustering coefficient for a node or entire subgraph
    fn clustering_coefficient(&self, _py: Python, _node_id: Option<NodeId>) -> PyResult<f64> {
        // Note: Clustering coefficient not yet implemented in core
        // This is a placeholder for future implementation
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Clustering coefficient not yet implemented in core - coming in future version",
        ))
    }

    /// Calculate transitivity of the subgraph
    fn transitivity(&self, _py: Python) -> PyResult<f64> {
        // Note: Transitivity not yet implemented in core
        // This is a placeholder for future implementation
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Transitivity not yet implemented in core - coming in future version",
        ))
    }

    // === ENTITY TYPE METHOD ===

    /// Return the entity type string
    fn entity_type(&self) -> PyResult<String> {
        Ok("Subgraph".to_string())
    }

    // === MISSING SUBGRAPH OPERATIONS ===

    /// BFS traversal - returns subgraph result
    fn bfs(&self, _py: Python, start: NodeId, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        let result = self.inner.bfs(start, max_depth);

        match result {
            Ok(boxed_subgraph) => {
                // Create concrete Subgraph from the trait object data
                use groggy::subgraphs::Subgraph;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    format!("bfs_from_{}", start),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("BFS error: {}", e))),
        }
    }

    /// DFS traversal - returns subgraph result
    fn dfs(&self, _py: Python, start: NodeId, max_depth: Option<usize>) -> PyResult<PySubgraph> {
        let result = self.inner.dfs(start, max_depth);

        match result {
            Ok(boxed_subgraph) => {
                // Create concrete Subgraph from the trait object data
                use groggy::subgraphs::Subgraph;
                let concrete_subgraph = Subgraph::new(
                    self.inner.graph(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    format!("dfs_from_{}", start),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("DFS error: {}", e))),
        }
    }

    /// Create subgraph representing shortest path between two nodes
    fn shortest_path_subgraph(
        &self,
        _py: Python,
        source: NodeId,
        target: NodeId,
    ) -> PyResult<Option<PySubgraph>> {
        match self.inner.shortest_path_subgraph(source, target) {
            Ok(Some(boxed_subgraph)) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::subgraphs::Subgraph;

                let concrete_subgraph = Subgraph::new(
                    self.inner.graph(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    format!("shortest_path_{}_{}", source, target),
                );
                Ok(Some(PySubgraph {
                    inner: concrete_subgraph,
                }))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Shortest path error: {}",
                e
            ))),
        }
    }

    /// Create induced subgraph from list of nodes
    fn induced_subgraph(&self, _py: Python, nodes: Vec<NodeId>) -> PyResult<PySubgraph> {
        match self.inner.induced_subgraph(&nodes) {
            Ok(boxed_subgraph) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::subgraphs::Subgraph;

                let concrete_subgraph = Subgraph::new(
                    self.inner.graph(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    "induced_subgraph".to_string(),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Induced subgraph error: {}",
                e
            ))),
        }
    }

    /// Create subgraph from list of edges
    fn subgraph_from_edges(&self, _py: Python, edges: Vec<EdgeId>) -> PyResult<PySubgraph> {
        match self.inner.subgraph_from_edges(&edges) {
            Ok(boxed_subgraph) => {
                // Create a concrete Subgraph from the trait object data
                use groggy::subgraphs::Subgraph;

                let concrete_subgraph = Subgraph::new(
                    self.inner.graph(),
                    boxed_subgraph.node_set().clone(),
                    boxed_subgraph.edge_set().clone(),
                    "subgraph_from_edges".to_string(),
                );
                Ok(PySubgraph {
                    inner: concrete_subgraph,
                })
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Subgraph from edges error: {}",
                e
            ))),
        }
    }

    /// Set operations - merge, intersect, subtract (placeholders)
    fn merge_with(&self, _py: Python, _other: &PySubgraph) -> PyResult<PySubgraph> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph set operations not yet implemented - requires subgraph algebra in core",
        ))
    }

    fn intersect_with(&self, _py: Python, _other: &PySubgraph) -> PyResult<PySubgraph> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph set operations not yet implemented - requires subgraph algebra in core",
        ))
    }

    fn subtract_from(&self, _py: Python, _other: &PySubgraph) -> PyResult<PySubgraph> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Subgraph set operations not yet implemented - requires subgraph algebra in core",
        ))
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
    pub fn collapse(
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
    ) -> PyResult<PyObject> {
        use crate::ffi::subgraphs::composer::{
            parse_edge_aggs_from_python, parse_node_aggs_from_python,
        };
        use groggy::subgraphs::composer::EdgeStrategy;
        use groggy::traits::subgraph_operations::NodeStrategy;

        // Parse input parameters
        let parsed_node_aggs = if let Some(aggs) = node_aggs {
            parse_node_aggs_from_python(aggs)?
        } else {
            Vec::new()
        };

        let parsed_edge_aggs = if let Some(aggs) = edge_aggs {
            parse_edge_aggs_from_python(aggs)?
        } else {
            Vec::new()
        };

        let strategy = EdgeStrategy::from_str(edge_strategy).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid edge strategy: {}", e))
        })?;

        let node_strategy = match node_strategy {
            "extract" => NodeStrategy::Extract,
            "collapse" => NodeStrategy::Collapse,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid node strategy '{}'. Must be 'extract' or 'collapse'",
                    node_strategy
                )))
            }
        };

        // Call the trait method
        let plan = self
            .inner
            .collapse(
                parsed_node_aggs,
                parsed_edge_aggs,
                strategy,
                node_strategy,
                preset,
                include_edge_count,
                mark_entity_type,
                entity_type.to_string(),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create plan: {}", e))
            })?;

        // Execute using appropriate method based on allow_missing_attributes
        let meta_node = if allow_missing_attributes {
            plan.add_to_graph_with_defaults(&self.inner).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create meta-node: {}",
                    e
                ))
            })?
        } else {
            plan.add_to_graph(&self.inner).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create meta-node: {}",
                    e
                ))
            })?
        };

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

            // Get graph reference to access edge endpoints
            let graph = self.inner.graph_ref();
            let graph_borrowed = graph.borrow();

            // Iterate through edges in the subgraph
            for &edge_id in self.inner.edge_set() {
                if let Ok((source, target)) = graph_borrowed.edge_endpoints(edge_id) {
                    result.push_str(&format!("\n  {:4}  {:6} → {}", edge_id, source, target));
                }
            }
        }

        result
    }

    // ========================================================================
    // PHASE 3: Cross-Type Conversions - Enable unified delegation architecture
    // ========================================================================

    /// Get nodes from this subgraph as a NodesAccessor
    /// Enables chaining like: subgraph.to_nodes().table().stats()
    pub fn to_nodes(&self) -> PyResult<crate::ffi::storage::accessors::PyNodesAccessor> {
        let node_ids: Vec<groggy::types::NodeId> = self.inner.node_set().iter().copied().collect();

        // Create a NodesAccessor using the same pattern as the getter method
        Ok(crate::ffi::storage::accessors::PyNodesAccessor {
            graph: self.inner.graph(),
            constrained_nodes: Some(node_ids),
        })
    }

    /// Get edges from this subgraph as an EdgesAccessor  
    /// Enables chaining like: subgraph.to_edges().to_nodes().connected_components()
    pub fn to_edges(&self) -> PyResult<crate::ffi::storage::accessors::PyEdgesAccessor> {
        let edge_ids: Vec<groggy::types::EdgeId> = self.inner.edge_set().iter().copied().collect();

        // Create an EdgesAccessor using the struct syntax
        Ok(crate::ffi::storage::accessors::PyEdgesAccessor {
            graph: self.inner.graph(),
            constrained_edges: Some(edge_ids),
        })
    }

    /// Convert this subgraph to its adjacency matrix representation
    /// Enables chaining like: subgraph.to_matrix().eigen().stats()
    pub fn to_matrix(&self) -> PyResult<crate::ffi::storage::matrix::PyGraphMatrix> {
        // For now, create a placeholder matrix
        // In full implementation, would convert subgraph to adjacency matrix
        let graph_ref = self.inner.graph();
        let graph_borrowed = graph_ref.borrow();

        // Get node IDs and create a mapping
        let node_ids: Vec<groggy::types::NodeId> = self.inner.node_set().iter().copied().collect();
        let n = node_ids.len();

        // Create adjacency matrix data (simplified - would be optimized in real implementation)
        let mut matrix_data = vec![vec![0.0f32; n]; n];

        // Fill adjacency matrix
        for (i, &node_i) in node_ids.iter().enumerate() {
            for (j, &node_j) in node_ids.iter().enumerate() {
                if i != j {
                    // Check if there's an edge between these nodes
                    if let Ok(has_edge) = graph_borrowed.has_edge_between(node_i, node_j) {
                        if has_edge {
                            matrix_data[i][j] = 1.0; // Unweighted for now
                        }
                    }
                }
            }
        }

        // Convert matrix data to NumArrays for each column
        Python::with_gil(|py| {
            let mut py_arrays: Vec<PyObject> = Vec::with_capacity(n);

            // Create a column for each node (column-major format)
            for col_idx in 0..n {
                let column_values: Vec<f64> = (0..n)
                    .map(|row_idx| matrix_data[row_idx][col_idx] as f64)
                    .collect();

                // Create NumArray since adjacency matrices are always numerical
                let num_array = PyNumArray::new(column_values);
                py_arrays.push(Py::new(py, num_array)?.to_object(py));
            }

            // Create GraphMatrix using the new constructor that accepts PyObject arrays
            let matrix = crate::ffi::storage::matrix::PyGraphMatrix::new(py, py_arrays)?;

            // Set column names based on node IDs
            let column_names: Vec<String> = node_ids
                .iter()
                .map(|&node_id| format!("node_{}", node_id))
                .collect();

            let mut inner_matrix = matrix.inner;
            inner_matrix.set_column_names(column_names);

            Ok(crate::ffi::storage::matrix::PyGraphMatrix::from_graph_matrix(inner_matrix))
        })
    }

    // === ADJACENCY METHODS (moved from Graph to Subgraph) ===

    /// Get adjacency list representation
    /// Returns: Dict mapping node_id -> list of connected node_ids
    fn adjacency_list(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        let result_dict = PyDict::new(py);
        let graph_ref = self.inner.graph();
        let graph_borrow = graph_ref.borrow();

        // For each node in subgraph, find connected nodes
        for &node_id in self.inner.node_set() {
            let mut neighbors = Vec::new();

            // Check all edges in subgraph to find neighbors
            for &edge_id in self.inner.edge_set() {
                if let Ok((source, target)) = graph_borrow.edge_endpoints(edge_id) {
                    if source == node_id && self.inner.contains_node(target) {
                        neighbors.push(target);
                    } else if target == node_id && self.inner.contains_node(source) {
                        neighbors.push(source);
                    }
                }
            }

            result_dict.set_item(node_id, neighbors)?;
        }

        Ok(result_dict.into())
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
                // Convert Python value to AttrValue using the same logic as before
                let attr_value = if let Ok(b) = default_item.extract::<bool>() {
                    AttrValue::Bool(b)
                } else if let Ok(i) = default_item.extract::<i64>() {
                    AttrValue::Int(i)
                } else if let Ok(f) = default_item.extract::<f64>() {
                    AttrValue::Float(f as f32)
                } else if let Ok(f) = default_item.extract::<f32>() {
                    AttrValue::Float(f)
                } else if let Ok(s) = default_item.extract::<String>() {
                    AttrValue::Text(s)
                } else if let Ok(vec) = default_item.extract::<Vec<f32>>() {
                    AttrValue::FloatVec(vec)
                } else if let Ok(vec) = default_item.extract::<Vec<f64>>() {
                    let f32_vec: Vec<f32> = vec.into_iter().map(|f| f as f32).collect();
                    AttrValue::FloatVec(f32_vec)
                } else if let Ok(bytes) = default_item.extract::<Vec<u8>>() {
                    AttrValue::Bytes(bytes)
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Unsupported default value type. Supported: int, float, str, bool, List[float], bytes"
                    ));
                };
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
