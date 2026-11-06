//! Components Array FFI Bindings
//!
//! Lazy array of connected components - only materializes PySubgraphs on access

use groggy::api::graph::Graph;
use groggy::storage::array::{ArrayIterator, ArrayOps, SubgraphLike};
use groggy::subgraphs::Subgraph;
use groggy::traits::SubgraphOperations;
use groggy::{EdgeId, NodeId};
use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::ffi::entities::meta_node::PyMetaNode;
use crate::ffi::subgraphs::subgraph::PySubgraph;

/// Lazy array of connected components - avoids creating hundreds of PySubgraphs immediately
#[pyclass(name = "ComponentsArray", unsendable)]
pub struct PyComponentsArray {
    // Store only the lightweight metadata, not materialized subgraphs!
    components_data: Vec<(HashSet<NodeId>, HashSet<EdgeId>)>,
    graph_ref: Rc<RefCell<Graph>>,
    source: groggy::GraphView,

    // Lazy cache for materialized components (only created when accessed)
    materialized_cache: RefCell<HashMap<usize, PySubgraph>>,
}

impl PyComponentsArray {
    /// Get the number of components (public accessor for internal use)
    pub fn len(&self) -> usize {
        self.components_data.len()
    }

    /// Check if the components array is empty
    pub fn is_empty(&self) -> bool {
        self.components_data.is_empty()
    }

    /// Create new ComponentsArray from trait objects (internal use)
    pub fn from_components(
        components: Vec<Box<dyn groggy::traits::SubgraphOperations>>,
        graph_ref: Rc<RefCell<Graph>>,
        source: groggy::GraphView,
    ) -> PyResult<Self> {
        // Extract lightweight data from trait objects
        let components_data: Vec<(HashSet<NodeId>, HashSet<EdgeId>)> = components
            .into_iter()
            .map(|comp| (comp.node_set().clone(), comp.edge_set().clone()))
            .collect();

        Ok(Self {
            source,
            components_data,
            graph_ref,
            materialized_cache: RefCell::new(HashMap::new()),
        })
    }
}

#[pymethods]
impl PyComponentsArray {
    /// Get the number of components
    fn __len__(&self) -> usize {
        self.components_data.len()
    }

    /// Lazy access to individual components - only materializes on demand
    fn __getitem__(&self, index: isize) -> PyResult<PySubgraph> {
        let len = self.components_data.len() as isize;

        // Handle negative indexing
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };

        if actual_index >= self.components_data.len() {
            return Err(PyIndexError::new_err(format!(
                "Component index {} out of range (0-{})",
                index,
                self.components_data.len() - 1
            )));
        }

        // Check cache first
        if let Some(cached) = self.materialized_cache.borrow().get(&actual_index) {
            return Ok(cached.clone());
        }

        // Lazy materialization - create PySubgraph on-demand
        let (nodes, edges) = &self.components_data[actual_index];
        let view = groggy::GraphView::from_membership(
            self.source.read_state().clone(),
            nodes.clone(),
            edges.clone(),
        )
        .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let subgraph = Subgraph::from_frozen_view(
            self.graph_ref.clone(),
            view,
            format!("component_{}", actual_index),
        );
        let py_subgraph = PySubgraph::from_core_subgraph(subgraph).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create component subgraph: {}", e))
        })?;

        // Cache for next time
        self.materialized_cache
            .borrow_mut()
            .insert(actual_index, py_subgraph.clone());

        Ok(py_subgraph)
    }

    /// Iteration support
    fn __iter__(slf: PyRef<Self>) -> PyResult<PyComponentsArrayIterator> {
        Ok(PyComponentsArrayIterator {
            components: slf.into(),
            index: 0,
        })
    }

    /// String representation showing count
    fn __repr__(&self) -> String {
        format!("ComponentsArray({} components)", self.components_data.len())
    }

    /// Collapse each component into a meta-node using the Subgraph collapse API.
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
        let selections = (0..self.components_data.len())
            .map(|index| {
                self.__getitem__(index as isize)
                    .map(|component| component.inner)
            })
            .collect::<PyResult<Vec<_>>>()?;
        collapse_many(
            py,
            selections,
            options,
            allow_missing_attributes,
            meta_targets.as_deref(),
        )
    }

    /// Convert to regular Python list (materializes all components)
    fn to_list(&self) -> PyResult<Vec<PySubgraph>> {
        let mut result = Vec::new();
        for i in 0..self.components_data.len() {
            result.push(self.__getitem__(i as isize)?);
        }
        Ok(result)
    }

    /// Get component sizes without materializing subgraphs
    fn sizes(&self) -> Vec<(usize, usize)> {
        self.components_data
            .iter()
            .map(|(nodes, edges)| (nodes.len(), edges.len()))
            .collect()
    }

    /// Get largest component without materializing others
    fn largest_component(&self) -> PyResult<PySubgraph> {
        if self.components_data.is_empty() {
            return Err(PyRuntimeError::new_err("No components available"));
        }

        // Find largest by node count
        let largest_index = self
            .components_data
            .iter()
            .enumerate()
            .max_by_key(|(_, (nodes, _))| nodes.len())
            .map(|(i, _)| i)
            .unwrap();

        self.__getitem__(largest_index as isize)
    }

    /// Direct delegation: Apply table() to each component and return PyTableArray
    fn table(&self) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        Python::with_gil(|py| {
            let tables = (0..self.components_data.len())
                .map(|index| self.__getitem__(index as isize)?.table(py))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(crate::ffi::storage::table_array::PyTableArray::new(tables))
        })
    }

    /// Direct delegation: Apply sample(k) to each component and return PySubgraphArray
    fn sample(&self, k: usize) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let mut sampled = Vec::new();

        for i in 0..self.components_data.len() {
            let component = self.__getitem__(i as isize)?;
            sampled.push(component.sample_k(k)?);
        }

        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            sampled,
        ))
    }

    /// Direct delegation: Apply neighborhood() to each component and return PySubgraphArray
    fn neighborhood(&self) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let mut neighborhoods = Vec::new();

        Python::with_gil(|py| {
            for i in 0..self.components_data.len() {
                if let Ok(component) = self.__getitem__(i as isize) {
                    // For simplicity, use first node as central and 1 hop
                    let node_ids: Vec<groggy::NodeId> =
                        component.inner.node_set().iter().take(1).copied().collect();
                    if !node_ids.is_empty() {
                        let node_list = PyList::new(py, &node_ids);
                        let node_any: &PyAny = node_list.as_ref();
                        match component.neighborhood(py, Some(node_any), 1) {
                            Ok(_neighborhood_result) => {
                                // For now, return the original component as placeholder
                                neighborhoods.push(component);
                            }
                            Err(_) => continue, // Skip failed components
                        }
                    } else {
                        neighborhoods.push(component); // No nodes, just return original
                    }
                }
            }
        });

        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            neighborhoods,
        ))
    }

    /// Direct delegation: Apply filter to components
    fn filter(&self, predicate: PyObject) -> PyResult<Self> {
        let filtered_data = Python::with_gil(|py| {
            if !predicate.as_ref(py).is_callable() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "component filter requires a callable",
                ));
            }
            let mut selected = Vec::new();
            for index in 0..self.components_data.len() {
                let component = self.__getitem__(index as isize)?;
                if predicate.call1(py, (component,))?.is_true(py)? {
                    selected.push(self.components_data[index].clone());
                }
            }
            Ok(selected)
        })?;
        Ok(Self {
            source: self.source.clone(),
            components_data: filtered_data,
            graph_ref: self.graph_ref.clone(),
            materialized_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Get viz accessor for visualization
    #[getter]
    fn viz(&self, py: Python) -> PyResult<Py<crate::ffi::viz_accessor::VizAccessor>> {
        use groggy::viz::streaming::GraphDataSource;

        let source = &self.source;
        let nodes = self
            .components_data
            .iter()
            .flat_map(|(nodes, _)| nodes.iter().copied())
            .collect();
        let edges = self
            .components_data
            .iter()
            .flat_map(|(_, edges)| edges.iter().copied())
            .collect();
        let view = groggy::GraphView::from_membership(source.read_state().clone(), nodes, edges)
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let graph_data_source = py
            .allow_threads(|| GraphDataSource::from_view(&view))
            .map_err(crate::ffi::utils::graph_error_to_py_err)?;
        let viz_accessor = crate::ffi::viz_accessor::VizAccessor::with_data_source(
            graph_data_source,
            "ComponentsArray".to_string(),
        );
        Py::new(py, viz_accessor)
    }
}

/// Iterator for ComponentsArray
#[pyclass]
pub struct PyComponentsArrayIterator {
    components: Py<PyComponentsArray>,
    index: usize,
}

#[pymethods]
impl PyComponentsArrayIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PySubgraph>> {
        let components = self.components.borrow(py);
        if self.index < components.components_data.len() {
            let result = components.__getitem__(self.index as isize)?;
            self.index += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
}

// =============================================================================
// BaseArray integration - implement ArrayOps for chaining support
// =============================================================================

impl ArrayOps<PySubgraph> for PyComponentsArray {
    fn len(&self) -> usize {
        self.components_data.len()
    }

    fn get(&self, _index: usize) -> Option<&PySubgraph> {
        // For ArrayOps, we can't return a reference because we create on demand
        // This is a limitation of our lazy materialization approach
        // We'll need to materialize and cache to support this properly
        None // TODO: Consider changing ArrayOps to return owned values for some types
    }

    fn iter(&self) -> ArrayIterator<PySubgraph>
    where
        PySubgraph: Clone + 'static,
    {
        // Materialize all components for the iterator
        let mut materialized: Vec<PySubgraph> = Vec::new();

        for i in 0..self.components_data.len() {
            // Use our existing lazy materialization logic
            if let Ok(py_subgraph) = self.__getitem__(i as isize) {
                materialized.push(py_subgraph);
            }
        }

        // Create iterator with graph reference for graph-aware operations
        ArrayIterator::with_graph(materialized, self.graph_ref.clone())
    }
}

// Implement SubgraphLike for PySubgraph to enable subgraph-specific operations
impl SubgraphLike for PySubgraph {
    // Marker trait - no methods required
    // This enables .filter_nodes(), .filter_edges(), .collapse() operations
}

// =============================================================================
// Python chaining iterator - wraps ArrayIterator for Python FFI
// =============================================================================

/// Python wrapper for ArrayIterator<PySubgraph> that supports method chaining
#[pyclass(name = "ComponentsIterator", unsendable)]
pub struct PyComponentsIterator {
    inner: ArrayIterator<PySubgraph>,
}

#[pymethods]
impl PyComponentsIterator {
    /// Filter nodes within subgraphs using a query string
    /// Enables: g.connected_components().iter().filter_nodes('age > 25')
    fn filter_nodes(slf: PyRefMut<Self>, query: &str) -> PyResult<Self> {
        let inner = slf.inner.clone(); // Clone the inner ArrayIterator
        Ok(Self {
            inner: inner.filter_nodes(query),
        })
    }

    /// Filter edges within subgraphs using a query string
    /// Enables: g.connected_components().iter().filter_edges('weight > 0.5')
    fn filter_edges(slf: PyRefMut<Self>, query: &str) -> PyResult<Self> {
        let inner = slf.inner.clone(); // Clone the inner ArrayIterator
        Ok(Self {
            inner: inner.filter_edges(query),
        })
    }

    /// Collapse subgraphs into meta-nodes with aggregations
    /// Enables: g.connected_components().iter().collapse({'avg_age': ('mean', 'age')})
    // TODO: Refactor to use config/builder pattern
    #[allow(clippy::too_many_arguments)]
    fn collapse(slf: PyRefMut<Self>, aggs: &PyDict) -> PyResult<PyMetaNodeIterator> {
        // Convert PyDict to HashMap<String, String> for now
        let mut agg_map = std::collections::HashMap::new();
        for (key, value) in aggs.iter() {
            let key_str = key.str()?.to_str()?;
            let value_str = value.str()?.to_str()?;
            agg_map.insert(key_str.to_string(), value_str.to_string());
        }

        let inner = slf.inner.clone(); // Clone the inner ArrayIterator
        let meta_iterator = inner.collapse(agg_map);

        Ok(PyMetaNodeIterator {
            inner: meta_iterator,
        })
    }

    /// Collect results back into a ComponentsArray-like structure
    /// Enables: g.connected_components().iter().filter_nodes().collect()
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<PySubgraph>> {
        // Extract the elements from our ArrayIterator
        let inner = slf.inner.clone(); // Clone the inner ArrayIterator
        Ok(inner.into_vec())
    }

    /// Apply table() to each component and return PyTableArray
    fn table(&mut self) -> PyResult<crate::ffi::storage::table_array::PyTableArray> {
        let components = self.inner.clone().into_vec();
        let mut tables = Vec::new();

        Python::with_gil(|py| {
            for component in components {
                match component.table(py) {
                    Ok(table) => tables.push(table),
                    Err(_) => continue, // Skip failed components
                }
            }
        });

        Ok(crate::ffi::storage::table_array::PyTableArray::new(tables))
    }

    /// Apply sample(k) to each component and return PySubgraphArray
    fn sample(
        &mut self,
        k: usize,
    ) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let components = self.inner.clone().into_vec();
        let mut sampled = Vec::new();

        for component in components {
            sampled.push(component.sample_k(k)?);
        }

        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            sampled,
        ))
    }

    /// Apply neighborhood() to each component and return PySubgraphArray (simplified version)
    fn neighborhood(&mut self) -> PyResult<crate::ffi::storage::subgraph_array::PySubgraphArray> {
        let components = self.inner.clone().into_vec();
        let mut neighborhoods = Vec::new();

        Python::with_gil(|py| {
            for component in components {
                // For simplicity, use first node as central and 1 hop
                let node_ids: Vec<groggy::NodeId> =
                    component.inner.node_set().iter().take(1).copied().collect();
                if !node_ids.is_empty() {
                    let node_list = PyList::new(py, &node_ids);
                    let node_any: &PyAny = node_list.as_ref();
                    match component.neighborhood(py, Some(node_any), 1) {
                        Ok(_neighborhood_result) => {
                            // For now, return the original component as placeholder
                            // In full implementation, would extract subgraph from neighborhood result
                            neighborhoods.push(component);
                        }
                        Err(_) => continue, // Skip failed components
                    }
                } else {
                    neighborhoods.push(component); // No nodes, just return original
                }
            }
        });

        Ok(crate::ffi::storage::subgraph_array::PySubgraphArray::new(
            neighborhoods,
        ))
    }
}

/// Python wrapper for ArrayIterator<MetaNode>
#[pyclass(name = "MetaNodeIterator", unsendable)]
pub struct PyMetaNodeIterator {
    inner: ArrayIterator<()>, // Placeholder for now
}

#[pymethods]
impl PyMetaNodeIterator {
    /// Collect meta-nodes into a list - placeholder implementation
    fn collect(slf: PyRefMut<Self>) -> PyResult<Vec<String>> {
        let inner = slf.inner.clone(); // Clone the inner ArrayIterator
        let _elements = inner.into_vec(); // Get the placeholder elements
                                          // Return placeholder result for now
        Ok(vec!["placeholder_meta_node".to_string()])
    }
}
