//! Neighborhood sampling FFI bindings
//!
//! Python bindings for neighborhood subgraph generation functionality.

use crate::ffi::core::subgraph::PySubgraph;
use groggy::core::neighborhood::{NeighborhoodResult, NeighborhoodStats, NeighborhoodSubgraph};
use groggy::core::traits::{GraphEntity, NeighborhoodOperations, SubgraphOperations};
use groggy::NodeId;
use pyo3::prelude::*;

/// Python wrapper for NeighborhoodSubgraph
#[pyclass(name = "NeighborhoodSubgraph", unsendable)]
#[derive(Clone)]
pub struct PyNeighborhoodSubgraph {
    pub inner: NeighborhoodSubgraph,
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
    fn subgraph(&self, _py: Python) -> PyResult<PySubgraph> {
        // Create PySubgraph using the same pattern as connected components
        // Create core Subgraph first, then wrap in PySubgraph
        let core_subgraph = groggy::core::subgraph::Subgraph::new(
            self.inner.graph_ref().clone(),
            self.inner.node_set().clone(),
            self.inner.edge_set().clone(),
            format!("neighborhood_hops_{}", self.inner.hops()),
        );
        PySubgraph::from_core_subgraph(core_subgraph)
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

/// Python wrapper for NeighborhoodResult
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
