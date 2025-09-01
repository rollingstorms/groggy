//! Component subgraph FFI bindings
//!
//! Python bindings for ComponentSubgraph - pure delegation to existing trait methods.

use groggy::core::component::ComponentSubgraph;
use groggy::core::traits::{ComponentOperations, SubgraphOperations};
use groggy::{EdgeId, NodeId};
use pyo3::prelude::*;

/// Python wrapper for ComponentSubgraph - Pure delegation to existing traits
#[pyclass(name = "ComponentSubgraph", unsendable)]
#[derive(Clone)]
pub struct PyComponentSubgraph {
    pub inner: ComponentSubgraph,
}

impl PyComponentSubgraph {
    /// Create from Rust ComponentSubgraph
    pub fn from_core_component(component: ComponentSubgraph) -> Self {
        Self { inner: component }
    }
}

#[pymethods]
impl PyComponentSubgraph {
    // === ComponentOperations - Specialized methods (these are unique) ===

    #[getter]
    fn component_id(&self) -> usize {
        self.inner.component_id()
    }

    #[getter]
    fn is_largest_component(&self) -> bool {
        self.inner.is_largest_component()
    }

    #[getter]
    fn component_size(&self) -> usize {
        self.inner.component_size()
    }

    #[getter]
    fn total_components(&self) -> usize {
        self.inner.total_components()
    }

    // === SubgraphOperations - Just delegate to existing trait methods ===

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count() // This calls SubgraphOperations::node_count()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count() // This calls SubgraphOperations::edge_count()
    }

    fn contains_node(&self, node_id: NodeId) -> bool {
        self.inner.contains_node(node_id) // This calls SubgraphOperations::contains_node()
    }

    fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.inner.contains_edge(edge_id) // This calls SubgraphOperations::contains_edge()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        self.inner.node_set().iter().copied().collect() // Use SubgraphOperations::node_set()
    }

    fn edge_ids(&self) -> Vec<EdgeId> {
        self.inner.edge_set().iter().copied().collect() // Use SubgraphOperations::edge_set()
    }

    // Note: density() is not available in SubgraphOperations trait
    // Use internal_density() for component-specific density calculation

    // === String representations ===

    fn __repr__(&self) -> String {
        format!(
            "ComponentSubgraph(id={}, nodes={}, edges={}, largest={})",
            self.inner.component_id(),
            self.inner.node_count(),
            self.inner.edge_count(),
            self.inner.is_largest_component()
        )
    }

    fn __str__(&self) -> String {
        let largest_indicator = if self.inner.is_largest_component() {
            " (largest)"
        } else {
            ""
        };

        format!(
            "Connected Component {} with {} nodes and {} edges{}",
            self.inner.component_id(),
            self.inner.node_count(),
            self.inner.edge_count(),
            largest_indicator
        )
    }
}
