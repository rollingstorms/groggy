// src_new/graph/core.rs

use pyo3::pyclass;
use pyo3::pymethods;

/// Main graph structure with delegated collections
#[pyclass]
pub struct FastGraph {
    pub attribute_manager: crate::graph::managers::attributes::AttributeManager,
    pub graph_store: std::sync::Arc<crate::storage::graph_store::GraphStore>,
    pub node_collection: crate::graph::nodes::collection::NodeCollection,
    pub edge_collection: crate::graph::edges::collection::EdgeCollection,
    pub directed: bool,
    pub info: crate::graph::types::GraphInfo,
}

#[pymethods]
impl FastGraph {
    /// Constructor for new graph instance
    #[new]
    pub fn new() -> Self {
        let attribute_manager = crate::graph::managers::attributes::AttributeManager::new();
        let graph_store = std::sync::Arc::new(crate::storage::graph_store::GraphStore::new());
        let node_collection = crate::graph::nodes::collection::NodeCollection::new(attribute_manager.clone(), graph_store.clone(), None);
        let edge_collection = crate::graph::edges::collection::EdgeCollection::new(attribute_manager.clone(), graph_store.clone(), None);
        let info = crate::graph::types::GraphInfo::default();
        Self {
            attribute_manager,
            graph_store,
            node_collection,
            edge_collection,
            directed: true,
            info,
        }
    }

    /// Get comprehensive graph information
    pub fn info(&self) -> crate::graph::types::GraphInfo {
        self.info.clone()
    }

    /// Get total size (nodes + edges)
    pub fn size(&self) -> usize {
        self.node_collection.size() + self.edge_collection.size()
    }

    /// Check if graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Returns NodeCollection instance
    pub fn nodes(&self) -> crate::graph::nodes::collection::NodeCollection {
        self.node_collection.clone()
    }

    /// Returns EdgeCollection instance
    pub fn edges(&self) -> crate::graph::edges::collection::EdgeCollection {
        self.edge_collection.clone()
    }

    /// Create subgraph with node/edge filters
    pub fn subgraph(&self, node_ids: Option<Vec<crate::graph::types::NodeId>>, edge_ids: Option<Vec<crate::graph::types::EdgeId>>) -> Self {
        let mut node_collection = self.node_collection.clone();
        let mut edge_collection = self.edge_collection.clone();
        if let Some(ids) = node_ids {
            node_collection.node_ids = ids;
        }
        if let Some(ids) = edge_ids {
            edge_collection.edge_ids = ids;
        }
        Self {
            attribute_manager: self.attribute_manager.clone(),
            graph_store: self.graph_store.clone(),
            node_collection,
            edge_collection,
            directed: self.directed,
            info: self.info.clone(),
        }
    }

    /// Get all subgraphs according to a given attr groups
    pub fn subgraphs(&self) {
        // TODO
    }
}
