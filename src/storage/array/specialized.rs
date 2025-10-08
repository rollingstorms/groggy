//! Specialized typed arrays built on BaseArray foundation

use crate::api::graph::Graph;
use crate::entities::meta_node::MetaNode;
use crate::storage::array::{
    ArrayIterator, ArrayOps, EdgeLike, MetaNodeLike, NodeIdLike, SubgraphLike,
};
use crate::subgraphs::subgraph::Subgraph;
use crate::types::{EdgeId, NodeId};
use std::cell::RefCell;
use std::rc::Rc;

// =============================================================================
// NodesArray - specialized array for NodeId collections
// =============================================================================

/// Typed array for NodeId collections with node-specific operations
pub struct NodesArray {
    /// Node IDs in this array
    node_ids: Vec<NodeId>,
    /// Optional reference to the parent graph for graph-aware operations
    graph_ref: Option<Rc<RefCell<Graph>>>,
    /// Optional name for debugging
    name: Option<String>,
}

impl NodesArray {
    /// Create a new NodesArray
    pub fn new(node_ids: Vec<NodeId>) -> Self {
        Self {
            node_ids,
            graph_ref: None,
            name: None,
        }
    }

    /// Create a new NodesArray with graph reference
    pub fn with_graph(node_ids: Vec<NodeId>, graph: Rc<RefCell<Graph>>) -> Self {
        Self {
            node_ids,
            graph_ref: Some(graph),
            name: None,
        }
    }

    /// Create a named NodesArray
    pub fn with_name(node_ids: Vec<NodeId>, name: String) -> Self {
        Self {
            node_ids,
            graph_ref: None,
            name: Some(name),
        }
    }

    /// Get the node IDs
    pub fn node_ids(&self) -> &Vec<NodeId> {
        &self.node_ids
    }

    /// Get the name (if any)
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }
}

impl ArrayOps<NodeId> for NodesArray {
    fn len(&self) -> usize {
        self.node_ids.len()
    }

    fn get(&self, index: usize) -> Option<&NodeId> {
        self.node_ids.get(index)
    }

    fn iter(&self) -> ArrayIterator<NodeId>
    where
        NodeId: Clone + 'static,
    {
        match &self.graph_ref {
            Some(graph) => ArrayIterator::with_graph(self.node_ids.clone(), graph.clone()),
            None => ArrayIterator::new(self.node_ids.clone()),
        }
    }
}

// Implement NodeIdLike for NodeId to enable node-specific operations
impl NodeIdLike for NodeId {
    // Marker trait - methods provided by ArrayIterator<T: NodeIdLike> impl
}

// =============================================================================
// EdgesArray - specialized array for EdgeId collections
// =============================================================================

/// Typed array for EdgeId collections with edge-specific operations
pub struct EdgesArray {
    /// Edge IDs in this array
    edge_ids: Vec<EdgeId>,
    /// Optional reference to the parent graph for graph-aware operations
    graph_ref: Option<Rc<RefCell<Graph>>>,
    /// Optional name for debugging
    #[allow(dead_code)]
    name: Option<String>,
}

impl EdgesArray {
    /// Create a new EdgesArray
    pub fn new(edge_ids: Vec<EdgeId>) -> Self {
        Self {
            edge_ids,
            graph_ref: None,
            name: None,
        }
    }

    /// Create a new EdgesArray with graph reference
    pub fn with_graph(edge_ids: Vec<EdgeId>, graph: Rc<RefCell<Graph>>) -> Self {
        Self {
            edge_ids,
            graph_ref: Some(graph),
            name: None,
        }
    }

    /// Get the edge IDs
    pub fn edge_ids(&self) -> &Vec<EdgeId> {
        &self.edge_ids
    }
}

impl ArrayOps<EdgeId> for EdgesArray {
    fn len(&self) -> usize {
        self.edge_ids.len()
    }

    fn get(&self, index: usize) -> Option<&EdgeId> {
        self.edge_ids.get(index)
    }

    fn iter(&self) -> ArrayIterator<EdgeId>
    where
        EdgeId: Clone + 'static,
    {
        match &self.graph_ref {
            Some(graph) => ArrayIterator::with_graph(self.edge_ids.clone(), graph.clone()),
            None => ArrayIterator::new(self.edge_ids.clone()),
        }
    }
}

// Implement EdgeLike for EdgeId to enable edge-specific operations
impl EdgeLike for EdgeId {
    // Marker trait - methods provided by ArrayIterator<T: EdgeLike> impl
}

// =============================================================================
// MetaNodeArray - specialized array for MetaNode collections
// =============================================================================

/// Typed array for MetaNode collections with meta-node-specific operations
pub struct MetaNodeArray {
    /// Meta-nodes in this array
    meta_nodes: Vec<MetaNode>,
    /// Optional reference to the parent graph for graph-aware operations
    graph_ref: Option<Rc<RefCell<Graph>>>,
    /// Optional name for debugging
    #[allow(dead_code)]
    name: Option<String>,
}

impl MetaNodeArray {
    /// Create a new MetaNodeArray
    pub fn new(meta_nodes: Vec<MetaNode>) -> Self {
        Self {
            meta_nodes,
            graph_ref: None,
            name: None,
        }
    }

    /// Create a new MetaNodeArray with graph reference
    pub fn with_graph(meta_nodes: Vec<MetaNode>, graph: Rc<RefCell<Graph>>) -> Self {
        Self {
            meta_nodes,
            graph_ref: Some(graph),
            name: None,
        }
    }

    /// Get the meta-nodes
    pub fn meta_nodes(&self) -> &Vec<MetaNode> {
        &self.meta_nodes
    }
}

impl ArrayOps<MetaNode> for MetaNodeArray {
    fn len(&self) -> usize {
        self.meta_nodes.len()
    }

    fn get(&self, index: usize) -> Option<&MetaNode> {
        self.meta_nodes.get(index)
    }

    fn iter(&self) -> ArrayIterator<MetaNode>
    where
        MetaNode: Clone + 'static,
    {
        match &self.graph_ref {
            Some(graph) => ArrayIterator::with_graph(self.meta_nodes.clone(), graph.clone()),
            None => ArrayIterator::new(self.meta_nodes.clone()),
        }
    }
}

// Implement MetaNodeLike for MetaNode to enable meta-node-specific operations
impl MetaNodeLike for MetaNode {
    // Marker trait - methods provided by ArrayIterator<T: MetaNodeLike> impl
}

// Implement SubgraphLike for Subgraph to enable subgraph-specific operations
impl SubgraphLike for Subgraph {
    // Marker trait - methods provided by ArrayIterator<T: SubgraphLike> impl
}

// =============================================================================
// Display implementations
// =============================================================================

impl std::fmt::Display for NodesArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name_str = self.name.as_deref().unwrap_or("NodesArray");
        write!(f, "{}[{}]", name_str, self.len())?;

        if !self.node_ids.is_empty() {
            write!(f, " [")?;
            for (i, node_id) in self.node_ids.iter().take(5).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", node_id)?;
            }
            if self.node_ids.len() > 5 {
                write!(f, ", ... ({} more)", self.node_ids.len() - 5)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

impl std::fmt::Display for EdgesArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EdgesArray[{}]", self.len())?;

        if !self.edge_ids.is_empty() {
            write!(f, " [")?;
            for (i, edge_id) in self.edge_ids.iter().take(5).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", edge_id)?;
            }
            if self.edge_ids.len() > 5 {
                write!(f, ", ... ({} more)", self.edge_ids.len() - 5)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}

impl std::fmt::Display for MetaNodeArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetaNodeArray[{}]", self.len())?;

        if !self.meta_nodes.is_empty() {
            write!(f, " [")?;
            for (i, meta_node) in self.meta_nodes.iter().take(3).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "MetaNode({})", meta_node.id())?;
            }
            if self.meta_nodes.len() > 3 {
                write!(f, ", ... ({} more)", self.meta_nodes.len() - 3)?;
            }
            write!(f, "]")?;
        }

        Ok(())
    }
}
