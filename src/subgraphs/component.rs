//! ComponentSubgraph - Connected component entity using existing efficient storage
//!
//! This module provides ComponentSubgraph which represents a connected component
//! using the same efficient storage pattern as the base Subgraph, with additional
//! component-specific metadata and operations.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::traits::{ComponentOperations, GraphEntity, SubgraphOperations};
use crate::types::{EdgeId, EntityId, NodeId};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// A connected component using our existing efficient subgraph storage
///
/// ComponentSubgraph owns one canonical `Subgraph` selection and its caches,
/// with specialized metadata kept separately. Clones share node/edge membership.
///
/// # Design Principles
/// - **Same Efficient Storage**: Identical foundation as base Subgraph
/// - **Component Metadata**: Adds component ID and analysis metadata
/// - **Algorithm Reuse**: All operations delegate to existing efficient algorithms
/// - **Trait Composability**: Implements both SubgraphOperations and ComponentOperations
#[derive(Debug, Clone)]
pub struct ComponentSubgraph {
    /// Canonical owner, selected membership, scope validation and topology caches.
    selection: crate::subgraphs::Subgraph,

    // Component-specific metadata (not stored in GraphPool - just behavior data)
    /// Unique component identifier within the parent graph's component structure
    component_id: usize,
    /// Whether this is the largest component in the graph
    is_largest: bool,
    /// Total number of components in the parent analysis (for context)
    total_components: usize,
}

impl ComponentSubgraph {
    /// Retain selected values/topology while preserving specialized metadata.
    pub fn freeze(mut self) -> GraphResult<Self> {
        self.selection = self.selection.freeze()?;
        Ok(self)
    }

    /// Borrow the canonical connected selection without copying its membership.
    pub fn as_subgraph(&self) -> &crate::subgraphs::Subgraph {
        &self.selection
    }

    /// Freeze current selected values/topology; specialized metadata stays on this wrapper.
    pub fn pin_view(&self) -> GraphResult<crate::subgraphs::GraphView> {
        self.selection.pin_view()
    }

    fn wrap_selection(&self, result: crate::subgraphs::Subgraph) -> Self {
        Self {
            is_largest: self.is_largest && result.nodes() == self.selection.nodes(),
            selection: result,
            component_id: self.component_id,
            total_components: self.total_components,
        }
    }

    /// Create a new ComponentSubgraph from existing component analysis
    ///
    /// # Arguments
    /// * `graph` - Reference to the shared graph infrastructure
    /// * `nodes` - Set of node IDs in this component
    /// * `edges` - Set of edge IDs in this component
    /// * `component_id` - Unique identifier for this component
    /// * `is_largest` - Whether this is the largest component
    /// * `total_components` - Total number of components in the analysis
    ///
    /// # Returns
    /// New ComponentSubgraph using efficient storage
    pub fn new(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        component_id: usize,
        is_largest: bool,
        total_components: usize,
    ) -> Self {
        Self {
            selection: crate::subgraphs::Subgraph::new(graph, nodes, edges, "component".into()),
            component_id,
            is_largest,
            total_components,
        }
    }

    /// Create ComponentSubgraph from connected components analysis result
    ///
    /// # Arguments
    /// * `graph` - Reference to shared graph infrastructure
    /// * `component_nodes` - Nodes in this component
    /// * `component_edges` - Edges in this component  
    /// * `component_id` - Component identifier
    /// * `all_component_sizes` - Sizes of all components for largest detection
    ///
    /// # Returns
    /// New ComponentSubgraph with computed metadata
    pub fn from_analysis(
        graph: Rc<RefCell<Graph>>,
        component_nodes: HashSet<NodeId>,
        component_edges: HashSet<EdgeId>,
        component_id: usize,
        all_component_sizes: &[usize],
    ) -> Self {
        let component_size = component_nodes.len();
        let is_largest = all_component_sizes
            .iter()
            .all(|&size| component_size >= size);
        let total_components = all_component_sizes.len();

        Self::new(
            graph,
            component_nodes,
            component_edges,
            component_id,
            is_largest,
            total_components,
        )
    }

    /// Get total number of components in the parent analysis
    pub fn total_components(&self) -> usize {
        self.total_components
    }
}

impl GraphEntity for ComponentSubgraph {
    fn entity_id(&self) -> EntityId {
        EntityId::Component(self.component_id)
    }

    fn entity_type(&self) -> &'static str {
        "component"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.selection.graph()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return nodes in this component as EntityNode wrappers
        use crate::entities::EntityNode;

        let entities: Vec<Box<dyn GraphEntity>> = self
            .selection
            .nodes()
            .iter()
            .map(|&node_id| {
                Box::new(EntityNode::new(node_id, self.selection.graph())) as Box<dyn GraphEntity>
            })
            .collect();
        Ok(entities)
    }

    fn summary(&self) -> String {
        let status = if self.is_largest {
            "largest"
        } else {
            "regular"
        };
        format!(
            "Component {}/{}: {} nodes, {} edges ({})",
            self.component_id,
            self.total_components,
            self.selection.nodes().len(),
            self.selection.edges().len(),
            status
        )
    }
}

impl SubgraphOperations for ComponentSubgraph {
    fn frozen_view(&self) -> Option<&crate::GraphView> {
        self.selection.frozen_view()
    }
    fn node_set(&self) -> &HashSet<NodeId> {
        self.selection.nodes()
    }
    fn edge_set(&self) -> &HashSet<EdgeId> {
        self.selection.edges()
    }
    fn neighbors(&self, node: NodeId) -> GraphResult<Vec<NodeId>> {
        self.selection.neighbors(node)
    }
    fn degree(&self, node: NodeId) -> GraphResult<usize> {
        self.selection.degree(node)
    }
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        Ok(self
            .selection
            .component_selections()?
            .into_iter()
            .map(|result| Box::new(self.wrap_selection(result)) as Box<dyn SubgraphOperations>)
            .collect())
    }
    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(self.wrap_selection(
            self.selection.bfs_selection(start, max_depth)?,
        )))
    }
    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(self.wrap_selection(
            self.selection.dfs_selection(start, max_depth)?,
        )))
    }
    fn shortest_path_subgraph(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        Ok(self
            .selection
            .path_selection(source, target)?
            .map(|result| Box::new(self.wrap_selection(result)) as Box<dyn SubgraphOperations>))
    }
    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(
            self.wrap_selection(self.selection.induced_selection(nodes)?),
        ))
    }
    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        Ok(Box::new(
            self.wrap_selection(self.selection.edge_selection(edges)?),
        ))
    }

    fn viz(&self) -> crate::viz::VizModule {
        use crate::subgraphs::visualization::SubgraphDataSource;
        use std::sync::Arc;

        // Create wrapper that extracts current component data
        let data_source = SubgraphDataSource::from_subgraph_operations(self);
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(data_source);

        crate::viz::VizModule::new(data_source)
    }
}

impl ComponentOperations for ComponentSubgraph {
    fn component_id(&self) -> usize {
        self.component_id
    }

    fn is_largest_component(&self) -> bool {
        self.is_largest
    }

    fn merge_with(
        &self,
        other: &dyn ComponentOperations,
    ) -> GraphResult<Box<dyn ComponentOperations>> {
        // Union of node and edge sets
        let mut merged_nodes = self.selection.nodes().clone();
        let other_nodes = other.node_set();
        merged_nodes.extend(other_nodes);

        let mut merged_edges = self.selection.edges().clone();
        let other_edges = other.edge_set();
        merged_edges.extend(other_edges);

        // Create new component with merged data
        let new_component_id = std::cmp::max(self.component_id, other.component_id());
        let merged_size = merged_nodes.len();
        let is_largest =
            merged_size >= std::cmp::max(self.selection.nodes().len(), other_nodes.len());

        let merged_component = ComponentSubgraph::new(
            self.selection.graph(),
            merged_nodes,
            merged_edges,
            new_component_id,
            is_largest,
            self.total_components.saturating_sub(1), // One fewer component after merge
        );

        Ok(Box::new(merged_component))
    }

    fn boundary_nodes(&self) -> GraphResult<Vec<NodeId>> {
        let mut boundary_nodes = Vec::new();
        let owner = self.selection.graph();
        let graph = owner.borrow();

        for &node_id in self.selection.nodes() {
            // Check if this node has any neighbors outside the component
            let neighbors = graph.neighbors(node_id)?;
            let has_external_neighbor = neighbors
                .iter()
                .any(|&neighbor| !self.selection.nodes().contains(&neighbor));

            if has_external_neighbor {
                boundary_nodes.push(node_id);
            }
        }

        Ok(boundary_nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_component_subgraph_creation() {
        // Test ComponentSubgraph creation and basic functionality
        let mut graph = Graph::new_manual();

        // Create a simple connected component
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();
        let edge1 = graph.add_edge(node1, node2).unwrap();
        let edge2 = graph.add_edge(node2, node3).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes = HashSet::from([node1, node2, node3]);
        let edges = HashSet::from([edge1, edge2]);

        let component = ComponentSubgraph::new(
            graph_rc, nodes, edges, 0,    // component_id
            true, // is_largest
            1,    // total_components
        );

        // Test ComponentOperations interface
        assert_eq!(component.component_id(), 0);
        assert!(component.is_largest_component());
        assert_eq!(component.component_size(), 3);
        assert_eq!(component.total_components(), 1);

        // Test SubgraphOperations interface
        assert_eq!(component.node_count(), 3);
        assert_eq!(component.edge_count(), 2);
        assert!(component.contains_node(node1));
        assert!(component.contains_edge(edge1));

        // Test GraphEntity interface
        assert_eq!(component.entity_type(), "component");

        // Test internal density calculation
        let density = component.internal_density();
        assert!(density > 0.0 && density <= 1.0);

        // ComponentSubgraph tests passed
    }

    #[test]
    fn test_component_algorithms() {
        // Test BFS and DFS on ComponentSubgraph
        let mut graph = Graph::new_manual();

        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();
        let node4 = graph.add_node().unwrap();

        let edge1 = graph.add_edge(node1, node2).unwrap();
        let edge2 = graph.add_edge(node2, node3).unwrap();
        let edge3 = graph.add_edge(node3, node4).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes = HashSet::from([node1, node2, node3, node4]);
        let edges = HashSet::from([edge1, edge2, edge3]);

        let component = ComponentSubgraph::new(graph_rc, nodes, edges, 0, true, 1);

        // Test BFS from node1
        let bfs_result = component.bfs(node1, Some(2)).unwrap();
        assert!(bfs_result.contains_node(node1));
        // BFS algorithm test completed

        // Test DFS from node1
        let dfs_result = component.dfs(node1, Some(2)).unwrap();
        assert!(dfs_result.contains_node(node1));
        // DFS algorithm test completed

        // Test boundary nodes (should be empty for a single component)
        let _boundary = component.boundary_nodes().unwrap();
        // Boundary nodes test completed

        // Component algorithm tests passed
    }
}
