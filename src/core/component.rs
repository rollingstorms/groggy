//! ComponentSubgraph - Connected component entity using existing efficient storage
//!
//! This module provides ComponentSubgraph which represents a connected component
//! using the same efficient storage pattern as the base Subgraph, with additional
//! component-specific metadata and operations.

use crate::api::graph::Graph;
use crate::core::traits::{ComponentOperations, GraphEntity, SubgraphOperations};
use crate::errors::GraphResult;
use crate::types::{EdgeId, EntityId, NodeId};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// A connected component using our existing efficient subgraph storage
///
/// ComponentSubgraph uses the same storage pattern as the base Subgraph:
/// - HashSet<NodeId> for efficient node lookups
/// - HashSet<EdgeId> for efficient edge lookups  
/// - Rc<RefCell<Graph>> for shared graph access
/// - Additional component metadata for specialized operations
///
/// # Design Principles
/// - **Same Efficient Storage**: Identical foundation as base Subgraph
/// - **Component Metadata**: Adds component ID and analysis metadata
/// - **Algorithm Reuse**: All operations delegate to existing efficient algorithms
/// - **Trait Composability**: Implements both SubgraphOperations and ComponentOperations
#[derive(Debug, Clone)]
pub struct ComponentSubgraph {
    /// Reference to our shared graph storage infrastructure
    graph: Rc<RefCell<Graph>>,
    /// Efficient node set - same as base Subgraph
    nodes: HashSet<NodeId>,
    /// Efficient edge set - same as base Subgraph  
    edges: HashSet<EdgeId>,

    // Component-specific metadata (not stored in GraphPool - just behavior data)
    /// Unique component identifier within the parent graph's component structure
    component_id: usize,
    /// Whether this is the largest component in the graph
    is_largest: bool,
    /// Total number of components in the parent analysis (for context)
    total_components: usize,
}

impl ComponentSubgraph {
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
            graph,
            nodes,
            edges,
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
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return nodes in this component as EntityNode wrappers
        use crate::core::node::EntityNode;

        let entities: Vec<Box<dyn GraphEntity>> = self
            .nodes
            .iter()
            .map(|&node_id| {
                Box::new(EntityNode::new(node_id, self.graph.clone())) as Box<dyn GraphEntity>
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
            self.nodes.len(),
            self.edges.len(),
            status
        )
    }
}

impl SubgraphOperations for ComponentSubgraph {
    // Use existing efficient storage - same as base Subgraph
    fn node_set(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    fn edge_set(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    // Delegate all core algorithms to existing efficient implementations
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // Components of a component are typically just itself, unless further subdivision
        // For now, return self as the only component
        Ok(vec![Box::new(self.clone()) as Box<dyn SubgraphOperations>])
    }

    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Use existing BFS implementation from base Subgraph pattern
        if !self.nodes.contains(&start) {
            return Err(crate::errors::GraphError::NodeNotFound {
                node_id: start,
                operation: "component_bfs".to_string(),
                suggestion: "Ensure start node is within this component".to_string(),
            });
        }

        use crate::core::traversal::TraversalEngine;
        let graph = self.graph.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.bfs(&graph.pool(), graph.space(), start, options)?;

        // Filter result to nodes that exist in this component
        let filtered_nodes: HashSet<NodeId> = result
            .nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: HashSet<EdgeId> = result
            .edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        // Create new ComponentSubgraph for the BFS result
        let bfs_component = ComponentSubgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            self.component_id, // Keep same component ID
            false,             // BFS result is not the largest
            self.total_components,
        );

        Ok(Box::new(bfs_component))
    }

    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Use existing DFS implementation pattern
        if !self.nodes.contains(&start) {
            return Err(crate::errors::GraphError::NodeNotFound {
                node_id: start,
                operation: "component_dfs".to_string(),
                suggestion: "Ensure start node is within this component".to_string(),
            });
        }

        use crate::core::traversal::TraversalEngine;
        let graph = self.graph.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.dfs(&graph.pool(), graph.space(), start, options)?;

        // Filter result to nodes that exist in this component
        let filtered_nodes: HashSet<NodeId> = result
            .nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: HashSet<EdgeId> = result
            .edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        // Create new ComponentSubgraph for the DFS result
        let dfs_component = ComponentSubgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            self.component_id, // Keep same component ID
            false,             // DFS result is not the largest
            self.total_components,
        );

        Ok(Box::new(dfs_component))
    }

    fn shortest_path_subgraph(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        if !self.nodes.contains(&source) || !self.nodes.contains(&target) {
            return Ok(None);
        }

        use crate::core::traversal::{PathFindingOptions, TraversalEngine};
        let graph = self.graph.borrow_mut();
        let options = PathFindingOptions::default();

        let mut traversal_engine = TraversalEngine::new();
        let x = if let Some(path_result) =
            traversal_engine.shortest_path(&graph.pool(), graph.space(), source, target, options)?
        {
            // Filter path to nodes/edges that exist in this component
            let filtered_nodes: HashSet<NodeId> = path_result
                .nodes
                .into_iter()
                .filter(|node| self.nodes.contains(node))
                .collect();
            let filtered_edges: HashSet<EdgeId> = path_result
                .edges
                .into_iter()
                .filter(|edge| self.edges.contains(edge))
                .collect();

            if !filtered_nodes.is_empty() {
                let path_component = ComponentSubgraph::new(
                    self.graph.clone(),
                    filtered_nodes,
                    filtered_edges,
                    self.component_id, // Keep same component ID
                    false,             // Path is not the largest
                    self.total_components,
                );
                Ok(Some(Box::new(path_component) as Box<dyn SubgraphOperations>))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        };
        x
    }

    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to nodes that exist in this component
        let component_nodes: HashSet<NodeId> = nodes
            .iter()
            .filter(|&&node| self.nodes.contains(&node))
            .cloned()
            .collect();

        if component_nodes.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No nodes from the requested set exist in this component".to_string(),
            ));
        }

        // Find induced edges using existing graph algorithms
        let graph = self.graph.borrow();
        let mut induced_edges = HashSet::new();

        for &node1 in &component_nodes {
            for &node2 in &component_nodes {
                if node1 < node2 {
                    // Avoid duplicate checking
                    if let Ok(has_edge) = graph.has_edge_between(node1, node2) {
                        if has_edge {
                            // Find the actual edge ID
                            if let Ok(edges) = graph.incident_edges(node1) {
                                for edge_id in edges {
                                    if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                        if (src == node1 && tgt == node2)
                                            || (src == node2 && tgt == node1)
                                        {
                                            if self.edges.contains(&edge_id) {
                                                induced_edges.insert(edge_id);
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let induced_component = ComponentSubgraph::new(
            self.graph.clone(),
            component_nodes,
            induced_edges,
            self.component_id, // Keep same component ID
            false,             // Induced subgraph is not the largest
            self.total_components,
        );

        Ok(Box::new(induced_component))
    }

    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to edges that exist in this component
        let component_edges: HashSet<EdgeId> = edges
            .iter()
            .filter(|&&edge| self.edges.contains(&edge))
            .cloned()
            .collect();

        if component_edges.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No edges from the requested set exist in this component".to_string(),
            ));
        }

        // Find nodes connected to these edges
        let graph = self.graph.borrow();
        let mut edge_nodes = HashSet::new();

        for &edge_id in &component_edges {
            if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                if self.nodes.contains(&src) && self.nodes.contains(&tgt) {
                    edge_nodes.insert(src);
                    edge_nodes.insert(tgt);
                }
            }
        }

        let edge_component = ComponentSubgraph::new(
            self.graph.clone(),
            edge_nodes,
            component_edges,
            self.component_id, // Keep same component ID
            false,             // Edge subgraph is not the largest
            self.total_components,
        );

        Ok(Box::new(edge_component))
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
        let mut merged_nodes = self.nodes.clone();
        let other_nodes = other.node_set();
        merged_nodes.extend(other_nodes);

        let mut merged_edges = self.edges.clone();
        let other_edges = other.edge_set();
        merged_edges.extend(other_edges);

        // Create new component with merged data
        let new_component_id = std::cmp::max(self.component_id, other.component_id());
        let merged_size = merged_nodes.len();
        let is_largest = merged_size >= std::cmp::max(self.nodes.len(), other_nodes.len());

        let merged_component = ComponentSubgraph::new(
            self.graph.clone(),
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
        let graph = self.graph.borrow();

        for &node_id in &self.nodes {
            // Check if this node has any neighbors outside the component
            let neighbors = graph.neighbors(node_id)?;
            let has_external_neighbor = neighbors
                .iter()
                .any(|&neighbor| !self.nodes.contains(&neighbor));

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
        let mut graph = Graph::new();

        // Create a simple connected component
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();
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

        println!("ComponentSubgraph tests passed!");
    }

    #[test]
    fn test_component_algorithms() {
        // Test BFS and DFS on ComponentSubgraph
        let mut graph = Graph::new();

        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();
        let node4 = graph.add_node();

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
        println!("BFS result has {} nodes", bfs_result.node_count());

        // Test DFS from node1
        let dfs_result = component.dfs(node1, Some(2)).unwrap();
        assert!(dfs_result.contains_node(node1));
        println!("DFS result has {} nodes", dfs_result.node_count());

        // Test boundary nodes (should be empty for a single component)
        let boundary = component.boundary_nodes().unwrap();
        println!("Found {} boundary nodes", boundary.len());

        println!("Component algorithm tests passed!");
    }
}
