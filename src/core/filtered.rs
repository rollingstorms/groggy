//! FilteredSubgraph - Filtered subgraph entity using existing efficient storage
//!
//! This module provides FilteredSubgraph which represents a subgraph created by
//! applying filter criteria, using the same efficient storage pattern as the base
//! Subgraph with additional filter-specific metadata and operations.

use crate::api::graph::Graph;
use crate::core::traits::filter_operations::FilterCriteria;
use crate::core::traits::{FilterOperations, GraphEntity, SubgraphOperations};
use crate::errors::GraphResult;
use crate::types::{EdgeId, EntityId, NodeId};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// A filtered subgraph using our existing efficient subgraph storage
///
/// FilteredSubgraph uses the same storage pattern as the base Subgraph:
/// - HashSet<NodeId> for efficient node lookups
/// - HashSet<EdgeId> for efficient edge lookups  
/// - Rc<RefCell<Graph>> for shared graph access
/// - Additional filter criteria for reapplication and combination
///
/// # Design Principles
/// - **Same Efficient Storage**: Identical foundation as base Subgraph
/// - **Filter Metadata**: Stores filter criteria for advanced operations
/// - **Algorithm Reuse**: All operations delegate to existing efficient algorithms
/// - **Trait Composability**: Implements both SubgraphOperations and FilterOperations
#[derive(Debug, Clone)]
pub struct FilteredSubgraph {
    /// Reference to our shared graph storage infrastructure
    graph: Rc<RefCell<Graph>>,
    /// Efficient node set - same as base Subgraph
    nodes: HashSet<NodeId>,
    /// Efficient edge set - same as base Subgraph  
    edges: HashSet<EdgeId>,

    // Filter-specific metadata (not stored in GraphPool - just behavior data)
    /// Filter criteria used to create this subgraph
    filter_criteria: FilterCriteria,
    /// Original graph size when filter was applied (for stats)
    original_node_count: usize,
    /// Original edge count when filter was applied
    original_edge_count: usize,
}

impl FilteredSubgraph {
    /// Create a new FilteredSubgraph from filter results
    ///
    /// # Arguments
    /// * `graph` - Reference to the shared graph infrastructure
    /// * `nodes` - Set of node IDs that passed the filter
    /// * `edges` - Set of edge IDs that passed the filter
    /// * `criteria` - Filter criteria used to create this subgraph
    /// * `original_node_count` - Total nodes in graph when filter was applied
    /// * `original_edge_count` - Total edges in graph when filter was applied
    ///
    /// # Returns
    /// New FilteredSubgraph using efficient storage
    pub fn new(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        criteria: FilterCriteria,
        original_node_count: usize,
        original_edge_count: usize,
    ) -> Self {
        Self {
            graph,
            nodes,
            edges,
            filter_criteria: criteria,
            original_node_count,
            original_edge_count,
        }
    }

    /// Create FilteredSubgraph by applying criteria to current graph
    ///
    /// # Arguments
    /// * `graph` - Reference to shared graph infrastructure
    /// * `criteria` - Filter criteria to apply
    ///
    /// # Returns
    /// New FilteredSubgraph with criteria applied to current graph state
    pub fn from_criteria(graph: Rc<RefCell<Graph>>, criteria: FilterCriteria) -> GraphResult<Self> {
        let mut filtered_nodes = HashSet::new();
        let mut filtered_edges = HashSet::new();

        let graph_borrow = graph.borrow();
        let original_node_count = graph_borrow.node_ids().len();
        let original_edge_count = graph_borrow.edge_ids().len();

        // Filter nodes
        for node_id in graph_borrow.node_ids() {
            if criteria.matches_node(node_id, &graph_borrow)? {
                filtered_nodes.insert(node_id);
            }
        }

        // Filter edges - only include if both endpoints are in filtered nodes
        for edge_id in graph_borrow.edge_ids() {
            let (src, tgt) = graph_borrow.edge_endpoints(edge_id)?;
            if filtered_nodes.contains(&src) && filtered_nodes.contains(&tgt) {
                // Only include edge if it also passes any edge criteria
                if criteria.matches_edge(edge_id, &graph_borrow)? {
                    filtered_edges.insert(edge_id);
                }
            }
        }

        drop(graph_borrow);

        Ok(Self::new(
            graph,
            filtered_nodes,
            filtered_edges,
            criteria,
            original_node_count,
            original_edge_count,
        ))
    }

    /// Get total node count when filter was originally applied
    pub fn original_node_count(&self) -> usize {
        self.original_node_count
    }

    /// Get total edge count when filter was originally applied
    pub fn original_edge_count(&self) -> usize {
        self.original_edge_count
    }
}

impl GraphEntity for FilteredSubgraph {
    fn entity_id(&self) -> EntityId {
        // Use hash of filter criteria as unique ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.filter_criteria).hash(&mut hasher);
        EntityId::Subgraph(hasher.finish() as crate::types::SubgraphId)
    }

    fn entity_type(&self) -> &'static str {
        "filtered_subgraph"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return nodes in this filtered subgraph as EntityNode wrappers
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
        let stats = self.filter_stats();
        format!(
            "FilteredSubgraph: {} nodes, {} edges (retained {:.1}% nodes, {:.1}% edges, complexity {})",
            self.nodes.len(),
            self.edges.len(),
            stats.node_retention_rate * 100.0,
            stats.edge_retention_rate * 100.0,
            stats.criteria_complexity
        )
    }
}

impl SubgraphOperations for FilteredSubgraph {
    // Use existing efficient storage - same as base Subgraph
    fn node_set(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    fn edge_set(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    // Delegate all core algorithms to existing efficient implementations
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // Use existing connected components algorithm with filtered data
        use crate::core::traversal::TraversalEngine;
        let graph = self.graph.borrow_mut();
        let options = crate::core::traversal::TraversalOptions::default();

        let mut traversal_engine = TraversalEngine::new();
        let result =
            traversal_engine.connected_components(&graph.pool(), graph.space(), options)?;

        let mut components = Vec::new();
        for component in result.components {
            let component_nodes: HashSet<NodeId> = component.nodes.into_iter().collect();
            let component_edges: HashSet<EdgeId> = component.edges.into_iter().collect();

            // Filter to only edges that are also in our filtered subgraph
            let filtered_component_edges: HashSet<EdgeId> =
                component_edges.intersection(&self.edges).cloned().collect();

            let component_subgraph = FilteredSubgraph::new(
                self.graph.clone(),
                component_nodes,
                filtered_component_edges,
                self.filter_criteria.clone(),
                self.original_node_count,
                self.original_edge_count,
            );

            components.push(Box::new(component_subgraph) as Box<dyn SubgraphOperations>);
        }

        Ok(components)
    }

    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(crate::errors::GraphError::NodeNotFound {
                node_id: start,
                operation: "filtered_bfs".to_string(),
                suggestion: "Ensure start node exists in filtered subgraph".to_string(),
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

        // Filter result to nodes/edges that exist in this filtered subgraph
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

        let bfs_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            self.filter_criteria.clone(),
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(bfs_subgraph))
    }

    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(crate::errors::GraphError::NodeNotFound {
                node_id: start,
                operation: "filtered_dfs".to_string(),
                suggestion: "Ensure start node exists in filtered subgraph".to_string(),
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

        // Filter result to nodes/edges that exist in this filtered subgraph
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

        let dfs_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            self.filter_criteria.clone(),
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(dfs_subgraph))
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
            // Filter path to nodes/edges that exist in this filtered subgraph
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
                let path_subgraph = FilteredSubgraph::new(
                    self.graph.clone(),
                    filtered_nodes,
                    filtered_edges,
                    self.filter_criteria.clone(),
                    self.original_node_count,
                    self.original_edge_count,
                );
                Ok(Some(Box::new(path_subgraph) as Box<dyn SubgraphOperations>))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        };
        x
    }

    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to nodes that exist in this filtered subgraph
        let filtered_nodes: HashSet<NodeId> = nodes
            .iter()
            .filter(|&&node| self.nodes.contains(&node))
            .cloned()
            .collect();

        if filtered_nodes.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No nodes from the requested set exist in this filtered subgraph".to_string(),
            ));
        }

        // Find induced edges using existing graph algorithms
        let graph = self.graph.borrow();
        let mut induced_edges = HashSet::new();

        for &node1 in &filtered_nodes {
            for &node2 in &filtered_nodes {
                if node1 < node2 {
                    // Avoid duplicate checking
                    if let Ok(edges) = graph.incident_edges(node1) {
                        for edge_id in edges {
                            if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                                if (src == node1 && tgt == node2) || (src == node2 && tgt == node1)
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

        let induced_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            filtered_nodes,
            induced_edges,
            self.filter_criteria.clone(),
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(induced_subgraph))
    }

    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to edges that exist in this filtered subgraph
        let filtered_edges: HashSet<EdgeId> = edges
            .iter()
            .filter(|&&edge| self.edges.contains(&edge))
            .cloned()
            .collect();

        if filtered_edges.is_empty() {
            return Err(crate::errors::GraphError::InvalidInput(
                "No edges from the requested set exist in this filtered subgraph".to_string(),
            ));
        }

        // Find nodes connected to these edges
        let graph = self.graph.borrow();
        let mut edge_nodes = HashSet::new();

        for &edge_id in &filtered_edges {
            if let Ok((src, tgt)) = graph.edge_endpoints(edge_id) {
                if self.nodes.contains(&src) && self.nodes.contains(&tgt) {
                    edge_nodes.insert(src);
                    edge_nodes.insert(tgt);
                }
            }
        }

        let edge_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            edge_nodes,
            filtered_edges,
            self.filter_criteria.clone(),
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(edge_subgraph))
    }
}

impl FilterOperations for FilteredSubgraph {
    fn filter_criteria(&self) -> &FilterCriteria {
        &self.filter_criteria
    }

    fn reapply_filter(&self) -> GraphResult<Box<dyn FilterOperations>> {
        let new_subgraph =
            FilteredSubgraph::from_criteria(self.graph.clone(), self.filter_criteria.clone())?;

        Ok(Box::new(new_subgraph))
    }

    fn and_filter(&self, other: &dyn FilterOperations) -> GraphResult<Box<dyn FilterOperations>> {
        // Create intersection of node and edge sets
        let intersected_nodes: HashSet<NodeId> =
            self.nodes.intersection(other.node_set()).cloned().collect();
        let intersected_edges: HashSet<EdgeId> =
            self.edges.intersection(other.edge_set()).cloned().collect();

        // Combine criteria with AND
        let combined_criteria = FilterCriteria::And(vec![
            self.filter_criteria.clone(),
            other.filter_criteria().clone(),
        ]);

        let and_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            intersected_nodes,
            intersected_edges,
            combined_criteria,
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(and_subgraph))
    }

    fn or_filter(&self, other: &dyn FilterOperations) -> GraphResult<Box<dyn FilterOperations>> {
        // Create union of node and edge sets
        let mut unioned_nodes = self.nodes.clone();
        unioned_nodes.extend(other.node_set());
        let mut unioned_edges = self.edges.clone();
        unioned_edges.extend(other.edge_set());

        // Combine criteria with OR
        let combined_criteria = FilterCriteria::Or(vec![
            self.filter_criteria.clone(),
            other.filter_criteria().clone(),
        ]);

        let or_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            unioned_nodes,
            unioned_edges,
            combined_criteria,
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(or_subgraph))
    }

    fn not_filter(&self) -> GraphResult<Box<dyn FilterOperations>> {
        // Create complement against full graph
        let graph_borrow = self.graph.borrow();
        let all_nodes: HashSet<NodeId> = graph_borrow.node_ids().into_iter().collect();
        let all_edges: HashSet<EdgeId> = graph_borrow.edge_ids().into_iter().collect();

        let complement_nodes: HashSet<NodeId> =
            all_nodes.difference(&self.nodes).cloned().collect();
        let complement_edges: HashSet<EdgeId> =
            all_edges.difference(&self.edges).cloned().collect();

        // Negate criteria
        let negated_criteria = FilterCriteria::Not(Box::new(self.filter_criteria.clone()));

        drop(graph_borrow);

        let not_subgraph = FilteredSubgraph::new(
            self.graph.clone(),
            complement_nodes,
            complement_edges,
            negated_criteria,
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(not_subgraph))
    }

    fn add_criteria(
        &self,
        additional_criteria: FilterCriteria,
    ) -> GraphResult<Box<dyn FilterOperations>> {
        // Combine with existing criteria using AND
        let combined_criteria =
            FilterCriteria::And(vec![self.filter_criteria.clone(), additional_criteria]);

        // Reapply combined criteria to current graph
        let new_subgraph = FilteredSubgraph::from_criteria(self.graph.clone(), combined_criteria)?;

        Ok(Box::new(new_subgraph))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::types::AttrValue;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_filtered_subgraph_creation() {
        // Test FilteredSubgraph creation and basic functionality
        let mut graph = Graph::new();

        // Create test nodes with attributes
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();

        graph
            .set_node_attr(
                node1,
                "type".to_string(),
                AttrValue::Text("important".to_string()),
            )
            .unwrap();
        graph
            .set_node_attr(
                node2,
                "type".to_string(),
                AttrValue::Text("normal".to_string()),
            )
            .unwrap();
        graph
            .set_node_attr(
                node3,
                "type".to_string(),
                AttrValue::Text("important".to_string()),
            )
            .unwrap();

        let _edge1 = graph.add_edge(node1, node2).unwrap(); // Should be filtered out (node2 is not important)
        let edge2 = graph.add_edge(node1, node3).unwrap(); // Should be included (both nodes important)

        let graph_rc = Rc::new(RefCell::new(graph));

        // Create filter criteria
        let criteria = FilterCriteria::NodeAttributeEquals {
            name: "type".to_string(),
            value: AttrValue::Text("important".to_string()),
        };

        // Create filtered subgraph
        let filtered = FilteredSubgraph::from_criteria(graph_rc, criteria.clone()).unwrap();

        // Test FilterOperations interface
        assert_eq!(filtered.filter_criteria().complexity_score(), 1);
        assert_eq!(filtered.node_count(), 2); // node1 and node3
        assert!(filtered.contains_node(node1));
        assert!(!filtered.contains_node(node2));
        assert!(filtered.contains_node(node3));

        // Test SubgraphOperations interface
        assert!(filtered.contains_edge(edge2)); // edge between important nodes

        // Test GraphEntity interface
        assert_eq!(filtered.entity_type(), "filtered_subgraph");

        // Test filter stats
        let stats = filtered.filter_stats();
        assert!(stats.node_retention_rate > 0.0);
        assert!(stats.criteria_complexity == 1);

        println!("FilteredSubgraph tests passed!");
    }

    #[test]
    fn test_filter_combination() {
        // Test combining filters with AND/OR operations
        let mut graph = Graph::new();

        let node1 = graph.add_node();
        let node2 = graph.add_node();

        graph
            .set_node_attr(node1, "type".to_string(), AttrValue::Text("A".to_string()))
            .unwrap();
        graph
            .set_node_attr(node1, "value".to_string(), AttrValue::Int(10))
            .unwrap();
        graph
            .set_node_attr(node2, "type".to_string(), AttrValue::Text("B".to_string()))
            .unwrap();
        graph
            .set_node_attr(node2, "value".to_string(), AttrValue::Int(20))
            .unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));

        // Create two different filters
        let filter1 = FilteredSubgraph::from_criteria(
            graph_rc.clone(),
            FilterCriteria::NodeAttributeEquals {
                name: "type".to_string(),
                value: AttrValue::Text("A".to_string()),
            },
        )
        .unwrap();

        let filter2 = FilteredSubgraph::from_criteria(
            graph_rc,
            FilterCriteria::NodeAttributeEquals {
                name: "value".to_string(),
                value: AttrValue::Int(20),
            },
        )
        .unwrap();

        // Test OR combination
        let or_result = filter1.or_filter(&filter2).unwrap();
        assert_eq!(or_result.node_count(), 2); // Should have both nodes

        // Test AND combination
        let and_result = filter1.and_filter(&filter2).unwrap();
        assert_eq!(and_result.node_count(), 0); // No nodes match both criteria

        println!("Filter combination tests passed!");
    }
}
