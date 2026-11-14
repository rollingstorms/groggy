//! FilteredSubgraph - Filtered subgraph entity using existing efficient storage
//!
//! This module provides FilteredSubgraph which represents a subgraph created by
//! applying filter criteria, using the same efficient storage pattern as the base
//! Subgraph with additional filter-specific metadata and operations.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::ownership::OwnerCell;
use crate::traits::filter_operations::FilterCriteria;
use crate::traits::{FilterOperations, GraphEntity, SubgraphOperations};
use crate::types::{EdgeId, EntityId, NodeId};
use std::collections::HashSet;
use std::sync::Arc;

/// A filtered subgraph using our existing efficient subgraph storage
///
/// FilteredSubgraph owns one canonical `Subgraph` selection and its caches,
/// with specialized metadata kept separately. Clones share node/edge membership.
///
/// # Design Principles
/// - **Same Efficient Storage**: Identical foundation as base Subgraph
/// - **Filter Metadata**: Stores filter criteria for advanced operations
/// - **Algorithm Reuse**: All operations delegate to existing efficient algorithms
/// - **Trait Composability**: Implements both SubgraphOperations and FilterOperations
#[derive(Debug, Clone)]
pub struct FilteredSubgraph {
    /// Canonical owner, selected membership, scope validation and topology caches.
    selection: crate::subgraphs::Subgraph,

    // Filter-specific metadata (not stored in GraphPool - just behavior data)
    /// Filter criteria used to create this subgraph
    filter_criteria: FilterCriteria,
    /// Original graph size when filter was applied (for stats)
    original_node_count: usize,
    /// Original edge count when filter was applied
    original_edge_count: usize,
}

impl FilteredSubgraph {
    fn with_view_criteria(
        &self,
        source: crate::GraphView,
        criteria: FilterCriteria,
    ) -> GraphResult<Self> {
        let mut nodes = Vec::new();
        for id in source.node_ids() {
            if criteria.matches_node_view(id, &source)? {
                nodes.push(id);
            }
        }
        let selected = source.select_nodes(&nodes)?;
        let mut edges = HashSet::new();
        for id in selected.edge_ids() {
            if criteria.matches_edge_view(id, &source)? {
                edges.insert(id);
            }
        }
        let view = crate::GraphView::from_membership(
            source.read_state().clone(),
            selected.nodes().clone(),
            edges,
        )?;
        let mut result =
            self.wrap_selection(self.selection.bound_selection(view, "filtered".into()));
        result.filter_criteria = criteria;
        result.original_node_count = source.node_count();
        result.original_edge_count = source.edge_count();
        Ok(result)
    }

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
            selection: result,
            filter_criteria: self.filter_criteria.clone(),
            original_node_count: self.original_node_count,
            original_edge_count: self.original_edge_count,
        }
    }

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
        graph: Arc<OwnerCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        criteria: FilterCriteria,
        original_node_count: usize,
        original_edge_count: usize,
    ) -> Self {
        Self {
            selection: crate::subgraphs::Subgraph::new(graph, nodes, edges, "filtered".into()),
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
    pub fn from_criteria(graph: Arc<OwnerCell<Graph>>, criteria: FilterCriteria) -> GraphResult<Self> {
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
        use std::hash::Hash;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.filter_criteria).hash(&mut hasher);
        EntityId::Subgraph(hasher.finish() as crate::types::SubgraphId)
    }

    fn entity_type(&self) -> &'static str {
        "filtered_subgraph"
    }

    fn graph_ref(&self) -> Arc<OwnerCell<Graph>> {
        self.selection.graph()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return nodes in this filtered subgraph as EntityNode wrappers
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
        let stats = self.filter_stats();
        format!(
            "FilteredSubgraph: {} nodes, {} edges (retained {:.1}% nodes, {:.1}% edges, complexity {})",
            self.selection.nodes().len(),
            self.selection.edges().len(),
            stats.node_retention_rate * 100.0,
            stats.edge_retention_rate * 100.0,
            stats.criteria_complexity
        )
    }
}

impl SubgraphOperations for FilteredSubgraph {
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

        // Create wrapper that extracts current filtered data
        let data_source = SubgraphDataSource::from_subgraph_operations(self);
        let data_source: Arc<dyn crate::viz::streaming::data_source::DataSource> =
            Arc::new(data_source);

        crate::viz::VizModule::new(data_source)
    }
}

impl FilterOperations for FilteredSubgraph {
    fn filter_criteria(&self) -> &FilterCriteria {
        &self.filter_criteria
    }

    fn reapply_filter(&self) -> GraphResult<Box<dyn FilterOperations>> {
        if self.frozen_view().is_some() {
            let source = self.selection.graph().borrow().view()?;
            return Ok(Box::new(
                self.with_view_criteria(source, self.filter_criteria.clone())?,
            ));
        }
        let new_subgraph =
            FilteredSubgraph::from_criteria(self.selection.graph(), self.filter_criteria.clone())?;

        Ok(Box::new(new_subgraph))
    }

    fn and_filter(&self, other: &dyn FilterOperations) -> GraphResult<Box<dyn FilterOperations>> {
        if self.frozen_view().is_some() || other.frozen_view().is_some() {
            let left = self.frozen_view().ok_or_else(|| {
                crate::GraphError::InvalidInput(
                    "Freeze both filter sources before combining them".into(),
                )
            })?;
            let right = other.frozen_view().ok_or_else(|| {
                crate::GraphError::InvalidInput(
                    "Freeze both filter sources before combining them".into(),
                )
            })?;
            let mut result = self.wrap_selection(
                self.selection
                    .bound_selection(left.intersect_with(right)?, "filtered".into()),
            );
            result.filter_criteria = FilterCriteria::And(vec![
                self.filter_criteria.clone(),
                other.filter_criteria().clone(),
            ]);
            return Ok(Box::new(result));
        }

        // Create intersection of node and edge sets
        let intersected_nodes: HashSet<NodeId> = self
            .selection
            .nodes()
            .intersection(other.node_set())
            .cloned()
            .collect();
        let intersected_edges: HashSet<EdgeId> = self
            .selection
            .edges()
            .intersection(other.edge_set())
            .cloned()
            .collect();

        // Combine criteria with AND
        let combined_criteria = FilterCriteria::And(vec![
            self.filter_criteria.clone(),
            other.filter_criteria().clone(),
        ]);

        let and_subgraph = FilteredSubgraph::new(
            self.selection.graph(),
            intersected_nodes,
            intersected_edges,
            combined_criteria,
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(and_subgraph))
    }

    fn or_filter(&self, other: &dyn FilterOperations) -> GraphResult<Box<dyn FilterOperations>> {
        if self.frozen_view().is_some() || other.frozen_view().is_some() {
            let left = self.frozen_view().ok_or_else(|| {
                crate::GraphError::InvalidInput(
                    "Freeze both filter sources before combining them".into(),
                )
            })?;
            let right = other.frozen_view().ok_or_else(|| {
                crate::GraphError::InvalidInput(
                    "Freeze both filter sources before combining them".into(),
                )
            })?;
            let mut result = self.wrap_selection(
                self.selection
                    .bound_selection(left.merge_with(right)?, "filtered".into()),
            );
            result.filter_criteria = FilterCriteria::Or(vec![
                self.filter_criteria.clone(),
                other.filter_criteria().clone(),
            ]);
            return Ok(Box::new(result));
        }

        // Create union of node and edge sets
        let mut unioned_nodes = self.selection.nodes().clone();
        unioned_nodes.extend(other.node_set());
        let mut unioned_edges = self.selection.edges().clone();
        unioned_edges.extend(other.edge_set());

        // Combine criteria with OR
        let combined_criteria = FilterCriteria::Or(vec![
            self.filter_criteria.clone(),
            other.filter_criteria().clone(),
        ]);

        let or_subgraph = FilteredSubgraph::new(
            self.selection.graph(),
            unioned_nodes,
            unioned_edges,
            combined_criteria,
            self.original_node_count,
            self.original_edge_count,
        );

        Ok(Box::new(or_subgraph))
    }

    fn not_filter(&self) -> GraphResult<Box<dyn FilterOperations>> {
        if let Some(view) = self.frozen_view() {
            let root = crate::GraphView::all(view.read_state().clone());
            let nodes: Vec<_> = root.nodes().difference(view.nodes()).copied().collect();
            let mut result = self.wrap_selection(
                self.selection
                    .bound_selection(root.select_nodes(&nodes)?, "filtered".into()),
            );
            result.filter_criteria = FilterCriteria::Not(Box::new(self.filter_criteria.clone()));
            return Ok(Box::new(result));
        }
        // Create complement against full graph
        let owner = self.selection.graph();
        let graph_borrow = owner.borrow();
        let all_nodes: HashSet<NodeId> = graph_borrow.node_ids().into_iter().collect();
        self.selection.validate_query_scope(&graph_borrow)?;
        let complement_nodes: HashSet<NodeId> = all_nodes
            .difference(self.selection.nodes())
            .copied()
            .collect();
        let mut complement_edges = HashSet::new();
        for edge in graph_borrow.edge_ids() {
            if self.selection.edges().contains(&edge) {
                continue;
            }
            let (source, target) = graph_borrow.edge_endpoints(edge)?;
            if complement_nodes.contains(&source) && complement_nodes.contains(&target) {
                complement_edges.insert(edge);
            }
        }

        // Negate criteria
        let negated_criteria = FilterCriteria::Not(Box::new(self.filter_criteria.clone()));

        drop(graph_borrow);

        let not_subgraph = FilteredSubgraph::new(
            self.selection.graph(),
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

        if let Some(view) = self.frozen_view() {
            return Ok(Box::new(self.with_view_criteria(
                crate::GraphView::all(view.read_state().clone()),
                combined_criteria,
            )?));
        }
        // Reapply combined criteria to current graph
        let new_subgraph =
            FilteredSubgraph::from_criteria(self.selection.graph(), combined_criteria)?;

        Ok(Box::new(new_subgraph))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::ownership::OwnerCell;
    use crate::types::AttrValue;
    use std::sync::Arc;

    #[test]
    fn test_filtered_subgraph_creation() {
        // Test FilteredSubgraph creation and basic functionality
        let mut graph = Graph::new_manual();

        // Create test nodes with attributes
        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();
        let node3 = graph.add_node().unwrap();

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

        let graph_rc = Arc::new(OwnerCell::new(graph));

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

        // FilteredSubgraph tests passed
    }

    #[test]
    fn test_filter_combination() {
        // Test combining filters with AND/OR operations
        let mut graph = Graph::new_manual();

        let node1 = graph.add_node().unwrap();
        let node2 = graph.add_node().unwrap();

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

        let graph_rc = Arc::new(OwnerCell::new(graph));

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

        // Filter combination tests passed
    }
}
