//! FilterOperations - Specialized interface for filtered subgraph entities
//!
//! This trait provides specialized operations for filtered subgraphs while leveraging
//! our existing efficient SubgraphOperations infrastructure. All filter operations
//! use the same storage pattern with filter-specific metadata and criteria.

use crate::core::traits::SubgraphOperations;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};

/// Specialized operations for filtered subgraph entities
///
/// This trait extends SubgraphOperations with filter-specific functionality.
/// Filtered subgraphs maintain the same efficient HashSet<NodeId> + HashSet<EdgeId> storage
/// with additional metadata for filter criteria and reapplication capabilities.
///
/// # Design Principles
/// - **Same Efficient Storage**: Uses identical storage pattern as base Subgraph
/// - **Filter Metadata**: Stores filter criteria for reapplication and combination
/// - **Algorithm Integration**: Leverages existing filtering algorithms
/// - **Composable Filters**: Can combine and chain multiple filter criteria
pub trait FilterOperations: SubgraphOperations {
    /// Get the filter criteria used to create this filtered subgraph
    ///
    /// # Returns
    /// Reference to the filter criteria applied to create this subgraph
    ///
    /// # Performance
    /// O(1) - Direct field access to existing FilterCriteria
    fn filter_criteria(&self) -> &FilterCriteria;

    /// Reapply the same filter to the current graph state
    ///
    /// # Returns
    /// New filtered subgraph using the same criteria on current graph data
    ///
    /// # Performance
    /// Uses existing efficient filtering algorithms with stored criteria
    fn reapply_filter(&self) -> GraphResult<Box<dyn FilterOperations>>;

    /// Combine this filter with another filter using AND logic
    ///
    /// # Arguments
    /// * `other` - Another filter to combine with
    ///
    /// # Returns
    /// New filtered subgraph containing entities that match BOTH filters
    ///
    /// # Performance
    /// Uses efficient set intersection on filtered results
    fn and_filter(&self, other: &dyn FilterOperations) -> GraphResult<Box<dyn FilterOperations>>;

    /// Combine this filter with another filter using OR logic
    ///
    /// # Arguments
    /// * `other` - Another filter to combine with
    ///
    /// # Returns
    /// New filtered subgraph containing entities that match EITHER filter
    ///
    /// # Performance
    /// Uses efficient set union on filtered results
    fn or_filter(&self, other: &dyn FilterOperations) -> GraphResult<Box<dyn FilterOperations>>;

    /// Create a NOT filter (inverse of this filter)
    ///
    /// # Returns
    /// New filtered subgraph containing entities that DON'T match this filter
    ///
    /// # Performance
    /// Uses efficient set difference against full graph
    fn not_filter(&self) -> GraphResult<Box<dyn FilterOperations>>;

    /// Add additional filter criteria to existing filter
    ///
    /// # Arguments
    /// * `additional_criteria` - New criteria to add to existing filter
    ///
    /// # Returns
    /// New filtered subgraph with combined criteria applied
    ///
    /// # Performance
    /// Creates combined criteria and applies using existing algorithms
    fn add_criteria(
        &self,
        additional_criteria: FilterCriteria,
    ) -> GraphResult<Box<dyn FilterOperations>>;

    /// Get statistics about the filtering results
    ///
    /// # Returns
    /// FilterStats showing filtering effectiveness and coverage
    ///
    /// # Performance
    /// O(1) calculation using existing efficient counts
    fn filter_stats(&self) -> FilterStats {
        // Calculate against full graph for comparison
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        let total_nodes = graph.node_ids().len();
        let total_edges = graph.edge_ids().len();

        FilterStats {
            filtered_nodes: self.node_count(),
            filtered_edges: self.edge_count(),
            total_nodes,
            total_edges,
            node_retention_rate: if total_nodes > 0 {
                self.node_count() as f64 / total_nodes as f64
            } else {
                0.0
            },
            edge_retention_rate: if total_edges > 0 {
                self.edge_count() as f64 / total_edges as f64
            } else {
                0.0
            },
            criteria_complexity: self.filter_criteria().complexity_score(),
        }
    }

    /// Check if a node would pass the filter criteria
    ///
    /// # Arguments
    /// * `node_id` - Node to test against filter
    ///
    /// # Returns
    /// true if the node matches the filter criteria
    ///
    /// # Performance
    /// Uses existing attribute lookup with criteria matching
    fn matches_node_filter(&self, node_id: NodeId) -> GraphResult<bool> {
        self.filter_criteria()
            .matches_node(node_id, &self.graph_ref().borrow())
    }

    /// Check if an edge would pass the filter criteria
    ///
    /// # Arguments
    /// * `edge_id` - Edge to test against filter
    ///
    /// # Returns
    /// true if the edge matches the filter criteria
    ///
    /// # Performance
    /// Uses existing attribute lookup with criteria matching
    fn matches_edge_filter(&self, edge_id: EdgeId) -> GraphResult<bool> {
        self.filter_criteria()
            .matches_edge(edge_id, &self.graph_ref().borrow())
    }
}

/// Filter criteria for creating and managing filtered subgraphs
#[derive(Debug, Clone)]
pub enum FilterCriteria {
    /// Filter nodes by attribute value
    NodeAttributeEquals { name: AttrName, value: AttrValue },
    /// Filter nodes by attribute range (for numeric attributes)
    NodeAttributeRange {
        name: AttrName,
        min: AttrValue,
        max: AttrValue,
    },
    /// Filter edges by attribute value
    EdgeAttributeEquals { name: AttrName, value: AttrValue },
    /// Filter edges by attribute range
    EdgeAttributeRange {
        name: AttrName,
        min: AttrValue,
        max: AttrValue,
    },
    /// Filter nodes by degree (number of connections)
    NodeDegreeRange { min: usize, max: usize },
    /// Combine multiple criteria with AND logic
    And(Vec<FilterCriteria>),
    /// Combine multiple criteria with OR logic
    Or(Vec<FilterCriteria>),
    /// Negate criteria
    Not(Box<FilterCriteria>),
}

impl FilterCriteria {
    /// Calculate complexity score for performance estimation
    pub fn complexity_score(&self) -> usize {
        match self {
            FilterCriteria::NodeAttributeEquals { .. } => 1,
            FilterCriteria::NodeAttributeRange { .. } => 2,
            FilterCriteria::EdgeAttributeEquals { .. } => 1,
            FilterCriteria::EdgeAttributeRange { .. } => 2,
            FilterCriteria::NodeDegreeRange { .. } => 3,
            FilterCriteria::And(criteria) => {
                criteria.iter().map(|c| c.complexity_score()).sum::<usize>() + 1
            }
            FilterCriteria::Or(criteria) => {
                criteria.iter().map(|c| c.complexity_score()).sum::<usize>() + 1
            }
            FilterCriteria::Not(criteria) => criteria.complexity_score() + 1,
        }
    }

    /// Check if a node matches this filter criteria
    pub fn matches_node(
        &self,
        node_id: NodeId,
        graph: &crate::api::graph::Graph,
    ) -> GraphResult<bool> {
        match self {
            FilterCriteria::NodeAttributeEquals { name, value } => {
                if let Some(node_value) = graph.get_node_attr(node_id, name)? {
                    Ok(node_value == *value)
                } else {
                    Ok(false)
                }
            }
            FilterCriteria::NodeAttributeRange { name, min, max } => {
                if let Some(node_value) = graph.get_node_attr(node_id, name)? {
                    Ok(node_value >= *min && node_value <= *max)
                } else {
                    Ok(false)
                }
            }
            FilterCriteria::NodeDegreeRange { min, max } => {
                let degree = graph.degree(node_id)?;
                Ok(degree >= *min && degree <= *max)
            }
            FilterCriteria::And(criteria) => {
                for criterion in criteria {
                    if !criterion.matches_node(node_id, graph)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            FilterCriteria::Or(criteria) => {
                for criterion in criteria {
                    if criterion.matches_node(node_id, graph)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            FilterCriteria::Not(criteria) => Ok(!criteria.matches_node(node_id, graph)?),
            _ => Ok(true), // Node criteria always pass for edges (node filtering already done)
        }
    }

    /// Check if an edge matches this filter criteria
    pub fn matches_edge(
        &self,
        edge_id: EdgeId,
        graph: &crate::api::graph::Graph,
    ) -> GraphResult<bool> {
        match self {
            FilterCriteria::EdgeAttributeEquals { name, value } => {
                if let Some(edge_value) = graph.get_edge_attr(edge_id, name)? {
                    Ok(edge_value == *value)
                } else {
                    Ok(false)
                }
            }
            FilterCriteria::EdgeAttributeRange { name, min, max } => {
                if let Some(edge_value) = graph.get_edge_attr(edge_id, name)? {
                    Ok(edge_value >= *min && edge_value <= *max)
                } else {
                    Ok(false)
                }
            }
            FilterCriteria::And(criteria) => {
                for criterion in criteria {
                    if !criterion.matches_edge(edge_id, graph)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            FilterCriteria::Or(criteria) => {
                for criterion in criteria {
                    if criterion.matches_edge(edge_id, graph)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            FilterCriteria::Not(criteria) => Ok(!criteria.matches_edge(edge_id, graph)?),
            _ => Ok(true), // Node criteria always pass for edges (node filtering already done)
        }
    }
}

/// Statistics about filtering operations and results
#[derive(Debug, Clone)]
pub struct FilterStats {
    /// Number of nodes after filtering
    pub filtered_nodes: usize,
    /// Number of edges after filtering
    pub filtered_edges: usize,
    /// Total nodes in original graph
    pub total_nodes: usize,
    /// Total edges in original graph
    pub total_edges: usize,
    /// Percentage of nodes retained after filtering
    pub node_retention_rate: f64,
    /// Percentage of edges retained after filtering
    pub edge_retention_rate: f64,
    /// Complexity score of the filter criteria
    pub criteria_complexity: usize,
}

impl FilterStats {
    /// Calculate filtering efficiency (how much data was filtered out)
    pub fn filtering_efficiency(&self) -> f64 {
        1.0 - ((self.filtered_nodes + self.filtered_edges) as f64
            / (self.total_nodes + self.total_edges) as f64)
    }

    /// Calculate selectivity (how selective the filter is)
    pub fn selectivity(&self) -> f64 {
        (self.node_retention_rate + self.edge_retention_rate) / 2.0
    }
}
