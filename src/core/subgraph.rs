//! Subgraph - A view into a Graph with full Graph API inheritance
//!
//! *** ARCHITECTURE OVERVIEW ***
//! A Subgraph represents a subset of nodes and edges from a parent Graph.
//! Unlike traditional graph libraries, Subgraphs in Groggy have FULL Graph API.
//!
//! DESIGN PHILOSOPHY:
//! - Subgraph IS-A Graph (through trait or delegation)
//! - All Graph operations work on Subgraphs: filter_nodes, bfs, dfs, etc.
//! - Infinite composability: subgraph.filter_nodes().bfs().filter_edges()
//! - Column access: subgraph[attr_name] -> Vec<AttrValue>
//! - True inheritance enables unprecedented power
//!
//! KEY BENEFITS:
//! - Recursive filtering and analysis
//! - Batch operations at any subgraph level
//! - Consistent API regardless of graph or subgraph context
//! - Performance: operations stay in Rust core

use crate::api::graph::Graph;
use crate::core::traversal::TraversalOptions;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// A Subgraph represents a subset of nodes and edges from a parent Graph
/// with full Graph API capabilities through delegation.
///
/// CORE CONCEPT: Subgraph IS-A Graph
/// - All Graph operations work: filter_nodes, bfs, dfs, algorithms
/// - Column access: subgraph[attr_name] -> Vec<AttrValue>
/// - Infinite composability: subgraph.filter().filter().bfs().set()
/// - Consistent API at every level
#[derive(Debug, Clone)]
pub struct Subgraph {
    /// Reference to the parent graph (shared across all subgraphs)
    /// This enables all Graph operations to work on Subgraphs
    /// Uses RefCell for interior mutability to allow batch operations
    graph: Rc<RefCell<Graph>>,

    /// Set of node IDs that are included in this subgraph
    /// Operations are filtered to only these nodes
    nodes: HashSet<NodeId>,

    /// Set of edge IDs that are included in this subgraph
    /// Usually induced edges (edges between subgraph nodes)
    edges: HashSet<EdgeId>,

    /// Metadata about how this subgraph was created
    /// Examples: "filter_nodes", "bfs_traversal", "batch_selection"
    subgraph_type: String,
}

impl Subgraph {
    /// Create a new Subgraph from a Graph with specific nodes and edges
    pub fn new(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        subgraph_type: String,
    ) -> Self {
        Self {
            graph,
            nodes,
            edges,
            subgraph_type,
        }
    }

    /// Create a new Subgraph with just nodes, calculating induced edges
    pub fn from_nodes(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        subgraph_type: String,
    ) -> GraphResult<Self> {
        let edges = Self::calculate_induced_edges(&graph, &nodes)?;
        Ok(Self::new(graph, nodes, edges, subgraph_type))
    }

    /// Calculate induced edges (edges where both endpoints are in the node set) - O(k) OPTIMIZED
    ///
    /// Uses columnar topology vectors for O(k) performance where k = number of active edges,
    /// which is much better than O(E) over all edges in the graph.
    fn calculate_induced_edges(
        graph: &Rc<RefCell<Graph>>,
        nodes: &HashSet<NodeId>,
    ) -> GraphResult<HashSet<EdgeId>> {
        let mut induced_edges = HashSet::new();
        let graph_borrow = graph.borrow();

        // Get columnar topology vectors (edge_ids, sources, targets) - O(1) if cached
        let (edge_ids, sources, targets) = graph_borrow.get_columnar_topology();

        // Iterate through parallel vectors - O(k) where k = active edges
        for i in 0..edge_ids.len() {
            let edge_id = edge_ids[i];
            let source = sources[i];
            let target = targets[i];

            // O(1) HashSet lookups instead of O(n) Vec::contains
            if nodes.contains(&source) && nodes.contains(&target) {
                induced_edges.insert(edge_id);
            }
        }

        Ok(induced_edges)
    }

    /// Get reference to the parent graph (for read operations)
    pub fn graph(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    /// Get the nodes in this subgraph
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    /// Get the edges in this subgraph  
    pub fn edges(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    /// Get the subgraph type metadata
    pub fn subgraph_type(&self) -> &str {
        &self.subgraph_type
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if a node is in this subgraph
    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains(&node_id)
    }

    /// Check if an edge is in this subgraph
    pub fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains(&edge_id)
    }

    /// Get all node IDs as a Vec (for compatibility with existing APIs)
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().copied().collect()
    }

    /// Get all edge IDs as a Vec (for compatibility with existing APIs)
    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.edges.iter().copied().collect()
    }
}

/// Core Graph operations implemented for Subgraph through delegation
/// This makes Subgraph truly behave like a Graph
impl Subgraph {
    /// Filter nodes within this subgraph using attribute filters
    /// This enables infinite composability: subgraph.filter_nodes().filter_nodes()
    pub fn filter_nodes_by_attributes(
        &self,
        filters: &std::collections::HashMap<AttrName, crate::core::query::AttributeFilter>,
    ) -> GraphResult<Subgraph> {
        let mut filtered_nodes = HashSet::new();
        let graph_borrow = self.graph.borrow();

        // Only consider nodes that are already in this subgraph
        for &node_id in &self.nodes {
            let mut matches_all = true;

            // Check all filters
            for (attr_name, filter) in filters {
                if let Some(attr_value) = graph_borrow.get_node_attr(node_id, attr_name)? {
                    if !filter.matches(&attr_value) {
                        matches_all = false;
                        break;
                    }
                } else {
                    // Node doesn't have the attribute, so it doesn't match
                    matches_all = false;
                    break;
                }
            }

            if matches_all {
                filtered_nodes.insert(node_id);
            }
        }
        drop(graph_borrow); // Release borrow before creating new subgraph

        // Create new subgraph with filtered nodes
        Self::from_nodes(
            self.graph.clone(),
            filtered_nodes,
            format!("{}_attr_filtered", self.subgraph_type),
        )
    }

    /// Filter nodes within this subgraph by a simple attribute value match
    /// This is a convenience method for the most common case
    pub fn filter_nodes_by_attribute(
        &self,
        attr_name: &AttrName,
        attr_value: &AttrValue,
    ) -> GraphResult<Subgraph> {
        let mut filtered_nodes = HashSet::new();
        let graph_borrow = self.graph.borrow();

        // Only consider nodes that are already in this subgraph
        for &node_id in &self.nodes {
            if let Some(node_attr_value) = graph_borrow.get_node_attr(node_id, attr_name)? {
                if &node_attr_value == attr_value {
                    filtered_nodes.insert(node_id);
                }
            }
        }
        drop(graph_borrow); // Release borrow before creating new subgraph

        // Create new subgraph with filtered nodes
        Self::from_nodes(
            self.graph.clone(),
            filtered_nodes,
            format!("{}_value_filtered", self.subgraph_type),
        )
    }

    /// Run BFS traversal starting from a node within this subgraph
    /// The traversal is constrained to nodes in this subgraph
    pub fn bfs(&self, start: NodeId, _options: TraversalOptions) -> GraphResult<Subgraph> {
        // Ensure start node is in this subgraph
        if !self.has_node(start) {
            return Err(GraphError::node_not_found(start, "subgraph BFS"));
        }

        // TODO: Implement constrained BFS within subgraph
        // For now, return a placeholder
        Ok(Subgraph::new(
            self.graph.clone(),
            HashSet::from([start]),
            HashSet::new(),
            format!("{}_bfs", self.subgraph_type),
        ))
    }

    /// Run DFS traversal starting from a node within this subgraph
    pub fn dfs(&self, start: NodeId, _options: TraversalOptions) -> GraphResult<Subgraph> {
        // Ensure start node is in this subgraph
        if !self.has_node(start) {
            return Err(GraphError::node_not_found(start, "subgraph DFS"));
        }

        // TODO: Implement constrained DFS within subgraph
        // For now, return a placeholder
        Ok(Subgraph::new(
            self.graph.clone(),
            HashSet::from([start]),
            HashSet::new(),
            format!("{}_dfs", self.subgraph_type),
        ))
    }

    /// Find connected components within this subgraph
    pub fn connected_components(&self) -> GraphResult<Vec<Subgraph>> {
        // TODO: Implement connected components analysis within subgraph
        // For now, return the subgraph itself as a single component
        Ok(vec![self.clone()])
    }

    /// Check if the subgraph is connected (has exactly one connected component)
    pub fn is_connected(&self) -> GraphResult<bool> {
        let components = self.connected_components()?;
        Ok(components.len() == 1 && !self.nodes.is_empty())
    }
}

/// Column access operations for bulk attribute extraction
impl Subgraph {
    /// Get all values for a node attribute within this subgraph
    /// This is the key column access operation: subgraph[attr_name] -> Vec<AttrValue>
    pub fn get_node_attribute_column(&self, attr_name: &AttrName) -> GraphResult<Vec<AttrValue>> {
        let mut values = Vec::new();
        let graph_borrow = self.graph.borrow();

        for &node_id in &self.nodes {
            if let Some(attr_value) = graph_borrow.get_node_attr(node_id, attr_name)? {
                values.push(attr_value);
            } else {
                // Handle missing attributes - could use None or skip
                // For now, we'll skip missing attributes
            }
        }

        Ok(values)
    }

    /// Get all values for an edge attribute within this subgraph
    pub fn get_edge_attribute_column(&self, attr_name: &AttrName) -> GraphResult<Vec<AttrValue>> {
        let mut values = Vec::new();
        let graph_borrow = self.graph.borrow();

        for &edge_id in &self.edges {
            if let Some(attr_value) = graph_borrow.get_edge_attr(edge_id, attr_name)? {
                values.push(attr_value);
            } else {
                // Handle missing attributes - skip for now
            }
        }

        Ok(values)
    }

    /// Get node attribute values for specific nodes within this subgraph
    /// This enables: g.nodes[[1,2,3]][attr_name] -> [val1, val2, val3]
    pub fn get_node_attributes_for_nodes(
        &self,
        node_ids: &[NodeId],
        attr_name: &AttrName,
    ) -> GraphResult<Vec<AttrValue>> {
        let mut values = Vec::new();
        let graph_borrow = self.graph.borrow();

        for &node_id in node_ids {
            // Ensure the node is in this subgraph
            if !self.has_node(node_id) {
                return Err(GraphError::node_not_found(
                    node_id,
                    "subgraph attribute access",
                ));
            }

            if let Some(attr_value) = graph_borrow.get_node_attr(node_id, attr_name)? {
                values.push(attr_value);
            }
        }

        Ok(values)
    }
}

/// Batch operations for setting attributes on multiple nodes/edges in subgraph
impl Subgraph {
    /// Set an attribute on all nodes in this subgraph
    /// This enables: subgraph.set_node_attribute_bulk("department", "Engineering")
    /// OPTIMIZED: Uses bulk API instead of O(n) individual calls
    pub fn set_node_attribute_bulk(
        &self,
        attr_name: &AttrName,
        attr_value: AttrValue,
    ) -> GraphResult<()> {
        let mut graph_borrow = self.graph.borrow_mut();

        // Use the optimized bulk API instead of individual calls
        let mut attrs_values = std::collections::HashMap::new();
        let node_value_pairs: Vec<(NodeId, AttrValue)> = self
            .nodes
            .iter()
            .map(|&node_id| (node_id, attr_value.clone()))
            .collect();

        attrs_values.insert(attr_name.clone(), node_value_pairs);

        // Single bulk operation instead of O(n) individual calls
        graph_borrow.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set attributes on specific nodes within this subgraph
    /// OPTIMIZED: Uses bulk API instead of O(n) individual calls
    pub fn set_node_attributes_for_nodes(
        &self,
        node_ids: &[NodeId],
        attr_name: &AttrName,
        attr_value: AttrValue,
    ) -> GraphResult<()> {
        // Validate all nodes exist in this subgraph upfront
        for &node_id in node_ids {
            if !self.has_node(node_id) {
                return Err(GraphError::node_not_found(
                    node_id,
                    "subgraph batch operation",
                ));
            }
        }

        // Use optimized bulk API
        let mut graph_borrow = self.graph.borrow_mut();
        let mut attrs_values = std::collections::HashMap::new();
        let node_value_pairs: Vec<(NodeId, AttrValue)> = node_ids
            .iter()
            .map(|&node_id| (node_id, attr_value.clone()))
            .collect();

        attrs_values.insert(attr_name.clone(), node_value_pairs);
        graph_borrow.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set multiple attributes on all nodes in this subgraph
    /// This enables: subgraph.set_bulk({attr1: val1, attr2: val2})
    /// OPTIMIZED: Uses bulk API instead of O(n*m) individual calls
    pub fn set_node_attributes_bulk(
        &self,
        attributes: std::collections::HashMap<AttrName, AttrValue>,
    ) -> GraphResult<()> {
        let mut graph_borrow = self.graph.borrow_mut();

        // Transform to bulk API format: HashMap<AttrName, Vec<(NodeId, AttrValue)>>
        let mut attrs_values = std::collections::HashMap::new();

        for (attr_name, attr_value) in attributes {
            let node_value_pairs: Vec<(NodeId, AttrValue)> = self
                .nodes
                .iter()
                .map(|&node_id| (node_id, attr_value.clone()))
                .collect();
            attrs_values.insert(attr_name, node_value_pairs);
        }

        // Single bulk operation instead of O(n*m) individual calls
        graph_borrow.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set an attribute on all edges in this subgraph
    pub fn set_edge_attribute_bulk(
        &self,
        attr_name: &AttrName,
        attr_value: AttrValue,
    ) -> GraphResult<()> {
        let mut graph_borrow = self.graph.borrow_mut();

        for &edge_id in &self.edges {
            graph_borrow.set_edge_attr(edge_id, attr_name.clone(), attr_value.clone())?;
        }

        Ok(())
    }
}

// Note: Index syntax (subgraph[attr_name]) is not implemented because we can't return
// references to computed Vec<AttrValue>. Use get_node_attribute_column() instead.

impl std::fmt::Display for Subgraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Subgraph(nodes={}, edges={}, type={})",
            self.node_count(),
            self.edge_count(),
            self.subgraph_type
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::types::AttrValue;

    #[test]
    fn test_subgraph_creation() {
        // Test basic subgraph creation and functionality
        let mut graph = Graph::new();

        // Add test nodes
        let node1 = graph.add_node();
        let node2 = graph.add_node();

        // Set attributes
        graph
            .set_node_attr(
                node1,
                "name".to_string(),
                AttrValue::Text("Alice".to_string()),
            )
            .unwrap();
        graph
            .set_node_attr(
                node2,
                "name".to_string(),
                AttrValue::Text("Bob".to_string()),
            )
            .unwrap();

        // Create subgraph
        let graph_rc = Rc::new(RefCell::new(graph));
        let node_subset = HashSet::from([node1, node2]);
        let subgraph = Subgraph::from_nodes(graph_rc, node_subset, "test".to_string()).unwrap();

        // Test basic properties
        assert_eq!(subgraph.node_count(), 2);
        assert!(subgraph.has_node(node1));
        assert!(subgraph.has_node(node2));

        // Test column access
        let names = subgraph
            .get_node_attribute_column(&"name".to_string())
            .unwrap();
        assert_eq!(names.len(), 2);
    }
}
