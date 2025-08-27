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
use crate::core::traits::{GraphEntity, SubgraphOperations};
use crate::core::traversal::TraversalOptions;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, EntityId, NodeId, SubgraphId};
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

    /// Unique identifier for this subgraph (for GraphEntity trait)
    /// TODO: Generate proper IDs through GraphPool storage
    subgraph_id: SubgraphId,
}

impl Subgraph {
    /// Create a new Subgraph from a Graph with specific nodes and edges
    pub fn new(
        graph: Rc<RefCell<Graph>>,
        nodes: HashSet<NodeId>,
        edges: HashSet<EdgeId>,
        subgraph_type: String,
    ) -> Self {
        // TODO: Generate proper subgraph ID through GraphPool
        // For now, use a simple hash-based ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        
        // Hash the node and edge sets by iterating over them in sorted order
        let mut sorted_nodes: Vec<_> = nodes.iter().collect();
        sorted_nodes.sort();
        for node in sorted_nodes {
            node.hash(&mut hasher);
        }
        
        let mut sorted_edges: Vec<_> = edges.iter().collect();
        sorted_edges.sort();
        for edge in sorted_edges {
            edge.hash(&mut hasher);
        }
        
        subgraph_type.hash(&mut hasher);
        let subgraph_id = hasher.finish() as SubgraphId;

        Self {
            graph,
            nodes,
            edges,
            subgraph_type,
            subgraph_id,
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
        if self.nodes.is_empty() {
            return Ok(vec![]);
        }
        
        // Use TraversalEngine via Graph method
        let mut graph = self.graph.borrow_mut();
        let nodes: Vec<NodeId> = self.nodes.iter().copied().collect();
        
        let result = graph.connected_components_for_subgraph(nodes)?;
        
        // Convert ConnectedComponentsResult to Vec<Subgraph>
        let mut components = Vec::new();
        for component_result in result.components {
            // Get nodes in this component
            let component_nodes: HashSet<NodeId> = component_result.nodes.into_iter().collect();
            
            // Find edges that are within this component
            let mut component_edges = HashSet::new();
            for &edge_id in &self.edges {
                if let Ok((source, target)) = graph.edge_endpoints(edge_id) {
                    if component_nodes.contains(&source) && component_nodes.contains(&target) {
                        component_edges.insert(edge_id);
                    }
                }
            }
            
            // Create subgraph for this component
            components.push(Subgraph::new(
                self.graph.clone(),
                component_nodes,
                component_edges,
                format!("{}_component_{}", self.subgraph_type, components.len()),
            ));
        }
        
        Ok(components)
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

/// GraphEntity trait implementation for Subgraph
/// This integrates Subgraph into the universal entity system
impl GraphEntity for Subgraph {
    fn entity_id(&self) -> EntityId {
        EntityId::Subgraph(self.subgraph_id)
    }

    fn entity_type(&self) -> &'static str {
        "subgraph"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return node entities contained in this subgraph
        // TODO: Create EntityNode wrappers for the nodes in this subgraph
        Ok(Vec::new()) // Placeholder for now
    }

    fn summary(&self) -> String {
        format!(
            "Subgraph(type={}, id={}, nodes={}, edges={})",
            self.subgraph_type,
            self.subgraph_id,
            self.nodes.len(),
            self.edges.len()
        )
    }
}

/// SubgraphOperations trait implementation for Subgraph
/// This provides the standard subgraph interface using existing efficient storage
impl SubgraphOperations for Subgraph {
    fn node_set(&self) -> &HashSet<NodeId> {
        &self.nodes
    }

    fn edge_set(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to nodes that exist in this subgraph
        let filtered_nodes: HashSet<NodeId> = nodes.iter()
            .filter(|&&node_id| self.nodes.contains(&node_id))
            .cloned()
            .collect();

        // Create induced subgraph using existing method
        let induced = Subgraph::from_nodes(
            self.graph.clone(),
            filtered_nodes,
            format!("{}_induced", self.subgraph_type)
        )?;

        Ok(Box::new(induced))
    }

    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to edges that exist in this subgraph
        let filtered_edges: HashSet<EdgeId> = edges.iter()
            .filter(|&&edge_id| self.edges.contains(&edge_id))
            .cloned()
            .collect();

        // Calculate nodes from edge endpoints using existing method
        let mut endpoint_nodes = HashSet::new();
        let graph_borrow = self.graph.borrow();
        for &edge_id in &filtered_edges {
            if let Ok((source, target)) = graph_borrow.edge_endpoints(edge_id) {
                if self.nodes.contains(&source) {
                    endpoint_nodes.insert(source);
                }
                if self.nodes.contains(&target) {
                    endpoint_nodes.insert(target);
                }
            }
        }

        let edge_subgraph = Subgraph::new(
            self.graph.clone(),
            endpoint_nodes,
            filtered_edges,
            format!("{}_from_edges", self.subgraph_type)
        );

        Ok(Box::new(edge_subgraph))
    }

    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // Use existing efficient TraversalEngine for connected components
        let mut graph = self.graph.borrow_mut();
        let nodes_vec: Vec<NodeId> = self.nodes.iter().cloned().collect();
        let options = crate::core::traversal::TraversalOptions::default();
        
        let result = graph.traversal_engine.connected_components_for_nodes(
            &graph.pool.borrow(),
            &graph.space,
            nodes_vec,
            options
        )?;

        let mut component_subgraphs = Vec::new();
        for (i, component) in result.components.into_iter().enumerate() {
            let component_nodes: std::collections::HashSet<NodeId> = component.nodes.into_iter().collect();
            let component_edges: std::collections::HashSet<EdgeId> = component.edges.into_iter().collect();
            
            let component_subgraph = Subgraph::new(
                self.graph.clone(),
                component_nodes,
                component_edges,
                format!("{}_component_{}", self.subgraph_type, i)
            );
            component_subgraphs.push(Box::new(component_subgraph) as Box<dyn SubgraphOperations>);
        }

        Ok(component_subgraphs)
    }

    fn bfs_subgraph(&self, start: NodeId, max_depth: Option<usize>) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(GraphError::NodeNotFound { 
                node_id: start,
                operation: "bfs_subgraph".to_string(),
                suggestion: "Ensure start node is within this subgraph".to_string(),
            });
        }

        // Use existing efficient TraversalEngine for BFS
        let mut graph = self.graph.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        
        let result = graph.traversal_engine.bfs(
            &graph.pool.borrow(),
            &mut graph.space,
            start,
            options
        )?;

        // Filter result to nodes that exist in this subgraph
        let filtered_nodes: std::collections::HashSet<NodeId> = result.nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: std::collections::HashSet<EdgeId> = result.edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        let bfs_subgraph = Subgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            format!("{}_bfs_from_{}", self.subgraph_type, start)
        );

        Ok(Box::new(bfs_subgraph))
    }

    fn dfs_subgraph(&self, start: NodeId, max_depth: Option<usize>) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(GraphError::NodeNotFound { 
                node_id: start,
                operation: "dfs_subgraph".to_string(),
                suggestion: "Ensure start node is within this subgraph".to_string(),
            });
        }

        // Use existing efficient TraversalEngine for DFS
        let mut graph = self.graph.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }
        
        let result = graph.traversal_engine.dfs(
            &graph.pool.borrow(),
            &mut graph.space,
            start,
            options
        )?;

        // Filter result to nodes that exist in this subgraph
        let filtered_nodes: std::collections::HashSet<NodeId> = result.nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: std::collections::HashSet<EdgeId> = result.edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        let dfs_subgraph = Subgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            format!("{}_dfs_from_{}", self.subgraph_type, start)
        );

        Ok(Box::new(dfs_subgraph))
    }

    fn shortest_path_subgraph(&self, source: NodeId, target: NodeId) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        if !self.nodes.contains(&source) || !self.nodes.contains(&target) {
            return Ok(None);
        }

        // Use existing efficient TraversalEngine for shortest path
        let mut graph = self.graph.borrow_mut();
        let options = crate::core::traversal::PathFindingOptions::default();
        
        if let Some(path_result) = graph.traversal_engine.shortest_path(
            &graph.pool.borrow(),
            &mut graph.space,
            source,
            target,
            options
        )? {
            // Filter path to nodes/edges that exist in this subgraph
            let filtered_nodes: std::collections::HashSet<NodeId> = path_result.nodes
                .into_iter()
                .filter(|node| self.nodes.contains(node))
                .collect();
            let filtered_edges: std::collections::HashSet<EdgeId> = path_result.edges
                .into_iter()
                .filter(|edge| self.edges.contains(edge))
                .collect();

            if !filtered_nodes.is_empty() {
                let path_subgraph = Subgraph::new(
                    self.graph.clone(),
                    filtered_nodes,
                    filtered_edges,
                    format!("{}_path_{}_{}", self.subgraph_type, source, target)
                );
                Ok(Some(Box::new(path_subgraph)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
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
