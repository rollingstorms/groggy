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
use crate::core::traversal::TraversalEngine;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, EntityId, NodeId, SubgraphId};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

/// Similarity metrics for comparing subgraphs
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityMetric {
    /// Jaccard similarity: |A ∩ B| / |A ∪ B|
    Jaccard,
    /// Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    Dice,
    /// Cosine similarity: A·B / (||A|| * ||B||)
    Cosine,
    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|)
    Overlap,
}

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
    pub fn calculate_induced_edges(
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

    /// Check if the subgraph is connected (has exactly one connected component)
    pub fn is_connected(&self) -> GraphResult<bool> {
        // Use the trait method instead of the old method
        use crate::core::traits::SubgraphOperations;
        let components = SubgraphOperations::connected_components(self)?;
        Ok(components.len() == 1 && !self.nodes.is_empty())
    }

    /// Check if there is a path between two nodes within this subgraph
    ///
    /// This is more efficient than `shortest_path_subgraph` when you only need
    /// to know if a path exists, not the actual path.
    ///
    /// # Arguments
    /// * `node1_id` - The starting node ID
    /// * `node2_id` - The destination node ID
    ///
    /// # Returns
    /// * `Ok(true)` if a path exists between the nodes within this subgraph
    /// * `Ok(false)` if no path exists or either node is not in this subgraph
    /// * `Err(GraphError)` if there's an error during traversal
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use groggy::core::subgraph::Subgraph;
    /// # use groggy::errors::GraphResult;
    /// # fn example() -> GraphResult<()> {
    /// # let subgraph: Subgraph = todo!(); // Placeholder for actual subgraph
    /// // Check if there's a path between node 1 and node 5 in the subgraph
    /// let path_exists = subgraph.has_path(1, 5)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn has_path(&self, node1_id: NodeId, node2_id: NodeId) -> GraphResult<bool> {
        // Quick checks first
        if node1_id == node2_id {
            return Ok(self.nodes.contains(&node1_id));
        }

        if !self.nodes.contains(&node1_id) || !self.nodes.contains(&node2_id) {
            return Ok(false);
        }

        // Use BFS to check for reachability - more efficient than Dijkstra for just connectivity
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(node1_id);
        visited.insert(node1_id);

        let graph = self.graph.borrow();

        while let Some(current_node) = queue.pop_front() {
            if current_node == node2_id {
                return Ok(true);
            }

            // Get neighbors of current node
            if let Ok(neighbors) = graph.neighbors(current_node) {
                for neighbor in neighbors {
                    // Only consider neighbors that are in this subgraph and not yet visited
                    if self.nodes.contains(&neighbor) && !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        Ok(false)
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
        let graph_borrow = self.graph.borrow_mut();

        // Use the optimized bulk API instead of individual calls
        let mut attrs_values = std::collections::HashMap::new();
        let node_value_pairs: Vec<(NodeId, AttrValue)> = self
            .nodes
            .iter()
            .map(|&node_id| (node_id, attr_value.clone()))
            .collect();

        attrs_values.insert(attr_name.clone(), node_value_pairs);

        // Single bulk operation instead of O(n) individual calls - now using trait system
        drop(graph_borrow); // Release the borrow before calling trait method
        self.set_node_attrs(attrs_values)?;

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
        let graph_borrow = self.graph.borrow_mut();
        let mut attrs_values = std::collections::HashMap::new();
        let node_value_pairs: Vec<(NodeId, AttrValue)> = node_ids
            .iter()
            .map(|&node_id| (node_id, attr_value.clone()))
            .collect();

        attrs_values.insert(attr_name.clone(), node_value_pairs);
        drop(graph_borrow); // Release the borrow before calling trait method
        self.set_node_attrs(attrs_values)?;

        Ok(())
    }

    /// Set multiple attributes on all nodes in this subgraph
    /// This enables: subgraph.set_bulk({attr1: val1, attr2: val2})
    /// OPTIMIZED: Uses bulk API instead of O(n*m) individual calls
    pub fn set_node_attributes_bulk(
        &self,
        attributes: std::collections::HashMap<AttrName, AttrValue>,
    ) -> GraphResult<()> {
        let graph_borrow = self.graph.borrow_mut();

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

        // Single bulk operation instead of O(n*m) individual calls - now using trait system
        drop(graph_borrow); // Release the borrow before calling trait method
        self.set_node_attrs(attrs_values)?;

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

    // === STRUCTURAL METRICS ===
    // Remember: All operations are on this subgraph (active nodes/edges)

    /// Calculate clustering coefficient for a node or average for all nodes
    /// Formula: 2 * triangles / (degree * (degree - 1)) for directed graphs
    pub fn clustering_coefficient(&self, node_id: Option<NodeId>) -> GraphResult<f64> {
        match node_id {
            Some(nid) => {
                if !self.has_node(nid) {
                    return Err(GraphError::InvalidInput(format!(
                        "Node {} not in this subgraph",
                        nid
                    )));
                }
                self.calculate_node_clustering_coefficient(nid)
            }
            None => {
                // Average clustering coefficient for all nodes in subgraph
                let mut total = 0.0;
                let mut count = 0;

                for &node_id in &self.nodes {
                    let coefficient = self.calculate_node_clustering_coefficient(node_id)?;
                    total += coefficient;
                    count += 1;
                }

                if count == 0 {
                    Ok(0.0)
                } else {
                    Ok(total / count as f64)
                }
            }
        }
    }

    /// Helper method to calculate clustering coefficient for a single node
    fn calculate_node_clustering_coefficient(&self, node_id: NodeId) -> GraphResult<f64> {
        let graph = self.graph.borrow();

        // Get neighbors of this node that are also in the subgraph
        let neighbors = graph.neighbors_filtered(node_id, &self.nodes)?;
        let degree = neighbors.len();

        if degree < 2 {
            return Ok(0.0); // Clustering coefficient is 0 for nodes with degree < 2
        }

        // Count triangles: edges between neighbors
        let mut triangles = 0;
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                if graph.has_edge_between_filtered(neighbors[i], neighbors[j], &self.edges)? {
                    triangles += 1;
                }
            }
        }

        // Calculate clustering coefficient
        let possible_edges = degree * (degree - 1) / 2; // For undirected graphs
        Ok(triangles as f64 / possible_edges as f64)
    }

    /// Calculate global clustering coefficient (transitivity)
    /// Formula: 3 * triangles / triads
    pub fn transitivity(&self) -> GraphResult<f64> {
        let mut total_triangles = 0;
        let mut total_triads = 0;

        let graph = self.graph.borrow();

        // Count each triangle only once by using node ordering
        let mut nodes_vec: Vec<_> = self.nodes.iter().copied().collect();
        nodes_vec.sort();

        for (i, &node_a) in nodes_vec.iter().enumerate() {
            let neighbors_a = graph.neighbors_filtered(node_a, &self.nodes)?;

            for (j, &node_b) in nodes_vec.iter().enumerate().skip(i + 1) {
                if !neighbors_a.contains(&node_b) {
                    continue; // No edge between a and b
                }

                for &node_c in nodes_vec.iter().skip(j + 1) {
                    if neighbors_a.contains(&node_c) {
                        let neighbors_b = graph.neighbors_filtered(node_b, &self.nodes)?;
                        if neighbors_b.contains(&node_c) {
                            // Triangle found: a-b-c
                            total_triangles += 1;
                        }
                    }
                }
            }
        }

        // Count triads (connected triples)
        for &node_id in &self.nodes {
            let neighbors = graph.neighbors_filtered(node_id, &self.nodes)?;
            let degree = neighbors.len();

            if degree >= 2 {
                total_triads += degree * (degree - 1) / 2;
            }
        }

        if total_triads == 0 {
            Ok(0.0)
        } else {
            Ok(3.0 * total_triangles as f64 / total_triads as f64)
        }
    }

    /// Calculate subgraph density
    /// Formula: 2 * edges / (nodes * (nodes - 1)) for undirected graphs
    pub fn density(&self) -> f64 {
        let node_count = self.nodes.len();

        if node_count < 2 {
            return 0.0;
        }

        let edge_count = self.edges.len();
        let max_possible_edges = node_count * (node_count - 1) / 2; // Undirected graph

        edge_count as f64 / max_possible_edges as f64
    }

    // === SUBGRAPH SET OPERATIONS ===

    /// Merge with another subgraph (union operation)
    /// Returns new subgraph containing all nodes and edges from both subgraphs
    pub fn merge_with(&self, other: &Subgraph) -> GraphResult<Subgraph> {
        // Union of nodes and edges
        let mut merged_nodes = self.nodes.clone();
        merged_nodes.extend(&other.nodes);

        let mut merged_edges = self.edges.clone();
        merged_edges.extend(&other.edges);

        // Create merged subgraph with union of nodes and edges
        Ok(Subgraph::new(
            self.graph.clone(),
            merged_nodes,
            merged_edges,
            format!("merge_{}_{}", self.subgraph_type, other.subgraph_type),
        ))
    }

    /// Intersect with another subgraph
    /// Returns new subgraph containing only nodes and edges present in both subgraphs
    pub fn intersect_with(&self, other: &Subgraph) -> GraphResult<Subgraph> {
        // Intersection of nodes
        let intersected_nodes: HashSet<NodeId> =
            self.nodes.intersection(&other.nodes).copied().collect();

        // Intersection of edges
        let intersected_edges: HashSet<EdgeId> =
            self.edges.intersection(&other.edges).copied().collect();

        Ok(Subgraph::new(
            self.graph.clone(),
            intersected_nodes,
            intersected_edges,
            format!(
                "intersection_{}_{}",
                self.subgraph_type, other.subgraph_type
            ),
        ))
    }

    /// Subtract another subgraph from this one (difference operation)
    /// Returns new subgraph with other's nodes and edges removed
    pub fn subtract_from(&self, other: &Subgraph) -> GraphResult<Subgraph> {
        // Remove other's nodes from this subgraph
        let remaining_nodes: HashSet<NodeId> =
            self.nodes.difference(&other.nodes).copied().collect();

        // Remove other's edges from this subgraph
        let remaining_edges: HashSet<EdgeId> =
            self.edges.difference(&other.edges).copied().collect();

        Ok(Subgraph::new(
            self.graph.clone(),
            remaining_nodes,
            remaining_edges,
            format!("difference_{}_{}", self.subgraph_type, other.subgraph_type),
        ))
    }

    /// Calculate similarity with another subgraph using specified metric
    pub fn calculate_similarity(
        &self,
        other: &Subgraph,
        metric: SimilarityMetric,
    ) -> GraphResult<f64> {
        match metric {
            SimilarityMetric::Jaccard => {
                // |A ∩ B| / |A ∪ B| for nodes
                let intersection_size = self.nodes.intersection(&other.nodes).count();
                let union_size = self.nodes.union(&other.nodes).count();

                if union_size == 0 {
                    Ok(0.0)
                } else {
                    Ok(intersection_size as f64 / union_size as f64)
                }
            }
            SimilarityMetric::Dice => {
                // 2 * |A ∩ B| / (|A| + |B|) for nodes
                let intersection_size = self.nodes.intersection(&other.nodes).count();
                let total_size = self.nodes.len() + other.nodes.len();

                if total_size == 0 {
                    Ok(0.0)
                } else {
                    Ok(2.0 * intersection_size as f64 / total_size as f64)
                }
            }
            SimilarityMetric::Overlap => {
                // |A ∩ B| / min(|A|, |B|) for nodes
                let intersection_size = self.nodes.intersection(&other.nodes).count();
                let min_size = self.nodes.len().min(other.nodes.len());

                if min_size == 0 {
                    Ok(0.0)
                } else {
                    Ok(intersection_size as f64 / min_size as f64)
                }
            }
            SimilarityMetric::Cosine => {
                // Simplified cosine similarity based on node presence (binary vectors)
                let intersection_size = self.nodes.intersection(&other.nodes).count();
                let magnitude_product =
                    (self.nodes.len() as f64).sqrt() * (other.nodes.len() as f64).sqrt();

                if magnitude_product == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(intersection_size as f64 / magnitude_product)
                }
            }
        }
    }

    /// Find overlapping regions with multiple other subgraphs
    /// Returns vector of intersection subgraphs with each input subgraph
    pub fn find_overlaps(&self, others: Vec<&Subgraph>) -> GraphResult<Vec<Subgraph>> {
        let mut overlaps = Vec::new();

        for other in others {
            let overlap = self.intersect_with(other)?;
            if !overlap.nodes.is_empty() || !overlap.edges.is_empty() {
                overlaps.push(overlap);
            }
        }

        Ok(overlaps)
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
        let filtered_nodes: HashSet<NodeId> = nodes
            .iter()
            .filter(|&&node_id| self.nodes.contains(&node_id))
            .cloned()
            .collect();

        // Create induced subgraph using existing method
        let induced = Subgraph::from_nodes(
            self.graph.clone(),
            filtered_nodes,
            format!("{}_induced", self.subgraph_type),
        )?;

        Ok(Box::new(induced))
    }

    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>> {
        // Filter to edges that exist in this subgraph
        let filtered_edges: HashSet<EdgeId> = edges
            .iter()
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
            format!("{}_from_edges", self.subgraph_type),
        );

        Ok(Box::new(edge_subgraph))
    }

    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // Use existing efficient TraversalEngine for connected components
        let graph = self.graph.borrow();
        let nodes_vec: Vec<NodeId> = self.nodes.iter().cloned().collect();
        let options = crate::core::traversal::TraversalOptions::default();

        // Use TraversalEngine directly - no Graph API indirection needed
        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.connected_components_for_nodes(
            &graph.pool(),
            graph.space(),
            nodes_vec,
            options,
        )?;

        let mut component_subgraphs = Vec::new();
        for (i, component) in result.components.into_iter().enumerate() {
            let component_nodes: std::collections::HashSet<NodeId> =
                component.nodes.into_iter().collect();
            let component_edges: std::collections::HashSet<EdgeId> =
                component.edges.into_iter().collect();

            let component_subgraph = Subgraph::new(
                self.graph.clone(),
                component_nodes,
                component_edges,
                format!("{}_component_{}", self.subgraph_type, i),
            );
            component_subgraphs.push(Box::new(component_subgraph) as Box<dyn SubgraphOperations>);
        }

        Ok(component_subgraphs)
    }

    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(GraphError::NodeNotFound {
                node_id: start,
                operation: "bfs_subgraph".to_string(),
                suggestion: "Ensure start node is within this subgraph".to_string(),
            });
        }

        // Use existing efficient TraversalEngine for BFS
        let graph = self.graph.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.bfs(&graph.pool(), graph.space(), start, options)?;

        // Filter result to nodes that exist in this subgraph
        let filtered_nodes: std::collections::HashSet<NodeId> = result
            .nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: std::collections::HashSet<EdgeId> = result
            .edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        let bfs_subgraph = Subgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            format!("{}_bfs_from_{}", self.subgraph_type, start),
        );

        Ok(Box::new(bfs_subgraph))
    }

    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>> {
        if !self.nodes.contains(&start) {
            return Err(GraphError::NodeNotFound {
                node_id: start,
                operation: "dfs_subgraph".to_string(),
                suggestion: "Ensure start node is within this subgraph".to_string(),
            });
        }

        // Use existing efficient TraversalEngine for DFS
        let graph = self.graph.borrow_mut();
        let mut options = crate::core::traversal::TraversalOptions::default();
        if let Some(depth) = max_depth {
            options.max_depth = Some(depth);
        }

        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let result = traversal_engine.dfs(&graph.pool(), graph.space(), start, options)?;

        // Filter result to nodes that exist in this subgraph
        let filtered_nodes: std::collections::HashSet<NodeId> = result
            .nodes
            .into_iter()
            .filter(|node| self.nodes.contains(node))
            .collect();
        let filtered_edges: std::collections::HashSet<EdgeId> = result
            .edges
            .into_iter()
            .filter(|edge| self.edges.contains(edge))
            .collect();

        let dfs_subgraph = Subgraph::new(
            self.graph.clone(),
            filtered_nodes,
            filtered_edges,
            format!("{}_dfs_from_{}", self.subgraph_type, start),
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

        // Use existing efficient TraversalEngine for shortest path
        let graph = self.graph.borrow_mut();
        let options = crate::core::traversal::PathFindingOptions::default();

        // Use TraversalEngine directly
        let mut traversal_engine = TraversalEngine::new();
        let x = if let Some(path_result) =
            traversal_engine.shortest_path(&graph.pool(), graph.space(), source, target, options)?
        {
            // Filter path to nodes/edges that exist in this subgraph
            let filtered_nodes: std::collections::HashSet<NodeId> = path_result
                .nodes
                .into_iter()
                .filter(|node| self.nodes.contains(node))
                .collect();
            let filtered_edges: std::collections::HashSet<EdgeId> = path_result
                .edges
                .into_iter()
                .filter(|edge| self.edges.contains(edge))
                .collect();

            if !filtered_nodes.is_empty() {
                let path_subgraph = Subgraph::new(
                    self.graph.clone(),
                    filtered_nodes,
                    filtered_edges,
                    format!("{}_path_{}_{}", self.subgraph_type, source, target),
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

    #[test]
    fn test_subgraph_algorithms() {
        // Test the core SubgraphOperations algorithms
        let mut graph = Graph::new();

        // Create a small connected graph for testing
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();
        let node4 = graph.add_node(); // Isolated node to test components

        let edge1 = graph.add_edge(node1, node2).unwrap();
        let edge2 = graph.add_edge(node2, node3).unwrap();
        let _edge3 = graph.add_edge(node3, node1).unwrap(); // Triangle

        let graph_rc = Rc::new(RefCell::new(graph));
        let node_subset = HashSet::from([node1, node2, node3, node4]);
        let subgraph = Subgraph::from_nodes(graph_rc, node_subset, "test".to_string()).unwrap();

        // Test BFS subgraph
        let bfs_result = subgraph.bfs(node1, Some(2)).unwrap();
        assert!(bfs_result.contains_node(node1));
        println!("BFS test: contains {} nodes", bfs_result.node_count());

        // Test DFS subgraph
        let dfs_result = subgraph.dfs(node1, Some(2)).unwrap();
        assert!(dfs_result.contains_node(node1));
        println!("DFS test: contains {} nodes", dfs_result.node_count());

        // Test connected components
        use crate::core::traits::SubgraphOperations;
        let components = SubgraphOperations::connected_components(&subgraph).unwrap();
        println!("Found {} connected components", components.len());
        assert!(components.len() >= 1); // At least the triangle should form one component

        // Test induced subgraph
        let induced = subgraph.induced_subgraph(&[node1, node2]).unwrap();
        assert_eq!(induced.node_count(), 2);
        assert!(induced.contains_node(node1));
        assert!(induced.contains_node(node2));

        // Test subgraph from edges
        let edge_subset = [edge1, edge2];
        let edge_subgraph = subgraph.subgraph_from_edges(&edge_subset).unwrap();
        assert!(edge_subgraph.contains_edge(edge1));
        assert!(edge_subgraph.contains_edge(edge2));

        // Test shortest path (if path exists)
        if let Some(path) = subgraph.shortest_path_subgraph(node1, node3).unwrap() {
            assert!(path.contains_node(node1));
            assert!(path.contains_node(node3));
            println!("Shortest path found with {} nodes", path.node_count());
        }
    }

    #[test]
    fn test_structural_metrics() {
        // Create a triangle graph for testing clustering coefficient
        let mut graph = Graph::new();
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();

        // Create edges to form a triangle
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();
        graph.add_edge(node3, node1).unwrap();

        // Create subgraph with all nodes
        let graph_rc = Rc::new(RefCell::new(graph));
        let all_nodes = HashSet::from([node1, node2, node3]);
        let subgraph = Subgraph::from_nodes(graph_rc, all_nodes, "triangle".to_string()).unwrap();

        // Test density - triangle graph should have density = 1.0
        let density = subgraph.density();
        assert!((density - 1.0).abs() < f64::EPSILON);

        // Test connectivity
        assert!(subgraph.is_connected().unwrap());

        // Test clustering coefficient - each node should have coefficient = 1.0 (complete triangle)
        let avg_clustering = subgraph.clustering_coefficient(None).unwrap();
        assert!((avg_clustering - 1.0).abs() < f64::EPSILON);

        // Test transitivity
        let transitivity = subgraph.transitivity().unwrap();
        assert!((transitivity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_subgraph_set_operations() {
        let mut graph = Graph::new();
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();
        let node4 = graph.add_node();

        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();
        graph.add_edge(node3, node4).unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));

        // Create two overlapping subgraphs
        let subgraph1_nodes = HashSet::from([node1, node2, node3]);
        let subgraph1 =
            Subgraph::from_nodes(graph_rc.clone(), subgraph1_nodes, "sub1".to_string()).unwrap();

        let subgraph2_nodes = HashSet::from([node2, node3, node4]);
        let subgraph2 =
            Subgraph::from_nodes(graph_rc.clone(), subgraph2_nodes, "sub2".to_string()).unwrap();

        // Test merge (union)
        let merged = subgraph1.merge_with(&subgraph2).unwrap();
        assert_eq!(merged.node_count(), 4); // All 4 nodes
        assert!(merged.has_node(node1));
        assert!(merged.has_node(node4));

        // Test intersection
        let intersection = subgraph1.intersect_with(&subgraph2).unwrap();
        assert_eq!(intersection.node_count(), 2); // node2, node3
        assert!(intersection.has_node(node2));
        assert!(intersection.has_node(node3));
        assert!(!intersection.has_node(node1));
        assert!(!intersection.has_node(node4));

        // Test subtraction
        let difference = subgraph1.subtract_from(&subgraph2).unwrap();
        assert_eq!(difference.node_count(), 1); // only node1
        assert!(difference.has_node(node1));
        assert!(!difference.has_node(node2));

        // Test similarity metrics
        let jaccard = subgraph1
            .calculate_similarity(&subgraph2, SimilarityMetric::Jaccard)
            .unwrap();
        assert!((jaccard - 0.5).abs() < f64::EPSILON); // |intersection| / |union| = 2/4 = 0.5

        let dice = subgraph1
            .calculate_similarity(&subgraph2, SimilarityMetric::Dice)
            .unwrap();
        assert!((dice - 2.0 / 3.0).abs() < f64::EPSILON); // 2 * 2 / (3 + 3) = 4/6 = 2/3
    }
}
