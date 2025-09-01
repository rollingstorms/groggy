//! SubgraphOperations - Shared interface for all subgraph-like entities
//!
//! This trait provides common operations for all subgraph types while leveraging
//! our existing efficient storage (HashSet<NodeId>, HashSet<EdgeId>) and algorithms.
//! All subgraph types use the same optimized foundation with specialized behaviors.

use crate::core::traits::GraphEntity;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::collections::{HashMap, HashSet};

/// Common operations that all subgraph-like entities support
///
/// This trait provides a unified interface over our existing efficient subgraph storage.
/// All implementations use the same HashSet<NodeId> and HashSet<EdgeId> foundation
/// with our existing optimized algorithms accessed through the Graph reference.
///
/// # Design Principles
/// - **Same Storage**: All subgraphs use HashSet<NodeId> + HashSet<EdgeId> + Rc<RefCell<Graph>>
/// - **Algorithm Reuse**: All operations delegate to existing optimized algorithms
/// - **Zero Copying**: Methods return references to existing data structures
/// - **Trait Objects**: Methods return Box<dyn SubgraphOperations> for composability
pub trait SubgraphOperations: GraphEntity {
    /// Reference to our efficient node set (no copying)
    ///
    /// # Returns
    /// Reference to the HashSet<NodeId> containing this subgraph's nodes
    ///
    /// # Performance
    /// O(1) - Direct reference to existing efficient storage
    fn node_set(&self) -> &HashSet<NodeId>;

    /// Reference to our efficient edge set (no copying)
    ///
    /// # Returns
    /// Reference to the HashSet<EdgeId> containing this subgraph's edges
    ///
    /// # Performance
    /// O(1) - Direct reference to existing efficient storage
    fn edge_set(&self) -> &HashSet<EdgeId>;

    /// Node count using our existing efficient structure
    ///
    /// # Returns
    /// Number of nodes in this subgraph
    ///
    /// # Performance
    /// O(1) - Direct .len() call on HashSet
    fn node_count(&self) -> usize {
        self.node_set().len()
    }

    /// Edge count using our existing efficient structure
    ///
    /// # Returns
    /// Number of edges in this subgraph
    ///
    /// # Performance
    /// O(1) - Direct .len() call on HashSet
    fn edge_count(&self) -> usize {
        self.edge_set().len()
    }

    /// Containment check using our existing HashSet lookups
    ///
    /// # Arguments
    /// * `node_id` - Node to check for membership
    ///
    /// # Returns
    /// true if node is in this subgraph
    ///
    /// # Performance
    /// O(1) - Direct HashSet.contains() call
    fn contains_node(&self, node_id: NodeId) -> bool {
        self.node_set().contains(&node_id)
    }

    /// Edge containment check using our existing HashSet lookups
    ///
    /// # Arguments
    /// * `edge_id` - Edge to check for membership
    ///
    /// # Returns
    /// true if edge is in this subgraph
    ///
    /// # Performance
    /// O(1) - Direct HashSet.contains() call
    fn contains_edge(&self, edge_id: EdgeId) -> bool {
        self.edge_set().contains(&edge_id)
    }

    /// Calculate subgraph density (ratio of actual edges to possible edges)
    ///
    /// # Returns
    /// Density value between 0.0 and 1.0, where 1.0 means fully connected
    ///
    /// # Performance
    /// O(1) - Uses existing efficient node_count() and edge_count() methods
    fn density(&self) -> f64 {
        let node_count = self.node_count();
        let edge_count = self.edge_count();

        if node_count <= 1 {
            return 0.0;
        }

        // Determine if parent graph is directed
        let is_directed = {
            let graph = self.graph_ref();
            let graph_borrowed = graph.borrow();
            graph_borrowed.is_directed()
        };

        let max_possible_edges = if is_directed {
            // For directed graphs: n(n-1)
            node_count * (node_count - 1)
        } else {
            // For undirected graphs: n(n-1)/2
            (node_count * (node_count - 1)) / 2
        };

        if max_possible_edges > 0 {
            edge_count as f64 / max_possible_edges as f64
        } else {
            0.0
        }
    }

    /// Node attribute access using GraphPool (no copying)
    ///
    /// # Arguments
    /// * `node_id` - Node whose attribute to retrieve
    /// * `name` - Attribute name
    ///
    /// # Returns
    /// Optional reference to attribute value in GraphPool
    ///
    /// # Performance
    /// O(1) - Direct lookup in optimized columnar storage
    fn get_node_attribute(
        &self,
        node_id: NodeId,
        name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        let x = graph.pool().get_node_attribute(node_id, name);
        x
    }

    /// Edge attribute access using GraphPool (no copying)
    ///
    /// # Arguments
    /// * `edge_id` - Edge whose attribute to retrieve
    /// * `name` - Attribute name
    ///
    /// # Returns
    /// Optional reference to attribute value in GraphPool
    ///
    /// # Performance
    /// O(1) - Direct lookup in optimized columnar storage
    fn get_edge_attribute(
        &self,
        edge_id: EdgeId,
        name: &AttrName,
    ) -> GraphResult<Option<AttrValue>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        let x = graph.pool().get_edge_attribute(edge_id, name);
        x
    }

    /// Topology queries using our existing efficient algorithms
    ///
    /// # Arguments
    /// * `node_id` - Node whose neighbors to find
    ///
    /// # Returns
    /// Vector of neighbor node IDs within this subgraph
    ///
    /// # Performance
    /// Uses existing optimized neighbor algorithm with subgraph filtering
    fn neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        graph.neighbors_filtered(node_id, self.node_set())
    }

    /// Node degree within this subgraph
    ///
    /// # Arguments
    /// * `node_id` - Node whose degree to calculate
    ///
    /// # Returns
    /// Number of edges connected to this node within the subgraph
    ///
    /// # Performance
    /// Uses existing optimized degree algorithm with subgraph filtering
    fn degree(&self, node_id: NodeId) -> GraphResult<usize> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        graph.degree_filtered(node_id, self.node_set())
    }

    /// Edge endpoints within this subgraph
    ///
    /// # Arguments
    /// * `edge_id` - Edge whose endpoints to retrieve
    ///
    /// # Returns
    /// Tuple of (source, target) node IDs
    ///
    /// # Performance
    /// Uses existing efficient edge endpoint lookup
    fn edge_endpoints(&self, edge_id: EdgeId) -> GraphResult<(NodeId, NodeId)> {
        let graph_ref = self.graph_ref();
        let graph = graph_ref.borrow();
        graph.edge_endpoints(edge_id)
    }

    /// Check for edge between nodes in this subgraph
    ///
    /// # Arguments
    /// * `source` - Source node ID
    /// * `target` - Target node ID
    ///
    /// # Returns
    /// true if edge exists between nodes within this subgraph
    ///
    /// # Performance
    /// Uses existing efficient edge existence check
    fn has_edge_between(&self, source: NodeId, target: NodeId) -> GraphResult<bool> {
        let binding = self.graph_ref();
        let graph = binding.borrow();
        graph.has_edge_between_filtered(source, target, self.edge_set())
    }

    // === SUBGRAPH CREATION (Returns trait objects for composability) ===

    /// Create induced subgraph from node subset
    ///
    /// # Arguments
    /// * `nodes` - Slice of node IDs to include in new subgraph
    ///
    /// # Returns
    /// New subgraph containing only specified nodes and edges between them
    ///
    /// # Performance
    /// Uses existing efficient induced subgraph algorithm
    fn induced_subgraph(&self, nodes: &[NodeId]) -> GraphResult<Box<dyn SubgraphOperations>>;

    /// Create subgraph from edge subset
    ///
    /// # Arguments
    /// * `edges` - Slice of edge IDs to include in new subgraph
    ///
    /// # Returns
    /// New subgraph containing specified edges and their endpoint nodes
    ///
    /// # Performance
    /// Uses existing efficient edge-based subgraph creation
    fn subgraph_from_edges(&self, edges: &[EdgeId]) -> GraphResult<Box<dyn SubgraphOperations>>;

    // === ALGORITHMS (Delegate to existing implementations, return trait objects) ===

    /// Find connected components within this subgraph
    ///
    /// # Returns
    /// Vector of subgraphs, each representing a connected component
    ///
    /// # Performance
    /// Uses existing optimized connected components algorithm on subgraph data
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>>;

    /// Breadth-first search from starting node within this subgraph
    ///
    /// # Arguments
    /// * `start` - Starting node for BFS
    /// * `max_depth` - Optional maximum depth to traverse
    ///
    /// # Returns
    /// Subgraph containing nodes reachable via BFS
    ///
    /// # Performance
    /// Uses existing efficient BFS algorithm with subgraph constraints
    fn bfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>>;

    /// Depth-first search from starting node within this subgraph
    ///
    /// # Arguments
    /// * `start` - Starting node for DFS
    /// * `max_depth` - Optional maximum depth to traverse
    ///
    /// # Returns
    /// Subgraph containing nodes reachable via DFS
    ///
    /// # Performance
    /// Uses existing efficient DFS algorithm with subgraph constraints
    fn dfs(
        &self,
        start: NodeId,
        max_depth: Option<usize>,
    ) -> GraphResult<Box<dyn SubgraphOperations>>;

    /// Find shortest path between nodes within this subgraph
    ///
    /// # Arguments
    /// * `source` - Starting node
    /// * `target` - Destination node
    ///
    /// # Returns
    /// Optional subgraph representing the shortest path (None if no path exists)
    ///
    /// # Performance
    /// Uses existing efficient shortest path algorithm with subgraph constraints
    fn shortest_path_subgraph(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> GraphResult<Option<Box<dyn SubgraphOperations>>>;

    // === HIERARCHICAL OPERATIONS (Integration with GraphPool storage) ===

    /// Collapse this subgraph into a meta-node stored in GraphPool
    ///
    /// # Arguments
    /// * `agg_functions` - Attribute aggregation functions (e.g., "mean", "sum", "count")
    ///
    /// # Returns
    /// NodeId of the created meta-node in GraphPool
    ///
    /// # Storage Integration
    /// - Creates new node in GraphPool
    /// - Stores subgraph reference using SubgraphRef attribute
    /// - Applies aggregation functions using existing bulk operations
    /// - All attributes stored in our efficient columnar storage
    fn collapse_to_node(&self, agg_functions: HashMap<AttrName, String>) -> GraphResult<NodeId> {
        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();

        // Create new meta-node using Graph API
        let meta_node_id = graph.add_node();

        // Store subgraph reference in GraphPool using new subgraph storage capability
        let subgraph_id = graph.pool_mut().store_subgraph(
            self.node_set().clone(),
            self.edge_set().clone(),
            self.entity_type().to_string(),
        )?;

        // Link meta-node to subgraph using our efficient attribute storage
        graph.set_node_attr(
            meta_node_id,
            "contained_subgraph".into(),
            AttrValue::SubgraphRef(subgraph_id),
        )?;

        // Apply aggregation functions using hierarchical operations
        drop(graph); // Release the borrow to avoid double borrowing

        for (attr_name, agg_func_str) in agg_functions {
            // Parse aggregation function
            let agg_func =
                crate::core::hierarchical::AggregationFunction::from_string(&agg_func_str)?;

            // Collect all values for this attribute from nodes in the subgraph
            let mut values = Vec::new();
            let binding = self.graph_ref();
            let graph = binding.borrow();

            for &node_id in self.node_set() {
                if let Some(value) = graph.pool().get_node_attribute(node_id, &attr_name)? {
                    values.push(value);
                }
            }

            // Apply aggregation and store result
            if !values.is_empty() {
                let aggregated_value = agg_func.aggregate(&values)?;
                drop(graph); // Release borrow

                let mut graph = binding.borrow_mut();
                graph.set_node_attr(meta_node_id, attr_name, aggregated_value)?;
            }
        }

        Ok(meta_node_id)
    }

    /// Get parent subgraph (if this is a sub-subgraph)
    ///
    /// # Returns
    /// Optional parent subgraph that contains this one
    fn parent_subgraph(&self) -> Option<Box<dyn SubgraphOperations>> {
        // TODO: Implement parent tracking for hierarchical subgraphs
        None
    }

    /// Get child subgraphs (if this contains other subgraphs)
    ///
    /// # Returns
    /// Vector of child subgraphs contained within this one
    fn child_subgraphs(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // TODO: Implement child tracking for hierarchical subgraphs
        Ok(Vec::new())
    }

    // === BULK ATTRIBUTE OPERATIONS (OPTIMIZED) ===

    /// Set node attributes in bulk using optimized vectorized operations
    ///
    /// # Arguments
    /// * `attrs_values` - HashMap mapping attribute names to vectors of (NodeId, AttrValue) pairs
    ///
    /// # Performance
    /// Uses optimized bulk operations for significant performance gains on large datasets
    fn set_node_attrs(
        &self,
        attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)>>,
    ) -> GraphResult<()> {
        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();

        // Batch validation - check all nodes exist upfront
        for node_values in attrs_values.values() {
            for &(node_id, _) in node_values {
                if !graph.space().contains_node(node_id) {
                    return Err(crate::errors::GraphError::node_not_found(
                        node_id,
                        "set bulk node attributes",
                    )
                    .into());
                }
            }
        }

        // Use optimized vectorized pool operation
        let index_changes = graph.pool_mut().set_bulk_attrs(attrs_values, true);

        // Update space attribute indices in bulk
        for (attr_name, entity_indices) in index_changes {
            for (node_id, new_index) in entity_indices {
                graph
                    .space_mut()
                    .set_node_attr_index(node_id, attr_name.clone(), new_index);
            }
        }

        Ok(())
    }

    /// Set edge attributes in bulk using optimized vectorized operations
    ///
    /// # Arguments
    /// * `attrs_values` - HashMap mapping attribute names to vectors of (EdgeId, AttrValue) pairs
    ///
    /// # Performance
    /// Uses optimized bulk operations for significant performance gains on large datasets
    fn set_edge_attrs(
        &self,
        attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>,
    ) -> GraphResult<()> {
        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();

        // Batch validation - check all edges exist upfront
        for edge_values in attrs_values.values() {
            for &(edge_id, _) in edge_values {
                if !graph.space().contains_edge(edge_id) {
                    return Err(crate::errors::GraphError::edge_not_found(
                        edge_id,
                        "set bulk edge attributes",
                    )
                    .into());
                }
            }
        }

        // Use optimized vectorized pool operation
        let index_changes = graph.pool_mut().set_bulk_attrs(attrs_values, false);

        // Update space attribute indices in bulk
        for (attr_name, entity_indices) in index_changes {
            for (edge_id, new_index) in entity_indices {
                graph
                    .space_mut()
                    .set_edge_attr_index(edge_id, attr_name.clone(), new_index);
            }
        }

        Ok(())
    }

    /// Create table from subgraph nodes with all their attributes
    ///
    /// # Returns
    /// GraphTable with node IDs and all node attributes as columns
    ///
    /// # Performance
    /// Uses optimized columnar access to build table efficiently
    fn nodes_table(&self) -> GraphResult<crate::core::table::GraphTable> {
        let binding = self.graph_ref();
        let graph = binding.borrow();

        // Sort node IDs for consistent table ordering
        let mut sorted_node_ids: Vec<_> = self.node_set().iter().copied().collect();
        sorted_node_ids.sort();

        // Collect all unique attribute names across all nodes
        let mut all_attrs = std::collections::HashSet::new();
        for &node_id in &sorted_node_ids {
            if let Ok(attrs) = graph.get_node_attrs(node_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        // Build columns: node_id first, then attributes
        let mut column_names = vec!["node_id".to_string()];
        column_names.extend(all_attrs.into_iter());

        let mut columns = Vec::new();

        for column_name in &column_names {
            let mut attr_values = Vec::new();

            if column_name == "node_id" {
                // Node ID column
                for &node_id in &sorted_node_ids {
                    attr_values.push(crate::types::AttrValue::Int(node_id as i64));
                }
            } else {
                // Attribute column
                for &node_id in &sorted_node_ids {
                    if let Ok(Some(attr_value)) = graph.get_node_attr(node_id, column_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Use null placeholder for missing attributes
                        attr_values.push(crate::types::AttrValue::Null);
                    }
                }
            }

            let graph_array = crate::core::array::GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }

        // Use existing GraphTable::from_arrays_standalone (no graph reference needed)
        crate::core::table::GraphTable::from_arrays_standalone(columns, Some(column_names))
    }

    /// Create table from subgraph edges with all their attributes  
    ///
    /// # Returns
    /// GraphTable with edge IDs and all edge attributes as columns
    ///
    /// # Performance
    /// Uses optimized columnar access to build table efficiently
    fn edges_table(&self) -> GraphResult<crate::core::table::GraphTable> {
        let binding = self.graph_ref();
        let graph = binding.borrow();

        // Sort edge IDs for consistent table ordering
        let mut sorted_edge_ids: Vec<_> = self.edge_set().iter().copied().collect();
        sorted_edge_ids.sort();

        // Collect all unique attribute names across all edges
        let mut all_attrs = std::collections::HashSet::new();
        for &edge_id in &sorted_edge_ids {
            if let Ok(attrs) = graph.get_edge_attrs(edge_id) {
                for attr_name in attrs.keys() {
                    all_attrs.insert(attr_name.clone());
                }
            }
        }

        // Build columns: edge_id, source, target, then attributes
        let mut column_names = vec![
            "edge_id".to_string(),
            "source".to_string(),
            "target".to_string(),
        ];
        column_names.extend(all_attrs.into_iter());

        let mut columns = Vec::new();

        for column_name in &column_names {
            let mut attr_values = Vec::new();

            if column_name == "edge_id" {
                // Edge ID column
                for &edge_id in &sorted_edge_ids {
                    attr_values.push(crate::types::AttrValue::Int(edge_id as i64));
                }
            } else if column_name == "source" {
                // Source node column
                for &edge_id in &sorted_edge_ids {
                    if let Ok((source, _target)) = graph.edge_endpoints(edge_id) {
                        attr_values.push(crate::types::AttrValue::Int(source as i64));
                    } else {
                        // Use null placeholder for missing endpoints
                        attr_values.push(crate::types::AttrValue::Null);
                    }
                }
            } else if column_name == "target" {
                // Target node column
                for &edge_id in &sorted_edge_ids {
                    if let Ok((_source, target)) = graph.edge_endpoints(edge_id) {
                        attr_values.push(crate::types::AttrValue::Int(target as i64));
                    } else {
                        // Use null placeholder for missing endpoints
                        attr_values.push(crate::types::AttrValue::Null);
                    }
                }
            } else {
                // Attribute column
                for &edge_id in &sorted_edge_ids {
                    if let Ok(Some(attr_value)) = graph.get_edge_attr(edge_id, column_name) {
                        attr_values.push(attr_value);
                    } else {
                        // Use null placeholder for missing attributes
                        attr_values.push(crate::types::AttrValue::Null);
                    }
                }
            }

            let graph_array = crate::core::array::GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }

        // Use existing GraphTable::from_arrays_standalone (no graph reference needed)
        crate::core::table::GraphTable::from_arrays_standalone(columns, Some(column_names))
    }
}

// Note: No default implementations provided to avoid conflicts.
// Each concrete subgraph type implements SubgraphOperations directly.
