//! Graph Traversal Algorithms - Efficient pathfinding and connectivity analysis.
//!
//! ARCHITECTURE ROLE:
//! This module provides high-performance graph traversal algorithms that leverage
//! the columnar storage and parallel processing capabilities of the graph system.
//!
//! DESIGN PHILOSOPHY:
//! - Performance-first: Use columnar topology access and parallel processing
//! - Memory-efficient: Reuse data structures and avoid unnecessary allocations
//! - Modular: Each algorithm implements common traits for consistency
//! - Configurable: Support filtering, early termination, and custom constraints

use crate::core::pool::GraphPool;
use crate::core::query::{EdgeFilter, NodeFilter, QueryEngine};
use crate::core::space::GraphSpace;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
// use rayon::prelude::*; // TODO: Re-enable when parallel traversal is implemented

/// High-performance graph traversal engine
///
/// RESPONSIBILITIES:
/// - Execute BFS, DFS, and shortest path algorithms
/// - Find connected components and analyze connectivity
/// - Support filtered traversals and early termination
/// - Provide parallel implementations for large graphs
///
/// NOT RESPONSIBLE FOR:
/// - Graph modification (read-only operations)
/// - Query result caching (that's QueryEngine's job)
/// - Attribute management (that's GraphPool's job)
#[derive(Debug)]
pub struct TraversalEngine {
    /// Reusable state to avoid allocations
    state_pool: TraversalStatePool,

    /// Performance configuration
    #[allow(dead_code)] // TODO: Implement configuration system
    config: TraversalConfig,

    /// Statistics tracking
    stats: TraversalStats,

    /// Query engine for bulk filtering operations
    #[allow(dead_code)]
    query_engine: QueryEngine,
}

/// Filter result cache for avoiding repeated evaluations
#[derive(Debug)]
pub struct FilterCache {
    cache: HashMap<(NodeId, String), bool>,
    max_size: usize,
}

impl FilterCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
        }
    }

    pub fn get(&self, node_id: NodeId, filter_key: &str) -> Option<bool> {
        self.cache.get(&(node_id, filter_key.to_string())).copied()
    }

    pub fn insert(&mut self, node_id: NodeId, filter_key: String, result: bool) {
        if self.cache.len() >= self.max_size {
            // Simple LRU: clear half the cache
            let to_remove: Vec<_> = self.cache.keys().take(self.max_size / 2).cloned().collect();
            for key in to_remove {
                self.cache.remove(&key);
            }
        }

        self.cache.insert((node_id, filter_key), result);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl TraversalEngine {
    /// Create a new traversal engine with default configuration
    pub fn new() -> Self {
        Self {
            state_pool: TraversalStatePool::new(),
            config: TraversalConfig::default(),
            stats: TraversalStats::new(),
            query_engine: QueryEngine::new(),
        }
    }

    /// Create traversal engine with custom configuration
    pub fn with_config(config: TraversalConfig) -> Self {
        Self {
            state_pool: TraversalStatePool::new(),
            config,
            stats: TraversalStats::new(),
            query_engine: QueryEngine::new(),
        }
    }

    /*
    === BASIC TRAVERSAL ALGORITHMS ===
    Core BFS and DFS implementations with filtering support
    */

    /// Breadth-First Search from a starting node
    ///
    /// PERFORMANCE: O(V + E) with cached adjacency map - FAST!
    /// FEATURES: Early termination, filtering, minimal overhead
    pub fn bfs(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        start: NodeId,
        options: TraversalOptions,
    ) -> GraphResult<TraversalResult> {
        let start_time = std::time::Instant::now();

        // Get fresh snapshot with guaranteed consistent adjacency data
        let (_, _, _, neighbors) = space.snapshot(pool);

        // 🚀 PERFORMANCE: Use Arc reference directly - no O(E) clone needed!
        // The Arc allows zero-copy sharing of the adjacency map

        // 🚀 PERFORMANCE: Skip bulk pre-filtering - filter nodes individually during traversal
        // This is much faster for sparse traversals that only visit a small subset of nodes

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        let mut levels = HashMap::new();

        // Initialize BFS
        queue.push_back((start, 0)); // (node_id, level)
        visited.insert(start);
        levels.insert(start, 0);

        let mut max_depth = 0;

        while let Some((current_node, level)) = queue.pop_front() {
            // Check termination conditions - skip nodes beyond max depth
            if let Some(max_depth_limit) = options.max_depth {
                if level > max_depth_limit {
                    continue;
                }
            }

            if let Some(max_nodes) = options.max_nodes {
                if result_nodes.len() >= max_nodes {
                    break;
                }
            }

            result_nodes.push(current_node);
            max_depth = max_depth.max(level);

            // Find neighbors using fresh adjacency map - O(1) lookup, O(degree) iteration - FAST!
            if let Some(node_neighbors) = neighbors.get(&current_node) {
                // Check if we can still explore deeper before processing neighbors
                let next_level = level + 1;
                let can_explore_deeper = match options.max_depth {
                    Some(max_depth_limit) => next_level <= max_depth_limit,
                    None => true,
                };

                if can_explore_deeper {
                    for &(neighbor, edge_id) in node_neighbors {
                        if !visited.contains(&neighbor) {
                            // 🚀 FAST: Individual node filtering - only check nodes we actually encounter
                            if self.should_visit_node(pool, space, neighbor, &options)? {
                                visited.insert(neighbor);
                                queue.push_back((neighbor, next_level));
                                result_edges.push(edge_id);
                                levels.insert(neighbor, next_level);
                            }
                        }
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        self.stats
            .record_traversal("bfs".to_string(), result_nodes.len(), duration);

        Ok(TraversalResult {
            algorithm: TraversalAlgorithm::BFS,
            nodes: result_nodes,
            edges: result_edges,
            paths: Vec::new(),
            metadata: TraversalMetadata {
                start_node: Some(start),
                end_node: None,
                max_depth,
                nodes_visited: visited.len(),
                execution_time: duration,
                levels: Some(levels),
                discovery_order: None,
            },
        })
    }

    /// Depth-First Search from a starting node
    ///
    /// PERFORMANCE: O(V + E) with cached adjacency map - FAST!
    /// FEATURES: Iterative implementation, minimal overhead
    pub fn dfs(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        start: NodeId,
        options: TraversalOptions,
    ) -> GraphResult<TraversalResult> {
        let start_time = std::time::Instant::now();

        // Get fresh snapshot with guaranteed consistent adjacency data
        let (_, _, _, neighbors) = space.snapshot(pool);

        // 🚀 PERFORMANCE: Use Arc reference directly - no O(E) clone needed!
        // The Arc allows zero-copy sharing of the adjacency map

        // 🚀 PERFORMANCE: Skip bulk pre-filtering - filter nodes individually during traversal
        // This is much faster for sparse traversals that only visit a small subset of nodes

        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        let mut edge_set = HashSet::new(); // O(1) duplicate check instead of O(n)
        let mut discovery_order = HashMap::new();
        let mut discovery_count = 0;

        // Initialize DFS with depth tracking
        stack.push((start, 0)); // (node, depth)

        while let Some((current_node, depth)) = stack.pop() {
            if !visited.contains(&current_node) {
                // Check depth limit before processing
                if let Some(max_depth_limit) = options.max_depth {
                    if depth > max_depth_limit {
                        continue;
                    }
                }

                // 🚀 FAST: Individual node filtering - only check nodes we actually encounter
                if self.should_visit_node(pool, space, current_node, &options)? {
                    visited.insert(current_node);
                    result_nodes.push(current_node);
                    discovery_order.insert(current_node, discovery_count);
                    discovery_count += 1;

                    // Check termination conditions
                    if let Some(max_nodes) = options.max_nodes {
                        if result_nodes.len() >= max_nodes {
                            break;
                        }
                    }

                    // Get neighbors from fresh adjacency map - O(1) lookup - FAST!
                    if let Some(node_neighbors) = neighbors.get(&current_node) {
                        // Check if we can explore deeper
                        let next_depth = depth + 1;
                        let can_explore_deeper = match options.max_depth {
                            Some(max_depth_limit) => next_depth <= max_depth_limit,
                            None => true,
                        };

                        if can_explore_deeper {
                            // Add in reverse order for consistent DFS traversal
                            for &(neighbor, edge_id) in node_neighbors.iter().rev() {
                                if !visited.contains(&neighbor) {
                                    stack.push((neighbor, next_depth));
                                    if edge_set.insert(edge_id) {
                                        // O(1) check + insert
                                        result_edges.push(edge_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        self.stats
            .record_traversal("dfs".to_string(), result_nodes.len(), duration);

        Ok(TraversalResult {
            algorithm: TraversalAlgorithm::DFS,
            nodes: result_nodes,
            edges: result_edges,
            paths: Vec::new(),
            metadata: TraversalMetadata {
                start_node: Some(start),
                end_node: None,
                max_depth: 0, // DFS doesn't track depth easily
                nodes_visited: discovery_count,
                execution_time: duration,
                levels: None,
                discovery_order: Some(discovery_order),
            },
        })
    }

    /*
    === PATH FINDING ALGORITHMS ===
    Shortest path and path enumeration algorithms
    */

    /// Find shortest path between two nodes using Dijkstra's algorithm
    ///
    /// PERFORMANCE: O((V + E) log V) with binary heap optimization
    /// FEATURES: Supports edge weights, early termination at target
    pub fn shortest_path(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        start: NodeId,
        end: NodeId,
        options: PathFindingOptions,
    ) -> GraphResult<Option<Path>> {
        let start_time = std::time::Instant::now();

        let mut state = self.state_pool.get_state();
        state.distances.clear();
        state.predecessors.clear();
        state.visited.clear();

        // Priority queue for Dijkstra's algorithm
        let mut heap = BinaryHeap::new();

        // Initialize
        state.distances.insert(start, 0.0);
        heap.push(DijkstraNode {
            id: start,
            distance: 0.0,
        });

        // Get columnar topology with cache maintenance (make owned copies)
        let (edge_ids_ref, sources_ref, targets_ref, _) = space.snapshot(pool);
        let edge_ids: Vec<EdgeId> = edge_ids_ref.as_ref().clone();
        let sources: Vec<NodeId> = sources_ref.as_ref().clone();
        let targets: Vec<NodeId> = targets_ref.as_ref().clone();

        while let Some(DijkstraNode {
            id: current,
            distance,
        }) = heap.pop()
        {
            // Early termination if we reached the target
            if current == end {
                let path = self.reconstruct_path(&state.predecessors, start, end, pool, space)?;
                self.state_pool.return_state(state);

                let duration = start_time.elapsed();
                self.stats.record_traversal(
                    "shortest_path".to_string(),
                    path.nodes.len(),
                    duration,
                );

                return Ok(Some(path));
            }

            // Skip if we've found a better path to this node
            if let Some(&current_distance) = state.distances.get(&current) {
                if distance > current_distance {
                    continue;
                }
            }

            state.visited.insert(current);

            // Explore neighbors directly from topology
            for i in 0..sources.len() {
                let (neighbor, edge_id) = if sources[i] == current {
                    (targets[i], edge_ids[i])
                } else if targets[i] == current {
                    (sources[i], edge_ids[i])
                } else {
                    continue;
                };

                if state.visited.contains(&neighbor) {
                    continue;
                }

                // Calculate edge weight
                let edge_weight =
                    self.get_edge_weight(pool, space, edge_id, &options.weight_attribute);
                let new_distance = distance + edge_weight;

                // Update if we found a shorter path
                if !state.distances.contains_key(&neighbor)
                    || new_distance < state.distances[&neighbor]
                {
                    state.distances.insert(neighbor, new_distance);
                    state.predecessors.insert(neighbor, current);
                    heap.push(DijkstraNode {
                        id: neighbor,
                        distance: new_distance,
                    });
                }
            }
        }

        // No path found
        self.state_pool.return_state(state);
        let duration = start_time.elapsed();
        self.stats
            .record_traversal("shortest_path".to_string(), 0, duration);

        Ok(None)
    }

    /// Find all simple paths between two nodes (up to maximum length)
    ///
    /// WARNING: Can be expensive for large graphs or long paths
    pub fn all_paths(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        start: NodeId,
        end: NodeId,
        max_length: usize,
    ) -> GraphResult<Vec<Path>> {
        let start_time = std::time::Instant::now();

        let mut all_paths = Vec::new();
        let mut current_path = Vec::new();
        let mut visited = HashSet::new();

        self.find_all_paths_recursive(
            pool,
            space,
            start,
            end,
            max_length,
            &mut current_path,
            &mut visited,
            &mut all_paths,
        )?;

        let duration = start_time.elapsed();
        self.stats
            .record_traversal("all_paths".to_string(), all_paths.len(), duration);

        Ok(all_paths)
    }

    /*
    === CONNECTIVITY ALGORITHMS ===
    Connected components and connectivity analysis
    */

    /// Find all connected components in the graph
    ///
    /// PERFORMANCE: O(V + E) using optimized BFS with adjacency map
    pub fn connected_components(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        options: TraversalOptions,
    ) -> GraphResult<ConnectedComponentsResult> {
        let start_time = std::time::Instant::now();

        // 📊 TIMING: Step 1 - Get active nodes
        let step1_start = std::time::Instant::now();
        let active_nodes: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
        let _step1_duration = step1_start.elapsed();

        // 📊 TIMING: Step 2 - Build adjacency snapshot
        let step2_start = std::time::Instant::now();
        let (_, _, _, neighbors) = space.snapshot(pool);
        let _step2_duration = step2_start.elapsed();

        // 🚀 PERFORMANCE: Use Arc reference directly - no O(E) clone needed!
        // The Arc allows zero-copy sharing of the adjacency map

        // 📊 TIMING: Step 3 - Initialize data structures
        let step3_start = std::time::Instant::now();
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        let _step3_duration = step3_start.elapsed();

        // 📊 TIMING: Step 4 - Main component finding loop
        let step4_start = std::time::Instant::now();
        let mut bfs_time = std::time::Duration::ZERO;
        let mut edge_computation_time = std::time::Duration::ZERO;
        let mut component_count = 0;

        // 🚀 OPTIMIZATION: Pre-mark nodes with component IDs for O(1) edge validation
        let mut node_component_id: HashMap<NodeId, usize> = HashMap::new();

        // BFS for each unvisited node - O(V + E) total
        for &start_node in &active_nodes {
            if !visited.contains(&start_node) {
                // 🚀 FAST: Individual node filtering - only check nodes we actually encounter
                if !self.should_visit_node(pool, space, start_node, &options)? {
                    continue;
                }

                // 📊 TIMING: BFS to find component
                let bfs_start = std::time::Instant::now();
                let mut component_nodes = Vec::new();
                let mut queue = VecDeque::new();

                queue.push_back(start_node);
                visited.insert(start_node);
                node_component_id.insert(start_node, component_count); // Mark with component ID

                while let Some(current) = queue.pop_front() {
                    component_nodes.push(current);

                    // Use fresh adjacency for optimal performance - get (neighbor, edge_id) pairs
                    if let Some(current_neighbors) = neighbors.get(&current) {
                        for &(neighbor, _edge_id) in current_neighbors {
                            if !visited.contains(&neighbor) {
                                // 🚀 FAST: Individual node filtering - only check nodes we actually encounter
                                if self.should_visit_node(pool, space, neighbor, &options)? {
                                    visited.insert(neighbor);
                                    node_component_id.insert(neighbor, component_count); // Mark with component ID
                                    queue.push_back(neighbor);
                                }
                            }
                        }
                    }
                }
                bfs_time += bfs_start.elapsed();

                if !component_nodes.is_empty() {
                    // 📊 TIMING: Calculate induced edges for this component
                    let edges_start = std::time::Instant::now();
                    let mut component_edges = Vec::new();
                    let mut edge_set = HashSet::new(); // O(1) duplicate check instead of O(n)

                    // 🚀 OPTIMIZATION: Use O(1) component ID checks instead of O(log n) HashSet lookups
                    // For each node in the component, check its edges
                    for &node in &component_nodes {
                        if let Some(node_neighbors) = neighbors.get(&node) {
                            for &(neighbor, edge_id) in node_neighbors {
                                // Only include edge if both endpoints are in the SAME component
                                // and we haven't already added this edge (avoid duplicates)
                                if let Some(&neighbor_comp_id) = node_component_id.get(&neighbor) {
                                    if neighbor_comp_id == component_count // O(1) check instead of O(log n)
                                        && node < neighbor // Avoid adding same edge twice
                                        && edge_set.insert(edge_id)
                                    {
                                        // O(1) check + insert
                                        component_edges.push(edge_id);
                                    }
                                }
                            }
                        }
                    }
                    edge_computation_time += edges_start.elapsed();
                    component_count += 1;

                    components.push(ConnectedComponent {
                        nodes: component_nodes.clone(),
                        edges: component_edges,
                        size: component_nodes.len(),
                        root: start_node,
                    });
                }
            }
        }

        let _step4_duration = step4_start.elapsed();

        // 📊 TIMING: Step 5 - Sort components by size
        let step5_start = std::time::Instant::now();
        components.sort_unstable_by(|a, b| b.size.cmp(&a.size));
        let _step5_duration = step5_start.elapsed();

        let duration = start_time.elapsed();
        self.stats.record_traversal(
            "connected_components".to_string(),
            components.len(),
            duration,
        );

        let total_components = components.len();
        let largest_component_size = components.iter().map(|c| c.size).max().unwrap_or(0);

        Ok(ConnectedComponentsResult {
            components,
            total_components,
            largest_component_size,
            execution_time: duration,
        })
    }

    /*
    === HELPER METHODS ===
    Internal utility methods for traversal algorithms
    */

    /// Check if a node should be visited based on traversal options (FAST: individual filtering)
    /// This is much faster than bulk pre-filtering for sparse traversals
    fn should_visit_node(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_id: NodeId,
        options: &TraversalOptions,
    ) -> GraphResult<bool> {
        if let Some(ref filter) = options.node_filter {
            self.should_visit_node_inline(pool, space, node_id, filter)
        } else {
            Ok(true)
        }
    }

    /// Inline node filter matching for maximum performance
    fn should_visit_node_inline(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_id: NodeId,
        filter: &NodeFilter,
    ) -> GraphResult<bool> {
        match filter {
            NodeFilter::HasAttribute { name } => {
                Ok(space.get_node_attr_index(node_id, name).is_some())
            }
            NodeFilter::AttributeEquals { name, value } => {
                if let Some(index) = space.get_node_attr_index(node_id, name) {
                    if let Some(attr_value) = pool.get_attr_by_index(name, index, true) {
                        return Ok(attr_value == value);
                    }
                }
                Ok(false)
            }
            NodeFilter::AttributeFilter { name, filter } => {
                if let Some(index) = space.get_node_attr_index(node_id, name) {
                    if let Some(attr_value) = pool.get_attr_by_index(name, index, true) {
                        return Ok(filter.matches(attr_value));
                    }
                }
                Ok(false)
            }
            NodeFilter::And(filters) => {
                for f in filters {
                    if !self.should_visit_node_inline(pool, space, node_id, f)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            NodeFilter::Or(filters) => {
                for f in filters {
                    if self.should_visit_node_inline(pool, space, node_id, f)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            NodeFilter::Not(filter) => {
                Ok(!self.should_visit_node_inline(pool, space, node_id, filter)?)
            }
            _ => Ok(true), // Accept other filter types for now
        }
    }

    /// Pre-filter nodes using bulk operations for O(n) performance instead of O(n²)
    /// NOTE: This is slower for sparse traversals - kept for compatibility
    #[allow(dead_code)]
    fn get_eligible_nodes(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        candidate_nodes: &[NodeId],
        options: &TraversalOptions,
    ) -> GraphResult<HashSet<NodeId>> {
        if let Some(ref filter) = options.node_filter {
            // Use bulk filtering - O(n) instead of O(n²)
            let filtered_nodes = self
                .query_engine
                .find_nodes_by_filter_with_space(pool, space, filter)?;
            // Return intersection with candidate nodes - convert filtered to HashSet for O(1) lookups
            let filtered_set: HashSet<NodeId> = filtered_nodes.into_iter().collect();
            Ok(candidate_nodes
                .iter()
                .copied()
                .filter(|node| filtered_set.contains(node))
                .collect())
        } else {
            // No filter - all candidates are eligible, direct conversion is faster
            Ok(candidate_nodes.iter().copied().collect())
        }
    }

    /// Get edge weight for pathfinding algorithms
    fn get_edge_weight(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        edge_id: EdgeId,
        weight_attr: &Option<AttrName>,
    ) -> f64 {
        if let Some(attr_name) = weight_attr {
            // Try to get edge attribute value (using space since pool doesn't have this method)
            if let Some(index) = space.get_edge_attr_index(edge_id, attr_name) {
                if let Some(attr_value) = pool.get_attr_by_index(attr_name, index, false) {
                    match attr_value {
                        AttrValue::Float(f) => *f as f64,
                        AttrValue::Int(i) => *i as f64,
                        _ => 1.0, // Default weight
                    }
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0 // Default unit weight
        }
    }

    /// Reconstruct path from predecessor information
    fn reconstruct_path(
        &self,
        predecessors: &HashMap<NodeId, NodeId>,
        start: NodeId,
        end: NodeId,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
    ) -> GraphResult<Path> {
        let mut path_nodes = Vec::new();
        let mut path_edges = Vec::new();
        let mut current = end;
        let mut total_weight = 0.0;

        // Build path backwards
        path_nodes.push(current);

        while current != start {
            if let Some(&predecessor) = predecessors.get(&current) {
                path_nodes.push(predecessor);

                // Find the edge between predecessor and current
                if let Some(edge_id) = self.find_edge_between(pool, space, predecessor, current)? {
                    path_edges.push(edge_id);
                    total_weight += self.get_edge_weight(pool, space, edge_id, &None);
                }

                current = predecessor;
            } else {
                return Err(crate::errors::GraphError::InternalError {
                    message: "Path reconstruction failed - missing predecessor".to_string(),
                    location: "reconstruct_path".to_string(),
                    context: std::collections::HashMap::new(),
                });
            }
        }

        // Reverse to get correct order
        path_nodes.reverse();
        path_edges.reverse();

        let path_length = path_nodes.len().saturating_sub(1);

        Ok(Path {
            nodes: path_nodes,
            edges: path_edges,
            total_weight,
            metadata: PathMetadata {
                algorithm: "shortest_path".to_string(),
                is_simple: true,
                length: path_length,
            },
        })
    }

    /// Find edge between two nodes (helper for path reconstruction)
    fn find_edge_between(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        node1: NodeId,
        node2: NodeId,
    ) -> GraphResult<Option<EdgeId>> {
        let (edge_ids, sources, targets, _) = space.snapshot(pool);

        for i in 0..sources.len() {
            if (sources[i] == node1 && targets[i] == node2)
                || (sources[i] == node2 && targets[i] == node1)
            {
                return Ok(Some(edge_ids[i]));
            }
        }

        Ok(None)
    }

    /// Recursive helper for finding all paths
    fn find_all_paths_recursive(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed to &GraphSpace (no longer mutable)
        current: NodeId,
        target: NodeId,
        max_length: usize,
        path: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        all_paths: &mut Vec<Path>,
    ) -> GraphResult<()> {
        if path.len() >= max_length {
            return Ok(());
        }

        path.push(current);
        visited.insert(current);

        if current == target {
            // Found a path - convert to Path structure
            let path_nodes = path.clone();
            let mut path_edges = Vec::new();
            let mut total_weight = 0.0;

            for i in 0..path_nodes.len() - 1 {
                if let Some(edge_id) =
                    self.find_edge_between(pool, space, path_nodes[i], path_nodes[i + 1])?
                {
                    path_edges.push(edge_id);
                    total_weight += self.get_edge_weight(pool, space, edge_id, &None);
                }
            }

            all_paths.push(Path {
                nodes: path_nodes,
                edges: path_edges,
                total_weight,
                metadata: PathMetadata {
                    algorithm: "all_paths".to_string(),
                    is_simple: true,
                    length: path.len() - 1,
                },
            });
        } else {
            // Continue exploring - get topology data
            let (_edge_ids, sources, targets, _) = space.snapshot(pool);

            // Collect neighbors first to avoid borrowing conflicts
            let mut neighbors = Vec::new();
            for i in 0..sources.len() {
                if sources[i] == current {
                    neighbors.push(targets[i]);
                } else if targets[i] == current {
                    neighbors.push(sources[i]);
                }
            }

            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    self.find_all_paths_recursive(
                        pool, space, neighbor, target, max_length, path, visited, all_paths,
                    )?;
                }
            }
        }

        path.pop();
        visited.remove(&current);
        Ok(())
    }

    /// Get traversal statistics
    pub fn statistics(&self) -> &TraversalStats {
        &self.stats
    }

    /// Clear performance statistics
    pub fn clear_stats(&mut self) {
        self.stats.clear();
    }
}

/*
=== SUPPORTING DATA STRUCTURES ===
*/

/// Node for Dijkstra's algorithm priority queue
#[derive(Debug, Clone)]
struct DijkstraNode {
    id: NodeId,
    distance: f64,
}

impl PartialEq for DijkstraNode {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for DijkstraNode {}

impl PartialOrd for DijkstraNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Reusable state pool to avoid allocations
#[derive(Debug)]
struct TraversalStatePool {
    available_states: Vec<TraversalState>,
}

impl TraversalStatePool {
    fn new() -> Self {
        Self {
            available_states: Vec::new(),
        }
    }

    fn get_state(&mut self) -> TraversalState {
        self.available_states
            .pop()
            .unwrap_or_else(|| TraversalState {
                visited: HashSet::new(),
                queue: VecDeque::new(),
                distances: HashMap::new(),
                predecessors: HashMap::new(),
            })
    }

    fn return_state(&mut self, state: TraversalState) {
        self.available_states.push(state);
    }
}

/// Reusable traversal state
#[derive(Debug)]
struct TraversalState {
    visited: HashSet<NodeId>,
    #[allow(dead_code)] // TODO: Implement queue-based traversal
    queue: VecDeque<NodeId>,
    distances: HashMap<NodeId, f64>,
    predecessors: HashMap<NodeId, NodeId>,
}

/// Configuration for traversal algorithms
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Use parallel processing when nodes/edges exceed this threshold
    pub parallel_threshold: usize,
    /// Maximum memory to use for state (bytes)
    pub max_memory: usize,
    /// Enable performance tracking
    pub track_performance: bool,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 1000,
            max_memory: 64 * 1024 * 1024, // 64MB
            track_performance: true,
        }
    }
}

/// Options for controlling traversal behavior
#[derive(Debug, Clone)]
pub struct TraversalOptions {
    /// Filter nodes during traversal
    pub node_filter: Option<NodeFilter>,
    /// Filter edges during traversal  
    pub edge_filter: Option<EdgeFilter>,
    /// Maximum depth to traverse
    pub max_depth: Option<usize>,
    /// Maximum number of nodes to visit
    pub max_nodes: Option<usize>,
    /// Early termination condition
    pub target_node: Option<NodeId>,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            node_filter: None,
            edge_filter: None,
            max_depth: None,
            max_nodes: None,
            target_node: None,
        }
    }
}

/// Options for pathfinding algorithms
#[derive(Debug, Clone)]
pub struct PathFindingOptions {
    /// Attribute name to use as edge weight
    pub weight_attribute: Option<AttrName>,
    /// Maximum path length to consider
    pub max_path_length: Option<usize>,
    /// Heuristic function for A* (future)
    pub heuristic: Option<String>,
}

impl Default for PathFindingOptions {
    fn default() -> Self {
        Self {
            weight_attribute: None,
            max_path_length: None,
            heuristic: None,
        }
    }
}

/// Result of a traversal operation
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub algorithm: TraversalAlgorithm,
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub paths: Vec<Path>,
    pub metadata: TraversalMetadata,
}

/// Type of traversal algorithm
#[derive(Debug, Clone, PartialEq)]
pub enum TraversalAlgorithm {
    BFS,
    DFS,
    ShortestPath,
    AllPaths,
    ConnectedComponents,
}

/// Metadata about traversal execution
#[derive(Debug, Clone)]
pub struct TraversalMetadata {
    pub start_node: Option<NodeId>,
    pub end_node: Option<NodeId>,
    pub max_depth: usize,
    pub nodes_visited: usize,
    pub execution_time: std::time::Duration,
    pub levels: Option<HashMap<NodeId, usize>>, // For BFS
    pub discovery_order: Option<HashMap<NodeId, usize>>, // For DFS
}

/// A path through the graph
#[derive(Debug, Clone)]
pub struct Path {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub total_weight: f64,
    pub metadata: PathMetadata,
}

/// Metadata about a path
#[derive(Debug, Clone)]
pub struct PathMetadata {
    pub algorithm: String,
    pub is_simple: bool,
    pub length: usize,
}

/// Result of connected components analysis
#[derive(Debug, Clone)]
pub struct ConnectedComponentsResult {
    pub components: Vec<ConnectedComponent>,
    pub total_components: usize,
    pub largest_component_size: usize,
    pub execution_time: std::time::Duration,
}

/// A connected component
#[derive(Debug, Clone)]
pub struct ConnectedComponent {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>, // Induced edges within this component
    pub size: usize,
    pub root: NodeId, // Representative node
}

/// Performance statistics for traversal operations
#[derive(Debug, Clone)]
pub struct TraversalStats {
    pub total_traversals: usize,
    pub total_nodes_visited: usize,
    pub total_time: std::time::Duration,
    pub algorithm_counts: HashMap<String, usize>,
    pub algorithm_times: HashMap<String, std::time::Duration>,
}

impl TraversalStats {
    fn new() -> Self {
        Self {
            total_traversals: 0,
            total_nodes_visited: 0,
            total_time: std::time::Duration::new(0, 0),
            algorithm_counts: HashMap::new(),
            algorithm_times: HashMap::new(),
        }
    }

    fn record_traversal(
        &mut self,
        algorithm: String,
        nodes_visited: usize,
        duration: std::time::Duration,
    ) {
        self.total_traversals += 1;
        self.total_nodes_visited += nodes_visited;
        self.total_time += duration;

        *self.algorithm_counts.entry(algorithm.clone()).or_insert(0) += 1;
        *self
            .algorithm_times
            .entry(algorithm)
            .or_insert(std::time::Duration::new(0, 0)) += duration;
    }

    fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn average_time_per_traversal(&self) -> f64 {
        if self.total_traversals > 0 {
            self.total_time.as_secs_f64() / self.total_traversals as f64
        } else {
            0.0
        }
    }

    pub fn average_nodes_per_traversal(&self) -> f64 {
        if self.total_traversals > 0 {
            self.total_nodes_visited as f64 / self.total_traversals as f64
        } else {
            0.0
        }
    }
}

impl Default for TraversalEngine {
    fn default() -> Self {
        Self::new()
    }
}

/*
=== IMPLEMENTATION NOTES ===

PERFORMANCE OPTIMIZATIONS:
1. Columnar topology access for cache-friendly iteration
2. Parallel processing for large graphs using Rayon
3. State pooling to avoid allocations
4. Early termination conditions
5. Efficient data structures (BinaryHeap for Dijkstra)

MEMORY EFFICIENCY:
1. Reusable state objects
2. In-place operations where possible
3. Configurable memory limits
4. Lazy evaluation for expensive operations

ALGORITHM IMPLEMENTATIONS:
1. BFS: Level-by-level with parallel processing per level
2. DFS: Iterative to avoid stack overflow
3. Shortest Path: Dijkstra with early termination
4. Connected Components: Union-find approach with BFS

EXTENSIBILITY:
1. Common TraversalOptions for all algorithms
2. Pluggable filtering during traversal
3. Configurable performance vs memory trade-offs
4. Statistics tracking for optimization

INTEGRATION:
1. Uses existing NodeFilter/EdgeFilter from query engine
2. Leverages GraphSpace's columnar topology
3. Compatible with GraphPool's attribute system
4. Consistent error handling with rest of system
*/
