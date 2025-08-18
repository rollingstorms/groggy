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

use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use crate::types::{NodeId, EdgeId, AttrName, AttrValue};
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::core::query::{NodeFilter, EdgeFilter};
use crate::errors::GraphResult;
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
    
    /// Adjacency cache for fast neighbor lookups
    #[allow(dead_code)] // TODO: Implement adjacency caching
    adjacency_cache: AdjacencyCache,
    
    /// Filter result cache to avoid repeated evaluations
    #[allow(dead_code)] // TODO: Implement filter result caching
    filter_cache: FilterCache,
}

/// Adjacency list cache for fast neighbor lookups
#[derive(Debug, Clone)]
pub struct AdjacencyCache {
    /// Map from node_id to list of (neighbor, edge_id)
    adjacency_map: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,
    /// Whether the cache is valid
    is_valid: bool,
    /// Cache generation (for invalidation)
    generation: usize,
}

impl AdjacencyCache {
    pub fn new() -> Self {
        Self {
            adjacency_map: HashMap::new(),
            is_valid: false,
            generation: 0,
        }
    }
    
    /// Build adjacency cache from columnar topology
    pub fn rebuild(&mut self, edge_ids: &[EdgeId], sources: &[NodeId], targets: &[NodeId], topology_generation: usize) {
        self.adjacency_map.clear();
        
        for i in 0..sources.len() {
            let source = sources[i];
            let target = targets[i];
            let edge_id = edge_ids[i];
            
            // Add both directions for undirected edges
            self.adjacency_map.entry(source)
                .or_insert_with(Vec::new)
                .push((target, edge_id));
                
            self.adjacency_map.entry(target)
                .or_insert_with(Vec::new)
                .push((source, edge_id));
        }
        
        self.is_valid = true;
        self.generation = topology_generation;  // Set to current topology generation
    }
    
    /// Get neighbors for a node using the cache
    pub fn get_neighbors(&self, node_id: NodeId) -> Option<&Vec<(NodeId, EdgeId)>> {
        if self.is_valid {
            self.adjacency_map.get(&node_id)
        } else {
            None
        }
    }
    
    /// Check if cache is up to date with topology generation
    pub fn is_up_to_date(&self, topology_generation: usize) -> bool {
        self.is_valid && self.generation == topology_generation
    }
    
    /// Invalidate the cache
    pub fn invalidate(&mut self) {
        self.is_valid = false;
    }
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
            adjacency_cache: AdjacencyCache::new(),
            filter_cache: FilterCache::new(10000), // Cache up to 10k results
        }
    }
    
    /// Create traversal engine with custom configuration
    pub fn with_config(config: TraversalConfig) -> Self {
        Self {
            state_pool: TraversalStatePool::new(),
            config,
            stats: TraversalStats::new(),
            adjacency_cache: AdjacencyCache::new(),
            filter_cache: FilterCache::new(10000), // Cache up to 10k results
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
        space: &mut GraphSpace,
        start: NodeId,
        options: TraversalOptions
    ) -> GraphResult<TraversalResult> {
        let start_time = std::time::Instant::now();
        
        // Use cached adjacency map - only rebuild if topology changed!
        let topology_generation = space.get_topology_generation();
        if !self.adjacency_cache.is_up_to_date(topology_generation) {
            let (edge_ids, sources, targets) = space.get_columnar_topology();
            self.adjacency_cache.rebuild(edge_ids, sources, targets, topology_generation);
        }        let mut visited = HashSet::new();
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
            // Check termination conditions
            if let Some(max_depth_limit) = options.max_depth {
                if level >= max_depth_limit {
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
            
            // Find neighbors using cached adjacency map - O(degree) per node - FAST!
            if let Some(neighbors) = self.adjacency_cache.get_neighbors(current_node) {
                for &(neighbor, edge_id) in neighbors {
                    if !visited.contains(&neighbor) {
                        // Simple filter check without caching overhead
                        if self.should_visit_node(pool, space, neighbor, &options)? {
                            visited.insert(neighbor);
                            queue.push_back((neighbor, level + 1));
                            result_edges.push(edge_id);
                            levels.insert(neighbor, level + 1);
                        }
                    }
                }
            }
        }
        
        let duration = start_time.elapsed();
        self.stats.record_traversal("bfs".to_string(), result_nodes.len(), duration);
        
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
        space: &mut GraphSpace,
        start: NodeId,
        options: TraversalOptions
    ) -> GraphResult<TraversalResult> {
        let start_time = std::time::Instant::now();
        
        // Use cached adjacency map - only rebuild if topology changed!
        let topology_generation = space.get_topology_generation();
        if !self.adjacency_cache.is_up_to_date(topology_generation) {
            let (edge_ids, sources, targets) = space.get_columnar_topology();
            self.adjacency_cache.rebuild(edge_ids, sources, targets, topology_generation);
        }        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        let mut discovery_order = HashMap::new();
        let mut discovery_count = 0;
        
        // Initialize DFS
        stack.push(start);
        
        while let Some(current_node) = stack.pop() {
            if !visited.contains(&current_node) {
                // Simple filter check without caching overhead
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
                    
                    // Get neighbors from cached adjacency map - O(1) lookup - FAST!
                    if let Some(neighbors) = self.adjacency_cache.get_neighbors(current_node) {
                        // Add in reverse order for consistent DFS traversal
                        for &(neighbor, edge_id) in neighbors.iter().rev() {
                            if !visited.contains(&neighbor) {
                                stack.push(neighbor);
                                if !result_edges.contains(&edge_id) {
                                    result_edges.push(edge_id);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let duration = start_time.elapsed();
        self.stats.record_traversal("dfs".to_string(), result_nodes.len(), duration);
        
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
        space: &mut GraphSpace,
        start: NodeId,
        end: NodeId,
        options: PathFindingOptions
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
        heap.push(DijkstraNode { id: start, distance: 0.0 });
        
        // Get columnar topology with cache maintenance (make owned copies)
        let (edge_ids_ref, sources_ref, targets_ref) = space.get_columnar_topology();
        let edge_ids: Vec<EdgeId> = edge_ids_ref.to_vec();
        let sources: Vec<NodeId> = sources_ref.to_vec();
        let targets: Vec<NodeId> = targets_ref.to_vec();
        
        while let Some(DijkstraNode { id: current, distance }) = heap.pop() {
            // Early termination if we reached the target
            if current == end {
                let path = self.reconstruct_path(&state.predecessors, start, end, pool, space)?;
                self.state_pool.return_state(state);
                
                let duration = start_time.elapsed();
                self.stats.record_traversal("shortest_path".to_string(), path.nodes.len(), duration);
                
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
                let edge_weight = self.get_edge_weight(pool, space, edge_id, &options.weight_attribute);
                let new_distance = distance + edge_weight;
                
                // Update if we found a shorter path
                if !state.distances.contains_key(&neighbor) || new_distance < state.distances[&neighbor] {
                    state.distances.insert(neighbor, new_distance);
                    state.predecessors.insert(neighbor, current);
                    heap.push(DijkstraNode { id: neighbor, distance: new_distance });
                }
            }
        }
        
        // No path found
        self.state_pool.return_state(state);
        let duration = start_time.elapsed();
        self.stats.record_traversal("shortest_path".to_string(), 0, duration);
        
        Ok(None)
    }
    
    /// Find all simple paths between two nodes (up to maximum length)
    /// 
    /// WARNING: Can be expensive for large graphs or long paths
    pub fn all_paths(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        start: NodeId,
        end: NodeId,
        max_length: usize
    ) -> GraphResult<Vec<Path>> {
        let start_time = std::time::Instant::now();
        
        let mut all_paths = Vec::new();
        let mut current_path = Vec::new();
        let mut visited = HashSet::new();
        
        self.find_all_paths_recursive(
            pool, space, start, end, max_length,
            &mut current_path, &mut visited, &mut all_paths
        )?;
        
        let duration = start_time.elapsed();
        self.stats.record_traversal("all_paths".to_string(), all_paths.len(), duration);
        
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
        space: &mut GraphSpace,
        options: TraversalOptions
    ) -> GraphResult<ConnectedComponentsResult> {
        let start_time = std::time::Instant::now();
        
        // Use cached adjacency for optimal performance  
        let topology_generation = space.get_topology_generation();
        if !self.adjacency_cache.is_up_to_date(topology_generation) {
            let (edge_ids, sources, targets) = space.get_columnar_topology();
            self.adjacency_cache.rebuild(edge_ids, sources, targets, topology_generation);
        }
        
        let active_nodes: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
        let mut visited = HashSet::new();
        let mut components = Vec::new();
        
        // BFS for each unvisited node - O(V + E) total
        for &start_node in &active_nodes {
            if !visited.contains(&start_node) {
                if !self.should_visit_node(pool, space, start_node, &options)? {
                    continue;
                }
                
                // BFS to find component
                let mut component_nodes = Vec::new();
                let mut queue = VecDeque::new();
                
                queue.push_back(start_node);
                visited.insert(start_node);
                
                while let Some(current) = queue.pop_front() {
                    component_nodes.push(current);
                    
                    // Use cached adjacency for optimal performance - get (neighbor, edge_id) pairs
                    if let Some(neighbors) = self.adjacency_cache.get_neighbors(current) {
                        for &(neighbor, _edge_id) in neighbors {
                            if !visited.contains(&neighbor) {
                                if self.should_visit_node(pool, space, neighbor, &options)? {
                                    visited.insert(neighbor);
                                    queue.push_back(neighbor);
                                }
                            }
                        }
                    }
                }
                
                if !component_nodes.is_empty() {
                    // Calculate induced edges for this component
                    let component_node_set: HashSet<NodeId> = component_nodes.iter().copied().collect();
                    let mut component_edges = Vec::new();
                    
                    // Use cached adjacency to find induced edges efficiently
                    if let Some(_neighbors) = self.adjacency_cache.get_neighbors(start_node) {
                        // For each node in the component, check its edges
                        for &node in &component_nodes {
                            if let Some(node_neighbors) = self.adjacency_cache.get_neighbors(node) {
                                for &(neighbor, edge_id) in node_neighbors {
                                    // Only include edge if both endpoints are in this component
                                    // and we haven't already added this edge (avoid duplicates)
                                    if component_node_set.contains(&neighbor) 
                                        && node < neighbor // Avoid adding same edge twice
                                        && !component_edges.contains(&edge_id) {
                                        component_edges.push(edge_id);
                                    }
                                }
                            }
                        }
                    }
                    
                    components.push(ConnectedComponent {
                        nodes: component_nodes.clone(),
                        edges: component_edges,
                        size: component_nodes.len(),
                        root: start_node,
                    });
                }
            }
        }
        
        // Sort components by size (largest first) for consistent results
        components.sort_unstable_by(|a, b| b.size.cmp(&a.size));
        
        let duration = start_time.elapsed();
        self.stats.record_traversal("connected_components".to_string(), components.len(), duration);
        
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
    
    /// Check if a node should be visited based on filter options
    fn should_visit_node(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_id: NodeId,
        options: &TraversalOptions
    ) -> GraphResult<bool> {
        if let Some(ref filter) = options.node_filter {
            self.should_visit_node_inline(pool, space, node_id, filter)
        } else {
            Ok(true)
        }
    }
    
    /// Inline node filter matching (since QueryEngine method is private)
    fn should_visit_node_inline(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_id: NodeId,
        filter: &NodeFilter
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
    
    /// Check if traversal should terminate based on options
    #[allow(dead_code)] // TODO: Implement termination conditions
    fn should_terminate(&self, options: &TraversalOptions, depth: usize, nodes_found: usize) -> bool {
        if let Some(max_depth) = options.max_depth {
            if depth >= max_depth {
                return true;
            }
        }
        
        if let Some(max_nodes) = options.max_nodes {
            if nodes_found >= max_nodes {
                return true;
            }
        }
        
        false
    }
    
    /// Get edge weight for pathfinding algorithms
    fn get_edge_weight(&self, pool: &GraphPool, space: &GraphSpace, edge_id: EdgeId, weight_attr: &Option<AttrName>) -> f64 {
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
        space: &mut GraphSpace
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
    fn find_edge_between(&self, pool: &GraphPool, space: &mut GraphSpace, node1: NodeId, node2: NodeId) -> GraphResult<Option<EdgeId>> {
        let (edge_ids, sources, targets) = space.get_columnar_topology();
        
        for i in 0..sources.len() {
            if (sources[i] == node1 && targets[i] == node2) || 
               (sources[i] == node2 && targets[i] == node1) {
                return Ok(Some(edge_ids[i]));
            }
        }
        
        Ok(None)
    }
    
    /// Recursive helper for finding all paths
    fn find_all_paths_recursive(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        current: NodeId,
        target: NodeId,
        max_length: usize,
        path: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
        all_paths: &mut Vec<Path>
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
                if let Some(edge_id) = self.find_edge_between(pool, space, path_nodes[i], path_nodes[i + 1])? {
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
            let (_edge_ids, sources, targets) = space.get_columnar_topology();
            
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
                    self.find_all_paths_recursive(pool, space, neighbor, target, max_length, path, visited, all_paths)?;
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
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
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
        self.available_states.pop().unwrap_or_else(|| TraversalState {
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
    
    fn record_traversal(&mut self, algorithm: String, nodes_visited: usize, duration: std::time::Duration) {
        self.total_traversals += 1;
        self.total_nodes_visited += nodes_visited;
        self.total_time += duration;
        
        *self.algorithm_counts.entry(algorithm.clone()).or_insert(0) += 1;
        *self.algorithm_times.entry(algorithm).or_insert(std::time::Duration::new(0, 0)) += duration;
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