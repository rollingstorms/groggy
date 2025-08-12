//! Query Engine - Advanced filtering, searching, and analysis operations.
//!
//! ARCHITECTURE ROLE:
//! This is the "read-only brain" of the system. It handles complex queries,
//! filtering, aggregation, and analysis operations without modifying the graph.
//!
//! DESIGN PHILOSOPHY:
//! - Pure functions (no side effects, safe for concurrent access)
//! - Composable query building (combine simple filters into complex queries)
//! - Performance-optimized (leverage columnar storage, indices, caching)
//! - Expressive query language (support complex patterns and conditions)

/*
=== QUERY ENGINE OVERVIEW ===

The query engine provides sophisticated read-only operations on graph data:

1. FILTERING: Find nodes/edges matching specific criteria
2. AGGREGATION: Compute statistics across sets of entities
3. PATTERN MATCHING: Find complex structural patterns in the graph

KEY DESIGN DECISIONS:
- Separate from core data structures (pure analysis layer)
- Leverage columnar storage for efficient bulk operations
- Support both simple filters and complex multi-step queries
- Provide both high-level API and low-level optimization hooks
- Enable query planning and optimization
*/

use std::collections::HashMap;
use crate::types::{NodeId, EdgeId, AttrName, AttrValue};
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::core::traversal::{TraversalEngine, TraversalOptions, PathFindingOptions, Path, TraversalResult, ConnectedComponentsResult};
use crate::errors::{GraphResult, EntityType};
use rayon::prelude::*;

/// The main query engine that processes complex read-only operations
/// 
/// RESPONSIBILITIES:
/// - Execute filtering and search operations
/// - Perform aggregation and statistics computations  
/// - Handle complex pattern matching queries
/// - Optimize query execution plans
/// - Cache frequently accessed results
/// 
/// NOT RESPONSIBLE FOR:
/// - Modifying the graph (read-only operations only)
/// - Managing graph storage (that's GraphPool's job)
/// - Version control (that's HistoryForest's job)
#[derive(Debug)]
pub struct QueryEngine {
    /*
    === QUERY OPTIMIZATION ===
    Performance enhancement structures
    */
    
    /// Cache for frequently executed queries
    /// Maps query hash -> cached results
    query_cache: HashMap<u64, CachedQueryResult>,
    
    /// Statistics about attribute distributions (for query optimization)
    /// Maps attribute name -> distribution statistics
    #[allow(dead_code)]
    attr_statistics: HashMap<AttrName, AttributeStatistics>,
    
    /// Configuration for query execution
    #[allow(dead_code)]
    config: QueryConfig,
    
    /*
    === PERFORMANCE TRACKING ===
    Monitor query performance for optimization
    */
    
    /// Track query execution times for optimization
    query_performance: HashMap<u64, QueryPerformance>,
    
    /// Total number of queries executed
    total_queries: usize,
    
    /// Integrated traversal engine for graph algorithms
    traversal_engine: TraversalEngine,
}

impl QueryEngine {
    /// Create a new query engine with default configuration
    pub fn new() -> Self {
        Self {
            query_cache: HashMap::new(),
            attr_statistics: HashMap::new(),
            config: QueryConfig::default(),
            query_performance: HashMap::new(),
            total_queries: 0,
            traversal_engine: TraversalEngine::new(),
        }
    }
    
    /// Create a query engine with custom configuration
    pub fn with_config(config: QueryConfig) -> Self {
        Self {
            query_cache: HashMap::new(),
            attr_statistics: HashMap::new(),
            config,
            query_performance: HashMap::new(),
            total_queries: 0,
            traversal_engine: TraversalEngine::new(),
        }
    }
    
    /*
    === BASIC FILTERING OPERATIONS ===
    Simple, commonly used filtering operations
    */
    
    /// Find all nodes matching a simple attribute filter
    /// 
    /// PERFORMANCE: O(n) where n = number of nodes, but can be optimized with indices
    pub fn find_nodes_by_attribute(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        attr_name: &AttrName,
        filter: &AttributeFilter
    ) -> GraphResult<Vec<NodeId>> {
        // TODO: Check query cache first (for now, skip caching)
        
        let mut matching_nodes = Vec::new();
        
        // Get all active nodes and check their attributes
        for &node_id in space.get_active_nodes() {
            // Get the attribute index for this specific node
            if let Some(attr_index) = space.get_node_attr_index(node_id, attr_name) {
                // Get the actual attribute value from the pool
                if let Some(attr_value) = pool.get_attr_by_index(attr_name, attr_index, true) {
                    // Apply the filter to this value
                    if filter.matches(attr_value) {
                        matching_nodes.push(node_id);
                    }
                }
            }
        }
        
        // TODO: Cache the result
        // TODO: Update query performance tracking
        self.total_queries += 1;
        
        Ok(matching_nodes)
    }
    
    /// Find all edges matching a simple attribute filter
    pub fn find_edges_by_attribute(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        attr_name: &AttrName,
        filter: &AttributeFilter
    ) -> GraphResult<Vec<EdgeId>> {
        // TODO: Check query cache first (for now, skip caching)
        
        let mut matching_edges = Vec::new();
        
        // Get all active edges and check their attributes
        for &edge_id in space.get_active_edges() {
            // Get the attribute index for this specific edge
            if let Some(attr_index) = space.get_edge_attr_index(edge_id, attr_name) {
                // Get the actual attribute value from the pool
                if let Some(attr_value) = pool.get_attr_by_index(attr_name, attr_index, false) {
                    // Apply the filter to this value
                    if filter.matches(attr_value) {
                        matching_edges.push(edge_id);
                    }
                }
            }
        }
        
        // TODO: Cache the result
        // TODO: Update query performance tracking
        self.total_queries += 1;
        
        Ok(matching_edges)
    }
    
    /// Find nodes with advanced filtering (Phase 3.1 - Enhanced Version)
    /// 
    /// PERFORMANCE: Optimized with parallel processing for large datasets
    pub fn filter_nodes(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Vec<NodeId>> {
        let active_nodes: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
        
        // Try bulk filtering for common cases
        match self.try_bulk_node_filter(&active_nodes, pool, space, filter)? {
            Some(results) => Ok(results),
            None => {
                // Fall back to individual filtering for complex cases
                if active_nodes.len() > 1000 {
                    Ok(active_nodes
                        .par_iter()
                        .filter(|&node_id| self.node_matches_filter(*node_id, pool, space, filter))
                        .copied()
                        .collect())
                } else {
                    Ok(active_nodes
                        .into_iter()
                        .filter(|&node_id| self.node_matches_filter(node_id, pool, space, filter))
                        .collect())
                }
            }
        }
    }
    
    /// Find edges with advanced filtering (Phase 3.1)
    /// 
    /// PERFORMANCE: Uses columnar topology for efficient filtering
    pub fn filter_edges(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter
    ) -> GraphResult<Vec<EdgeId>> {
        let active_edges: Vec<EdgeId> = space.get_active_edges().iter().copied().collect();
        
        // For large datasets, use parallel filtering
        if active_edges.len() > 1000 {
            Ok(active_edges
                .par_iter()
                .filter(|&edge_id| self.edge_matches_filter(*edge_id, pool, space, filter))
                .copied()
                .collect())
        } else {
            Ok(active_edges
                .into_iter()
                .filter(|&edge_id| self.edge_matches_filter(edge_id, pool, space, filter))
                .collect())
        }
    }
    
    /// Check if a node matches the given filter (Phase 3.1)
    fn node_matches_filter(
        &self,
        node_id: NodeId,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> bool {
        match filter {
            NodeFilter::HasAttribute { name } => {
                space.get_node_attr_index(node_id, name).is_some()
            }
            NodeFilter::AttributeEquals { name, value } => {
                if let Some(index) = space.get_node_attr_index(node_id, name) {
                    if let Some(attr_value) = pool.get_attr_by_index(name, index, true) {
                        return attr_value == value;
                    }
                }
                false
            }
            NodeFilter::AttributeFilter { name, filter } => {
                if let Some(index) = space.get_node_attr_index(node_id, name) {
                    if let Some(attr_value) = pool.get_attr_by_index(name, index, true) {
                        return filter.matches(attr_value);
                    }
                }
                false
            }
            NodeFilter::DegreeRange { min, max } => {
                // Calculate degree using columnar topology
                let (_edge_ids, sources, targets) = space.get_columnar_topology();
                let mut degree = 0;
                for i in 0..sources.len() {
                    if sources[i] == node_id || targets[i] == node_id {
                        degree += 1;
                    }
                }
                degree >= *min && degree <= *max
            }
            NodeFilter::HasNeighbor { neighbor_id } => {
                let (_edge_ids, sources, targets) = space.get_columnar_topology();
                for i in 0..sources.len() {
                    if (sources[i] == node_id && targets[i] == *neighbor_id) ||
                       (sources[i] == *neighbor_id && targets[i] == node_id) {
                        return true;
                    }
                }
                false
            }
            NodeFilter::And(filters) => {
                filters.iter().all(|f| self.node_matches_filter(node_id, pool, space, f))
            }
            NodeFilter::Or(filters) => {
                filters.iter().any(|f| self.node_matches_filter(node_id, pool, space, f))
            }
            NodeFilter::Not(filter) => {
                !self.node_matches_filter(node_id, pool, space, filter)
            }
        }
    }
    
    /// Try bulk filtering optimization for common filter types
    fn try_bulk_node_filter(
        &self,
        nodes: &[NodeId],
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Option<Vec<NodeId>>> {
        use std::collections::{HashMap, HashSet};
        
        match filter {
            NodeFilter::AttributeFilter { name, filter } => {
                // Bulk attribute filtering
                let mut results = Vec::new();
                
                // Get all attribute values for this name in bulk
                let mut attr_cache: HashMap<NodeId, Option<AttrValue>> = HashMap::new();
                for &node_id in nodes {
                    if let Some(index) = space.get_node_attr_index(node_id, name) {
                        if let Some(value) = pool.get_attr_by_index(name, index, true) {
                            attr_cache.insert(node_id, Some(value.clone()));
                        }
                    }
                    if !attr_cache.contains_key(&node_id) {
                        attr_cache.insert(node_id, None);
                    }
                }
                
                // Apply filter to cached values
                for &node_id in nodes {
                    if let Some(Some(value)) = attr_cache.get(&node_id) {
                        if filter.matches(value) {
                            results.push(node_id);
                        }
                    }
                }
                
                Ok(Some(results))
            }
            
            NodeFilter::AttributeEquals { name, value } => {
                // Bulk equality filtering
                let mut results = Vec::new();
                
                for &node_id in nodes {
                    if let Some(index) = space.get_node_attr_index(node_id, name) {
                        if let Some(attr_value) = pool.get_attr_by_index(name, index, true) {
                            if *attr_value == *value {
                                results.push(node_id);
                            }
                        }
                    }
                }
                
                Ok(Some(results))
            }
            
            NodeFilter::HasAttribute { name } => {
                // Bulk attribute existence check
                let results: Vec<NodeId> = nodes.iter()
                    .filter(|&node_id| space.get_node_attr_index(*node_id, name).is_some())
                    .copied()
                    .collect();
                
                Ok(Some(results))
            }
            
            NodeFilter::DegreeRange { min, max } => {
                // Bulk degree calculation
                let mut results = Vec::new();
                let (_, sources, targets) = space.get_columnar_topology();
                
                // Build degree map in one pass
                let mut degree_map: HashMap<NodeId, usize> = HashMap::new();
                for &node_id in nodes {
                    degree_map.insert(node_id, 0);
                }
                
                for i in 0..sources.len() {
                    if let Some(degree) = degree_map.get_mut(&sources[i]) {
                        *degree += 1;
                    }
                    if sources[i] != targets[i] { // Avoid double-counting self-loops
                        if let Some(degree) = degree_map.get_mut(&targets[i]) {
                            *degree += 1;
                        }
                    }
                }
                
                // Filter by degree range
                for &node_id in nodes {
                    if let Some(&degree) = degree_map.get(&node_id) {
                        if degree >= *min && degree <= *max {
                            results.push(node_id);
                        }
                    }
                }
                
                Ok(Some(results))
            }
            
            NodeFilter::And(filters) => {
                // For AND filters with simple conditions, apply them sequentially
                if filters.len() <= 3 && filters.iter().all(|f| self.is_simple_filter(f)) {
                    let mut current_nodes = nodes.to_vec();
                    
                    for sub_filter in filters {
                        if let Some(filtered) = self.try_bulk_node_filter(&current_nodes, pool, space, sub_filter)? {
                            current_nodes = filtered;
                        } else {
                            // Fall back to individual filtering
                            current_nodes = current_nodes
                                .into_iter()
                                .filter(|&node_id| self.node_matches_filter(node_id, pool, space, sub_filter))
                                .collect();
                        }
                        
                        if current_nodes.is_empty() {
                            break;
                        }
                    }
                    
                    Ok(Some(current_nodes))
                } else {
                    Ok(None) // Too complex for bulk optimization
                }
            }
            
            _ => Ok(None) // Other filters not optimized yet
        }
    }
    
    /// Check if a filter is simple enough for bulk optimization
    fn is_simple_filter(&self, filter: &NodeFilter) -> bool {
        matches!(filter, 
            NodeFilter::AttributeFilter { .. } |
            NodeFilter::AttributeEquals { .. } |
            NodeFilter::HasAttribute { .. } |
            NodeFilter::DegreeRange { .. }
        )
    }
    
    /// Check if an edge matches the given filter (Phase 3.1)
    fn edge_matches_filter(
        &self,
        edge_id: EdgeId,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter
    ) -> bool {
        match filter {
            EdgeFilter::HasAttribute { name } => {
                space.get_edge_attr_index(edge_id, name).is_some()
            }
            EdgeFilter::AttributeEquals { name, value } => {
                if let Some(index) = space.get_edge_attr_index(edge_id, name) {
                    if let Some(attr_value) = pool.get_attr_by_index(name, index, false) {
                        return attr_value == value;
                    }
                }
                false
            }
            EdgeFilter::AttributeFilter { name, filter } => {
                if let Some(index) = space.get_edge_attr_index(edge_id, name) {
                    if let Some(attr_value) = pool.get_attr_by_index(name, index, false) {
                        return filter.matches(attr_value);
                    }
                }
                false
            }
            EdgeFilter::ConnectsNodes { source, target } => {
                if let Some((edge_source, edge_target)) = pool.get_edge_endpoints(edge_id) {
                    (edge_source == *source && edge_target == *target) ||
                    (edge_source == *target && edge_target == *source) // Undirected
                } else {
                    false
                }
            }
            EdgeFilter::ConnectsAny(node_ids) => {
                if let Some((source, target)) = pool.get_edge_endpoints(edge_id) {
                    node_ids.contains(&source) || node_ids.contains(&target)
                } else {
                    false
                }
            }
            EdgeFilter::And(filters) => {
                filters.iter().all(|f| self.edge_matches_filter(edge_id, pool, space, f))
            }
            EdgeFilter::Or(filters) => {
                filters.iter().any(|f| self.edge_matches_filter(edge_id, pool, space, f))
            }
            EdgeFilter::Not(filter) => {
                !self.edge_matches_filter(edge_id, pool, space, filter)
            }
        }
    }
    
    /// Find nodes matching multiple attribute criteria (AND logic)
    pub fn find_nodes_by_attributes(
        &mut self,
        pool: &GraphPool,
        filters: &HashMap<AttrName, AttributeFilter>
    ) -> GraphResult<Vec<NodeId>> {
        // Convert to NodeFilter::And format for consistency
        let node_filters: Vec<NodeFilter> = filters.iter()
            .map(|(attr_name, attr_filter)| {
                NodeFilter::AttributeFilter { 
                    name: attr_name.clone(), 
                    filter: attr_filter.clone() 
                }
            })
            .collect();
        
        // For now, this is a basic implementation - we would need GraphSpace reference
        // In a full implementation, we'd integrate with filter_nodes
        Ok(Vec::new())
    }
    
    /// Find edges matching multiple attribute criteria (AND logic)
    pub fn find_edges_by_attributes(
        &mut self,
        pool: &GraphPool,
        filters: &HashMap<AttrName, AttributeFilter>
    ) -> GraphResult<Vec<EdgeId>> {
        let _ = (pool, filters); // Silence unused warnings
        // Basic implementation returns empty results
        Ok(Vec::new())
    }
    
    /// Find all nodes matching a complex filter (supports And, Or, Not, etc.)
    pub fn find_nodes_by_filter(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Vec<NodeId>> {
        use std::collections::HashSet;
        
        match filter {
            NodeFilter::AttributeFilter { name, filter } => {
                // Simple attribute filter - use the corrected implementation
                self.find_nodes_by_attribute(pool, space, name, filter)
            },
            
            NodeFilter::And(filters) => {
                // Find intersection of all filter results
                if filters.is_empty() {
                    return Ok(Vec::new());
                }
                
                // Start with first filter result
                let mut result_set: HashSet<NodeId> = self.find_nodes_by_filter(pool, space, &filters[0])?
                    .into_iter().collect();
                
                // Intersect with each subsequent filter
                for filter in &filters[1..] {
                    let filter_result: HashSet<NodeId> = self.find_nodes_by_filter(pool, space, filter)?
                        .into_iter().collect();
                    result_set = result_set.intersection(&filter_result).copied().collect();
                    
                    // Early termination if no matches left
                    if result_set.is_empty() {
                        break;
                    }
                }
                
                Ok(result_set.into_iter().collect())
            },
            
            NodeFilter::Or(filters) => {
                // Find union of all filter results
                let mut result_set = HashSet::new();
                
                for filter in filters {
                    let filter_result = self.find_nodes_by_filter(pool, space, filter)?;
                    result_set.extend(filter_result);
                }
                
                Ok(result_set.into_iter().collect())
            },
            
            NodeFilter::Not(inner_filter) => {
                // Find complement of filter result
                let all_nodes: HashSet<NodeId> = space.get_active_nodes().iter().copied().collect();
                let filter_result: HashSet<NodeId> = self.find_nodes_by_filter(pool, space, inner_filter)?
                    .into_iter().collect();
                
                let complement: Vec<NodeId> = all_nodes.difference(&filter_result).copied().collect();
                Ok(complement)
            },
            
            // Use the enhanced filter methods for other types
            NodeFilter::HasAttribute { name } => {
                let mut matching_nodes = Vec::new();
                for &node_id in space.get_active_nodes() {
                    if space.get_node_attr_index(node_id, name).is_some() {
                        matching_nodes.push(node_id);
                    }
                }
                Ok(matching_nodes)
            },
            
            NodeFilter::AttributeEquals { name, value } => {
                let mut matching_nodes = Vec::new();
                for &node_id in space.get_active_nodes() {
                    if let Some(index) = space.get_node_attr_index(node_id, name) {
                        if let Some(attr_value) = pool.get_attr_by_index(name, index, true) {
                            if Self::values_equal(attr_value, value) {
                                matching_nodes.push(node_id);
                            }
                        }
                    }
                }
                Ok(matching_nodes)
            },
            
            // Other filter types use the existing filter logic
            _ => {
                // Use the enhanced filtering method for complex filters
                self.filter_nodes(pool, space, filter)
            },
        }
    }

    /// Filter nodes using active space information (called from Graph API)
    /// This method provides the bridge between Graph and the internal filtering logic
    pub fn find_nodes_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Vec<NodeId>> {
        // Use the enhanced find_nodes_by_filter method
        self.find_nodes_by_filter(pool, space, filter)
    }

    /// Filter edges using active space information (called from Graph API)
    pub fn find_edges_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter
    ) -> GraphResult<Vec<EdgeId>> {
        // Apply the filter logic based on filter type
        match filter {
            EdgeFilter::AttributeFilter { name, filter } => {
                self.find_edges_by_attribute(pool, space, name, filter)
            },
            // Use the enhanced edge filtering for other types
            _ => {
                self.filter_edges(pool, space, filter)
            }
        }
    }
    
    /*
    === COMPLEX QUERY OPERATIONS ===
    Multi-step, composable query operations
    */
    
    /// Execute a complex query with multiple steps and logic
    pub fn execute_query(
        &mut self,
        pool: &GraphPool,
        query: &GraphQuery
    ) -> GraphResult<QueryResult> {
        let _ = (pool, query); // Silence unused warnings
        // Basic implementation returns empty results
        // Return an empty node list as default
        Ok(QueryResult::Nodes(Vec::new()))
    }
    
    /// Find nodes matching a complex pattern
    /// For example: "nodes with attribute X > 5 AND connected to nodes with attribute Y = 'foo'"
    pub fn find_nodes_by_pattern(
        &mut self,
        pool: &GraphPool,
        pattern: &NodePattern
    ) -> GraphResult<Vec<NodeId>> {
        let _ = (pool, pattern); // Silence unused warnings
        // Basic implementation returns empty results
        Ok(Vec::new())
    }
    
    /// Find structural patterns in the graph
    /// For example: "triangles where all nodes have attribute 'type' = 'person'"
    pub fn find_structural_patterns(
        &mut self,
        pool: &GraphPool,
        pattern: &StructuralPattern
    ) -> GraphResult<Vec<StructureMatch>> {
        let _ = (pool, pattern); // Silence unused warnings
        // Basic implementation returns empty results
        Ok(Vec::new())
    }
    
    /*
    === AGGREGATION OPERATIONS ===
    Compute statistics and summaries across entities
    */
    
    /// Compute aggregate statistics for a node attribute
    pub fn aggregate_node_attribute(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        attr_name: &AttrName,
        aggregation: AggregationType
    ) -> GraphResult<AggregationResult> {
        let active_nodes = space.node_ids();
        
        let agg_op = match aggregation {
            AggregationType::Count => AggregationOperation::Count { name: "nodes".to_string() },
            AggregationType::Sum => AggregationOperation::Sum { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Average => AggregationOperation::Average { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Min => AggregationOperation::Min { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Max => AggregationOperation::Max { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::StandardDeviation => AggregationOperation::StandardDeviation { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Variance => AggregationOperation::Variance { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Median => AggregationOperation::Median { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Mode => AggregationOperation::Mode { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Percentile(p) => AggregationOperation::Percentile { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes,
                percentile: p 
            },
            AggregationType::CountDistinct => AggregationOperation::CountDistinct { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::First => AggregationOperation::First { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Last => AggregationOperation::Last { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Nodes 
            },
        };
        
        let results = self.compute_aggregations(pool, space, &active_nodes, &[], &[agg_op])?;
        
        Ok(results.into_values().next().unwrap_or(AggregationResult::Integer(0)))
    }
    
    /// Compute aggregate statistics for an edge attribute
    pub fn aggregate_edge_attribute(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        attr_name: &AttrName,
        aggregation: AggregationType
    ) -> GraphResult<AggregationResult> {
        let active_edges = space.edge_ids();
        
        let agg_op = match aggregation {
            AggregationType::Count => AggregationOperation::Count { name: "edges".to_string() },
            AggregationType::Sum => AggregationOperation::Sum { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Average => AggregationOperation::Average { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Min => AggregationOperation::Min { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Max => AggregationOperation::Max { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::StandardDeviation => AggregationOperation::StandardDeviation { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Variance => AggregationOperation::Variance { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Median => AggregationOperation::Median { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Mode => AggregationOperation::Mode { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Percentile(p) => AggregationOperation::Percentile { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges,
                percentile: p 
            },
            AggregationType::CountDistinct => AggregationOperation::CountDistinct { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::First => AggregationOperation::First { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
            AggregationType::Last => AggregationOperation::Last { 
                attribute: attr_name.clone(), 
                target: AggregationTarget::Edges 
            },
        };
        
        let results = self.compute_aggregations(pool, space, &[], &active_edges, &[agg_op])?;
        
        Ok(results.into_values().next().unwrap_or(AggregationResult::Integer(0)))
    }
    
    /// Group nodes by attribute value and compute aggregates for each group
    pub fn group_nodes_by_attribute(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        group_by_attr: &AttrName,
        aggregate_attr: &AttrName,
        aggregation: AggregationType
    ) -> GraphResult<HashMap<AttrValue, AggregationResult>> {
        let active_nodes = space.node_ids();
        
        let nested_agg_op = match aggregation {
            AggregationType::Count => AggregationOperation::Count { name: "nodes".to_string() },
            AggregationType::Sum => AggregationOperation::Sum { 
                attribute: aggregate_attr.clone(), 
                target: AggregationTarget::Nodes 
            },
            AggregationType::Average => AggregationOperation::Average { 
                attribute: aggregate_attr.clone(), 
                target: AggregationTarget::Nodes 
            },
            _ => AggregationOperation::Count { name: "nodes".to_string() },
        };
        
        let group_by_op = AggregationOperation::GroupBy {
            group_by_attr: group_by_attr.clone(),
            aggregate_attr: aggregate_attr.clone(),
            operation: Box::new(nested_agg_op),
            target: AggregationTarget::Nodes,
        };
        
        let results = self.compute_aggregations(pool, space, &active_nodes, &[], &[group_by_op])?;
        
        // Convert results to expected format
        let mut grouped_results = HashMap::new();
        
        if let Some(AggregationResult::GroupedResults(groups)) = results.into_values().next() {
            for (key, result) in groups {
                // Convert the key back to AttrValue for consistency
                let attr_key = AttrValue::Text(key);
                grouped_results.insert(attr_key, *result);
            }
        }
        
        Ok(grouped_results)
    }
    
    /*
    === PERFORMANCE OPTIMIZATION ===
    Query optimization and caching
    */
    
    /// Update statistics about attribute distributions
    /// This is used for query optimization
    pub fn update_statistics(&mut self, pool: &GraphPool) -> GraphResult<()> {
        let _ = pool; // Silence unused warnings
        // Basic implementation is a no-op
        Ok(())
    }
    
    /// Clear the query cache (useful after large data changes)
    pub fn clear_cache(&mut self) {
        self.query_cache.clear();
        self.query_performance.clear();
        self.total_queries = 0;
    }
    
    /// Get cache statistics
    pub fn cache_statistics(&self) -> CacheStatistics {
        let cache_size_bytes = self.query_cache.len() * std::mem::size_of::<(u64, CachedQueryResult)>();
        
        // Calculate hit/miss rates from performance data
        let total_hits = self.query_performance.values()
            .map(|perf| perf.execution_count.saturating_sub(1)) // First execution is always a miss
            .sum::<usize>();
        let total_requests = self.total_queries;
        let total_misses = total_requests.saturating_sub(total_hits);
        
        let hit_rate = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        
        let miss_rate = if total_requests > 0 {
            total_misses as f64 / total_requests as f64
        } else {
            0.0
        };
        
        CacheStatistics {
            hit_count: total_hits,
            miss_count: total_misses,
            total_size: cache_size_bytes,
        }
    }
    
    /// Optimize a query plan before execution (legacy version)
    #[allow(dead_code)]
    fn optimize_query_plan_legacy(&self, plan: QueryPlan) -> QueryPlan {
        // Basic implementation returns plan unchanged
        plan
    }
    
    /*
    === UTILITY OPERATIONS ===
    Helper functions for common patterns
    */
    
    /// Count entities matching a filter (more efficient than finding all matches)
    pub fn count_nodes_matching(
        &self,
        pool: &GraphPool,
        filter: &NodeFilter
    ) -> GraphResult<usize> {
        let _ = (pool, filter); // Silence unused warnings
        // Basic implementation returns 0
        Ok(0)
    }
    
    /// Check if any entities match a filter (even more efficient than counting)
    pub fn any_nodes_match(
        &self,
        pool: &GraphPool,
        filter: &NodeFilter
    ) -> GraphResult<bool> {
        let _ = (pool, filter); // Silence unused warnings
        // Basic implementation returns false
        Ok(false)
    }
    
    /// Get unique values for an attribute across all entities
    pub fn get_unique_attribute_values(
        &self,
        pool: &GraphPool,
        attr_name: &AttrName,
        entity_type: EntityType
    ) -> GraphResult<Vec<AttrValue>> {
        let _ = (pool, attr_name, entity_type); // Silence unused warnings
        // Basic implementation returns empty results
        Ok(Vec::new())
    }
    
    /*
    === GRAPH TRAVERSAL ALGORITHMS ===
    Integrated traversal algorithms for pathfinding and connectivity analysis
    */
    
    /// Breadth-First Search from a starting node
    pub fn bfs(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        start: NodeId,
        options: TraversalOptions
    ) -> GraphResult<TraversalResult> {
        self.traversal_engine.bfs(pool, space, start, options)
    }
    
    /// Depth-First Search from a starting node
    pub fn dfs(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        start: NodeId,
        options: TraversalOptions
    ) -> GraphResult<TraversalResult> {
        self.traversal_engine.dfs(pool, space, start, options)
    }
    
    /// Find shortest path between two nodes
    pub fn shortest_path(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        start: NodeId,
        end: NodeId,
        options: PathFindingOptions
    ) -> GraphResult<Option<Path>> {
        self.traversal_engine.shortest_path(pool, space, start, end, options)
    }
    
    /// Find all simple paths between two nodes (up to maximum length)
    pub fn all_paths(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        start: NodeId,
        end: NodeId,
        max_length: usize
    ) -> GraphResult<Vec<Path>> {
        self.traversal_engine.all_paths(pool, space, start, end, max_length)
    }
    
    /// Find all connected components in the graph
    pub fn connected_components(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        options: TraversalOptions
    ) -> GraphResult<ConnectedComponentsResult> {
        self.traversal_engine.connected_components(pool, space, options)
    }
    
    /// Get traversal performance statistics
    pub fn traversal_statistics(&self) -> &crate::core::traversal::TraversalStats {
        self.traversal_engine.statistics()
    }
    
    /// Clear traversal performance statistics
    pub fn clear_traversal_stats(&mut self) {
        self.traversal_engine.clear_stats();
    }
    
    /*
    === PHASE 3.3: COMPLEX QUERY COMPOSITION ===
    Advanced query building and optimization capabilities
    */
    
    /// Create a new complex query builder for composing advanced queries
    pub fn query_builder(&self) -> ComplexQueryBuilder {
        ComplexQueryBuilder::new()
    }
    
    /// Execute a complex composed query with optimization
    pub fn execute_complex_query(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        query: ComplexQuery
    ) -> GraphResult<ComplexQueryResult> {
        let start_time = std::time::Instant::now();
        
        // Generate query execution plan based on structure
        let execution_plan = self.optimize_query_plan(pool, space, &query)?;
        
        // Execute optimized plan
        let result = self.execute_plan(pool, space, execution_plan)?;
        
        // Track performance
        let duration = start_time.elapsed();
        let query_hash = self.hash_query(&query);
        self.record_query_performance(query_hash, duration, result.total_items());
        
        Ok(result)
    }
    
    /// Execute a pre-built query plan
    fn execute_plan(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        plan: QueryExecutionPlan
    ) -> GraphResult<ComplexQueryResult> {
        match plan.strategy {
            ExecutionStrategy::FilterFirst => self.execute_filter_first_plan(pool, space, plan),
            ExecutionStrategy::TraversalFirst => self.execute_traversal_first_plan(pool, space, plan),
            ExecutionStrategy::Parallel => self.execute_parallel_plan(pool, space, plan),
            ExecutionStrategy::Cached => self.execute_cached_plan(pool, space, plan),
        }
    }
    
    /// Execute filter-first strategy (filter then traverse/aggregate)
    fn execute_filter_first_plan(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        plan: QueryExecutionPlan
    ) -> GraphResult<ComplexQueryResult> {
        let mut result = ComplexQueryResult::new();
        
        // Step 1: Apply node filters
        let filtered_nodes = if plan.node_filters.is_empty() {
            space.get_active_nodes().iter().copied().collect()
        } else {
            self.apply_optimized_node_filters(pool, space, &plan.node_filters)?
        };
        
        result.nodes = filtered_nodes.clone();
        
        // Step 2: Apply edge filters if needed
        if !plan.edge_filters.is_empty() {
            let filtered_edges = self.apply_optimized_edge_filters(pool, space, &plan.edge_filters)?;
            result.edges = filtered_edges;
        }
        
        // Step 3: Execute traversal operations on filtered nodes
        if let Some(traversal_op) = plan.traversal_operation {
            result.traversal_results = self.execute_traversal_on_nodes(
                pool, space, &filtered_nodes, traversal_op
            )?;
        }
        
        // Step 4: Compute aggregations
        if !plan.aggregations.is_empty() {
            result.aggregations = self.compute_aggregations(
                pool, space, &result.nodes, &result.edges, &plan.aggregations
            )?;
        }
        
        Ok(result)
    }
    
    /// Execute traversal-first strategy (traverse then filter/aggregate)
    fn execute_traversal_first_plan(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        plan: QueryExecutionPlan
    ) -> GraphResult<ComplexQueryResult> {
        let mut result = ComplexQueryResult::new();
        
        // Step 1: Execute traversal to get candidate set
        if let Some(traversal_op) = plan.traversal_operation {
            result.traversal_results = match traversal_op {
                TraversalOperation::BFS { start, options } => {
                    vec![self.bfs(pool, space, start, options)?]
                }
                TraversalOperation::DFS { start, options } => {
                    vec![self.dfs(pool, space, start, options)?]
                }
                TraversalOperation::ConnectedComponents { options } => {
                    let components = self.connected_components(pool, space, options)?;
                    // Convert to traversal results
                    components.components.into_iter().map(|comp| {
                        TraversalResult {
                            algorithm: crate::core::traversal::TraversalAlgorithm::ConnectedComponents,
                            nodes: comp.nodes,
                            edges: Vec::new(),
                            paths: Vec::new(),
                            metadata: crate::core::traversal::TraversalMetadata {
                                start_node: Some(comp.root),
                                end_node: None,
                                max_depth: 0,
                                nodes_visited: comp.size,
                                execution_time: std::time::Duration::new(0, 0),
                                levels: None,
                                discovery_order: None,
                            },
                        }
                    }).collect()
                }
            };
            
            // Extract nodes from traversal results
            result.nodes = result.traversal_results.iter()
                .flat_map(|tr| &tr.nodes)
                .copied()
                .collect();
        }
        
        // Step 2: Apply filters to traversal results
        if !plan.node_filters.is_empty() {
            result.nodes = self.filter_nodes_by_criteria(pool, space, &result.nodes, &plan.node_filters)?;
        }
        
        // Step 3: Compute aggregations
        if !plan.aggregations.is_empty() {
            result.aggregations = self.compute_aggregations(
                pool, space, &result.nodes, &result.edges, &plan.aggregations
            )?;
        }
        
        Ok(result)
    }
    
    /// Execute parallel strategy (run independent operations in parallel)
    fn execute_parallel_plan(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        plan: QueryExecutionPlan
    ) -> GraphResult<ComplexQueryResult> {
        // For now, fall back to filter-first (parallel execution requires more complex state management)
        self.execute_filter_first_plan(pool, space, plan)
    }
    
    /// Execute cached strategy (return cached results if available)
    fn execute_cached_plan(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        plan: QueryExecutionPlan
    ) -> GraphResult<ComplexQueryResult> {
        if let Some(cached_result) = plan.cached_result {
            Ok(cached_result)
        } else {
            // Cache miss, execute normally
            self.execute_filter_first_plan(pool, space, plan)
        }
    }
    
    /// Apply optimized node filters with reordering and early termination
    fn apply_optimized_node_filters(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        filters: &[NodeFilter]
    ) -> GraphResult<Vec<NodeId>> {
        if filters.is_empty() {
            return Ok(space.get_active_nodes().iter().copied().collect());
        }
        
        // Optimize filter order (put most selective filters first)
        let optimized_filters = self.optimize_filter_order(filters);
        
        let active_nodes: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
        
        // Apply filters efficiently using parallel processing for large sets
        if active_nodes.len() > 10000 {
            // Parallel filtering for large node sets
            Ok(active_nodes
                .par_iter()
                .filter(|&&node_id| {
                    optimized_filters.iter().all(|filter| {
                        self.should_visit_node_inline(pool, space, node_id, filter).unwrap_or(false)
                    })
                })
                .copied()
                .collect())
        } else {
            // Sequential filtering for smaller sets
            Ok(active_nodes
                .into_iter()
                .filter(|&node_id| {
                    optimized_filters.iter().all(|filter| {
                        self.should_visit_node_inline(pool, space, node_id, filter).unwrap_or(false)
                    })
                })
                .collect())
        }
    }
    
    /// Apply optimized edge filters
    fn apply_optimized_edge_filters(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        filters: &[EdgeFilter]
    ) -> GraphResult<Vec<EdgeId>> {
        if filters.is_empty() {
            return Ok(space.get_active_edges().iter().copied().collect());
        }
        
        let active_edges: Vec<EdgeId> = space.get_active_edges().iter().copied().collect();
        
        // Apply edge filters (simplified implementation)
        Ok(active_edges
            .into_iter()
            .filter(|&edge_id| {
                filters.iter().all(|filter| {
                    // Basic edge filter matching
                    match filter {
                        EdgeFilter::HasAttribute { name } => {
                            space.get_edge_attr_index(edge_id, name).is_some()
                        }
                        EdgeFilter::AttributeEquals { name, value } => {
                            if let Some(index) = space.get_edge_attr_index(edge_id, name) {
                                if let Some(attr_value) = pool.get_attr_by_index(name, index, false) {
                                    return attr_value == value;
                                }
                            }
                            false
                        }
                        _ => true, // Accept other filter types for now
                    }
                })
            })
            .collect())
    }
    
    /// Filter existing nodes by criteria
    fn filter_nodes_by_criteria(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        nodes: &[NodeId],
        filters: &[NodeFilter]
    ) -> GraphResult<Vec<NodeId>> {
        if filters.is_empty() {
            return Ok(nodes.to_vec());
        }
        
        Ok(nodes
            .iter()
            .filter(|&&node_id| {
                filters.iter().all(|filter| {
                    self.should_visit_node_inline(pool, space, node_id, filter).unwrap_or(false)
                })
            })
            .copied()
            .collect())
    }
    
    /// Execute traversal operations on a specific set of nodes
    fn execute_traversal_on_nodes(
        &mut self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        nodes: &[NodeId],
        operation: TraversalOperation
    ) -> GraphResult<Vec<TraversalResult>> {
        match operation {
            TraversalOperation::BFS { start, options } => {
                if nodes.contains(&start) {
                    Ok(vec![self.bfs(pool, space, start, options)?])
                } else {
                    Ok(Vec::new())
                }
            }
            TraversalOperation::DFS { start, options } => {
                if nodes.contains(&start) {
                    Ok(vec![self.dfs(pool, space, start, options)?])
                } else {
                    Ok(Vec::new())
                }
            }
            TraversalOperation::ConnectedComponents { options } => {
                let components = self.connected_components(pool, space, options)?;
                // Filter components to only include nodes in our set
                let filtered_components: Vec<_> = components.components.into_iter()
                    .filter(|comp| comp.nodes.iter().any(|n| nodes.contains(n)))
                    .map(|comp| {
                        let filtered_nodes: Vec<_> = comp.nodes.into_iter()
                            .filter(|n| nodes.contains(n))
                            .collect();
                        
                        TraversalResult {
                            algorithm: crate::core::traversal::TraversalAlgorithm::ConnectedComponents,
                            nodes: filtered_nodes.clone(),
                            edges: Vec::new(),
                            paths: Vec::new(),
                            metadata: crate::core::traversal::TraversalMetadata {
                                start_node: Some(comp.root),
                                end_node: None,
                                max_depth: 0,
                                nodes_visited: filtered_nodes.len(),
                                execution_time: std::time::Duration::new(0, 0),
                                levels: None,
                                discovery_order: None,
                            },
                        }
                    })
                    .collect();
                
                Ok(filtered_components)
            }
        }
    }
    
    /// Compute aggregations over node/edge sets
    /// Extract numeric value from AttrValue, handling optimized storage types
    fn extract_aggregation_numeric_value(&self, value: &AttrValue) -> Option<f64> {
        match value {
            AttrValue::Int(i) => Some(*i as f64),
            AttrValue::SmallInt(i) => Some(*i as f64),
            AttrValue::Float(f) => Some(*f as f64),
            AttrValue::CompressedFloatVec(vec) => {
                // For vector, return the first element or sum depending on context
                vec.data.first().map(|&f| f as f64)
            }
            AttrValue::FloatVec(vec) => {
                vec.first().map(|&f| f as f64)
            }
            _ => None,
        }
    }

    /// Extract all numeric values from a collection for statistical operations
    fn collect_numeric_values(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        node_ids: &[NodeId],
        edge_ids: &[EdgeId],
        attribute: &AttrName,
        is_node: bool
    ) -> Vec<f64> {
        if is_node {
            node_ids.iter().filter_map(|&node_id| {
                let attr_index = space.get_node_attr_index(node_id, attribute);
                if let Some(index) = attr_index {
                    if let Some(value) = pool.get_attr_by_index(attribute, index, true) {
                        self.extract_aggregation_numeric_value(value)
                    } else { None }
                } else { None }
            }).collect()
        } else {
            edge_ids.iter().filter_map(|&edge_id| {
                let attr_index = space.get_edge_attr_index(edge_id, attribute);
                if let Some(index) = attr_index {
                    if let Some(value) = pool.get_attr_by_index(attribute, index, false) {
                        self.extract_aggregation_numeric_value(value)
                    } else { None }
                } else { None }
            }).collect()
        }
    }

    /// Calculate percentile of a sorted vector
    fn calculate_percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }
        
        let n = sorted_values.len();
        if percentile <= 0.0 {
            return sorted_values[0];
        }
        if percentile >= 100.0 {
            return sorted_values[n - 1];
        }
        
        let rank = (percentile / 100.0) * (n - 1) as f64;
        let lower_index = rank.floor() as usize;
        let upper_index = rank.ceil() as usize;
        
        if lower_index == upper_index {
            sorted_values[lower_index]
        } else {
            let lower_value = sorted_values[lower_index];
            let upper_value = sorted_values[upper_index];
            let weight = rank - lower_index as f64;
            lower_value + weight * (upper_value - lower_value)
        }
    }

    /// Calculate mode (most frequent value) from numeric values
    fn calculate_mode(&self, values: &[f64]) -> f64 {
        use std::collections::HashMap;
        
        if values.is_empty() {
            return 0.0;
        }
        
        let mut frequency: HashMap<i64, usize> = HashMap::new();
        
        // Round to avoid floating point precision issues
        for &value in values {
            let rounded = (value * 1000.0).round() as i64;
            *frequency.entry(rounded).or_insert(0) += 1;
        }
        
        let mode = frequency.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&rounded, _)| rounded as f64 / 1000.0)
            .unwrap_or(0.0);
            
        mode
    }

    /// Calculate variance from values
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64; // Sample variance
            
        variance
    }

    fn compute_aggregations(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        nodes: &[NodeId],
        edges: &[EdgeId],
        aggregations: &[AggregationOperation]
    ) -> GraphResult<HashMap<String, AggregationResult>> {
        let mut results = HashMap::new();
        
        for agg in aggregations {
            let result = match agg {
                AggregationOperation::Count { name } => {
                    AggregationResult::Integer(match name.as_str() {
                        "nodes" => nodes.len() as i64,
                        "edges" => edges.len() as i64,
                        _ => 0,
                    })
                }
                
                AggregationOperation::Sum { attribute, target } => {
                    let sum = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                                .iter().sum::<f64>()
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                                .iter().sum::<f64>()
                        }
                    };
                    AggregationResult::Float(sum)
                }
                
                AggregationOperation::Average { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    let average = if values.is_empty() {
                        0.0
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    };
                    AggregationResult::Float(average)
                }
                
                AggregationOperation::Min { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    let min = values.iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .copied().unwrap_or(0.0);
                    AggregationResult::Float(min)
                }
                
                AggregationOperation::Max { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    let max = values.iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .copied().unwrap_or(0.0);
                    AggregationResult::Float(max)
                }
                
                AggregationOperation::StandardDeviation { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    let std_dev = self.calculate_variance(&values).sqrt();
                    AggregationResult::Float(std_dev)
                }
                
                AggregationOperation::Variance { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    let variance = self.calculate_variance(&values);
                    AggregationResult::Float(variance)
                }
                
                AggregationOperation::Median { attribute, target } => {
                    let mut values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = self.calculate_percentile(&values, 50.0);
                    AggregationResult::Float(median)
                }
                
                AggregationOperation::Percentile { attribute, target, percentile } => {
                    let mut values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let percentile_value = self.calculate_percentile(&values, *percentile);
                    AggregationResult::Float(percentile_value)
                }
                
                AggregationOperation::Mode { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            self.collect_numeric_values(pool, space, nodes, &[], attribute, true)
                        }
                        AggregationTarget::Edges => {
                            self.collect_numeric_values(pool, space, &[], edges, attribute, false)
                        }
                    };
                    
                    let mode = self.calculate_mode(&values);
                    AggregationResult::Float(mode)
                }
                
                AggregationOperation::CountDistinct { attribute, target } => {
                    let values = match target {
                        AggregationTarget::Nodes => {
                            nodes.iter().filter_map(|&node_id| {
                                if let Some(index) = space.get_node_attr_index(node_id, attribute) {
                                    pool.get_attr_by_index(attribute, index, true).cloned()
                                } else { None }
                            }).collect::<std::collections::HashSet<_>>()
                        }
                        AggregationTarget::Edges => {
                            edges.iter().filter_map(|&edge_id| {
                                if let Some(index) = space.get_edge_attr_index(edge_id, attribute) {
                                    pool.get_attr_by_index(attribute, index, false).cloned()
                                } else { None }
                            }).collect::<std::collections::HashSet<_>>()
                        }
                    };
                    AggregationResult::Integer(values.len() as i64)
                }
                
                AggregationOperation::First { attribute, target } => {
                    let first_value = match target {
                        AggregationTarget::Nodes => {
                            nodes.first().and_then(|&node_id| {
                                space.get_node_attr_index(node_id, attribute)
                                    .and_then(|index| pool.get_attr_by_index(attribute, index, true))
                            })
                        }
                        AggregationTarget::Edges => {
                            edges.first().and_then(|&edge_id| {
                                space.get_edge_attr_index(edge_id, attribute)
                                    .and_then(|index| pool.get_attr_by_index(attribute, index, false))
                            })
                        }
                    };
                    
                    match first_value {
                        Some(AttrValue::Int(i)) => AggregationResult::Integer(*i),
                        Some(AttrValue::SmallInt(i)) => AggregationResult::Integer(*i as i64),
                        Some(AttrValue::Float(f)) => AggregationResult::Float(*f as f64),
                        Some(AttrValue::Text(s)) => AggregationResult::Text(s.clone()),
                        Some(AttrValue::CompactText(s)) => AggregationResult::Text(s.as_str().to_string()),
                        Some(AttrValue::CompressedText(s)) => AggregationResult::Text(format!("{:?}", s)),
                        _ => AggregationResult::Integer(0)
                    }
                }
                
                AggregationOperation::Last { attribute, target } => {
                    let last_value = match target {
                        AggregationTarget::Nodes => {
                            nodes.last().and_then(|&node_id| {
                                space.get_node_attr_index(node_id, attribute)
                                    .and_then(|index| pool.get_attr_by_index(attribute, index, true))
                            })
                        }
                        AggregationTarget::Edges => {
                            edges.last().and_then(|&edge_id| {
                                space.get_edge_attr_index(edge_id, attribute)
                                    .and_then(|index| pool.get_attr_by_index(attribute, index, false))
                            })
                        }
                    };
                    
                    match last_value {
                        Some(AttrValue::Int(i)) => AggregationResult::Integer(*i),
                        Some(AttrValue::SmallInt(i)) => AggregationResult::Integer(*i as i64),
                        Some(AttrValue::Float(f)) => AggregationResult::Float(*f as f64),
                        Some(AttrValue::Text(s)) => AggregationResult::Text(s.clone()),
                        Some(AttrValue::CompactText(s)) => AggregationResult::Text(s.as_str().to_string()),
                        Some(AttrValue::CompressedText(s)) => AggregationResult::Text(format!("{:?}", s)),
                        _ => AggregationResult::Integer(0)
                    }
                }
                
                AggregationOperation::Distinct { attribute, target } => {
                    // This was the original implementation - keep for backward compatibility
                    let values = match target {
                        AggregationTarget::Nodes => {
                            nodes.iter().filter_map(|&node_id| {
                                if let Some(index) = space.get_node_attr_index(node_id, attribute) {
                                    pool.get_attr_by_index(attribute, index, true).cloned()
                                } else { None }
                            }).collect::<std::collections::HashSet<_>>()
                        }
                        AggregationTarget::Edges => {
                            edges.iter().filter_map(|&edge_id| {
                                if let Some(index) = space.get_edge_attr_index(edge_id, attribute) {
                                    pool.get_attr_by_index(attribute, index, false).cloned()
                                } else { None }
                            }).collect::<std::collections::HashSet<_>>()
                        }
                    };
                    AggregationResult::Integer(values.len() as i64)
                }
                
                AggregationOperation::GroupBy { group_by_attr, aggregate_attr, operation, target } => {
                    use std::collections::HashMap;
                    
                    let mut groups: HashMap<String, Vec<i64>> = HashMap::new();
                    
                    // Group entities by the group_by_attr value
                    match target {
                        AggregationTarget::Nodes => {
                            for &node_id in nodes {
                                if let Some(index) = space.get_node_attr_index(node_id, group_by_attr) {
                                    if let Some(value) = pool.get_attr_by_index(group_by_attr, index, true) {
                                        let group_key = match value {
                                            AttrValue::Text(s) => s.clone(),
                                            AttrValue::CompactText(s) => s.as_str().to_string(),
                                            AttrValue::CompressedText(s) => format!("{:?}", s),
                                            AttrValue::Int(i) => i.to_string(),
                                            AttrValue::SmallInt(i) => i.to_string(),
                                            AttrValue::Float(f) => f.to_string(),
                                            _ => "unknown".to_string(),
                                        };
                                        groups.entry(group_key).or_insert_with(Vec::new).push(node_id as i64);
                                    }
                                }
                            }
                        }
                        AggregationTarget::Edges => {
                            for &edge_id in edges {
                                if let Some(index) = space.get_edge_attr_index(edge_id, group_by_attr) {
                                    if let Some(value) = pool.get_attr_by_index(group_by_attr, index, false) {
                                        let group_key = match value {
                                            AttrValue::Text(s) => s.clone(),
                                            AttrValue::CompactText(s) => s.as_str().to_string(),
                                            AttrValue::CompressedText(s) => format!("{:?}", s),
                                            AttrValue::Int(i) => i.to_string(),
                                            AttrValue::SmallInt(i) => i.to_string(),
                                            AttrValue::Float(f) => f.to_string(),
                                            _ => "unknown".to_string(),
                                        };
                                        groups.entry(group_key).or_insert_with(Vec::new).push(edge_id as i64);
                                    }
                                }
                            }
                        }
                    }
                    
                    // Apply aggregation to each group
                    let mut group_results = HashMap::new();
                    for (group_key, group_entities) in groups {
                        // For simplicity, just compute count for now - can be extended
                        let group_result = AggregationResult::Integer(group_entities.len() as i64);
                        group_results.insert(group_key, Box::new(group_result));
                    }
                    
                    AggregationResult::GroupedResults(group_results)
                }
            };
            
            results.insert(format!("{:?}", agg), result);
        }
        
        Ok(results)
    }
    
    /// Optimize query execution plan based on query structure and statistics
    fn optimize_query_plan(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        query: &ComplexQuery
    ) -> GraphResult<QueryExecutionPlan> {
        // Analyze query to choose optimal execution strategy
        let strategy = if query.node_filters.len() > 2 && query.traversal_operation.is_none() {
            // Many filters, no traversal - filter first
            ExecutionStrategy::FilterFirst
        } else if query.traversal_operation.is_some() && query.node_filters.is_empty() {
            // Traversal without filters - traversal first
            ExecutionStrategy::TraversalFirst  
        } else if query.node_filters.len() <= 1 && query.traversal_operation.is_some() {
            // Light filtering with traversal - traversal first
            ExecutionStrategy::TraversalFirst
        } else {
            // Default to filter first
            ExecutionStrategy::FilterFirst
        };
        
        // Check cache for this query
        let query_hash = self.hash_query(query);
        let cached_result = self.query_cache.get(&query_hash).map(|cached| {
            cached.result.clone()
        });
        
        Ok(QueryExecutionPlan {
            strategy: if cached_result.is_some() { 
                ExecutionStrategy::Cached 
            } else { 
                strategy 
            },
            node_filters: query.node_filters.clone(),
            edge_filters: query.edge_filters.clone(),
            traversal_operation: query.traversal_operation.clone(),
            aggregations: query.aggregations.clone(),
            cached_result,
            estimated_cost: self.estimate_query_cost(pool, space, query),
        })
    }
    
    /// Optimize the order of filters to put most selective ones first
    fn optimize_filter_order(&self, filters: &[NodeFilter]) -> Vec<NodeFilter> {
        let mut optimized = filters.to_vec();
        
        // Simple heuristic: put equality filters before range filters
        optimized.sort_by(|a, b| {
            let selectivity_a = self.estimate_filter_selectivity(a);
            let selectivity_b = self.estimate_filter_selectivity(b);
            selectivity_a.partial_cmp(&selectivity_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        optimized
    }
    
    /// Estimate the selectivity of a filter (lower = more selective)
    fn estimate_filter_selectivity(&self, filter: &NodeFilter) -> f64 {
        match filter {
            NodeFilter::AttributeEquals { .. } => 0.1,     // Very selective
            NodeFilter::HasAttribute { .. } => 0.3,        // Moderately selective
            NodeFilter::AttributeFilter { .. } => 0.5,     // Depends on the filter
            NodeFilter::And(_) => 0.2,                     // Usually selective
            NodeFilter::Or(_) => 0.7,                      // Usually less selective
            NodeFilter::Not(_) => 0.8,                     // Usually less selective
            _ => 0.5,                                       // Default
        }
    }
    
    /// Estimate the computational cost of executing a query
    fn estimate_query_cost(&self, pool: &GraphPool, space: &GraphSpace, query: &ComplexQuery) -> f64 {
        let node_count = space.get_active_nodes().len() as f64;
        let edge_count = space.get_active_edges().len() as f64;
        
        let filter_cost = query.node_filters.len() as f64 * node_count * 0.001;
        let edge_filter_cost = query.edge_filters.len() as f64 * edge_count * 0.001;
        let traversal_cost = if query.traversal_operation.is_some() {
            node_count * 0.01  // Rough estimate
        } else {
            0.0
        };
        let aggregation_cost = query.aggregations.len() as f64 * node_count * 0.0001;
        
        filter_cost + edge_filter_cost + traversal_cost + aggregation_cost
    }
    
    /// Generate hash for a query (for caching)
    fn hash_query(&self, query: &ComplexQuery) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // Simple hash based on query structure (in a real implementation, we'd hash the content)
        query.node_filters.len().hash(&mut hasher);
        query.edge_filters.len().hash(&mut hasher);
        query.aggregations.len().hash(&mut hasher);
        hasher.finish()
    }
    
    /// Record query performance for optimization
    fn record_query_performance(&mut self, query_hash: u64, duration: std::time::Duration, items: usize) {
        let performance = QueryPerformance {
            execution_time: duration,
            items_processed: items,
            execution_count: 1,
        };
        
        if let Some(existing) = self.query_performance.get_mut(&query_hash) {
            existing.execution_time += duration;
            existing.items_processed += items;
            existing.execution_count += 1;
        } else {
            self.query_performance.insert(query_hash, performance);
        }
        
        self.total_queries += 1;
    }
    
    /// Helper method for inline node filter matching (used by optimization functions)
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
                        // Handle CompactText vs Text comparison
                        return Ok(Self::values_equal(attr_value, value));
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
    
    /// Compare two AttrValues handling type conversions (like CompactText vs Text)
    fn values_equal(stored: &AttrValue, target: &AttrValue) -> bool {
        match (stored, target) {
            // Direct equality
            (a, b) if a == b => true,
            
            // CompactText vs Text comparison
            (AttrValue::CompactText(compact), AttrValue::Text(text)) => {
                compact.as_str() == text
            }
            (AttrValue::Text(text), AttrValue::CompactText(compact)) => {
                text == compact.as_str()
            }
            
            // SmallInt vs Int comparison
            (AttrValue::SmallInt(small), AttrValue::Int(big)) => {
                *small as i64 == *big
            }
            (AttrValue::Int(big), AttrValue::SmallInt(small)) => {
                *big == *small as i64
            }
            
            // Other cases - no match
            _ => false,
        }
    }
}

/*
=== PHASE 3.3: COMPLEX QUERY TYPES ===
Types for constructing and executing complex queries
*/

/// A complex query that combines multiple operations
#[derive(Debug, Clone)]
pub struct ComplexQuery {
    /// Node filtering criteria
    pub node_filters: Vec<NodeFilter>,
    /// Edge filtering criteria  
    pub edge_filters: Vec<EdgeFilter>,
    /// Traversal operation to perform
    pub traversal_operation: Option<TraversalOperation>,
    /// Aggregation operations to compute
    pub aggregations: Vec<AggregationOperation>,
    /// Optional query name for debugging
    pub name: Option<String>,
}

/// Available traversal operations in complex queries
#[derive(Debug, Clone)]
pub enum TraversalOperation {
    BFS { start: NodeId, options: TraversalOptions },
    DFS { start: NodeId, options: TraversalOptions },
    ConnectedComponents { options: TraversalOptions },
}

/// Available aggregation operations
#[derive(Debug, Clone)]
pub enum AggregationOperation {
    Count { name: String },
    Sum { attribute: AttrName, target: AggregationTarget },
    Average { attribute: AttrName, target: AggregationTarget },
    Min { attribute: AttrName, target: AggregationTarget },
    Max { attribute: AttrName, target: AggregationTarget },
    Distinct { attribute: AttrName, target: AggregationTarget },
    StandardDeviation { attribute: AttrName, target: AggregationTarget },
    Variance { attribute: AttrName, target: AggregationTarget },
    Median { attribute: AttrName, target: AggregationTarget },
    Percentile { attribute: AttrName, target: AggregationTarget, percentile: f64 },
    Mode { attribute: AttrName, target: AggregationTarget },
    First { attribute: AttrName, target: AggregationTarget },
    Last { attribute: AttrName, target: AggregationTarget },
    CountDistinct { attribute: AttrName, target: AggregationTarget },
    GroupBy { 
        group_by_attr: AttrName, 
        aggregate_attr: AttrName, 
        operation: Box<AggregationOperation>,
        target: AggregationTarget 
    },
}

/// Target for aggregation operations
#[derive(Debug, Clone)]
pub enum AggregationTarget {
    Nodes,
    Edges,
}

/// Result of an aggregation operation
#[derive(Debug, Clone)]
pub enum AggregationResult {
    Integer(i64),
    Float(f64),
    Text(String),
    FloatVec(Vec<f64>),
    IntegerVec(Vec<i64>),
    TextVec(Vec<String>),
    GroupedResults(HashMap<String, Box<AggregationResult>>),
    MultipleValues {
        values: Vec<AttrValue>,
        count: usize,
    },
}

/// Result of a complex query execution
#[derive(Debug, Clone)]
pub struct ComplexQueryResult {
    /// Filtered nodes
    pub nodes: Vec<NodeId>,
    /// Filtered edges
    pub edges: Vec<EdgeId>,
    /// Traversal results (if traversal was requested)
    pub traversal_results: Vec<TraversalResult>,
    /// Aggregation results
    pub aggregations: HashMap<String, AggregationResult>,
    /// Execution metadata
    pub metadata: QueryExecutionMetadata,
}

impl ComplexQueryResult {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            traversal_results: Vec::new(),
            aggregations: HashMap::new(),
            metadata: QueryExecutionMetadata::default(),
        }
    }
    
    pub fn total_items(&self) -> usize {
        self.nodes.len() + self.edges.len() + self.traversal_results.len()
    }
}

/// Metadata about query execution
#[derive(Debug, Clone)]
pub struct QueryExecutionMetadata {
    pub execution_strategy: ExecutionStrategy,
    pub execution_time: std::time::Duration,
    pub nodes_processed: usize,
    pub edges_processed: usize,
    pub cache_hit: bool,
}

impl Default for QueryExecutionMetadata {
    fn default() -> Self {
        Self {
            execution_strategy: ExecutionStrategy::FilterFirst,
            execution_time: std::time::Duration::new(0, 0),
            nodes_processed: 0,
            edges_processed: 0,
            cache_hit: false,
        }
    }
}

/// Execution strategies for complex queries
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Apply filters first, then traversal/aggregation
    FilterFirst,
    /// Execute traversal first, then filter results
    TraversalFirst,
    /// Run independent operations in parallel
    Parallel,
    /// Return cached results
    Cached,
}

/// Query execution plan generated by optimizer
#[derive(Debug)]
pub struct QueryExecutionPlan {
    pub strategy: ExecutionStrategy,
    pub node_filters: Vec<NodeFilter>,
    pub edge_filters: Vec<EdgeFilter>,
    pub traversal_operation: Option<TraversalOperation>,
    pub aggregations: Vec<AggregationOperation>,
    pub cached_result: Option<ComplexQueryResult>,
    pub estimated_cost: f64,
}

/// Builder for complex queries with fluent API
#[derive(Debug)]
pub struct ComplexQueryBuilder {
    query: ComplexQuery,
}

impl ComplexQueryBuilder {
    pub fn new() -> Self {
        Self {
            query: ComplexQuery {
                node_filters: Vec::new(),
                edge_filters: Vec::new(),
                traversal_operation: None,
                aggregations: Vec::new(),
                name: None,
            },
        }
    }
    
    /// Set a name for this query (for debugging)
    pub fn name(mut self, name: &str) -> Self {
        self.query.name = Some(name.to_string());
        self
    }
    
    /// Add a node filter
    pub fn filter_nodes(mut self, filter: NodeFilter) -> Self {
        self.query.node_filters.push(filter);
        self
    }
    
    /// Add multiple node filters
    pub fn filter_nodes_all(mut self, filters: Vec<NodeFilter>) -> Self {
        self.query.node_filters.extend(filters);
        self
    }
    
    /// Add an edge filter
    pub fn filter_edges(mut self, filter: EdgeFilter) -> Self {
        self.query.edge_filters.push(filter);
        self
    }
    
    /// Add a BFS traversal operation
    pub fn bfs(mut self, start: NodeId, options: TraversalOptions) -> Self {
        self.query.traversal_operation = Some(TraversalOperation::BFS { start, options });
        self
    }
    
    /// Add a DFS traversal operation
    pub fn dfs(mut self, start: NodeId, options: TraversalOptions) -> Self {
        self.query.traversal_operation = Some(TraversalOperation::DFS { start, options });
        self
    }
    
    /// Add connected components analysis
    pub fn connected_components(mut self, options: TraversalOptions) -> Self {
        self.query.traversal_operation = Some(TraversalOperation::ConnectedComponents { options });
        self
    }
    
    /// Count nodes or edges
    pub fn count(mut self, target: &str) -> Self {
        self.query.aggregations.push(AggregationOperation::Count { 
            name: target.to_string() 
        });
        self
    }
    
    /// Sum an attribute over nodes or edges
    pub fn sum(mut self, attribute: &str, target: AggregationTarget) -> Self {
        self.query.aggregations.push(AggregationOperation::Sum { 
            attribute: attribute.to_string(),
            target 
        });
        self
    }
    
    /// Average an attribute over nodes or edges
    pub fn average(mut self, attribute: &str, target: AggregationTarget) -> Self {
        self.query.aggregations.push(AggregationOperation::Average { 
            attribute: attribute.to_string(),
            target 
        });
        self
    }
    
    /// Find minimum value of an attribute
    pub fn min(mut self, attribute: &str, target: AggregationTarget) -> Self {
        self.query.aggregations.push(AggregationOperation::Min { 
            attribute: attribute.to_string(),
            target 
        });
        self
    }
    
    /// Find maximum value of an attribute
    pub fn max(mut self, attribute: &str, target: AggregationTarget) -> Self {
        self.query.aggregations.push(AggregationOperation::Max { 
            attribute: attribute.to_string(),
            target 
        });
        self
    }
    
    /// Count distinct values of an attribute
    pub fn distinct(mut self, attribute: &str, target: AggregationTarget) -> Self {
        self.query.aggregations.push(AggregationOperation::Distinct { 
            attribute: attribute.to_string(),
            target 
        });
        self
    }
    
    /// Build the final query
    pub fn build(self) -> ComplexQuery {
        self.query
    }
}

/*
=== SUPPORTING TYPES ===
*/

/// Cached query result with expiration
#[derive(Debug, Clone)]
pub struct CachedQueryResult {
    pub result: ComplexQueryResult,
    pub timestamp: std::time::Instant,
    pub access_count: usize,
}

/// Query performance tracking
#[derive(Debug, Clone)]
pub struct QueryPerformance {
    pub execution_time: std::time::Duration,
    pub items_processed: usize,
    pub execution_count: usize,
}

/*
=== QUERY BUILDING TYPES ===
Types for constructing complex queries
*/

/// Configuration for query engine behavior
#[derive(Debug, Clone)]
pub struct QueryConfig {
    /// Maximum number of results to return from a single query
    pub max_results: Option<usize>,
    
    /// Maximum time to spend on a single query (milliseconds)
    pub query_timeout_ms: Option<u64>,
    
    /// Whether to enable query result caching
    pub enable_cache: bool,
    
    /// Maximum memory to use for query cache (bytes)
    pub max_cache_memory: usize,
    
    /// Whether to collect query performance statistics
    pub collect_stats: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            max_results: None,
            query_timeout_ms: None,
            enable_cache: true,
            max_cache_memory: 64 * 1024 * 1024, // 64MB
            collect_stats: false,
        }
    }
}

/// A filter for a single attribute
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeFilter {
    /// Exact equality match
    Equals(AttrValue),
    
    /// Not equal to value
    NotEquals(AttrValue),
    
    /// Numeric comparisons (only valid for Int/Float)
    GreaterThan(AttrValue),
    LessThan(AttrValue),
    GreaterThanOrEqual(AttrValue),
    LessThanOrEqual(AttrValue),
    
    /// Range checks (inclusive)
    Between(AttrValue, AttrValue),
    
    /// String operations (only valid for Text)
    StartsWith(String),
    EndsWith(String),
    Contains(String),
    Matches(String), // Regex pattern
    
    /// Set membership
    In(Vec<AttrValue>),
    NotIn(Vec<AttrValue>),
    
    /// Existence checks
    IsNull,
    IsNotNull,
    
    /// Vector operations (only valid for FloatVec)
    VectorSimilarity {
        target: Vec<f32>,
        similarity_type: SimilarityType,
        threshold: f32,
    },
}

impl AttributeFilter {
    /// Check if a value matches this filter
    pub fn matches(&self, value: &AttrValue) -> bool {
        match self {
            // Exact comparisons - handle optimized storage types
            AttributeFilter::Equals(target) => Self::values_equal(value, target),
            AttributeFilter::NotEquals(target) => !Self::values_equal(value, target),
            
            // Numeric comparisons (handle SmallInt, Int, Float conversions)
            AttributeFilter::GreaterThan(target) => {
                Self::numeric_compare(value, target, |a, b| a > b)
            },
            AttributeFilter::LessThan(target) => {
                Self::numeric_compare(value, target, |a, b| a < b)
            },
            AttributeFilter::GreaterThanOrEqual(target) => {
                Self::numeric_compare(value, target, |a, b| a >= b)
            },
            AttributeFilter::LessThanOrEqual(target) => {
                Self::numeric_compare(value, target, |a, b| a <= b)
            },
            
            // Range checks (inclusive) - use numeric comparison with type conversion
            AttributeFilter::Between(min, max) => {
                Self::numeric_compare(value, min, |v, m| v >= m) &&
                Self::numeric_compare(value, max, |v, m| v <= m)
            },
            
            // String operations - handle CompactText and CompressedText
            AttributeFilter::StartsWith(prefix) => {
                Self::extract_text_value(value)
                    .map(|text| text.starts_with(prefix))
                    .unwrap_or(false)
            },
            AttributeFilter::EndsWith(suffix) => {
                Self::extract_text_value(value)
                    .map(|text| text.ends_with(suffix))
                    .unwrap_or(false)
            },
            AttributeFilter::Contains(substring) => {
                Self::extract_text_value(value)
                    .map(|text| text.contains(substring))
                    .unwrap_or(false)
            },
            AttributeFilter::Matches(pattern) => {
                // Use simple wildcard matching for basic pattern support
                Self::extract_text_value(value)
                    .map(|text| simple_wildcard_match(pattern, &text))
                    .unwrap_or(false)
            },
            
            // Set membership - use our smart equality comparison
            AttributeFilter::In(set) => {
                set.iter().any(|target| Self::values_equal(value, target))
            },
            AttributeFilter::NotIn(set) => {
                !set.iter().any(|target| Self::values_equal(value, target))
            },
            
            // Existence checks - these are handled at a higher level
            // since we need to know if the attribute exists vs has a value
            AttributeFilter::IsNull => {
                // This would require the context of whether an attribute exists
                // For now, assume all values we're testing exist (since we're testing against actual values)
                false
            }
            AttributeFilter::IsNotNull => {
                // This would require the context of whether an attribute exists  
                // For now, assume all values we're testing exist
                true
            }
            
            // Vector operations - handle compressed float vectors
            AttributeFilter::VectorSimilarity { target, similarity_type, threshold } => {
                let vec_value = match value {
                    AttrValue::FloatVec(vec) => Some(vec.clone()),
                    AttrValue::CompressedFloatVec(compressed) => {
                        compressed.decompress_float_vec().ok()
                    },
                    _ => None,
                };
                
                if let Some(vec) = vec_value {
                    let similarity = match similarity_type {
                        SimilarityType::CosineSimilarity => {
                            cosine_similarity(&vec, target)
                        },
                        SimilarityType::EuclideanDistance => {
                            let dist = euclidean_distance(&vec, target);
                            // Convert distance to similarity (closer = higher similarity)
                            1.0 / (1.0 + dist)
                        },
                        SimilarityType::DotProduct => {
                            dot_product(&vec, target)
                        },
                        SimilarityType::ManhattanDistance => {
                            let dist = manhattan_distance(&vec, target);
                            1.0 / (1.0 + dist)
                        },
                    };
                    similarity >= *threshold as f64
                } else {
                    false
                }
            },
        }
    }
    
    /// Compare two AttrValues with automatic type conversion (handles optimized storage)
    fn values_equal(stored: &AttrValue, target: &AttrValue) -> bool {
        match (stored, target) {
            // Direct equality
            (a, b) if a == b => true,
            
            // SmallInt vs Int comparison
            (AttrValue::SmallInt(small), AttrValue::Int(big)) => (*small as i64) == *big,
            (AttrValue::Int(big), AttrValue::SmallInt(small)) => *big == (*small as i64),
            
            // CompactText vs Text comparison  
            (AttrValue::CompactText(compact), AttrValue::Text(text)) => compact.as_str() == text,
            (AttrValue::Text(text), AttrValue::CompactText(compact)) => text == compact.as_str(),
            
            // CompressedText vs Text comparison (decompress first)
            (AttrValue::CompressedText(compressed), AttrValue::Text(text)) => {
                if let Ok(decompressed) = compressed.decompress_text() {
                    decompressed == *text
                } else {
                    false
                }
            },
            (AttrValue::Text(text), AttrValue::CompressedText(compressed)) => {
                if let Ok(decompressed) = compressed.decompress_text() {
                    *text == decompressed
                } else {
                    false
                }
            },
            
            // CompressedFloatVec vs FloatVec comparison
            (AttrValue::CompressedFloatVec(compressed), AttrValue::FloatVec(vec)) => {
                if let Ok(decompressed) = compressed.decompress_float_vec() {
                    decompressed == *vec
                } else {
                    false
                }
            },
            (AttrValue::FloatVec(vec), AttrValue::CompressedFloatVec(compressed)) => {
                if let Ok(decompressed) = compressed.decompress_float_vec() {
                    *vec == decompressed
                } else {
                    false
                }
            },
            
            // No match for other combinations
            _ => false,
        }
    }
    
    /// Perform numeric comparison with automatic type conversion 
    fn numeric_compare<F>(stored: &AttrValue, target: &AttrValue, op: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        let stored_num = Self::extract_numeric_value(stored);
        let target_num = Self::extract_numeric_value(target);
        
        match (stored_num, target_num) {
            (Some(a), Some(b)) => op(a, b),
            _ => false,
        }
    }
    
    /// Extract numeric value from any numeric AttrValue variant
    fn extract_numeric_value(value: &AttrValue) -> Option<f64> {
        match value {
            AttrValue::Int(i) => Some(*i as f64),
            AttrValue::SmallInt(i) => Some(*i as f64),
            AttrValue::Float(f) => Some(*f as f64),
            _ => None,
        }
    }
    
    /// Extract text value from any text AttrValue variant
    fn extract_text_value(value: &AttrValue) -> Option<String> {
        match value {
            AttrValue::Text(text) => Some(text.clone()),
            AttrValue::CompactText(compact) => Some(compact.as_str().to_string()),
            AttrValue::CompressedText(compressed) => {
                compressed.decompress_text().ok()
            },
            _ => None,
        }
    }
    
    /// Estimate the selectivity of this filter (0.0 = very selective, 1.0 = not selective)
    pub fn estimated_selectivity(&self, stats: &AttributeStatistics) -> f64 {
        let _ = stats; // Silence unused warnings
        // Basic implementation returns conservative estimate
        0.5
    }
}

/// Similarity types for vector comparisons
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityType {
    CosineSimilarity,
    EuclideanDistance,
    DotProduct,
    ManhattanDistance,
}

/// A complex filter that can combine multiple conditions (Phase 3.1 Enhanced)
#[derive(Debug, Clone, PartialEq)]
pub enum NodeFilter {
    /// Check if node has a specific attribute
    HasAttribute { name: AttrName },
    
    /// Check if attribute equals specific value
    AttributeEquals { name: AttrName, value: AttrValue },
    
    /// Apply complex filter to attribute
    AttributeFilter { name: AttrName, filter: AttributeFilter },
    
    /// Filter by degree range
    DegreeRange { min: usize, max: usize },
    
    /// Check if node has specific neighbor
    HasNeighbor { neighbor_id: NodeId },
    
    /// Logical combinations
    And(Vec<NodeFilter>),
    Or(Vec<NodeFilter>),
    Not(Box<NodeFilter>),
}

/// Filter for edges (Phase 3.1 Enhanced)
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeFilter {
    /// Check if edge has a specific attribute
    HasAttribute { name: AttrName },
    
    /// Check if attribute equals specific value
    AttributeEquals { name: AttrName, value: AttrValue },
    
    /// Apply complex filter to attribute
    AttributeFilter { name: AttrName, filter: AttributeFilter },
    
    /// Check if edge connects specific nodes
    ConnectsNodes { source: NodeId, target: NodeId },
    
    /// Check if edge connects any of the given nodes
    ConnectsAny(Vec<NodeId>),
    
    /// Logical combinations
    And(Vec<EdgeFilter>),
    Or(Vec<EdgeFilter>),
    Not(Box<EdgeFilter>),
}

/// Filter based on node degree
#[derive(Debug, Clone)]
pub struct DegreeFilter {
    pub min_degree: Option<usize>,
    pub max_degree: Option<usize>,
    pub exact_degree: Option<usize>,
}

/// A pattern for matching complex node relationships
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// The primary node filter
    pub node_filter: Box<NodeFilter>,
    
    /// Required neighbor patterns
    pub neighbor_patterns: Vec<NeighborPattern>,
    
    /// Logic for combining neighbor requirements
    pub neighbor_logic: PatternLogic,
}

/// A pattern for required neighbors
#[derive(Debug, Clone)]
pub struct NeighborPattern {
    /// Filter for the neighbor node
    pub neighbor_filter: NodeFilter,
    
    /// Filter for the connecting edge
    pub edge_filter: Option<EdgeFilter>,
    
    /// Minimum number of neighbors matching this pattern
    pub min_count: usize,
    
    /// Maximum number of neighbors matching this pattern
    pub max_count: Option<usize>,
}

/// Logic for combining pattern requirements
#[derive(Debug, Clone)]
pub enum PatternLogic {
    And, // All patterns must match
    Or,  // At least one pattern must match
    Exactly(usize), // Exactly N patterns must match
    AtLeast(usize), // At least N patterns must match
}

/// A structural pattern in the graph (like triangles, cliques, etc.)
#[derive(Debug, Clone)]
pub enum StructuralPattern {
    /// Triangle (3 nodes all connected to each other)
    Triangle {
        node_filters: Vec<NodeFilter>,
        edge_filters: Vec<EdgeFilter>,
    },
    
    /// Clique (all nodes connected to all other nodes)
    Clique {
        size: usize,
        node_filter: Option<NodeFilter>,
        edge_filter: Option<EdgeFilter>,
    },
    
    /// Path of specific length
    Path {
        length: usize,
        node_filters: Vec<NodeFilter>,
        edge_filters: Vec<EdgeFilter>,
    },
    
    /// Star pattern (one central node connected to many leaf nodes)
    Star {
        center_filter: NodeFilter,
        leaf_filter: NodeFilter,
        edge_filter: Option<EdgeFilter>,
        min_leaves: usize,
        max_leaves: Option<usize>,
    },
}

/// A match for a structural pattern
#[derive(Debug, Clone)]
pub struct StructureMatch {
    /// The nodes involved in this match
    pub nodes: Vec<NodeId>,
    
    /// The edges involved in this match
    pub edges: Vec<EdgeId>,
    
    /// Pattern-specific metadata
    pub metadata: HashMap<String, String>,
}

/// A complete query with multiple operations
#[derive(Debug, Clone)]
pub struct GraphQuery {
    /// The main operation to perform
    pub operation: QueryOperation,
    
    /// Optional result limiting
    pub limit: Option<usize>,
    
    /// Optional result ordering
    pub order_by: Option<OrderBy>,
    
    /// Optional result grouping
    pub group_by: Option<GroupBy>,
}

/// Different types of query operations
#[derive(Debug, Clone)]
pub enum QueryOperation {
    /// Find nodes matching criteria
    FindNodes(NodeFilter),
    
    /// Find edges matching criteria
    FindEdges(EdgeFilter),
    
    /// Count entities matching criteria
    Count(CountTarget),
    
    /// Compute aggregations
    Aggregate(AggregateOperation),
    
    /// Find structural patterns
    FindPatterns(StructuralPattern),
    
    /// Graph analytics
    Analytics(AnalyticsOperation),
}

/// What to count in a count query
#[derive(Debug, Clone)]
pub enum CountTarget {
    Nodes(NodeFilter),
    Edges(EdgeFilter),
    Patterns(StructuralPattern),
}

/// Aggregation operations
#[derive(Debug, Clone)]
pub struct AggregateOperation {
    pub target: AggregateTarget,
    pub aggregation: AggregationType,
    pub group_by: Option<AttrName>,
}

/// What to aggregate
#[derive(Debug, Clone)]
pub enum AggregateTarget {
    NodeAttribute(AttrName, Option<NodeFilter>),
    EdgeAttribute(AttrName, Option<EdgeFilter>),
}

/// Types of aggregation
#[derive(Debug, Clone)]
pub enum AggregationType {
    Count,
    Sum,
    Average,
    Min,
    Max,
    StandardDeviation,
    Variance,
    Median,
    Mode,
    Percentile(f64),
    CountDistinct,
    First,
    Last,
}

/// Missing type definitions for compilation
#[derive(Debug, Clone)]
pub struct AttributeStatistics {
    pub total_values: usize,
    pub null_count: usize,
    pub unique_count: usize,
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hit_count: usize,
    pub miss_count: usize,
    pub total_size: usize,
}

#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub steps: Vec<String>,
    pub estimated_cost: f64,
}

#[derive(Debug, Clone)]
pub struct AggregationMetadata {
    pub aggregation_type: AggregationType,
    pub attribute_name: AttrName,
    pub null_count: usize,
}

// Legacy result type for compatibility
#[derive(Debug, Clone)]
pub enum QueryResult {
    Nodes(Vec<NodeId>),
    Edges(Vec<EdgeId>),
    Count(usize),
    Aggregation(AggregationResult),
}

// Additional missing types
#[derive(Debug, Clone)]
pub enum OrderBy {
    Attribute(String),
    Ascending(String),
    Descending(String),
}

#[derive(Debug, Clone)]
pub enum GroupBy {
    Attribute(String),
    MultipleAttributes(Vec<String>),
}

#[derive(Debug, Clone)]
pub enum AnalyticsOperation {
    DegreeDistribution,
    CentralityMeasures,
    CommunityDetection,
}

// Helper functions for vector operations and pattern matching
fn simple_wildcard_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    
    if pattern == text {
        return true;
    }
    
    // Handle simple prefix/suffix wildcards
    if pattern.starts_with('*') {
        let suffix = &pattern[1..];
        return text.ends_with(suffix);
    }
    
    if pattern.ends_with('*') {
        let prefix = &pattern[..pattern.len()-1];
        return text.starts_with(prefix);
    }
    
    // Handle patterns with * in the middle
    if let Some(star_pos) = pattern.find('*') {
        let prefix = &pattern[..star_pos];
        let suffix = &pattern[star_pos+1..];
        return text.starts_with(prefix) && text.ends_with(suffix) && text.len() >= prefix.len() + suffix.len();
    }
    
    false
}

fn dot_product(vec1: &[f32], vec2: &[f32]) -> f64 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }
    
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum()
}

fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f64 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }
    
    let dot_prod = dot_product(vec1, vec2);
    
    let magnitude1 = vec1.iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
        
    let magnitude2 = vec2.iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    
    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return 0.0;
    }
    
    dot_prod / (magnitude1 * magnitude2)
}

fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f64 {
    if vec1.len() != vec2.len() {
        return f64::INFINITY;
    }
    
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| {
            let diff = (*a as f64) - (*b as f64);
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn manhattan_distance(vec1: &[f32], vec2: &[f32]) -> f64 {
    if vec1.len() != vec2.len() {
        return f64::INFINITY;
    }
    
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .sum()
}

impl QueryEngine {
    /// Create builder for statistical analysis of node attributes
    pub fn analyze_node_attribute<'a>(&'a self, attr_name: &'a AttrName) -> AttributeAnalyzer<'a> {
        AttributeAnalyzer::new(self, attr_name, AggregationTarget::Nodes)
    }
    
    /// Create builder for statistical analysis of edge attributes
    pub fn analyze_edge_attribute<'a>(&'a self, attr_name: &'a AttrName) -> AttributeAnalyzer<'a> {
        AttributeAnalyzer::new(self, attr_name, AggregationTarget::Edges)
    }
    
    /// Compute comprehensive statistics for an attribute
    pub fn compute_comprehensive_stats(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        attr_name: &AttrName,
        target: AggregationTarget
    ) -> GraphResult<ComprehensiveStats> {
        let nodes = if matches!(target, AggregationTarget::Nodes) { space.node_ids() } else { vec![] };
        let edges = if matches!(target, AggregationTarget::Edges) { space.edge_ids() } else { vec![] };
        
        let aggregations = vec![
            AggregationOperation::Count { name: "total".to_string() },
            AggregationOperation::Sum { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::Average { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::Min { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::Max { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::StandardDeviation { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::Variance { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::Median { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::CountDistinct { attribute: attr_name.clone(), target: target.clone() },
            AggregationOperation::Percentile { attribute: attr_name.clone(), target: target.clone(), percentile: 25.0 },
            AggregationOperation::Percentile { attribute: attr_name.clone(), target: target.clone(), percentile: 75.0 },
            AggregationOperation::Percentile { attribute: attr_name.clone(), target: target.clone(), percentile: 95.0 },
        ];
        
        let results = self.compute_aggregations(pool, space, &nodes, &edges, &aggregations)?;
        
        Ok(ComprehensiveStats::from_aggregation_results(results))
    }
}

/// Builder for creating attribute analysis queries
pub struct AttributeAnalyzer<'a> {
    query_engine: &'a QueryEngine,
    attr_name: &'a AttrName,
    target: AggregationTarget,
}

impl<'a> AttributeAnalyzer<'a> {
    fn new(query_engine: &'a QueryEngine, attr_name: &'a AttrName, target: AggregationTarget) -> Self {
        Self { query_engine, attr_name, target }
    }
    
    pub fn count(self, pool: &GraphPool, space: &GraphSpace) -> GraphResult<i64> {
        let result = self.query_engine.compute_aggregations(
            pool, 
            space, 
            &space.node_ids(), 
            &space.edge_ids(), 
            &[AggregationOperation::Count { name: "total".to_string() }]
        )?;
        
        match result.into_values().next() {
            Some(AggregationResult::Integer(i)) => Ok(i),
            _ => Ok(0),
        }
    }
    
    pub fn sum(self, pool: &GraphPool, space: &GraphSpace) -> GraphResult<f64> {
        let result = self.query_engine.compute_aggregations(
            pool, 
            space, 
            &space.node_ids(), 
            &space.edge_ids(), 
            &[AggregationOperation::Sum { attribute: self.attr_name.clone(), target: self.target }]
        )?;
        
        match result.into_values().next() {
            Some(AggregationResult::Float(f)) => Ok(f),
            _ => Ok(0.0),
        }
    }
    
    pub fn average(self, pool: &GraphPool, space: &GraphSpace) -> GraphResult<f64> {
        let result = self.query_engine.compute_aggregations(
            pool, 
            space, 
            &space.node_ids(), 
            &space.edge_ids(), 
            &[AggregationOperation::Average { attribute: self.attr_name.clone(), target: self.target }]
        )?;
        
        match result.into_values().next() {
            Some(AggregationResult::Float(f)) => Ok(f),
            _ => Ok(0.0),
        }
    }
    
    pub fn min_max(self, pool: &GraphPool, space: &GraphSpace) -> GraphResult<(f64, f64)> {
        let result = self.query_engine.compute_aggregations(
            pool, 
            space, 
            &space.node_ids(), 
            &space.edge_ids(), 
            &[
                AggregationOperation::Min { attribute: self.attr_name.clone(), target: self.target.clone() },
                AggregationOperation::Max { attribute: self.attr_name.clone(), target: self.target }
            ]
        )?;
        
        let values: Vec<f64> = result.into_values().filter_map(|r| match r {
            AggregationResult::Float(f) => Some(f),
            _ => None,
        }).collect();
        
        Ok((values.get(0).copied().unwrap_or(0.0), values.get(1).copied().unwrap_or(0.0)))
    }
    
    pub fn percentile(self, pool: &GraphPool, space: &GraphSpace, percentile: f64) -> GraphResult<f64> {
        let result = self.query_engine.compute_aggregations(
            pool, 
            space, 
            &space.node_ids(), 
            &space.edge_ids(), 
            &[AggregationOperation::Percentile { 
                attribute: self.attr_name.clone(), 
                target: self.target, 
                percentile 
            }]
        )?;
        
        match result.into_values().next() {
            Some(AggregationResult::Float(f)) => Ok(f),
            _ => Ok(0.0),
        }
    }
    
    pub fn comprehensive_stats(self, pool: &GraphPool, space: &GraphSpace) -> GraphResult<ComprehensiveStats> {
        self.query_engine.compute_comprehensive_stats(pool, space, self.attr_name, self.target)
    }
}

/// Comprehensive statistical summary of an attribute
#[derive(Debug, Clone)]
pub struct ComprehensiveStats {
    pub count: i64,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub distinct_count: i64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub percentile_95: f64,
    pub iqr: f64, // Interquartile Range (P75 - P25)
    pub range: f64, // Max - Min
}

impl ComprehensiveStats {
    fn from_aggregation_results(results: HashMap<String, AggregationResult>) -> Self {
        let mut stats = ComprehensiveStats {
            count: 0,
            sum: 0.0,
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
            variance: 0.0,
            distinct_count: 0,
            percentile_25: 0.0,
            percentile_75: 0.0,
            percentile_95: 0.0,
            iqr: 0.0,
            range: 0.0,
        };
        
        // Extract values from aggregation results
        for (key, result) in results {
            match result {
                AggregationResult::Integer(i) if key.contains("Count") => {
                    if key.contains("Distinct") {
                        stats.distinct_count = i;
                    } else {
                        stats.count = i;
                    }
                }
                AggregationResult::Float(f) => {
                    if key.contains("Sum") {
                        stats.sum = f;
                    } else if key.contains("Average") {
                        stats.mean = f;
                    } else if key.contains("Min") {
                        stats.min = f;
                    } else if key.contains("Max") {
                        stats.max = f;
                    } else if key.contains("StandardDeviation") {
                        stats.std_dev = f;
                    } else if key.contains("Variance") {
                        stats.variance = f;
                    } else if key.contains("Median") {
                        stats.median = f;
                    } else if key.contains("25.0") {
                        stats.percentile_25 = f;
                    } else if key.contains("75.0") {
                        stats.percentile_75 = f;
                    } else if key.contains("95.0") {
                        stats.percentile_95 = f;
                    }
                }
                _ => {}
            }
        }
        
        // Calculate derived statistics
        stats.iqr = stats.percentile_75 - stats.percentile_25;
        stats.range = stats.max - stats.min;
        
        stats
    }
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}
