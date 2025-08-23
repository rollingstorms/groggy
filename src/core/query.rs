//! Query Engine - Core filtering operations for graph data.

use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::errors::GraphResult;
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use rayon::prelude::*; // Re-enabled - RefCell eliminated, now safe for parallel processing
use std::collections::HashSet;

/// The main query engine for filtering operations
#[derive(Debug)]
pub struct QueryEngine {}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self {}
    }

    /// Find nodes using active space information (called from Graph API)
    pub fn find_nodes_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed from &mut to &
        filter: &NodeFilter,
    ) -> GraphResult<Vec<NodeId>> {
        self.filter_nodes(pool, space, filter)
    }

    /// Find nodes with filtering - PERFORMANCE OPTIMIZED using bulk columnar operations
    pub fn filter_nodes(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Changed from &mut to &
        filter: &NodeFilter,
    ) -> GraphResult<Vec<NodeId>> {
        let active_nodes: Vec<NodeId> = space.node_ids(); // Direct call, no RefCell clone

        // Use bulk columnar filtering for maximum performance
        self.filter_nodes_columnar(&active_nodes, pool, space, filter)
    }

    /// Columnar filtering on any subset of nodes - THE CORE METHOD (ULTRA-OPTIMIZED)
    fn filter_nodes_columnar(
        &self,
        nodes: &[NodeId],
        pool: &GraphPool,
        space: &GraphSpace, // Changed from &mut to &
        filter: &NodeFilter,
    ) -> GraphResult<Vec<NodeId>> {
        let start_time = std::time::Instant::now();

        match filter {
            NodeFilter::AttributeFilter { name, filter } => {
                // OPTIMIZED: Pre-allocate result vector to avoid multiple allocations
                let mut matching_nodes = Vec::with_capacity(nodes.len() / 4); // estimate 25% match rate
                let attr_start = std::time::Instant::now();
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                let _attr_time = attr_start.elapsed();

                // ULTRA-OPTIMIZED: Parallel iterator processing with minimal allocations
                let parallel_results: Vec<NodeId> = if node_attr_pairs.len() > 1000 {
                    // Use parallel processing for large datasets
                    node_attr_pairs
                        .into_par_iter()
                        .filter_map(|(node_id, attr_opt)| {
                            attr_opt.and_then(|attr_value| {
                                if filter.matches(attr_value) {
                                    Some(node_id)
                                } else {
                                    None
                                }
                            })
                        })
                        .collect()
                } else {
                    // Use sequential processing for small datasets to avoid overhead
                    node_attr_pairs
                        .into_iter()
                        .filter_map(|(node_id, attr_opt)| {
                            attr_opt.and_then(|attr_value| {
                                if filter.matches(attr_value) {
                                    Some(node_id)
                                } else {
                                    None
                                }
                            })
                        })
                        .collect()
                };
                matching_nodes.extend(parallel_results);
                let _total_time = start_time.elapsed();
                Ok(matching_nodes)
            }

            NodeFilter::AttributeEquals { name, value } => {
                // OPTIMIZED: Pre-allocate and avoid iterator chain
                let mut matching_nodes = Vec::with_capacity(nodes.len() / 4);
                let attr_start = std::time::Instant::now();
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                let _attr_time = attr_start.elapsed();

                // PARALLEL: Use parallel processing for large datasets
                let parallel_results: Vec<NodeId> = if node_attr_pairs.len() > 1000 {
                    node_attr_pairs
                        .into_par_iter()
                        .filter_map(|(node_id, attr_opt)| {
                            if let Some(attr_value) = attr_opt {
                                if *attr_value == *value {
                                    Some(node_id)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    node_attr_pairs
                        .into_iter()
                        .filter_map(|(node_id, attr_opt)| {
                            if let Some(attr_value) = attr_opt {
                                if *attr_value == *value {
                                    Some(node_id)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                };
                matching_nodes.extend(parallel_results);
                let _total_time = start_time.elapsed();
                Ok(matching_nodes)
            }

            NodeFilter::HasAttribute { name } => {
                // OPTIMIZED: Use bulk attribute lookup with parallel processing
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);

                let results: Vec<NodeId> = if node_attr_pairs.len() > 1000 {
                    node_attr_pairs
                        .into_par_iter()
                        .filter_map(|(node_id, attr_opt)| {
                            if attr_opt.is_some() {
                                Some(node_id)
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    node_attr_pairs
                        .into_iter()
                        .filter_map(|(node_id, attr_opt)| {
                            if attr_opt.is_some() {
                                Some(node_id)
                            } else {
                                None
                            }
                        })
                        .collect()
                };

                Ok(results)
            }

            NodeFilter::And(filters) => {
                if filters.is_empty() {
                    return Ok(Vec::new());
                }

                let mut result_set: HashSet<NodeId> = self
                    .filter_nodes_columnar(nodes, pool, space, &filters[0])?
                    .into_iter()
                    .collect();

                for sub_filter in &filters[1..] {
                    if result_set.is_empty() {
                        break;
                    }
                    let candidates: Vec<NodeId> = result_set.iter().copied().collect();
                    let filter_result: HashSet<NodeId> = self
                        .filter_nodes_columnar(&candidates, pool, space, sub_filter)?
                        .into_iter()
                        .collect();
                    result_set = result_set.intersection(&filter_result).copied().collect();
                }
                Ok(result_set.into_iter().collect())
            }

            NodeFilter::DegreeRange { min, max } => {
                // OPTIMIZED: Bulk degree calculation using columnar topology
                let (_, sources, targets, _) = space.snapshot(pool);
                let mut matching_nodes = Vec::with_capacity(nodes.len() / 4);

                // Count degrees for all nodes at once using columnar scan
                for &node_id in nodes {
                    let mut degree = 0;
                    for i in 0..sources.len() {
                        if sources[i] == node_id || targets[i] == node_id {
                            degree += 1;
                        }
                    }
                    if degree >= *min && degree <= *max {
                        matching_nodes.push(node_id);
                    }
                }
                Ok(matching_nodes)
            }

            NodeFilter::HasNeighbor { neighbor_id } => {
                // OPTIMIZED: Bulk neighbor check using columnar topology
                let (_, sources, targets, _) = space.snapshot(pool);
                let mut matching_nodes = Vec::with_capacity(nodes.len() / 4);

                // Check neighbor relationships for all nodes at once using columnar scan
                for &node_id in nodes {
                    let mut has_neighbor = false;
                    for i in 0..sources.len() {
                        if (sources[i] == node_id && targets[i] == *neighbor_id)
                            || (sources[i] == *neighbor_id && targets[i] == node_id)
                        {
                            has_neighbor = true;
                            break;
                        }
                    }
                    if has_neighbor {
                        matching_nodes.push(node_id);
                    }
                }
                Ok(matching_nodes)
            }

            NodeFilter::Or(filters) => {
                if filters.is_empty() {
                    return Ok(Vec::new());
                }

                let mut result_set = HashSet::new();
                for sub_filter in filters {
                    let filter_results =
                        self.filter_nodes_columnar(nodes, pool, space, sub_filter)?;
                    result_set.extend(filter_results);
                }
                Ok(result_set.into_iter().collect())
            }

            NodeFilter::Not(filter) => {
                let matching_nodes: HashSet<NodeId> = self
                    .filter_nodes_columnar(nodes, pool, space, filter)?
                    .into_iter()
                    .collect();
                let non_matching: Vec<NodeId> = nodes
                    .iter()
                    .copied()
                    .filter(|node_id| !matching_nodes.contains(node_id))
                    .collect();
                Ok(non_matching)
            }
        }
    }

    /// Check if a node matches the given filter with pre-fetched topology
    #[allow(dead_code)]
    fn node_matches_filter_with_topology(
        &self,
        node_id: NodeId,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter,
        topology_data: &Option<(Vec<EdgeId>, Vec<NodeId>, Vec<NodeId>)>,
    ) -> bool {
        match filter {
            NodeFilter::HasAttribute { name } => space.get_node_attr_index(node_id, name).is_some(),
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
                if let Some((_, sources, targets)) = topology_data {
                    let mut degree = 0;
                    for i in 0..sources.len() {
                        if sources[i] == node_id || targets[i] == node_id {
                            degree += 1;
                        }
                    }
                    degree >= *min && degree <= *max
                } else {
                    false
                }
            }
            NodeFilter::HasNeighbor { neighbor_id } => {
                if let Some((_, sources, targets)) = topology_data {
                    for i in 0..sources.len() {
                        if (sources[i] == node_id && targets[i] == *neighbor_id)
                            || (sources[i] == *neighbor_id && targets[i] == node_id)
                        {
                            return true;
                        }
                    }
                }
                false
            }
            NodeFilter::And(filters) => filters.iter().all(|f| {
                self.node_matches_filter_with_topology(node_id, pool, space, f, topology_data)
            }),
            NodeFilter::Or(filters) => filters.iter().any(|f| {
                self.node_matches_filter_with_topology(node_id, pool, space, f, topology_data)
            }),
            NodeFilter::Not(filter) => {
                !self.node_matches_filter_with_topology(node_id, pool, space, filter, topology_data)
            }
        }
    }

    /// Check if a node matches the given filter (legacy method)
    #[allow(dead_code)]
    fn node_matches_filter(
        &self,
        node_id: NodeId,
        pool: &GraphPool,
        space: &mut GraphSpace,
        filter: &NodeFilter,
    ) -> bool {
        // Get topology if needed
        let topology_data = match filter {
            NodeFilter::DegreeRange { .. } | NodeFilter::HasNeighbor { .. } => {
                let (edge_ids, sources, targets, _) = space.snapshot(pool);
                Some((
                    edge_ids.as_ref().clone(),
                    sources.as_ref().clone(),
                    targets.as_ref().clone(),
                ))
            }
            _ => None,
        };

        self.node_matches_filter_with_topology(node_id, pool, space, filter, &topology_data)
    }

    /// Check if an edge matches the given filter
    fn edge_matches_filter(
        &self,
        edge_id: EdgeId,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter,
    ) -> bool {
        self.edge_matches_filter_impl(edge_id, pool, space, filter)
    }

    /// Implementation of edge filter matching
    fn edge_matches_filter_impl(
        &self,
        edge_id: EdgeId,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter,
    ) -> bool {
        match filter {
            EdgeFilter::HasAttribute { name } => space.get_edge_attr_index(edge_id, name).is_some(),
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
                    (edge_source == *source && edge_target == *target)
                        || (edge_source == *target && edge_target == *source)
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
            EdgeFilter::And(filters) => filters
                .iter()
                .all(|f| self.edge_matches_filter(edge_id, pool, space, f)),
            EdgeFilter::Or(filters) => filters
                .iter()
                .any(|f| self.edge_matches_filter(edge_id, pool, space, f)),
            EdgeFilter::Not(filter) => !self.edge_matches_filter(edge_id, pool, space, filter),
        }
    }

    /// Find edges with filtering
    pub fn filter_edges(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Already &
        filter: &EdgeFilter,
    ) -> GraphResult<Vec<EdgeId>> {
        let start_time = std::time::Instant::now();
        let active_edges: Vec<EdgeId> = space.edge_ids(); // Direct call, no RefCell clone
        let _num_edges = active_edges.len();

        let results: Vec<EdgeId> = active_edges
            .into_iter()
            .filter(|&edge_id| self.edge_matches_filter(edge_id, pool, space, filter))
            .collect();

        let _total_time = start_time.elapsed();
        Ok(results)
    }

    /// Find all edges matching a filter (used by Graph API)
    pub fn find_edges_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace, // Already &
        filter: &EdgeFilter,
    ) -> GraphResult<Vec<EdgeId>> {
        self.filter_edges(pool, space, filter)
    }
}

/// Simple attribute filter for basic comparisons
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeFilter {
    Equals(AttrValue),
    NotEquals(AttrValue),
    GreaterThan(AttrValue),
    LessThan(AttrValue),
    GreaterThanOrEqual(AttrValue),
    LessThanOrEqual(AttrValue),
}

impl AttributeFilter {
    /// Check if a value matches this filter
    pub fn matches(&self, value: &AttrValue) -> bool {
        match self {
            AttributeFilter::Equals(target) => value == target,
            AttributeFilter::NotEquals(target) => value != target,
            AttributeFilter::GreaterThan(target) => {
                // Flexible numeric comparison - handle all numeric type combinations
                self.compare_numeric(value, target, |a, b| a > b)
            }
            AttributeFilter::LessThan(target) => {
                // Flexible numeric comparison - handle all numeric type combinations
                self.compare_numeric(value, target, |a, b| a < b)
            }
            AttributeFilter::GreaterThanOrEqual(target) => {
                // Flexible numeric comparison - handle all numeric type combinations
                self.compare_numeric(value, target, |a, b| a >= b)
            }
            AttributeFilter::LessThanOrEqual(target) => {
                // Flexible numeric comparison - handle all numeric type combinations
                self.compare_numeric(value, target, |a, b| a <= b)
            }
        }
    }

    /// Helper method for flexible numeric comparisons
    /// OPTIMIZED: Use pattern matching to avoid allocations and improve performance
    fn compare_numeric<F>(&self, value: &AttrValue, target: &AttrValue, op: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        // OPTIMIZED: Direct pattern matching for common cases to avoid conversions
        match (value, target) {
            // Int comparisons - most common case
            (AttrValue::Int(a), AttrValue::Int(b)) => op(*a as f64, *b as f64),
            (AttrValue::SmallInt(a), AttrValue::SmallInt(b)) => op(*a as f64, *b as f64),
            (AttrValue::Float(a), AttrValue::Float(b)) => op(*a as f64, *b as f64),

            // Mixed comparisons
            (AttrValue::Int(a), AttrValue::Float(b)) => op(*a as f64, *b as f64),
            (AttrValue::Float(a), AttrValue::Int(b)) => op(*a as f64, *b as f64),
            (AttrValue::SmallInt(a), AttrValue::Int(b)) => op(*a as f64, *b as f64),
            (AttrValue::Int(a), AttrValue::SmallInt(b)) => op(*a as f64, *b as f64),
            (AttrValue::SmallInt(a), AttrValue::Float(b)) => op(*a as f64, *b as f64),
            (AttrValue::Float(a), AttrValue::SmallInt(b)) => op(*a as f64, *b as f64),

            // Non-numeric types
            _ => false,
        }
    }
}

/// Node filter for graph queries
#[derive(Debug, Clone, PartialEq)]
pub enum NodeFilter {
    HasAttribute {
        name: AttrName,
    },
    AttributeEquals {
        name: AttrName,
        value: AttrValue,
    },
    AttributeFilter {
        name: AttrName,
        filter: AttributeFilter,
    },
    DegreeRange {
        min: usize,
        max: usize,
    },
    HasNeighbor {
        neighbor_id: NodeId,
    },
    And(Vec<NodeFilter>),
    Or(Vec<NodeFilter>),
    Not(Box<NodeFilter>),
}

/// Edge filter for graph queries
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeFilter {
    HasAttribute {
        name: AttrName,
    },
    AttributeEquals {
        name: AttrName,
        value: AttrValue,
    },
    AttributeFilter {
        name: AttrName,
        filter: AttributeFilter,
    },
    ConnectsNodes {
        source: NodeId,
        target: NodeId,
    },
    ConnectsAny(Vec<NodeId>),
    And(Vec<EdgeFilter>),
    Or(Vec<EdgeFilter>),
    Not(Box<EdgeFilter>),
}

impl Default for QueryEngine {
    fn default() -> Self {
        Self::new()
    }
}
