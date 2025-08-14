//! Query Engine - Core filtering operations for graph data.

use crate::types::{NodeId, EdgeId, AttrName, AttrValue};
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::errors::GraphResult;
use rayon::prelude::*;
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
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Vec<NodeId>> {
        self.filter_nodes(pool, space, filter)
    }

    /// Find nodes with filtering
    pub fn filter_nodes(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Vec<NodeId>> {
        // OPTIMIZED: Use the new optimized space method for active nodes
        match filter {
            NodeFilter::AttributeFilter { name, filter } => {
                // OPTIMIZED: Use the existing bulk method which is fastest overall
                let mut matching_nodes = Vec::with_capacity(space.get_active_nodes().len() / 4);
                let active_nodes_vec: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, &active_nodes_vec);
                
                // OPTIMIZED: Process results efficiently with pre-allocated capacity
                for (node_id, attr_opt) in node_attr_pairs {
                    if let Some(attr_value) = attr_opt {
                        if filter.matches(attr_value) {
                            matching_nodes.push(node_id);
                        }
                    }
                }
                Ok(matching_nodes)
            }
            
            NodeFilter::AttributeEquals { name, value } => {
                // OPTIMIZED: Use the existing bulk method which is fastest overall
                let mut matching_nodes = Vec::with_capacity(space.get_active_nodes().len() / 4);
                let active_nodes_vec: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, &active_nodes_vec);
                
                // OPTIMIZED: Process results efficiently with pre-allocated capacity
                for (node_id, attr_opt) in node_attr_pairs {
                    if let Some(attr_value) = attr_opt {
                        if *attr_value == *value {
                            matching_nodes.push(node_id);
                        }
                    }
                }
                Ok(matching_nodes)
            }
            
            NodeFilter::HasAttribute { name } => {
                // STREAMING: Check attribute existence directly
                let mut results = Vec::with_capacity(space.get_active_nodes().len() / 2);
                
                for &node_id in space.get_active_nodes() {
                    if space.get_node_attr_index(node_id, name).is_some() {
                        results.push(node_id);
                    }
                }
                Ok(results)
            }
            
            // For complex filters, fall back to collecting the active nodes
            _ => {
                let active_nodes_vec: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
                self.filter_nodes_columnar(&active_nodes_vec, pool, space, filter)
            }
        }
    }

    /// Columnar filtering on any subset of nodes - THE CORE METHOD
    fn filter_nodes_columnar(
        &self,
        nodes: &[NodeId],
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &NodeFilter
    ) -> GraphResult<Vec<NodeId>> {
        match filter {
            NodeFilter::AttributeFilter { name, filter } => {
                // OPTIMIZED: Pre-allocate result vector to avoid multiple allocations
                let mut matching_nodes = Vec::with_capacity(nodes.len() / 4); // estimate 25% match rate
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                
                for (node_id, attr_opt) in node_attr_pairs {
                    if let Some(attr_value) = attr_opt {
                        if filter.matches(attr_value) {
                            matching_nodes.push(node_id);
                        }
                    }
                }
                Ok(matching_nodes)
            }
            
            NodeFilter::AttributeEquals { name, value } => {
                // OPTIMIZED: Pre-allocate and avoid iterator chain
                let mut matching_nodes = Vec::with_capacity(nodes.len() / 4);
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                
                for (node_id, attr_opt) in node_attr_pairs {
                    if let Some(attr_value) = attr_opt {
                        if *attr_value == *value {
                            matching_nodes.push(node_id);
                        }
                    }
                }
                Ok(matching_nodes)
            }
            
            NodeFilter::HasAttribute { name } => {
                // OPTIMIZED: Use bulk attribute lookup and pre-allocate
                let mut results = Vec::with_capacity(nodes.len() / 2);
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                
                for (node_id, attr_opt) in node_attr_pairs {
                    if attr_opt.is_some() {
                        results.push(node_id);
                    }
                }
                Ok(results)
            }
            
            NodeFilter::And(filters) => {
                if filters.is_empty() {
                    return Ok(Vec::new());
                }
                
                let mut result_set: HashSet<NodeId> = 
                    self.filter_nodes_columnar(nodes, pool, space, &filters[0])?
                        .into_iter()
                        .collect();
                
                for sub_filter in &filters[1..] {
                    if result_set.is_empty() {
                        break;
                    }
                    let candidates: Vec<NodeId> = result_set.iter().copied().collect();
                    let filter_result: HashSet<NodeId> = 
                        self.filter_nodes_columnar(&candidates, pool, space, sub_filter)?
                            .into_iter()
                            .collect();
                    result_set = result_set.intersection(&filter_result).copied().collect();
                }
                Ok(result_set.into_iter().collect())
            }
            
            _ => {
                if nodes.len() > 1000 {
                    Ok(nodes
                        .par_iter()
                        .filter(|&node_id| self.node_matches_filter(*node_id, pool, space, filter))
                        .copied()
                        .collect())
                } else {
                    Ok(nodes
                        .iter()
                        .filter(|&node_id| self.node_matches_filter(*node_id, pool, space, filter))
                        .copied()
                        .collect())
                }
            }
        }
    }

    /// Check if a node matches the given filter
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

    /// Check if an edge matches the given filter
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
                    (edge_source == *target && edge_target == *source)
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

    /// Find edges with filtering
    pub fn filter_edges(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter
    ) -> GraphResult<Vec<EdgeId>> {
        let active_edges: Vec<EdgeId> = space.get_active_edges().iter().copied().collect();
        
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

    /// Find all edges matching a filter (used by Graph API)
    pub fn find_edges_by_filter_with_space(
        &mut self,
        pool: &GraphPool,
        space: &GraphSpace,
        filter: &EdgeFilter
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
    HasAttribute { name: AttrName },
    AttributeEquals { name: AttrName, value: AttrValue },
    AttributeFilter { name: AttrName, filter: AttributeFilter },
    DegreeRange { min: usize, max: usize },
    HasNeighbor { neighbor_id: NodeId },
    And(Vec<NodeFilter>),
    Or(Vec<NodeFilter>),
    Not(Box<NodeFilter>),
}

/// Edge filter for graph queries
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeFilter {
    HasAttribute { name: AttrName },
    AttributeEquals { name: AttrName, value: AttrValue },
    AttributeFilter { name: AttrName, filter: AttributeFilter },
    ConnectsNodes { source: NodeId, target: NodeId },
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
