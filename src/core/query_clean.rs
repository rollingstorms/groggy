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
        let active_nodes: Vec<NodeId> = space.get_active_nodes().iter().copied().collect();
        self.filter_nodes_columnar(&active_nodes, pool, space, filter)
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
                // Bulk optimization: Single operation for all nodes
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                let matching_nodes: Vec<NodeId> = node_attr_pairs
                    .into_iter()
                    .filter_map(|(node_id, attr_opt)| {
                        attr_opt.filter(|attr_value| filter.matches(attr_value))
                               .map(|_| node_id)
                    })
                    .collect();
                Ok(matching_nodes)
            }
            
            NodeFilter::AttributeEquals { name, value } => {
                // Bulk equality check
                let node_attr_pairs = space.get_attributes_for_nodes(pool, name, nodes);
                let matching_nodes: Vec<NodeId> = node_attr_pairs
                    .into_iter()
                    .filter_map(|(node_id, attr_opt)| {
                        attr_opt.filter(|attr_value| **attr_value == *value)
                               .map(|_| node_id)
                    })
                    .collect();
                Ok(matching_nodes)
            }
            
            NodeFilter::HasAttribute { name } => {
                let results: Vec<NodeId> = nodes.iter()
                    .filter(|&node_id| space.get_node_attr_index(*node_id, name).is_some())
                    .copied()
                    .collect();
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
    GreaterThan(AttrValue),
    LessThan(AttrValue),
}

impl AttributeFilter {
    /// Check if a value matches this filter
    pub fn matches(&self, value: &AttrValue) -> bool {
        match self {
            AttributeFilter::Equals(target) => value == target,
            AttributeFilter::GreaterThan(target) => {
                match (value, target) {
                    (AttrValue::Int(a), AttrValue::Int(b)) => a > b,
                    (AttrValue::Float(a), AttrValue::Float(b)) => a > b,
                    _ => false,
                }
            }
            AttributeFilter::LessThan(target) => {
                match (value, target) {
                    (AttrValue::Int(a), AttrValue::Int(b)) => a < b,
                    (AttrValue::Float(a), AttrValue::Float(b)) => a < b,
                    _ => false,
                }
            }
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
