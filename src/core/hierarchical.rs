//! Hierarchical Operations - Meta-nodes and graph hierarchy support
//!
//! This module provides hierarchical graph operations, allowing subgraphs to be
//! collapsed into meta-nodes and expanded back into subgraphs. This enables
//! multi-level graph analysis and visualization.
//!
//! # Design Principles
//! - **Storage Integration**: Uses existing GraphPool for meta-node storage
//! - **Attribute Aggregation**: Flexible aggregation functions (sum, mean, max, etc.)
//! - **Efficient Navigation**: Fast hierarchy traversal with cached metadata
//! - **Trait Consistency**: Follows same patterns as other graph operations

use crate::api::graph::Graph;
use crate::core::traits::{GraphEntity, SubgraphOperations};
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EntityId, NodeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// Attribute aggregation functions for collapsing subgraphs into meta-nodes
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationFunction {
    /// Sum all numeric values
    Sum,
    /// Calculate mean of all numeric values
    Mean,
    /// Take maximum value
    Max,
    /// Take minimum value
    Min,
    /// Count non-null values
    Count,
    /// Take first encountered value
    First,
    /// Take last encountered value
    Last,
    /// Concatenate text values with separator
    Concat(String),
}

impl AggregationFunction {
    /// Parse aggregation function from string
    pub fn from_string(s: &str) -> GraphResult<Self> {
        match s.to_lowercase().as_str() {
            "sum" => Ok(AggregationFunction::Sum),
            "mean" | "avg" | "average" => Ok(AggregationFunction::Mean),
            "max" | "maximum" => Ok(AggregationFunction::Max),
            "min" | "minimum" => Ok(AggregationFunction::Min),
            "count" => Ok(AggregationFunction::Count),
            "first" => Ok(AggregationFunction::First),
            "last" => Ok(AggregationFunction::Last),
            s if s.starts_with("concat:") => {
                let separator = s.strip_prefix("concat:").unwrap_or(",");
                Ok(AggregationFunction::Concat(separator.to_string()))
            }
            _ => Err(GraphError::InvalidInput(format!(
                "Unknown aggregation function: {}",
                s
            ))),
        }
    }
}

impl fmt::Display for AggregationFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregationFunction::Sum => write!(f, "sum"),
            AggregationFunction::Mean => write!(f, "mean"),
            AggregationFunction::Max => write!(f, "max"),
            AggregationFunction::Min => write!(f, "min"),
            AggregationFunction::Count => write!(f, "count"),
            AggregationFunction::First => write!(f, "first"),
            AggregationFunction::Last => write!(f, "last"),
            AggregationFunction::Concat(sep) => write!(f, "concat:{}", sep),
        }
    }
}

impl AggregationFunction {
    /// Apply aggregation function to a list of attribute values
    pub fn aggregate(&self, values: &[AttrValue]) -> GraphResult<AttrValue> {
        if values.is_empty() {
            return Ok(AttrValue::Null);
        }

        match self {
            AggregationFunction::Sum => self.aggregate_sum(values),
            AggregationFunction::Mean => self.aggregate_mean(values),
            AggregationFunction::Max => self.aggregate_max(values),
            AggregationFunction::Min => self.aggregate_min(values),
            AggregationFunction::Count => Ok(AttrValue::Int(values.len() as i64)),
            AggregationFunction::First => Ok(values[0].clone()),
            AggregationFunction::Last => Ok(values[values.len() - 1].clone()),
            AggregationFunction::Concat(separator) => self.aggregate_concat(values, separator),
        }
    }

    fn aggregate_sum(&self, values: &[AttrValue]) -> GraphResult<AttrValue> {
        let mut sum = 0.0;
        let mut has_float = false;

        for value in values {
            match value {
                AttrValue::Int(i) => sum += *i as f64,
                AttrValue::Float(f) => {
                    sum += *f as f64;
                    has_float = true;
                }
                AttrValue::SmallInt(i) => sum += *i as f64,
                _ => continue, // Skip non-numeric values
            }
        }

        if has_float {
            Ok(AttrValue::Float(sum as f32))
        } else {
            Ok(AttrValue::Int(sum as i64))
        }
    }

    fn aggregate_mean(&self, values: &[AttrValue]) -> GraphResult<AttrValue> {
        let sum_result = self.aggregate_sum(values)?;
        let count = values
            .iter()
            .filter(|v| {
                matches!(
                    v,
                    AttrValue::Int(_) | AttrValue::Float(_) | AttrValue::SmallInt(_)
                )
            })
            .count();

        if count == 0 {
            return Ok(AttrValue::Null);
        }

        match sum_result {
            AttrValue::Int(sum) => Ok(AttrValue::Float((sum as f64 / count as f64) as f32)),
            AttrValue::Float(sum) => Ok(AttrValue::Float(sum / count as f32)),
            _ => Ok(AttrValue::Null),
        }
    }

    fn aggregate_max(&self, values: &[AttrValue]) -> GraphResult<AttrValue> {
        let mut max_val: Option<f64> = None;
        let mut has_float = false;

        for value in values {
            let num_val = match value {
                AttrValue::Int(i) => *i as f64,
                AttrValue::Float(f) => {
                    has_float = true;
                    *f as f64
                }
                AttrValue::SmallInt(i) => *i as f64,
                _ => continue,
            };

            max_val = Some(max_val.map_or(num_val, |current| current.max(num_val)));
        }

        match max_val {
            Some(val) if has_float => Ok(AttrValue::Float(val as f32)),
            Some(val) => Ok(AttrValue::Int(val as i64)),
            None => Ok(AttrValue::Null),
        }
    }

    fn aggregate_min(&self, values: &[AttrValue]) -> GraphResult<AttrValue> {
        let mut min_val: Option<f64> = None;
        let mut has_float = false;

        for value in values {
            let num_val = match value {
                AttrValue::Int(i) => *i as f64,
                AttrValue::Float(f) => {
                    has_float = true;
                    *f as f64
                }
                AttrValue::SmallInt(i) => *i as f64,
                _ => continue,
            };

            min_val = Some(min_val.map_or(num_val, |current| current.min(num_val)));
        }

        match min_val {
            Some(val) if has_float => Ok(AttrValue::Float(val as f32)),
            Some(val) => Ok(AttrValue::Int(val as i64)),
            None => Ok(AttrValue::Null),
        }
    }

    fn aggregate_concat(&self, values: &[AttrValue], separator: &str) -> GraphResult<AttrValue> {
        let text_values: Vec<String> = values
            .iter()
            .filter_map(|v| match v {
                AttrValue::Text(s) => Some(s.clone()),
                AttrValue::CompactText(s) => Some(s.as_str().to_string()),
                AttrValue::Int(i) => Some(i.to_string()),
                AttrValue::Float(f) => Some(f.to_string()),
                AttrValue::SmallInt(i) => Some(i.to_string()),
                AttrValue::Bool(b) => Some(b.to_string()),
                _ => None,
            })
            .collect();

        Ok(AttrValue::Text(text_values.join(separator)))
    }
}

/// A meta-node represents a collapsed subgraph as a single node
/// This follows the GraphEntity pattern and can be treated as both a node and a subgraph reference
#[derive(Debug, Clone)]
pub struct MetaNode {
    /// The node ID of this meta-node in the graph
    node_id: NodeId,
    /// Reference to the graph containing this meta-node
    graph_ref: Rc<RefCell<Graph>>,
    /// Cached metadata for efficient access
    contained_subgraph_id: Option<usize>, // SubgraphId from GraphPool
}

impl MetaNode {
    /// Create a new MetaNode from an existing node that contains a subgraph reference
    pub fn new(node_id: NodeId, graph_ref: Rc<RefCell<Graph>>) -> GraphResult<Self> {
        let contained_subgraph_id = {
            let graph = graph_ref.borrow();
            let x = match graph.get_node_attr(node_id, &"contained_subgraph".into())? {
                Some(AttrValue::SubgraphRef(id)) => Some(id),
                _ => None,
            };
            x
        };

        Ok(MetaNode {
            node_id,
            graph_ref,
            contained_subgraph_id,
        })
    }

    /// Get the node ID of this meta-node
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Check if this meta-node contains a subgraph
    pub fn has_contained_subgraph(&self) -> bool {
        self.contained_subgraph_id.is_some()
    }

    /// Get the contained subgraph ID
    pub fn contained_subgraph_id(&self) -> Option<usize> {
        self.contained_subgraph_id
    }

    /// Expand this meta-node back into its contained subgraph
    pub fn expand_to_subgraph(&self) -> GraphResult<Option<Box<dyn SubgraphOperations>>> {
        if let Some(subgraph_id) = self.contained_subgraph_id {
            let (nodes, edges, subgraph_type) = {
                let graph = self.graph_ref.borrow();
                let x = graph.pool().get_subgraph(subgraph_id)?;
                x
            };

            // Create appropriate subgraph type
            let subgraph: Box<dyn SubgraphOperations> =
                Box::new(crate::core::subgraph::Subgraph::new(
                    self.graph_ref.clone(),
                    nodes,
                    edges,
                    subgraph_type,
                ));

            Ok(Some(subgraph))
        } else {
            Ok(None)
        }
    }

    /// Get aggregated attributes of the contained subgraph
    pub fn aggregated_attributes(&self) -> GraphResult<HashMap<AttrName, AttrValue>> {
        let graph = self.graph_ref.borrow();
        let x = graph.pool().get_all_node_attributes(self.node_id);
        x
    }

    /// Re-aggregate attributes from the contained subgraph using specified functions
    pub fn re_aggregate(
        &self,
        agg_functions: HashMap<AttrName, AggregationFunction>,
    ) -> GraphResult<()> {
        if let Some(subgraph) = self.expand_to_subgraph()? {
            // Get all attributes from nodes in the contained subgraph
            for (attr_name, agg_func) in agg_functions {
                let mut values = Vec::new();

                for &node_id in subgraph.node_set() {
                    if let Some(value) = subgraph.get_node_attribute(node_id, &attr_name)? {
                        values.push(value);
                    }
                }

                if !values.is_empty() {
                    let aggregated_value = agg_func.aggregate(&values)?;
                    let mut graph = self.graph_ref.borrow_mut();
                    graph.set_node_attr(self.node_id, attr_name, aggregated_value)?;
                }
            }
        }
        Ok(())
    }
}

impl GraphEntity for MetaNode {
    fn entity_id(&self) -> EntityId {
        EntityId::Node(self.node_id)
    }

    fn entity_type(&self) -> &'static str {
        "meta_node"
    }

    fn graph_ref(&self) -> Rc<RefCell<Graph>> {
        self.graph_ref.clone()
    }

    fn related_entities(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // Return entities from the contained subgraph if it exists
        if let Some(subgraph) = self.expand_to_subgraph()? {
            subgraph.related_entities()
        } else {
            // Return neighboring meta-nodes
            let graph = self.graph_ref.borrow();
            let neighbor_ids = graph.neighbors(self.node_id)?;

            let entities: Vec<Box<dyn GraphEntity>> = neighbor_ids
                .into_iter()
                .filter_map(|neighbor_id| {
                    // Try to create MetaNode, fallback to regular EntityNode
                    MetaNode::new(neighbor_id, self.graph_ref.clone())
                        .ok()
                        .map(|meta_node| Box::new(meta_node) as Box<dyn GraphEntity>)
                })
                .collect();

            Ok(entities)
        }
    }

    fn summary(&self) -> String {
        if let Some(subgraph_id) = self.contained_subgraph_id {
            format!(
                "MetaNode(id={}, contains_subgraph={})",
                self.node_id, subgraph_id
            )
        } else {
            format!("MetaNode(id={}, empty)", self.node_id)
        }
    }
}

/// Operations for managing hierarchical graph structures
pub trait HierarchicalOperations: GraphEntity {
    /// Collapse this entity into a meta-node with attribute aggregation
    fn collapse_to_meta_node(
        &self,
        agg_functions: HashMap<AttrName, AggregationFunction>,
    ) -> GraphResult<MetaNode>;

    /// Get parent meta-node if this entity is contained within one
    fn parent_meta_node(&self) -> GraphResult<Option<MetaNode>>;

    /// Get child meta-nodes if this entity contains them
    fn child_meta_nodes(&self) -> GraphResult<Vec<MetaNode>>;

    /// Get the hierarchy level of this entity (0 = root level)
    fn hierarchy_level(&self) -> GraphResult<usize>;

    /// Navigate up the hierarchy to the root level
    fn to_root(&self) -> GraphResult<Box<dyn GraphEntity>>;

    /// Get all entities at the same hierarchy level
    fn siblings(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>>;
}

// Default implementations for SubgraphOperations that support hierarchical operations
impl<T: SubgraphOperations> HierarchicalOperations for T {
    fn collapse_to_meta_node(
        &self,
        agg_functions: HashMap<AttrName, AggregationFunction>,
    ) -> GraphResult<MetaNode> {
        // Use the existing collapse_to_node method, then create MetaNode wrapper
        let agg_strings: HashMap<AttrName, String> = agg_functions
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string())) // Use proper string conversion
            .collect();

        let meta_node_id = self.collapse_to_node(agg_strings)?;
        MetaNode::new(meta_node_id, self.graph_ref())
    }

    fn parent_meta_node(&self) -> GraphResult<Option<MetaNode>> {
        // TODO: Implement parent tracking in future iteration
        Ok(None)
    }

    fn child_meta_nodes(&self) -> GraphResult<Vec<MetaNode>> {
        // Look for nodes in this subgraph that are meta-nodes
        let mut meta_nodes = Vec::new();

        for &node_id in self.node_set() {
            if let Ok(meta_node) = MetaNode::new(node_id, self.graph_ref()) {
                if meta_node.has_contained_subgraph() {
                    meta_nodes.push(meta_node);
                }
            }
        }

        Ok(meta_nodes)
    }

    fn hierarchy_level(&self) -> GraphResult<usize> {
        // TODO: Implement hierarchy level calculation in future iteration
        Ok(0)
    }

    fn to_root(&self) -> GraphResult<Box<dyn GraphEntity>> {
        // For now, return self as root (TODO: implement proper hierarchy traversal)
        Ok(Box::new(crate::core::node::EntityNode::new(
            self.node_set().iter().next().copied().unwrap_or(0),
            self.graph_ref(),
        )))
    }

    fn siblings(&self) -> GraphResult<Vec<Box<dyn GraphEntity>>> {
        // TODO: Implement sibling discovery in future iteration
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use crate::core::subgraph::Subgraph;
    use std::collections::HashSet;

    #[test]
    fn test_aggregation_functions() {
        // Test numeric aggregations
        let values = vec![
            AttrValue::Int(10),
            AttrValue::Int(20),
            AttrValue::Float(30.5),
            AttrValue::SmallInt(15),
        ];

        // Test sum
        let sum_func = AggregationFunction::Sum;
        let result = sum_func.aggregate(&values).unwrap();
        match result {
            AttrValue::Float(f) => assert!((f - 75.5).abs() < f32::EPSILON),
            _ => panic!("Expected float result for sum"),
        }

        // Test mean
        let mean_func = AggregationFunction::Mean;
        let result = mean_func.aggregate(&values).unwrap();
        match result {
            AttrValue::Float(f) => assert!((f - 18.875).abs() < f32::EPSILON),
            _ => panic!("Expected float result for mean"),
        }

        // Test max
        let max_func = AggregationFunction::Max;
        let result = max_func.aggregate(&values).unwrap();
        match result {
            AttrValue::Float(f) => assert!((f - 30.5).abs() < f32::EPSILON),
            _ => panic!("Expected float result for max"),
        }

        // Test min
        let min_func = AggregationFunction::Min;
        let result = min_func.aggregate(&values).unwrap();
        match result {
            AttrValue::Float(f) => assert!((f - 10.0).abs() < f32::EPSILON),
            _ => panic!("Expected float result for min"),
        }
    }

    #[test]
    fn test_text_aggregation() {
        let values = vec![
            AttrValue::Text("hello".to_string()),
            AttrValue::Text("world".to_string()),
            AttrValue::Int(42),
        ];

        // Test concat
        let concat_func = AggregationFunction::Concat(" ".to_string());
        let result = concat_func.aggregate(&values).unwrap();
        match result {
            AttrValue::Text(s) => assert_eq!(s, "hello world 42"),
            _ => panic!("Expected text result for concat"),
        }

        // Test first/last
        let first_func = AggregationFunction::First;
        let result = first_func.aggregate(&values).unwrap();
        match result {
            AttrValue::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("Expected text result for first"),
        }
    }

    #[test]
    fn test_aggregation_function_parsing() {
        assert_eq!(
            AggregationFunction::from_string("sum").unwrap(),
            AggregationFunction::Sum
        );
        assert_eq!(
            AggregationFunction::from_string("mean").unwrap(),
            AggregationFunction::Mean
        );
        assert_eq!(
            AggregationFunction::from_string("avg").unwrap(),
            AggregationFunction::Mean
        );
        assert_eq!(
            AggregationFunction::from_string("concat:;").unwrap(),
            AggregationFunction::Concat(";".to_string())
        );

        // Test error case
        assert!(AggregationFunction::from_string("invalid").is_err());
    }

    #[test]
    fn test_meta_node_creation() {
        let mut graph = Graph::new();
        let node_id = graph.add_node();

        // Test node without subgraph reference (should work but be empty)
        let graph_rc = Rc::new(RefCell::new(graph));
        let meta_node = MetaNode::new(node_id, graph_rc.clone()).unwrap();

        assert_eq!(meta_node.node_id(), node_id);
        assert!(!meta_node.has_contained_subgraph());
        assert_eq!(meta_node.contained_subgraph_id(), None);
        assert_eq!(meta_node.entity_type(), "meta_node");

        // Test expansion of empty meta-node
        let expanded = meta_node.expand_to_subgraph().unwrap();
        assert!(expanded.is_none());
    }

    #[test]
    fn test_subgraph_collapse_to_node() {
        // Create a small graph for testing
        let mut graph = Graph::new();
        let node1 = graph.add_node();
        let node2 = graph.add_node();
        let node3 = graph.add_node();

        // Add some test attributes
        graph
            .set_node_attr(node1, "value".into(), AttrValue::Int(10))
            .unwrap();
        graph
            .set_node_attr(node2, "value".into(), AttrValue::Int(20))
            .unwrap();
        graph
            .set_node_attr(node3, "value".into(), AttrValue::Int(30))
            .unwrap();

        // Create edges
        graph.add_edge(node1, node2).unwrap();
        graph.add_edge(node2, node3).unwrap();

        // Create subgraph and test collapse
        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes = HashSet::from([node1, node2, node3]);
        let subgraph =
            Subgraph::from_nodes(graph_rc.clone(), nodes, "test_subgraph".to_string()).unwrap();

        // Test collapse with aggregation
        let mut agg_functions = HashMap::new();
        agg_functions.insert("value".into(), "sum".to_string());

        let meta_node_id = subgraph.collapse_to_node(agg_functions).unwrap();

        // Verify meta-node was created with aggregated attributes
        let graph_ref = graph_rc.borrow();

        // Check that meta-node has the subgraph reference
        let subgraph_ref = graph_ref
            .get_node_attr(meta_node_id, &"contained_subgraph".into())
            .unwrap();
        assert!(matches!(subgraph_ref, Some(AttrValue::SubgraphRef(_))));

        // Check aggregated value
        let aggregated_value = graph_ref
            .get_node_attr(meta_node_id, &"value".into())
            .unwrap();
        match aggregated_value {
            Some(AttrValue::Int(sum)) => assert_eq!(sum, 60), // 10 + 20 + 30
            Some(AttrValue::SmallInt(sum)) => assert_eq!(sum, 60), // Handle SmallInt case
            Some(AttrValue::Float(sum)) => assert_eq!(sum as i64, 60), // Handle float case
            _ => panic!("Expected aggregated sum of 60, got {:?}", aggregated_value),
        }
    }

    #[test]
    fn test_hierarchical_operations_trait() {
        let mut graph = Graph::new();
        let node1 = graph.add_node();
        let node2 = graph.add_node();

        graph.add_edge(node1, node2).unwrap();
        graph
            .set_node_attr(node1, "weight".into(), AttrValue::Float(1.5))
            .unwrap();
        graph
            .set_node_attr(node2, "weight".into(), AttrValue::Float(2.5))
            .unwrap();

        let graph_rc = Rc::new(RefCell::new(graph));
        let nodes = HashSet::from([node1, node2]);
        let subgraph = Subgraph::from_nodes(graph_rc, nodes, "test_subgraph".to_string()).unwrap();

        // Test collapse_to_meta_node
        let mut agg_functions = HashMap::new();
        agg_functions.insert("weight".into(), AggregationFunction::Mean);

        let meta_node = subgraph.collapse_to_meta_node(agg_functions).unwrap();
        assert!(meta_node.has_contained_subgraph());
        assert_eq!(meta_node.entity_type(), "meta_node");

        // Test child_meta_nodes (should be empty for this simple case)
        let children = subgraph.child_meta_nodes().unwrap();
        assert!(children.is_empty());

        // Test hierarchy_level (returns 0 for now)
        let level = subgraph.hierarchy_level().unwrap();
        assert_eq!(level, 0);
    }
}
