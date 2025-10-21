//! MetaGraph Composer - Clean API for Meta-Node Creation
//!
//! This module provides an intuitive, builder-pattern API for composing meta-graphs
//! through subgraph collapse operations. It replaces the complex EdgeAggregationConfig
//! system with a simpler, more discoverable interface.

#![allow(clippy::wrong_self_convention)]

use crate::entities::MetaNode;
use crate::errors::{GraphError, GraphResult};
use crate::traits::{NodeStrategy, SubgraphOperations};
use crate::types::{AttrName, NodeId};
use std::collections::HashMap;

/// Edge strategies for meta-node creation
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeStrategy {
    /// Aggregate parallel edges between subgraph and external nodes (default)
    /// Combine attributes using aggregation functions, create single meta-edge per target
    Aggregate,

    /// Keep all external edges as-is, copying them to the meta-node
    /// Preserves original edge attributes and allows multiple edges to same target
    KeepExternal,

    /// Drop all edges - isolate the meta-node completely
    /// Useful for pure hierarchical grouping without connectivity
    DropAll,

    /// Contract edges through the subgraph (advanced)
    /// External A -> subgraph -> External B becomes A -> B
    /// Subgraph becomes pure aggregation node
    ContractAll,
}

impl Default for EdgeStrategy {
    fn default() -> Self {
        EdgeStrategy::Aggregate
    }
}

impl EdgeStrategy {
    /// Parse edge strategy from string (intentionally not implementing FromStr trait to avoid ambiguity)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> GraphResult<Self> {
        match s.to_lowercase().as_str() {
            "aggregate" => Ok(EdgeStrategy::Aggregate),
            "keep_external" => Ok(EdgeStrategy::KeepExternal),
            "drop_all" => Ok(EdgeStrategy::DropAll),
            "contract_all" => Ok(EdgeStrategy::ContractAll),
            _ => Err(GraphError::InvalidInput(format!(
                "Unknown edge strategy: '{}'. Valid options: aggregate, keep_external, drop_all, contract_all",
                s
            ))),
        }
    }
}

/// Node aggregation specification - flexible input format
#[derive(Debug, Clone)]
pub enum NodeAggregation {
    /// Simple: target_attr = function applied to same-named source
    /// Example: "salary" -> "mean" means avg_salary = mean(salary)
    Simple(String, String), // (target_attr, function)

    /// Tuple: target_attr = function applied to different source_attr
    /// Example: ("avg_salary", "mean", "salary")
    Tuple(String, String, String), // (target_attr, function, source_attr)

    /// Count: target_attr = count of nodes (no source needed)
    /// Example: "size" -> Count means size = count(nodes)
    Count(String), // (target_attr)
}

impl NodeAggregation {
    /// Get the target attribute name
    pub fn target_attr(&self) -> &str {
        match self {
            NodeAggregation::Simple(target, _) => target,
            NodeAggregation::Tuple(target, _, _) => target,
            NodeAggregation::Count(target) => target,
        }
    }

    /// Get the aggregation function
    pub fn function(&self) -> &str {
        match self {
            NodeAggregation::Simple(_, func) => func,
            NodeAggregation::Tuple(_, func, _) => func,
            NodeAggregation::Count(_) => "count",
        }
    }

    /// Get the source attribute name (defaults to target for Simple)
    pub fn source_attr(&self) -> &str {
        match self {
            NodeAggregation::Simple(target, _) => target, // Same as target
            NodeAggregation::Tuple(_, _, source) => source,
            NodeAggregation::Count(_) => "", // No source needed
        }
    }
}

/// Edge aggregation function specification
#[derive(Debug, Clone)]
pub struct EdgeAggregation {
    /// Attribute name to aggregate
    pub attr_name: AttrName,
    /// Aggregation function name ("mean", "sum", "concat", etc.)
    pub function: String,
}

/// Preset configurations for common use cases
#[derive(Debug, Clone)]
pub struct ComposerPreset {
    pub edge_strategy: EdgeStrategy,
    pub edge_aggs: Vec<EdgeAggregation>,
    pub include_edge_count: bool,
    pub entity_type: String,
}

/// Built-in preset configurations
pub fn get_preset(name: &str) -> GraphResult<ComposerPreset> {
    match name.to_lowercase().as_str() {
        "social_network" => Ok(ComposerPreset {
            edge_strategy: EdgeStrategy::Aggregate,
            edge_aggs: vec![
                EdgeAggregation {
                    attr_name: "weight".into(),
                    function: "mean".to_string(),
                },
                EdgeAggregation {
                    attr_name: "type".into(),
                    function: "concat".to_string(),
                },
            ],
            include_edge_count: true,
            entity_type: "community".to_string(),
        }),

        "org_hierarchy" => Ok(ComposerPreset {
            edge_strategy: EdgeStrategy::Aggregate,
            edge_aggs: vec![
                EdgeAggregation {
                    attr_name: "reports_to".into(),
                    function: "first".to_string(),
                },
                EdgeAggregation {
                    attr_name: "weight".into(),
                    function: "sum".to_string(),
                },
            ],
            include_edge_count: false,
            entity_type: "department".to_string(),
        }),

        "flow_network" => Ok(ComposerPreset {
            edge_strategy: EdgeStrategy::ContractAll,
            edge_aggs: vec![
                EdgeAggregation {
                    attr_name: "capacity".into(),
                    function: "min".to_string(),
                },
                EdgeAggregation {
                    attr_name: "flow".into(),
                    function: "sum".to_string(),
                },
            ],
            include_edge_count: false,
            entity_type: "junction".to_string(),
        }),

        _ => Err(GraphError::InvalidInput(format!(
            "Unknown preset: '{}'. Available presets: social_network, org_hierarchy, flow_network",
            name
        ))),
    }
}

/// A plan for creating a meta-node (not yet executed)
///
/// This provides a clean separation between configuration and execution,
/// allowing users to preview, modify, and validate plans before applying them.
#[derive(Debug, Clone)]
pub struct MetaNodePlan {
    /// Node aggregation specifications
    pub node_aggs: Vec<NodeAggregation>,

    /// Edge aggregation specifications  
    pub edge_aggs: Vec<EdgeAggregation>,

    /// Edge handling strategy
    pub edge_strategy: EdgeStrategy,

    /// Node handling strategy  
    pub node_strategy: NodeStrategy,

    /// Include edge count in meta-edges
    pub include_edge_count: bool,

    /// Mark nodes/edges with entity_type
    pub mark_entity_type: bool,

    /// Entity type to use for marking
    pub entity_type: String,

    /// Source subgraph node IDs (for validation)
    pub source_nodes: std::collections::HashSet<NodeId>,

    /// Source subgraph edge IDs (for validation)
    pub source_edges: std::collections::HashSet<crate::types::EdgeId>,
}

impl MetaNodePlan {
    /// Create a new meta-node plan
    pub fn new(
        source_nodes: std::collections::HashSet<NodeId>,
        source_edges: std::collections::HashSet<crate::types::EdgeId>,
    ) -> Self {
        Self {
            node_aggs: Vec::new(),
            edge_aggs: Vec::new(),
            edge_strategy: EdgeStrategy::default(),
            node_strategy: NodeStrategy::default(), // Default to Extract behavior
            include_edge_count: true,
            mark_entity_type: true,
            entity_type: "meta".to_string(),
            source_nodes,
            source_edges,
        }
    }

    /// Apply a preset configuration
    pub fn with_preset(mut self, preset_name: &str) -> GraphResult<Self> {
        let preset = get_preset(preset_name)?;
        self.edge_strategy = preset.edge_strategy;
        self.edge_aggs = preset.edge_aggs;
        self.include_edge_count = preset.include_edge_count;
        self.entity_type = preset.entity_type;
        Ok(self)
    }

    /// Add node aggregation
    pub fn with_node_agg(
        mut self,
        target: String,
        function: String,
        source: Option<String>,
    ) -> Self {
        let agg = match (source, function.as_str()) {
            (_, "count") => NodeAggregation::Count(target),
            (Some(src), _) => NodeAggregation::Tuple(target, function, src),
            (None, _) => NodeAggregation::Simple(target, function),
        };
        self.node_aggs.push(agg);
        self
    }

    /// Add edge aggregation
    pub fn with_edge_agg(mut self, attr_name: String, function: String) -> Self {
        self.edge_aggs.push(EdgeAggregation {
            attr_name: attr_name.into(),
            function,
        });
        self
    }

    /// Set edge strategy
    pub fn with_edge_strategy(mut self, strategy: EdgeStrategy) -> Self {
        self.edge_strategy = strategy;
        self
    }

    /// Set entity type
    pub fn with_entity_type(mut self, entity_type: String) -> Self {
        self.entity_type = entity_type;
        self
    }

    /// Preview what the plan will create (without executing)
    pub fn preview(&self) -> ComposerPreview {
        ComposerPreview {
            meta_node_attributes: self
                .node_aggs
                .iter()
                .map(|agg| (agg.target_attr().to_string(), agg.function().to_string()))
                .collect(),
            meta_edges_count: match self.edge_strategy {
                EdgeStrategy::DropAll => 0,
                _ => self.estimate_meta_edges_count(),
            },
            edge_strategy: self.edge_strategy.clone(),
            will_include_edge_count: self.include_edge_count,
            entity_type: self.entity_type.clone(),
        }
    }

    /// Estimate the number of meta-edges that will be created
    fn estimate_meta_edges_count(&self) -> usize {
        // This is a rough estimate - actual count depends on graph structure
        // In a real implementation, we'd analyze the actual external connections
        match self.edge_strategy {
            EdgeStrategy::DropAll => 0,
            EdgeStrategy::Aggregate => self.source_nodes.len().min(10), // Rough estimate
            EdgeStrategy::KeepExternal => self.source_edges.len(),
            EdgeStrategy::ContractAll => self.source_edges.len(),
        }
    }
}

/// Preview of what a MetaNodePlan will create
#[derive(Debug, Clone)]
pub struct ComposerPreview {
    /// Attributes that will be created on the meta-node
    pub meta_node_attributes: HashMap<String, String>, // attr_name -> function

    /// Estimated number of meta-edges
    pub meta_edges_count: usize,

    /// Edge strategy being used
    pub edge_strategy: EdgeStrategy,

    /// Whether edge count will be included
    pub will_include_edge_count: bool,

    /// Entity type that will be assigned
    pub entity_type: String,
}

impl MetaNodePlan {
    /// Execute the plan and create the meta-node in the graph (strict - fails on missing attributes)
    ///
    /// This converts the plan into actual graph modifications, creating the meta-node
    /// with aggregated attributes and meta-edges according to the configured strategy.
    ///
    /// # Arguments
    /// * `subgraph` - The subgraph to collapse (implements SubgraphOperations)
    ///
    /// # Returns
    /// A `MetaNode` representing the newly created meta-node
    pub fn add_to_graph<T: SubgraphOperations>(&self, subgraph: &T) -> GraphResult<MetaNode> {
        use crate::subgraphs::hierarchical::AggregationFunction;
        use crate::types::AttrValue;
        use std::collections::HashMap;

        let binding = subgraph.graph_ref();
        let mut graph = binding.borrow_mut();

        // Store subgraph reference in GraphPool first
        let subgraph_id = graph.pool_mut().store_subgraph(
            subgraph.node_set().clone(),
            subgraph.edge_set().clone(),
            subgraph.entity_type().to_string(),
        )?;

        // Create meta-node atomically with all required attributes
        let meta_node_id = graph.add_node();
        graph.set_node_attr_internal(
            meta_node_id,
            "entity_type".to_string(),
            AttrValue::Text("meta".to_string()),
        )?;
        graph.set_node_attr_internal(
            meta_node_id,
            "contains_subgraph".to_string(),
            AttrValue::SubgraphRef(subgraph_id),
        )?;

        // Mark original nodes as absorbed, but preserve existing meta-nodes
        drop(graph); // Release borrow for bulk operation
        {
            let binding = subgraph.graph_ref();
            let graph_ref = binding.borrow();
            let mut entity_type_attrs = Vec::new();

            for &node_id in subgraph.node_set() {
                // Check if node is already a meta-node
                let current_entity_type = graph_ref
                    .get_node_attr(node_id, &"entity_type".into())
                    .unwrap_or(None);

                let new_entity_type = match current_entity_type {
                    Some(AttrValue::Text(ref s)) if s == "meta" => {
                        // Preserve meta-nodes
                        AttrValue::Text("meta".to_string())
                    }
                    Some(AttrValue::CompactText(ref s)) if s.as_str() == "meta" => {
                        // Preserve meta-nodes
                        AttrValue::Text("meta".to_string())
                    }
                    _ => {
                        // Mark non-meta nodes as base
                        AttrValue::Text("base".to_string())
                    }
                };

                entity_type_attrs.push((node_id, new_entity_type));
            }

            drop(graph_ref);
            drop(binding);

            let mut bulk_attrs = HashMap::new();
            bulk_attrs.insert("entity_type".into(), entity_type_attrs);
            subgraph.set_node_attrs(bulk_attrs)?;
        }

        // Calculate aggregated attributes with proper source->target mapping
        let mut aggregated_attrs = Vec::new();
        for agg in &self.node_aggs {
            let target_attr = agg.target_attr();
            let source_attr = agg.source_attr();
            let function_str = agg.function();
            let agg_func = AggregationFunction::from_string(function_str)?;

            // Special handling for Count aggregation
            let aggregated_value = if matches!(agg_func, AggregationFunction::Count) {
                AttrValue::Int(subgraph.node_set().len() as i64)
            } else if source_attr.is_empty() {
                // Count doesn't need a source attribute
                AttrValue::Int(0)
            } else {
                // Collect values from the source attribute
                let mut values = Vec::new();
                let binding = subgraph.graph_ref();
                let graph = binding.borrow();

                for &node_id in subgraph.node_set() {
                    if let Some(value) = graph.get_node_attr(node_id, &source_attr.into())? {
                        values.push(value);
                    }
                }

                drop(graph);
                drop(binding);

                if values.is_empty() {
                    // Provide sensible defaults based on aggregation function
                    match function_str {
                        "sum" => AttrValue::Int(0),
                        "mean" | "min" | "max" => AttrValue::Float(0.0),
                        "first" | "concat" => AttrValue::Text("".to_string()),
                        _ => AttrValue::Int(0),
                    }
                } else {
                    // Apply aggregation function to collected values
                    agg_func.aggregate(&values)?
                }
            };

            aggregated_attrs.push((target_attr.into(), aggregated_value));
        }

        // Set all aggregated attributes in bulk
        if !aggregated_attrs.is_empty() {
            let mut bulk_attrs = HashMap::new();
            for (attr_name, aggregated_value) in aggregated_attrs {
                let node_value_pairs = vec![(meta_node_id, aggregated_value)];
                bulk_attrs.insert(attr_name, node_value_pairs);
            }
            subgraph.set_node_attrs(bulk_attrs)?;
        }

        // Handle edges with existing edge configuration
        let edge_config = self.convert_to_edge_config()?;
        subgraph.create_meta_edges_with_config(meta_node_id, &edge_config)?;

        // Create MetaNode wrapper
        let meta_node = MetaNode::new(meta_node_id, subgraph.graph_ref())?;

        Ok(meta_node)
    }

    /// Execute the plan with defaults for missing attributes (lenient)
    pub fn add_to_graph_with_defaults<T: SubgraphOperations>(
        &self,
        subgraph: &T,
    ) -> GraphResult<MetaNode> {
        use crate::subgraphs::hierarchical::AggregationFunction;
        use crate::types::AttrValue;
        use std::collections::HashMap;

        let binding = subgraph.graph_ref();
        let mut graph = binding.borrow_mut();

        // Store subgraph reference in GraphPool first
        let subgraph_id = graph.pool_mut().store_subgraph(
            subgraph.node_set().clone(),
            subgraph.edge_set().clone(),
            subgraph.entity_type().to_string(),
        )?;

        // Create meta-node atomically with all required attributes
        let meta_node_id = graph.add_node();
        graph.set_node_attr_internal(
            meta_node_id,
            "entity_type".to_string(),
            AttrValue::Text("meta".to_string()),
        )?;
        graph.set_node_attr_internal(
            meta_node_id,
            "contains_subgraph".to_string(),
            AttrValue::SubgraphRef(subgraph_id),
        )?;

        // Mark original nodes as absorbed, but preserve existing meta-nodes
        drop(graph); // Release borrow for bulk operation
        {
            let binding = subgraph.graph_ref();
            let graph_ref = binding.borrow();
            let mut entity_type_attrs = Vec::new();

            for &node_id in subgraph.node_set() {
                // Check if node is already a meta-node
                let current_entity_type = graph_ref
                    .get_node_attr(node_id, &"entity_type".into())
                    .unwrap_or(None);

                let new_entity_type = match current_entity_type {
                    Some(AttrValue::Text(ref s)) if s == "meta" => {
                        // Preserve meta-nodes
                        AttrValue::Text("meta".to_string())
                    }
                    Some(AttrValue::CompactText(ref s)) if s.as_str() == "meta" => {
                        // Preserve meta-nodes
                        AttrValue::Text("meta".to_string())
                    }
                    _ => {
                        // Mark non-meta nodes as base
                        AttrValue::Text("base".to_string())
                    }
                };

                entity_type_attrs.push((node_id, new_entity_type));
            }

            drop(graph_ref);
            drop(binding);

            let mut bulk_attrs = HashMap::new();
            bulk_attrs.insert("entity_type".into(), entity_type_attrs);
            subgraph.set_node_attrs(bulk_attrs)?;
        }

        // Calculate aggregated attributes with proper source->target mapping (LENIENT MODE)
        let mut aggregated_attrs = Vec::new();
        for agg in &self.node_aggs {
            let target_attr = agg.target_attr();
            let source_attr = agg.source_attr();
            let function_str = agg.function();
            let agg_func = AggregationFunction::from_string(function_str)?;

            // Special handling for Count aggregation
            let aggregated_value = if matches!(agg_func, AggregationFunction::Count) {
                AttrValue::Int(subgraph.node_set().len() as i64)
            } else if source_attr.is_empty() {
                // Count doesn't need a source attribute
                AttrValue::Int(0)
            } else {
                // Collect values from the source attribute
                let mut values = Vec::new();
                let binding = subgraph.graph_ref();
                let graph = binding.borrow();

                for &node_id in subgraph.node_set() {
                    if let Some(value) = graph.get_node_attr(node_id, &source_attr.into())? {
                        values.push(value);
                    }
                }

                drop(graph);
                drop(binding);

                if values.is_empty() {
                    // LENIENT MODE: Provide sensible defaults based on aggregation function
                    match function_str {
                        "sum" => AttrValue::Int(0),
                        "mean" | "min" | "max" => AttrValue::Float(0.0),
                        "first" | "concat" => AttrValue::Text("".to_string()),
                        _ => AttrValue::Int(0),
                    }
                } else {
                    // Apply aggregation function to collected values
                    agg_func.aggregate(&values)?
                }
            };

            aggregated_attrs.push((target_attr.into(), aggregated_value));
        }

        // Set all aggregated attributes in bulk
        if !aggregated_attrs.is_empty() {
            let mut bulk_attrs = HashMap::new();
            for (attr_name, aggregated_value) in aggregated_attrs {
                let node_value_pairs = vec![(meta_node_id, aggregated_value)];
                bulk_attrs.insert(attr_name, node_value_pairs);
            }
            subgraph.set_node_attrs(bulk_attrs)?;
        }

        // Handle edges with existing edge configuration
        let edge_config = self.convert_to_edge_config()?;
        subgraph.create_meta_edges_with_config(meta_node_id, &edge_config)?;

        // Handle original nodes based on node strategy AFTER all processing is complete
        match self.node_strategy {
            NodeStrategy::Extract => {
                // Extract: Mark original nodes as absorbed but keep them in graph (current behavior)
                // This is already done above in the current implementation, so nothing more needed
            }
            NodeStrategy::Collapse => {
                // Collapse: Remove original nodes from the graph completely
                let binding = subgraph.graph_ref();
                let mut graph = binding.borrow_mut();
                let nodes_to_remove: Vec<crate::types::NodeId> =
                    subgraph.node_set().iter().copied().collect();
                for node_id in nodes_to_remove {
                    graph.remove_node(node_id)?;
                }
            }
        }

        // Create MetaNode wrapper
        let meta_node = MetaNode::new(meta_node_id, subgraph.graph_ref())?;

        Ok(meta_node)
    }

    /// Convert node aggregations to HashMap format expected by existing code
    #[allow(dead_code)]
    fn convert_to_agg_functions(&self) -> HashMap<AttrName, String> {
        let mut result = HashMap::new();

        for agg in &self.node_aggs {
            let target = agg.target_attr();
            let function = agg.function();
            result.insert(target.into(), function.to_string());
        }

        result
    }

    /// Convert to EdgeAggregationConfig for existing implementation
    fn convert_to_edge_config(
        &self,
    ) -> GraphResult<crate::traits::subgraph_operations::EdgeAggregationConfig> {
        use crate::traits::subgraph_operations::{
            EdgeAggregationConfig, EdgeAggregationFunction, ExternalEdgeStrategy, MetaEdgeStrategy,
        };

        // Convert edge strategy
        let external_strategy = match self.edge_strategy {
            EdgeStrategy::Aggregate => ExternalEdgeStrategy::Aggregate,
            EdgeStrategy::KeepExternal => ExternalEdgeStrategy::Copy,
            EdgeStrategy::DropAll => ExternalEdgeStrategy::None,
            EdgeStrategy::ContractAll => ExternalEdgeStrategy::Aggregate, // TODO: Implement contract properly
        };

        // Convert edge aggregations
        let mut edge_aggregation = HashMap::new();
        for edge_agg in &self.edge_aggs {
            let function = EdgeAggregationFunction::from_string(&edge_agg.function)?;
            edge_aggregation.insert(edge_agg.attr_name.clone(), function);
        }

        Ok(EdgeAggregationConfig {
            edge_to_external: external_strategy,
            edge_to_meta: MetaEdgeStrategy::Auto,
            edge_aggregation,
            default_aggregation: EdgeAggregationFunction::First, // Safer default than Sum
            min_edge_count: 1,
            include_edge_count: self.include_edge_count,
            mark_entity_type: self.mark_entity_type,
            node_strategy: self.node_strategy, // Use the stored node_strategy
        })
    }
}
