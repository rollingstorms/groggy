//! SubgraphOperations - Shared interface for all subgraph-like entities
//!
//! This trait provides common operations for all subgraph types while leveraging
//! our existing efficient storage (HashSet<NodeId>, HashSet<EdgeId>) and algorithms.
//! All subgraph types use the same optimized foundation with specialized behaviors.

use crate::traits::GraphEntity;
use crate::errors::{GraphError, GraphResult};
use crate::types::{AttrName, AttrValue, EdgeId, NodeId};
use crate::subgraphs::composer::{MetaNodePlan, NodeAggregation, EdgeAggregation, EdgeStrategy, ComposerPreview};
use std::collections::{HashMap, HashSet};

/// Enhanced aggregation specification supporting multiple syntax forms
#[derive(Debug, Clone)]
pub struct AggregationSpec {
    /// Target attribute name for the result
    pub target_attr: AttrName,
    /// Aggregation function to apply
    pub function: String,
    /// Source attribute name (None for count/node-based functions)
    pub source_attr: Option<AttrName>,
    /// Default value if source attribute is missing
    pub default_value: Option<AttrValue>,
}

/// Strategy for handling edges to external nodes during meta-node creation
#[derive(Debug, Clone, PartialEq)]
pub enum ExternalEdgeStrategy {
    /// Create separate meta-edges for each original edge (preserves all attributes)
    Copy,
    /// Create single meta-edge with aggregated attributes (default)
    Aggregate,
    /// Create single meta-edge with only count information
    Count,
    /// No meta-edges to external nodes created
    None,
}

impl Default for ExternalEdgeStrategy {
    fn default() -> Self {
        Self::Aggregate
    }
}

/// Strategy for handling edges between meta-nodes
#[derive(Debug, Clone, PartialEq)]
pub enum MetaEdgeStrategy {
    /// Automatically create meta-to-meta edges based on subgraph connections (default)
    Auto,
    /// Only create meta-to-meta edges when explicitly requested
    Explicit,
    /// No meta-to-meta edges created automatically
    None,
}

impl Default for MetaEdgeStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Edge attribute aggregation function
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeAggregationFunction {
    Sum,
    Mean,
    Max,
    Min,
    Count,
    Concat,
    ConcatUnique,
    First,
    Last,
}

impl Default for EdgeAggregationFunction {
    fn default() -> Self {
        Self::Sum
    }
}

impl EdgeAggregationFunction {
    /// Parse aggregation function from string
    pub fn from_string(s: &str) -> GraphResult<Self> {
        match s.to_lowercase().as_str() {
            "sum" => Ok(Self::Sum),
            "mean" | "avg" => Ok(Self::Mean),
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            "count" => Ok(Self::Count),
            "concat" => Ok(Self::Concat),
            "concat_unique" => Ok(Self::ConcatUnique),
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            _ => Err(GraphError::InvalidInput(format!("Unknown edge aggregation function: {}", s))),
        }
    }

    /// Apply aggregation function to a collection of attribute values
    pub fn aggregate(&self, values: &[AttrValue]) -> GraphResult<AttrValue> {
        if values.is_empty() {
            return Err(GraphError::InvalidInput("Cannot aggregate empty values".to_string()));
        }

        match self {
            Self::Sum => {
                match values.first().unwrap() {
                    AttrValue::Int(_) => {
                        let sum: i64 = values.iter()
                            .filter_map(|v| if let AttrValue::Int(i) = v { Some(*i) } else { None })
                            .sum();
                        Ok(AttrValue::Int(sum))
                    }
                    AttrValue::Float(_) => {
                        let sum: f32 = values.iter()
                            .filter_map(|v| if let AttrValue::Float(f) = v { Some(*f) } else { None })
                            .sum();
                        Ok(AttrValue::Float(sum))
                    }
                    _ => Err(GraphError::InvalidInput("Sum aggregation only supported for numeric types".to_string())),
                }
            }
            Self::Mean => {
                match values.first().unwrap() {
                    AttrValue::Int(_) => {
                        let nums: Vec<i64> = values.iter()
                            .filter_map(|v| if let AttrValue::Int(i) = v { Some(*i) } else { None })
                            .collect();
                        let mean = nums.iter().sum::<i64>() as f32 / nums.len() as f32;
                        Ok(AttrValue::Float(mean))
                    }
                    AttrValue::Float(_) => {
                        let nums: Vec<f32> = values.iter()
                            .filter_map(|v| if let AttrValue::Float(f) = v { Some(*f) } else { None })
                            .collect();
                        let mean = nums.iter().sum::<f32>() / nums.len() as f32;
                        Ok(AttrValue::Float(mean))
                    }
                    _ => Err(GraphError::InvalidInput("Mean aggregation only supported for numeric types".to_string())),
                }
            }
            Self::Max => {
                match values.first().unwrap() {
                    AttrValue::Int(_) => {
                        let max = values.iter()
                            .filter_map(|v| if let AttrValue::Int(i) = v { Some(*i) } else { None })
                            .max()
                            .unwrap();
                        Ok(AttrValue::Int(max))
                    }
                    AttrValue::Float(_) => {
                        let max = values.iter()
                            .filter_map(|v| if let AttrValue::Float(f) = v { Some(*f) } else { None })
                            .fold(f32::NEG_INFINITY, |a, b| a.max(b));
                        Ok(AttrValue::Float(max))
                    }
                    _ => Err(GraphError::InvalidInput("Max aggregation only supported for numeric types".to_string())),
                }
            }
            Self::Min => {
                match values.first().unwrap() {
                    AttrValue::Int(_) => {
                        let min = values.iter()
                            .filter_map(|v| if let AttrValue::Int(i) = v { Some(*i) } else { None })
                            .min()
                            .unwrap();
                        Ok(AttrValue::Int(min))
                    }
                    AttrValue::Float(_) => {
                        let min = values.iter()
                            .filter_map(|v| if let AttrValue::Float(f) = v { Some(*f) } else { None })
                            .fold(f32::INFINITY, |a, b| a.min(b));
                        Ok(AttrValue::Float(min))
                    }
                    _ => Err(GraphError::InvalidInput("Min aggregation only supported for numeric types".to_string())),
                }
            }
            Self::Count => Ok(AttrValue::Int(values.len() as i64)),
            Self::Concat => {
                let concatenated = values.iter()
                    .filter_map(|v| if let AttrValue::Text(s) = v { Some(s.as_str()) } else { None })
                    .collect::<Vec<&str>>()
                    .join(",");
                Ok(AttrValue::Text(concatenated))
            }
            Self::ConcatUnique => {
                let mut unique_vals = std::collections::HashSet::new();
                for val in values {
                    if let AttrValue::Text(s) = val {
                        unique_vals.insert(s.as_str());
                    }
                }
                let concatenated = unique_vals.into_iter().collect::<Vec<&str>>().join(",");
                Ok(AttrValue::Text(concatenated))
            }
            Self::First => Ok(values.first().unwrap().clone()),
            Self::Last => Ok(values.last().unwrap().clone()),
        }
    }
}

/// Configuration for edge aggregation during meta-node creation
#[derive(Debug, Clone)]
pub struct EdgeAggregationConfig {
    /// Strategy for handling edges to external nodes
    pub edge_to_external: ExternalEdgeStrategy,
    /// Strategy for handling edges between meta-nodes
    pub edge_to_meta: MetaEdgeStrategy,
    /// Per-attribute aggregation functions (attribute_name -> function)
    pub edge_aggregation: HashMap<AttrName, EdgeAggregationFunction>,
    /// Default aggregation function for unlisted attributes
    pub default_aggregation: EdgeAggregationFunction,
    /// Minimum number of parallel edges required to create a meta-edge
    pub min_edge_count: u32,
    /// Whether to include edge_count attribute on meta-edges
    pub include_edge_count: bool,
    /// Whether to mark meta-edges with entity_type="meta"
    pub mark_entity_type: bool,
}

impl Default for EdgeAggregationConfig {
    fn default() -> Self {
        Self {
            edge_to_external: ExternalEdgeStrategy::default(),
            edge_to_meta: MetaEdgeStrategy::default(),
            edge_aggregation: HashMap::new(),
            default_aggregation: EdgeAggregationFunction::default(),
            min_edge_count: 1,
            include_edge_count: true,
            mark_entity_type: true,
        }
    }
}

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
        // Use the proper space-mapped attribute access instead of direct pool access
        match graph.get_node_attr(node_id, name) {
            Ok(value) => Ok(value),
            Err(e) => Err(e.into()),
        }
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




    /// Collapse this subgraph into a meta-node with enhanced missing attribute handling
    ///
    /// # Arguments
    /// * `agg_functions` - Attribute aggregation functions (e.g., "mean", "sum", "count")
    /// * `defaults` - Default values for missing attributes (advanced usage)
    ///
    /// # Returns
    /// NodeId of the created meta-node in GraphPool
    ///
    /// # Behavior
    /// - Errors by default when aggregating non-existent attributes (strict validation)
    /// - Uses provided defaults for missing attributes when specified
    /// - Count aggregation always works regardless of attribute existence
    /// Collapse subgraph with both defaults for missing attributes AND edge configuration
    fn collapse_to_node_with_defaults_and_edge_config(
        &self,
        agg_functions: HashMap<AttrName, String>,
        defaults: HashMap<AttrName, AttrValue>,
        edge_config: &EdgeAggregationConfig,
    ) -> GraphResult<NodeId> {
        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();

        // Store subgraph reference in GraphPool first
        let subgraph_id = graph.pool_mut().store_subgraph(
            self.node_set().clone(),
            self.edge_set().clone(),
            self.entity_type().to_string(),
        )?;

        // WORF SAFETY: Create meta-node atomically with all required attributes
        let meta_node_id = graph.add_node();
        graph.set_node_attr_internal(meta_node_id, "entity_type".to_string(), AttrValue::Text("meta".to_string()))?;
        graph.set_node_attr_internal(meta_node_id, "contains_subgraph".to_string(), AttrValue::SubgraphRef(subgraph_id))?;

        // Mark original nodes as absorbed first - use bulk operation
        drop(graph); // Release borrow for bulk operation
        {
            let entity_type_attrs = self.node_set().iter()
                .map(|&node_id| (node_id, AttrValue::Text("base".to_string())))
                .collect();
            let mut bulk_attrs = std::collections::HashMap::new();
            bulk_attrs.insert("entity_type".into(), entity_type_attrs);
            self.set_node_attrs(bulk_attrs)?;
        }
        
        // Reacquire mutable borrow
        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();

        // Release the mutable borrow temporarily
        drop(graph);
        
        // Calculate aggregated attributes with defaults for missing values (requires separate borrows)
        let mut aggregated_attrs = Vec::new();
        for (attr_name, agg_function) in agg_functions {
            let aggregated_value = self.aggregate_attribute_with_defaults(&attr_name, &agg_function, &defaults)?;
            aggregated_attrs.push((attr_name, aggregated_value));
        }
        
        // Set all aggregated attributes in bulk
        if !aggregated_attrs.is_empty() {
            let mut bulk_attrs = std::collections::HashMap::new();
            for (attr_name, aggregated_value) in aggregated_attrs {
                let node_value_pairs = vec![(meta_node_id, aggregated_value)];
                bulk_attrs.insert(attr_name.into(), node_value_pairs);
            }
            self.set_node_attrs(bulk_attrs)?;
        }

        
        // Create meta-edges according to edge configuration
        self.create_meta_edges_with_config(meta_node_id, edge_config)?;

        Ok(meta_node_id)
    }

    fn collapse_to_node_with_defaults(
        &self, 
        agg_functions: HashMap<AttrName, String>,
        defaults: HashMap<AttrName, AttrValue>
    ) -> GraphResult<NodeId> {
        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();

        // Store subgraph reference in GraphPool first
        let subgraph_id = graph.pool_mut().store_subgraph(
            self.node_set().clone(),
            self.edge_set().clone(),
            self.entity_type().to_string(),
        )?;

        // WORF SAFETY: Create meta-node atomically with all required attributes
        let meta_node_id = graph.create_meta_node(subgraph_id)?;

        // Release the mutable borrow before the aggregation loop
        drop(graph);

        // Calculate all aggregated attributes first, then set them in bulk
        let mut aggregated_attrs = Vec::new();
        for (attr_name, agg_func_str) in agg_functions {
            // Parse aggregation function
            let agg_func =
                crate::subgraphs::hierarchical::AggregationFunction::from_string(&agg_func_str)?;

            // Special handling for Count aggregation - count all nodes in subgraph
            let aggregated_value = if matches!(agg_func, crate::subgraphs::hierarchical::AggregationFunction::Count) {
                AttrValue::Int(self.node_set().len() as i64)
            } else {
                // Collect all values for this attribute from nodes in the subgraph
                let mut values = Vec::new();
                {
                    let binding = self.graph_ref();
                    let graph = binding.borrow();

                    for &node_id in self.node_set() {
                        if let Some(value) = graph.get_node_attr(node_id, &attr_name)? {
                            values.push(value);
                        }
                    }
                } // graph borrow automatically released here

                // Enhanced missing attribute handling
                if values.is_empty() {
                    // Check if we have a default value for this attribute
                    if let Some(default_value) = defaults.get(&attr_name) {
                        // Use the provided default value
                        default_value.clone()
                    } else {
                        // Strict validation: error on missing attributes
                        return Err(GraphError::InvalidInput(format!(
                            "Attribute '{}' not found on any nodes in subgraph and no default provided. \
                             Available attributes can be checked or provide a default value in the defaults parameter.",
                            attr_name
                        )));
                    }
                } else {
                    // Apply aggregation with collected values
                    agg_func.aggregate(&values)?
                }
            };

            aggregated_attrs.push((attr_name, aggregated_value));
        }

        // Set all aggregated attributes on the meta-node in bulk
        if !aggregated_attrs.is_empty() {
            let mut bulk_attrs = std::collections::HashMap::new();
            for (attr_name, aggregated_value) in aggregated_attrs {
                let node_value_pairs = vec![(meta_node_id, aggregated_value)];
                bulk_attrs.insert(attr_name.into(), node_value_pairs);
            }
            self.set_node_attrs(bulk_attrs)?;
        }

        Ok(meta_node_id)
    }

    /// Create meta-edges with configurable aggregation strategy
    /// 
    /// This enhanced method provides full control over edge aggregation behavior:
    /// - External edge handling: copy, aggregate, count, or none
    /// - Meta-to-meta edge strategy: auto, explicit, or none
    /// - Per-attribute aggregation functions with defaults
    /// Aggregate attribute with defaults for missing values
    fn aggregate_attribute_with_defaults(
        &self,
        attr_name: &str,
        agg_function: &str,
        defaults: &HashMap<AttrName, AttrValue>
    ) -> GraphResult<AttrValue> {
        use crate::subgraphs::hierarchical::AggregationFunction;
        
        let attr_name: AttrName = attr_name.into();
        let agg_func = AggregationFunction::from_string(agg_function)?;
        
        // Special handling for Count aggregation - count all nodes in subgraph
        if matches!(agg_func, AggregationFunction::Count) {
            return Ok(AttrValue::Int(self.node_set().len() as i64));
        }
        
        // Handle missing attributes
        if let Some(default_value) = defaults.get(&attr_name) {
            return Ok(default_value.clone());
        }
        
        // Collect all values for this attribute from nodes in the subgraph
        let mut values = Vec::new();
        let binding = self.graph_ref();
        let graph = binding.borrow();
        
        for &node_id in self.node_set() {
            if let Some(value) = graph.get_node_attr(node_id, &attr_name)? {
                values.push(value);
            }
        }
        
        // If no values found, provide sensible defaults based on aggregation function
        if values.is_empty() {
            let default = match agg_function {
                "sum" => AttrValue::Int(0),
                "mean" | "min" | "max" => AttrValue::Float(0.0),
                "first" | "concat" => AttrValue::Text("".to_string()),
                _ => AttrValue::Int(0),
            };
            Ok(default)
        } else {
            // Release borrow before aggregation
            drop(graph);
            drop(binding);
            
            // Apply aggregation with collected values
            agg_func.aggregate(&values)
        }
    }

    /// - Edge filtering based on count and attributes
    ///
    /// # Arguments
    /// * `meta_node_id` - The meta-node to create edges from
    /// * `config` - Edge aggregation configuration
    fn create_meta_edges_with_config(&self, meta_node_id: NodeId, config: &EdgeAggregationConfig) -> GraphResult<()> {
        // Skip edge creation if configured to do nothing
        if config.edge_to_external == ExternalEdgeStrategy::None {
            return Ok(());
        }

        let binding = self.graph_ref();
        let mut graph = binding.borrow_mut();
        
        // Track edges to create: target_node -> (edge_count, aggregated_attributes)
        let mut meta_edges: std::collections::HashMap<NodeId, (u32, std::collections::HashMap<AttrName, Vec<AttrValue>>)> = std::collections::HashMap::new();
        
        // Iterate through all nodes in the collapsed subgraph
        for &source_node in self.node_set() {
            // Get all incident edges for this node
            if let Ok(incident_edges) = graph.incident_edges(source_node) {
                for edge_id in incident_edges {
                    if let Ok((edge_source, edge_target)) = graph.edge_endpoints(edge_id) {
                        // Determine the external target node
                        let external_target = if edge_source == source_node {
                            edge_target
                        } else if edge_target == source_node {
                            edge_source  // For undirected graphs, edges can go both ways
                        } else {
                            continue; // This shouldn't happen, but skip if it does
                        };
                        
                        // Skip edges within the subgraph (these are now internal to the meta-node)
                        if self.node_set().contains(&external_target) {
                            continue;
                        }
                        
                        // This is an edge to an external node - create/aggregate meta-edge
                        let entry = meta_edges.entry(external_target).or_insert((0, std::collections::HashMap::new()));
                        entry.0 += 1; // Count parallel edges
                        
                        // Collect edge attributes for aggregation
                        if let Ok(edge_attrs) = graph.get_edge_attrs(edge_id) {
                            for (attr_name, attr_value) in edge_attrs {
                                entry.1.entry(attr_name).or_insert(Vec::new()).push(attr_value);
                            }
                        }
                    }
                }
            }
        }
        
        // Create the aggregated meta-edges based on strategy
        for (target_node, (edge_count, attr_collections)) in meta_edges {
            // Skip if below minimum edge count threshold
            if edge_count < config.min_edge_count {
                continue;
            }

                // DEFENSIVE CHECK: Validate that target_node still exists before creating edges
                if !graph.contains_node(target_node) {
                    eprintln!("Warning: Skipping meta-edge to non-existent node {}", target_node);
                    continue;
                }            match config.edge_to_external {
                ExternalEdgeStrategy::Copy => {
                    // Create separate meta-edges for each original edge (TODO: implement properly)
                    // For now, fall back to aggregate behavior
                    self.create_aggregated_meta_edge(&mut *graph, meta_node_id, target_node, edge_count, attr_collections, config)?;
                }
                ExternalEdgeStrategy::Aggregate => {
                    self.create_aggregated_meta_edge(&mut *graph, meta_node_id, target_node, edge_count, attr_collections, config)?;
                }
                ExternalEdgeStrategy::Count => {
                    self.create_count_meta_edge(&mut *graph, meta_node_id, target_node, edge_count, config)?;
                }
                ExternalEdgeStrategy::None => {
                    // Already handled above, skip
                }
            }
        }
        
        Ok(())
    }

    /// Helper method to create an aggregated meta-edge
    fn create_aggregated_meta_edge(
        &self,
        graph: &mut crate::api::graph::Graph,
        meta_node_id: NodeId,
        target_node: NodeId,
        edge_count: u32,
        attr_collections: std::collections::HashMap<AttrName, Vec<AttrValue>>,
        config: &EdgeAggregationConfig,
    ) -> GraphResult<()> {
        // DEFENSIVE CHECK: Validate both nodes exist before creating edge
        if !graph.contains_node(meta_node_id) {
            return Err(GraphError::InvalidInput(
                format!("Meta-node {} does not exist", meta_node_id)
            ));
        }
        if !graph.contains_node(target_node) {
            return Err(GraphError::InvalidInput(
                format!("Target node {} does not exist", target_node)
            ));
        }

        // Create meta-edge from meta-node to target
        let meta_edge_id = graph.add_edge(meta_node_id, target_node)?;
        
        // Set edge count attribute if enabled
        if config.include_edge_count {
            graph.set_edge_attr(meta_edge_id, "edge_count".to_string(), AttrValue::Int(edge_count as i64))?;
        }
        
        // Aggregate edge attributes using configured functions
        for (attr_name, values) in attr_collections {
            if values.is_empty() {
                continue;
            }

            let aggregated_value = if values.len() == 1 {
                // Single value - no aggregation needed
                values.into_iter().next().unwrap()
            } else {
                // Multiple values - use configured aggregation function
                let agg_function = config.edge_aggregation.get(&attr_name)
                    .unwrap_or(&config.default_aggregation);
                
                agg_function.aggregate(&values)?
            };
            
            graph.set_edge_attr(meta_edge_id, attr_name, aggregated_value)?;
        }
        
        // Mark this as a meta-edge if enabled
        if config.mark_entity_type {
            graph.set_edge_attr(meta_edge_id, "entity_type".to_string(), AttrValue::Text("meta".to_string()))?;
        }
        
        // Add explicit source and target attributes to meta-edges
        graph.set_edge_attr(meta_edge_id, "source".to_string(), AttrValue::Int(meta_node_id as i64))?;
        graph.set_edge_attr(meta_edge_id, "target".to_string(), AttrValue::Int(target_node as i64))?;

        Ok(())
    }

    /// Helper method to create a count-only meta-edge
    fn create_count_meta_edge(
        &self,
        graph: &mut crate::api::graph::Graph,
        meta_node_id: NodeId,
        target_node: NodeId,
        edge_count: u32,
        config: &EdgeAggregationConfig,
    ) -> GraphResult<()> {
        // DEFENSIVE CHECK: Validate both nodes exist before creating edge
        if !graph.contains_node(meta_node_id) {
            return Err(GraphError::InvalidInput(
                format!("Meta-node {} does not exist", meta_node_id)
            ));
        }
        if !graph.contains_node(target_node) {
            return Err(GraphError::InvalidInput(
                format!("Target node {} does not exist", target_node)
            ));
        }

        // Create meta-edge from meta-node to target
        let meta_edge_id = graph.add_edge(meta_node_id, target_node)?;
        
        // Only set edge count - no other attributes preserved
        graph.set_edge_attr(meta_edge_id, "edge_count".to_string(), AttrValue::Int(edge_count as i64))?;
        
        // Mark this as a meta-edge if enabled
        if config.mark_entity_type {
            graph.set_edge_attr(meta_edge_id, "entity_type".to_string(), AttrValue::Text("meta".to_string()))?;
        }
        
        // Add explicit source and target attributes to meta-edges
        graph.set_edge_attr(meta_edge_id, "source".to_string(), AttrValue::Int(meta_node_id as i64))?;
        graph.set_edge_attr(meta_edge_id, "target".to_string(), AttrValue::Int(target_node as i64))?;

        Ok(())
    }

    /// Create meta-edges for a collapsed subgraph (legacy method)
    ///
    /// # Arguments
    /// * `meta_node_id` - The meta-node representing the collapsed subgraph
    ///
    /// # Behavior
    /// Creates two types of meta-edges:
    /// 1. Child-to-External: Edges from collapsed nodes to external nodes
    /// 2. Meta-to-Meta: Edges from collapsed nodes to other meta-nodes
    ///
    /// # Edge Aggregation
    /// - Multiple parallel edges are aggregated using sum by default
    /// - Edge attributes are aggregated (weight, etc.)
    fn create_meta_edges(&self, meta_node_id: NodeId) -> GraphResult<()> {
        // Use default configuration for backward compatibility
        self.create_meta_edges_with_config(meta_node_id, &EdgeAggregationConfig::default())
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
    fn nodes_table(&self) -> GraphResult<crate::storage::table::GraphTable> {
        let binding = self.graph_ref();
        let graph = binding.borrow();

        // Create index-aligned table where table row index = node_id
        // This ensures g.nodes[g.table()['column'] == value] works correctly
        let node_set = self.node_set();
        let max_node_id = node_set.iter().max().copied().unwrap_or(0);

        // Create a sparse representation where table[node_id] = node_data
        // Missing nodes will have null values
        let table_size = (max_node_id + 1) as usize;

        // Collect all unique attribute names across all nodes
        let mut all_attrs = std::collections::HashSet::new();
        for &node_id in node_set.iter() {
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
            let mut attr_values = Vec::with_capacity(table_size);

            if column_name == "node_id" {
                // Node ID column - sparse array where table[i] = i if node exists, null otherwise
                for node_id in 0..table_size {
                    if node_set.contains(&node_id) {
                        attr_values.push(crate::types::AttrValue::Int(node_id as i64));
                    } else {
                        attr_values.push(crate::types::AttrValue::Null);
                    }
                }
            } else {
                // Attribute column - sparse array where table[node_id] = attribute_value
                for node_id in 0..table_size {
                    if node_set.contains(&node_id) {
                        if let Ok(Some(attr_value)) = graph.get_node_attr(node_id, column_name) {
                            attr_values.push(attr_value);
                        } else {
                            // Use null placeholder for missing attributes
                            attr_values.push(crate::types::AttrValue::Null);
                        }
                    } else {
                        // Node doesn't exist in subgraph - null value
                        attr_values.push(crate::types::AttrValue::Null);
                    }
                }
            }

            let graph_array = crate::storage::array::GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }

        // Use existing GraphTable::from_arrays_standalone (no graph reference needed)
        crate::storage::table::GraphTable::from_arrays_standalone(columns, Some(column_names))
    }

    /// Create table from subgraph edges with all their attributes  
    ///
    /// # Returns
    /// GraphTable with edge IDs and all edge attributes as columns
    ///
    /// # Performance
    /// Uses optimized columnar access to build table efficiently
    fn edges_table(&self) -> GraphResult<crate::storage::table::GraphTable> {
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

            let graph_array = crate::storage::array::GraphArray::from_vec(attr_values);
            columns.push(graph_array);
        }

        // Use existing GraphTable::from_arrays_standalone (no graph reference needed)
        crate::storage::table::GraphTable::from_arrays_standalone(columns, Some(column_names))
    }
    
    /// Modern MetaGraph Composer API - Clean interface for meta-node creation
    /// 
    /// This is the new, intuitive way to create meta-nodes with flexible configuration.
    /// It replaces the complex EdgeAggregationConfig system with a simple, discoverable API.
    /// 
    /// # Arguments
    /// * `node_aggs` - Node aggregation specifications (flexible input)
    /// * `edge_aggs` - Edge aggregation specifications  
    /// * `edge_strategy` - How to handle edges ("aggregate", "keep_external", "drop_all", "contract_all")
    /// * `preset` - Optional preset configuration
    /// * `include_edge_count` - Add edge_count attribute to meta-edges
    /// * `mark_entity_type` - Mark meta-nodes/edges with entity_type
    /// * `entity_type` - Entity type to use for marking
    /// 
    /// # Returns
    /// A `MetaNodePlan` that can be previewed, modified, and executed with `.add_to_graph()`
    /// 
    /// # Examples
    /// ```rust
    /// // Basic usage
    /// let plan = subgraph.collapse(
    ///     vec![("avg_salary", "mean", Some("salary"))],
    ///     vec![("weight", "mean")],
    ///     EdgeStrategy::Aggregate,
    ///     None,
    ///     true,
    ///     true,
    ///     "meta".to_string()
    /// )?;
    /// let meta_node = plan.add_to_graph()?;
    /// ```
    fn collapse(
        &self,
        node_aggs: Vec<(String, String, Option<String>)>, // (target, function, source)
        edge_aggs: Vec<(String, String)>, // (attr_name, function)  
        edge_strategy: EdgeStrategy,
        preset: Option<String>,
        include_edge_count: bool,
        mark_entity_type: bool,
        entity_type: String,
    ) -> GraphResult<MetaNodePlan> {
        // Create base plan
        let mut plan = MetaNodePlan::new(
            self.node_set().clone(),
            self.edge_set().clone(),
        );
        
        // Apply preset if provided
        if let Some(preset_name) = preset {
            plan = plan.with_preset(&preset_name)?;
        }
        
        // Add node aggregations
        for (target, function, source) in node_aggs {
            plan = plan.with_node_agg(target, function, source);
        }
        
        // Add edge aggregations
        for (attr_name, function) in edge_aggs {
            plan = plan.with_edge_agg(attr_name, function);
        }
        
        // Configure plan
        plan.edge_strategy = edge_strategy;
        plan.include_edge_count = include_edge_count;
        plan.mark_entity_type = mark_entity_type;
        plan.entity_type = entity_type;
        
        Ok(plan)
    }
}

// Note: No default implementations provided to avoid conflicts.
// Each concrete subgraph type implements SubgraphOperations directly.
