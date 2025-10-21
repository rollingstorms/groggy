//! RealtimeVizAccessor trait and DataSource implementation
//!
//! This module provides the bridge between DataSource and Realtime Engine.
//! This is the core of Phase 1 - a clean, testable adapter.

use super::engine_messages::{ControlMsg, Edge, EngineSnapshot, GraphMeta, Node, NodePosition};
use crate::errors::GraphResult;
use crate::types::{AttrValue, EdgeId, NodeId};
use crate::viz::realtime::{LayoutKind, VizConfig};
use crate::viz::streaming::data_source::{DataSource, LayoutAlgorithm};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Statistics for a column of numeric values
#[derive(Debug, Clone)]
struct ColumnStats {
    min: f64,
    max: f64,
}

/// Trait for accessing data sources in realtime visualization
pub trait RealtimeVizAccessor: Send + Sync {
    /// Get initial snapshot of all data
    fn initial_snapshot(&self) -> GraphResult<EngineSnapshot>;

    /// Apply control message (optional server-side reactions)
    fn apply_control(&self, control: ControlMsg) -> GraphResult<()>;

    /// Get current node count
    fn node_count(&self) -> usize;

    /// Get current edge count
    fn edge_count(&self) -> usize;

    /// Check if positions are available
    fn has_positions(&self) -> bool;

    /// Get table data window for nodes or edges
    fn get_table_data(
        &self,
        data_type: super::engine_messages::TableDataType,
        offset: usize,
        window_size: usize,
        sort_columns: Vec<super::engine_messages::SortColumn>,
    ) -> GraphResult<crate::viz::realtime::server::ws_bridge::TableDataWindow>;
}

/// Implementation of RealtimeVizAccessor for DataSource
pub struct DataSourceRealtimeAccessor {
    /// DataSource being accessed
    data_source: Arc<dyn DataSource>,
    /// Layout algorithm
    layout_algorithm: RwLock<LayoutAlgorithm>,
    /// Embedding dimensions
    embedding_dimensions: RwLock<usize>,
    /// Verbosity level for debug output (0=quiet, 1=info, 2=verbose, 3=debug)
    verbose: u8,
    /// Visualization styling configuration
    viz_config: Option<VizConfig>,
}

impl DataSourceRealtimeAccessor {
    /// Create new accessor from DataSource
    pub fn new(data_source: Arc<dyn DataSource>) -> Self {
        Self::with_verbosity(data_source, 0)
    }

    /// Convert complex AttrValue types to JSON-safe representations for realtime viz
    /// This prevents "[object Object]" display issues in the browser
    fn sanitize_attributes_for_realtime(
        &self,
        attributes: HashMap<String, AttrValue>,
    ) -> HashMap<String, AttrValue> {
        let mut sanitized = HashMap::new();

        for (key, value) in attributes {
            let safe_value = match value {
                // Convert all types to Text for consistent JSON serialization
                // This prevents the {"Text": "value"} wrapper issue
                AttrValue::Float(f) => AttrValue::Text(f.to_string()),
                AttrValue::Int(i) => AttrValue::Text(i.to_string()),
                AttrValue::Text(s) => AttrValue::Text(s),
                AttrValue::Bool(b) => AttrValue::Text(b.to_string()),
                AttrValue::SmallInt(i) => AttrValue::Text(i.to_string()),
                AttrValue::CompactText(s) => AttrValue::Text(s.as_str().to_string()),
                AttrValue::Null => AttrValue::Text("null".to_string()),

                // Complex vector types - convert to string representations for display
                AttrValue::FloatVec(vec) => {
                    if vec.len() <= 10 {
                        AttrValue::Text(format!(
                            "[{}]",
                            vec.iter()
                                .map(|f| format!("{:.2}", f))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    } else {
                        AttrValue::Text(format!(
                            "[{} values: {:.2}..{:.2}]",
                            vec.len(),
                            vec.first().unwrap_or(&0.0),
                            vec.last().unwrap_or(&0.0)
                        ))
                    }
                }
                AttrValue::IntVec(vec) => {
                    if vec.len() <= 10 {
                        AttrValue::Text(format!(
                            "[{}]",
                            vec.iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    } else {
                        AttrValue::Text(format!(
                            "[{} values: {}..{}]",
                            vec.len(),
                            vec.first().unwrap_or(&0),
                            vec.last().unwrap_or(&0)
                        ))
                    }
                }
                AttrValue::TextVec(vec) => {
                    if vec.len() <= 5 {
                        AttrValue::Text(format!("[{}]", vec.join(", ")))
                    } else {
                        AttrValue::Text(format!(
                            "[{} items: {}, ...]",
                            vec.len(),
                            vec.first().map(|s| s.as_str()).unwrap_or("")
                        ))
                    }
                }
                AttrValue::BoolVec(vec) => {
                    let true_count = vec.iter().filter(|&&b| b).count();
                    AttrValue::Text(format!(
                        "[{} bools: {} true, {} false]",
                        vec.len(),
                        true_count,
                        vec.len() - true_count
                    ))
                }

                // Complex reference types - convert to descriptive strings
                AttrValue::SubgraphRef(id) => AttrValue::Text(format!("Subgraph({})", id)),
                AttrValue::NodeArray(ref arr) => {
                    AttrValue::Text(format!("NodeArray({} nodes)", arr.len()))
                }
                AttrValue::EdgeArray(ref arr) => {
                    AttrValue::Text(format!("EdgeArray({} edges)", arr.len()))
                }

                // Binary/compressed data - show metadata only
                AttrValue::Bytes(ref bytes) => {
                    AttrValue::Text(format!("Bytes({} bytes)", bytes.len()))
                }
                AttrValue::CompressedText(ref data) => {
                    let ratio = data.compression_ratio();
                    AttrValue::Text(format!(
                        "CompressedText({} bytes, {:.1}x compression)",
                        data.data.len(),
                        1.0 / ratio
                    ))
                }
                AttrValue::CompressedFloatVec(ref data) => {
                    let ratio = data.compression_ratio();
                    AttrValue::Text(format!(
                        "CompressedFloatVec({} bytes, {:.1}x compression)",
                        data.data.len(),
                        1.0 / ratio
                    ))
                }

                // JSON - convert to string representation
                AttrValue::Json(ref json) => match serde_json::to_string(json) {
                    Ok(json_str) => {
                        if json_str.len() <= 100 {
                            AttrValue::Text(json_str)
                        } else {
                            AttrValue::Text(format!("JSON({} chars)", json_str.len()))
                        }
                    }
                    Err(_) => AttrValue::Text("JSON(parse error)".to_string()),
                },
            };

            sanitized.insert(key, safe_value);
        }

        if self.verbose >= 3 {
            // Debug: Sanitized complex attributes for realtime display
        }

        sanitized
    }

    /// Create new accessor with verbosity level
    pub fn with_verbosity(data_source: Arc<dyn DataSource>, verbose: u8) -> Self {
        Self {
            data_source,
            layout_algorithm: RwLock::new(LayoutAlgorithm::ForceDirected {
                charge: -30.0,
                distance: 30.0,
                iterations: 100,
            }),
            embedding_dimensions: RwLock::new(2),
            verbose,
            viz_config: None,
        }
    }

    /// Create new accessor with specific layout
    pub fn with_layout(data_source: Arc<dyn DataSource>, layout: LayoutAlgorithm) -> Self {
        Self {
            data_source,
            layout_algorithm: RwLock::new(layout),
            embedding_dimensions: RwLock::new(2),
            verbose: 0,
            viz_config: None,
        }
    }

    /// Create new accessor with specific layout and VizConfig
    pub fn with_layout_and_config(
        data_source: Arc<dyn DataSource>,
        layout: LayoutAlgorithm,
        viz_config: Option<VizConfig>,
    ) -> Self {
        Self {
            data_source,
            layout_algorithm: RwLock::new(layout),
            embedding_dimensions: RwLock::new(2),
            verbose: 0,
            viz_config,
        }
    }

    /// Map categorical string value to color using palette
    fn value_to_categorical_color(&self, value: &str) -> String {
        // Distinct color palette (10 colors)
        const PALETTE: &[&str] = &[
            "#e74c3c", // Red
            "#3498db", // Blue
            "#2ecc71", // Green
            "#f39c12", // Orange
            "#9b59b6", // Purple
            "#1abc9c", // Turquoise
            "#e67e22", // Carrot
            "#34495e", // Dark gray
            "#16a085", // Green sea
            "#d35400", // Pumpkin
        ];

        // Simple hash to map string to color index
        let hash: usize = value.bytes().map(|b| b as usize).sum();
        let index = hash % PALETTE.len();
        PALETTE[index].to_string()
    }

    /// Map numeric value to color using gradient
    fn value_to_gradient_color(&self, value: f64, min: f64, max: f64, gradient: &str) -> String {
        // Normalize value to [0, 1]
        let t = if max > min {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.5
        };

        match gradient {
            "grayscale" | "bw" | "blackwhite" => {
                // Black (0,0,0) to White (255,255,255)
                let intensity = (t * 255.0) as u8;
                format!("#{:02x}{:02x}{:02x}", intensity, intensity, intensity)
            }
            "redcyan" | "red-cyan" => {
                // Red (255,0,0) to Cyan (0,255,255)
                let r = ((1.0 - t) * 255.0) as u8;
                let g = (t * 255.0) as u8;
                let b = (t * 255.0) as u8;
                format!("#{:02x}{:02x}{:02x}", r, g, b)
            }
            _ => {
                // Default grayscale
                let intensity = (t * 255.0) as u8;
                format!("#{:02x}{:02x}{:02x}", intensity, intensity, intensity)
            }
        }
    }

    /// Resolve a VizParameter<String> for a specific node
    fn resolve_string_param(
        &self,
        param: &crate::viz::realtime::VizParameter<String>,
        node_idx: usize,
        attributes: &HashMap<String, AttrValue>,
    ) -> Option<String> {
        use crate::viz::realtime::VizParameter;
        match param {
            VizParameter::Array(arr) => arr.get(node_idx).cloned(),
            VizParameter::Column(col_name) => {
                // Try to extract value from attributes
                attributes.get(col_name).map(|attr| match attr {
                    AttrValue::Text(s) => s.clone(),
                    AttrValue::CompactText(s) => s.as_str().to_string(),
                    _ => format!("{:?}", attr), // Convert other types to string
                })
            }
            VizParameter::Value(val) => Some(val.clone()),
            VizParameter::None => None,
        }
    }

    /// Resolve a VizParameter<f64> for a specific node
    fn resolve_f64_param(
        &self,
        param: &crate::viz::realtime::VizParameter<f64>,
        node_idx: usize,
        attributes: &HashMap<String, AttrValue>,
    ) -> Option<f64> {
        use crate::viz::realtime::VizParameter;
        match param {
            VizParameter::Array(arr) => arr.get(node_idx).cloned(),
            VizParameter::Column(col_name) => {
                // Try to extract numeric value from attributes
                let result = attributes.get(col_name).and_then(|attr| match attr {
                    AttrValue::Float(f) => Some(*f as f64),
                    AttrValue::Int(i) => Some(*i as f64),
                    AttrValue::SmallInt(i) => Some(*i as f64),
                    AttrValue::Text(s) => s.parse::<f64>().ok(),
                    AttrValue::CompactText(s) => {
                        // Try to parse as f64
                        s.as_str().parse::<f64>().ok()
                    }
                    _ => None,
                });
                result
            }
            VizParameter::Value(val) => Some(*val),
            VizParameter::None => None,
        }
    }

    /// Scale a value to a range [min, max] based on column statistics
    fn scale_value(&self, value: f64, col_name: &str, range: Option<(f64, f64)>) -> f64 {
        // Default range for auto-scaling (5-20px for node sizes)
        let (min_val, max_val) = range.unwrap_or((5.0, 20.0));

        // Get min/max from all values in this column for normalization
        if let Some(stats) = self.get_column_stats(col_name) {
            let data_min = stats.min;
            let data_max = stats.max;

            if data_max > data_min {
                // Normalize to [0, 1] then scale to [min_val, max_val]
                let normalized = (value - data_min) / (data_max - data_min);
                min_val + normalized * (max_val - min_val)
            } else {
                // All values are the same, return midpoint
                (min_val + max_val) / 2.0
            }
        } else {
            // No stats available, return as-is
            value
        }
    }

    /// Get min/max statistics for a column
    fn get_column_stats(&self, col_name: &str) -> Option<ColumnStats> {
        // Collect all values from the column across all nodes
        let graph_nodes = self.data_source.get_graph_nodes();
        let mut values: Vec<f64> = Vec::new();

        for node in &graph_nodes {
            if let Some(attr) = node.attributes.get(col_name) {
                let val = match attr {
                    AttrValue::Float(f) => Some(*f as f64),
                    AttrValue::Int(i) => Some(*i as f64),
                    AttrValue::SmallInt(i) => Some(*i as f64),
                    AttrValue::Text(s) => s.parse::<f64>().ok(),
                    AttrValue::CompactText(s) => s.as_str().parse::<f64>().ok(),
                    _ => None,
                };
                if let Some(v) = val {
                    values.push(v);
                }
            }
        }

        if values.is_empty() {
            return None;
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Some(ColumnStats { min, max })
    }

    /// Auto-assign curvature to multi-edges between same node pairs and self-loops
    fn apply_auto_curvature(&self, edges: &mut [Edge]) {
        use std::collections::HashMap;

        // Group edges by (source, target) pair (treating undirected as same)
        let mut edge_groups: HashMap<(NodeId, NodeId), Vec<usize>> = HashMap::new();
        let mut self_loop_count: HashMap<NodeId, usize> = HashMap::new();
        let mut self_loop_curvatures: Vec<(usize, f64)> = Vec::new();

        for (idx, edge) in edges.iter().enumerate() {
            // Check for self-loops (source == target)
            if edge.source == edge.target {
                let count = self_loop_count.entry(edge.source).or_insert(0);
                *count += 1;
                // Self-loops get large curvature to render as circular arcs
                // Multiple self-loops on same node get different angles
                let angle_offset = (*count - 1) as f64 * 60.0; // 60 degree spacing
                let curvature = 2.0 + angle_offset / 180.0; // Base curvature 2.0 for self-loops
                self_loop_curvatures.push((idx, curvature));
                continue;
            }

            // Normalize the pair so (a,b) and (b,a) are treated as the same
            let pair = if edge.source <= edge.target {
                (edge.source, edge.target)
            } else {
                (edge.target, edge.source)
            };
            edge_groups.entry(pair).or_insert_with(Vec::new).push(idx);
        }

        // Apply self-loop curvatures
        for (idx, curvature) in self_loop_curvatures {
            edges[idx].curvature = Some(curvature);
        }

        // For each group with multiple edges, assign incrementing curvature
        for indices in edge_groups.values() {
            if indices.len() > 1 {
                // Assign alternating positive/negative curvature
                for (i, &idx) in indices.iter().enumerate() {
                    let curvature = if i % 2 == 0 {
                        (i / 2 + 1) as f64 * 0.5
                    } else {
                        -((i / 2 + 1) as f64 * 0.5)
                    };
                    edges[idx].curvature = Some(curvature);
                }
            }
        }
    }

    /// Convert DataSource nodes to engine nodes
    fn convert_nodes(&self) -> Vec<Node> {
        if self.verbose >= 3 {
            // Converting nodes - supports_graph_view
        }

        if !self.data_source.supports_graph_view() {
            if self.verbose >= 2 {
                // Debug message
            }
            return Vec::new();
        }

        let graph_nodes = self.data_source.get_graph_nodes();
        if self.verbose >= 3 {
            // Debug message
        }

        let engine_nodes: Vec<Node> = graph_nodes
            .into_iter()
            .enumerate()
            .map(|(idx, graph_node)| {
                let node_id = graph_node.id.parse().unwrap_or(0);
                if self.verbose >= 3 {
                    // Debug message
                }

                // Sanitize attributes to prevent [object Object] display issues
                let sanitized_attributes =
                    self.sanitize_attributes_for_realtime(graph_node.attributes.clone());

                // Create base node
                let mut node = Node::new(node_id, sanitized_attributes.clone());

                // Apply styling from VizConfig if present
                if let Some(ref config) = self.viz_config {
                    // Handle node_color with gradient/categorical support
                    use crate::viz::realtime::VizParameter;
                    match &config.node_color {
                        VizParameter::Column(col_name) => {
                            // Check if this should use a gradient or categorical mapping
                            if let Some(ref scale_type) = config.color_scale_type {
                                if scale_type == "linear" || scale_type == "gradient" {
                                    // Gradient: map numeric values to colors
                                    if let Some(value) = self.resolve_f64_param(
                                        &VizParameter::Column(col_name.clone()),
                                        idx,
                                        &sanitized_attributes,
                                    ) {
                                        if let Some(stats) = self.get_column_stats(col_name) {
                                            let gradient_name =
                                                if let Some(ref palette) = config.color_palette {
                                                    palette
                                                        .first()
                                                        .map(|s| s.as_str())
                                                        .unwrap_or("grayscale")
                                                } else {
                                                    "grayscale"
                                                };
                                            node.color = Some(self.value_to_gradient_color(
                                                value,
                                                stats.min,
                                                stats.max,
                                                gradient_name,
                                            ));
                                        }
                                    }
                                } else if scale_type == "categorical" {
                                    // Categorical: hash string values to distinct colors
                                    if let Some(value) = self.resolve_string_param(
                                        &VizParameter::Column(col_name.clone()),
                                        idx,
                                        &sanitized_attributes,
                                    ) {
                                        node.color = Some(self.value_to_categorical_color(&value));
                                    }
                                } else {
                                    // Other scale types: treat as direct string
                                    node.color = self.resolve_string_param(
                                        &config.node_color,
                                        idx,
                                        &sanitized_attributes,
                                    );
                                }
                            } else {
                                // No scale_type: treat as direct string
                                node.color = self.resolve_string_param(
                                    &config.node_color,
                                    idx,
                                    &sanitized_attributes,
                                );
                            }
                        }
                        _ => {
                            // Direct value or array
                            node.color = self.resolve_string_param(
                                &config.node_color,
                                idx,
                                &sanitized_attributes,
                            );
                        }
                    }

                    // Resolve size and apply scaling if range is specified
                    if let Some(raw_size) =
                        self.resolve_f64_param(&config.node_size, idx, &sanitized_attributes)
                    {
                        // Check if we need to scale based on column reference
                        use crate::viz::realtime::VizParameter;
                        if let VizParameter::Column(ref col_name) = config.node_size {
                            node.size =
                                Some(self.scale_value(raw_size, col_name, config.node_size_range));
                        } else {
                            node.size = Some(raw_size);
                        }
                    }

                    node.shape =
                        self.resolve_string_param(&config.node_shape, idx, &sanitized_attributes);
                    node.opacity =
                        self.resolve_f64_param(&config.node_opacity, idx, &sanitized_attributes);
                    node.border_color = self.resolve_string_param(
                        &config.node_border_color,
                        idx,
                        &sanitized_attributes,
                    );
                    node.border_width = self.resolve_f64_param(
                        &config.node_border_width,
                        idx,
                        &sanitized_attributes,
                    );
                    node.label =
                        self.resolve_string_param(&config.node_label, idx, &sanitized_attributes);
                    node.label_size =
                        self.resolve_f64_param(&config.label_size, idx, &sanitized_attributes);
                    node.label_color =
                        self.resolve_string_param(&config.label_color, idx, &sanitized_attributes);
                }

                node
            })
            .collect();

        if self.verbose >= 2 {
            // Debug message
        }

        // Debug: Print first node to verify styling is in the snapshot
        engine_nodes
    }

    /// Convert DataSource edges to engine edges
    fn convert_edges(&self) -> Vec<Edge> {
        if self.verbose >= 3 {
            // Debug message
        }

        if !self.data_source.supports_graph_view() {
            if self.verbose >= 2 {
                // Debug message
            }
            return Vec::new();
        }

        let graph_edges = self.data_source.get_graph_edges();
        if self.verbose >= 3 {
            // Debug message
        }

        let mut engine_edges: Vec<Edge> = graph_edges
            .into_iter()
            .enumerate()
            .map(|(idx, graph_edge)| {
                // Parse the actual edge ID from the graph data
                let edge_id = graph_edge.id.parse().unwrap_or(idx as EdgeId);
                let source_id = graph_edge.source.parse().unwrap_or(0);
                let target_id = graph_edge.target.parse().unwrap_or(0);
                if self.verbose >= 3 {
                    // Debug message
                }

                // Sanitize edge attributes to prevent [object Object] display issues
                let sanitized_attributes =
                    self.sanitize_attributes_for_realtime(graph_edge.attributes.clone());

                // Create base edge with actual edge ID
                let mut edge =
                    Edge::new(edge_id, source_id, target_id, sanitized_attributes.clone());

                // Apply styling from VizConfig if present
                if let Some(ref config) = self.viz_config {
                    // Handle edge_color with gradient/categorical support
                    use crate::viz::realtime::VizParameter;
                    match &config.edge_color {
                        VizParameter::Column(col_name) => {
                            if let Some(ref scale_type) = config.color_scale_type {
                                if scale_type == "linear" || scale_type == "gradient" {
                                    if let Some(value) = self.resolve_f64_param(
                                        &VizParameter::Column(col_name.clone()),
                                        idx,
                                        &sanitized_attributes,
                                    ) {
                                        if let Some(stats) = self.get_column_stats(col_name) {
                                            let gradient_name =
                                                if let Some(ref palette) = config.color_palette {
                                                    palette
                                                        .first()
                                                        .map(|s| s.as_str())
                                                        .unwrap_or("grayscale")
                                                } else {
                                                    "grayscale"
                                                };
                                            edge.color = Some(self.value_to_gradient_color(
                                                value,
                                                stats.min,
                                                stats.max,
                                                gradient_name,
                                            ));
                                        }
                                    }
                                } else if scale_type == "categorical" {
                                    if let Some(value) = self.resolve_string_param(
                                        &VizParameter::Column(col_name.clone()),
                                        idx,
                                        &sanitized_attributes,
                                    ) {
                                        edge.color = Some(self.value_to_categorical_color(&value));
                                    }
                                } else {
                                    edge.color = self.resolve_string_param(
                                        &config.edge_color,
                                        idx,
                                        &sanitized_attributes,
                                    );
                                }
                            } else {
                                edge.color = self.resolve_string_param(
                                    &config.edge_color,
                                    idx,
                                    &sanitized_attributes,
                                );
                            }
                        }
                        _ => {
                            edge.color = self.resolve_string_param(
                                &config.edge_color,
                                idx,
                                &sanitized_attributes,
                            );
                        }
                    }

                    // Resolve width and apply scaling if range is specified
                    if let Some(raw_width) =
                        self.resolve_f64_param(&config.edge_width, idx, &sanitized_attributes)
                    {
                        use crate::viz::realtime::VizParameter;
                        if let VizParameter::Column(ref col_name) = config.edge_width {
                            edge.width = Some(self.scale_value(
                                raw_width,
                                col_name,
                                config.edge_width_range,
                            ));
                        } else {
                            edge.width = Some(raw_width);
                        }
                    }

                    edge.opacity =
                        self.resolve_f64_param(&config.edge_opacity, idx, &sanitized_attributes);
                    edge.style =
                        self.resolve_string_param(&config.edge_style, idx, &sanitized_attributes);

                    // Edge label support
                    edge.label =
                        self.resolve_string_param(&config.edge_label, idx, &sanitized_attributes);
                    edge.label_size =
                        self.resolve_f64_param(&config.edge_label_size, idx, &sanitized_attributes);
                    edge.label_color = self.resolve_string_param(
                        &config.edge_label_color,
                        idx,
                        &sanitized_attributes,
                    );
                }

                edge
            })
            .collect();

        // Auto-assign curvature to multi-edges between the same node pairs
        self.apply_auto_curvature(&mut engine_edges);

        if self.verbose >= 2 {
            // Debug message
        }
        engine_edges
    }

    /// Generate or compute positions for nodes
    fn compute_positions(&self, nodes: &[Node]) -> Vec<NodePosition> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Try to use DataSource layout computation
        let positions = self
            .data_source
            .compute_layout(self.layout_algorithm.read().unwrap().clone());

        if !positions.is_empty() {
            // Convert DataSource positions to engine positions
            positions
                .into_iter()
                .map(|pos| NodePosition {
                    node_id: pos.node_id.parse().unwrap_or(0),
                    coords: vec![pos.position.x, pos.position.y],
                })
                .collect()
        } else {
            // Generate default positions if DataSource can't provide layout
            self.generate_default_positions(nodes)
        }
    }

    /// Generate default positions for nodes (fallback)
    fn generate_default_positions(&self, nodes: &[Node]) -> Vec<NodePosition> {
        use std::f64::consts::PI;

        nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let angle = 2.0 * PI * i as f64 / nodes.len() as f64;
                let radius = 100.0;
                NodePosition {
                    node_id: node.id,
                    coords: vec![radius * angle.cos(), radius * angle.sin()],
                }
            })
            .collect()
    }

    /// Create graph metadata
    fn create_meta(&self, node_count: usize, edge_count: usize, has_positions: bool) -> GraphMeta {
        let _ds_meta = self.data_source.get_graph_metadata();

        GraphMeta {
            node_count,
            edge_count,
            dimensions: *self.embedding_dimensions.read().unwrap(),
            layout_method: format!("{:?}", *self.layout_algorithm.read().unwrap()),
            embedding_method: "default".to_string(),
            has_positions,
        }
    }
}

impl RealtimeVizAccessor for DataSourceRealtimeAccessor {
    fn initial_snapshot(&self) -> GraphResult<EngineSnapshot> {
        if self.verbose >= 3 {
            // Debug message
        }

        // Convert DataSource data to engine format
        let nodes = self.convert_nodes();
        let edges = self.convert_edges();
        let positions = self.compute_positions(&nodes);
        let has_positions = !positions.is_empty();
        let meta = self.create_meta(nodes.len(), edges.len(), has_positions);

        if self.verbose >= 1 {
            // Debug message
        }

        Ok(EngineSnapshot {
            nodes,
            edges,
            positions,
            meta,
        })
    }

    fn apply_control(&self, control: ControlMsg) -> GraphResult<()> {
        if self.verbose >= 3 {
            // Debug message
        }

        match control {
            ControlMsg::ChangeEmbedding { method, k, params } => {
                *self.embedding_dimensions.write().unwrap() = k;
                if self.verbose >= 2 {
                    // Debug message
                }

                if method == "rotation" {
                    let _rotation_x = params
                        .get("rotation_x")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);
                    let _rotation_y = params
                        .get("rotation_y")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0);

                    if self.verbose >= 3 {
                        // Debug message
                    }
                }

                // Engine is authoritative for runtime updates; accessor only acknowledges control inputs.
            }
            ControlMsg::ChangeLayout { algorithm, params } => {
                if self.verbose >= 2 {
                    // Debug message
                }

                match algorithm.parse::<LayoutKind>() {
                    Ok(LayoutKind::Honeycomb) => {
                        *self.layout_algorithm.write().unwrap() = LayoutAlgorithm::Honeycomb {
                            cell_size: params
                                .get("cell_size")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(40.0),
                            energy_optimization: params
                                .get("energy_optimization")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(true),
                            iterations: params
                                .get("iterations")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(500),
                        };
                    }
                    Ok(LayoutKind::ForceDirected) => {
                        *self.layout_algorithm.write().unwrap() = LayoutAlgorithm::ForceDirected {
                            charge: params
                                .get("charge")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(-30.0),
                            distance: params
                                .get("distance")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(30.0),
                            iterations: params
                                .get("iterations")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(100),
                        };
                    }
                    Ok(LayoutKind::Circular) => {
                        *self.layout_algorithm.write().unwrap() = LayoutAlgorithm::Circular {
                            radius: params.get("radius").and_then(|s| s.parse().ok()),
                            start_angle: params
                                .get("start_angle")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(0.0),
                        };
                    }
                    Ok(LayoutKind::Grid) => {
                        *self.layout_algorithm.write().unwrap() = LayoutAlgorithm::Grid {
                            columns: params
                                .get("columns")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or_else(|| {
                                    (self.data_source.get_graph_nodes().len() as f64)
                                        .sqrt()
                                        .ceil() as usize
                                }),
                            cell_size: params
                                .get("cell_size")
                                .and_then(|s| s.parse().ok())
                                .unwrap_or(80.0),
                        };
                    }
                    Err(_err) => {
                        if self.verbose >= 1 {
                            // Debug message
                        }
                    }
                }

                // Engine recomputes layout and will stream envelopes downstream.
            }
            _ => {
                if self.verbose >= 2 {
                    // Debug message
                }
            }
        }

        Ok(())
    }

    fn node_count(&self) -> usize {
        if self.data_source.supports_graph_view() {
            self.data_source.get_graph_metadata().node_count
        } else {
            self.data_source.total_rows()
        }
    }

    fn edge_count(&self) -> usize {
        if self.data_source.supports_graph_view() {
            self.data_source.get_graph_metadata().edge_count
        } else {
            0
        }
    }

    fn has_positions(&self) -> bool {
        // Check if DataSource can provide layout
        !self
            .data_source
            .compute_layout(self.layout_algorithm.read().unwrap().clone())
            .is_empty()
    }

    fn get_table_data(
        &self,
        data_type: super::engine_messages::TableDataType,
        offset: usize,
        window_size: usize,
        sort_columns: Vec<super::engine_messages::SortColumn>,
    ) -> GraphResult<crate::viz::realtime::server::ws_bridge::TableDataWindow> {
        use super::engine_messages::TableDataType;
        use crate::viz::realtime::server::ws_bridge::TableDataWindow;

        let snapshot = self.initial_snapshot()?;

        match data_type {
            TableDataType::Nodes => {
                let mut nodes = snapshot.nodes.clone();

                // Apply sorting if sort_columns is not empty
                if !sort_columns.is_empty() {
                    nodes.sort_by(|a, b| {
                        for sort_col in &sort_columns {
                            let ordering = if sort_col.column == "ID" {
                                // Sort by node ID
                                if sort_col.direction == "desc" {
                                    b.id.cmp(&a.id)
                                } else {
                                    a.id.cmp(&b.id)
                                }
                            } else {
                                // Sort by attribute
                                let a_val = a.attributes.get(&sort_col.column);
                                let b_val = b.attributes.get(&sort_col.column);

                                let cmp = match (a_val, b_val) {
                                    (Some(av), Some(bv)) => {
                                        av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (Some(_), None) => std::cmp::Ordering::Less,
                                    (None, Some(_)) => std::cmp::Ordering::Greater,
                                    (None, None) => std::cmp::Ordering::Equal,
                                };

                                if sort_col.direction == "desc" {
                                    cmp.reverse()
                                } else {
                                    cmp
                                }
                            };

                            if ordering != std::cmp::Ordering::Equal {
                                return ordering;
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                }

                let total_rows = nodes.len();
                let end = std::cmp::min(offset + window_size, total_rows);
                let nodes_window = &nodes[offset..end];

                // Build headers from all unique attributes across all nodes in window
                let mut headers = vec!["ID".to_string()];
                let mut attr_keys = std::collections::HashSet::new();
                for node in nodes_window {
                    for key in node.attributes.keys() {
                        attr_keys.insert(key.clone());
                    }
                }
                let mut sorted_keys: Vec<_> = attr_keys.into_iter().collect();
                sorted_keys.sort();
                headers.extend(sorted_keys);

                // Build rows
                let mut rows = Vec::new();
                for node in nodes_window {
                    let mut row = vec![serde_json::Value::Number(node.id.into())];
                    for key in headers.iter().skip(1) {
                        let value = node
                            .attributes
                            .get(key)
                            .map(attr_value_to_json)
                            .unwrap_or(serde_json::Value::Null);
                        row.push(value);
                    }
                    rows.push(row);
                }

                Ok(TableDataWindow {
                    headers,
                    rows,
                    total_rows,
                    start_offset: offset,
                    data_type: "nodes".to_string(),
                })
            }
            TableDataType::Edges => {
                let mut edges = snapshot.edges.clone();

                // Apply sorting if sort_columns is not empty
                if !sort_columns.is_empty() {
                    edges.sort_by(|a, b| {
                        for sort_col in &sort_columns {
                            let ordering = if sort_col.column == "ID" {
                                // Sort by edge ID
                                if sort_col.direction == "desc" {
                                    b.id.cmp(&a.id)
                                } else {
                                    a.id.cmp(&b.id)
                                }
                            } else if sort_col.column == "Source" {
                                // Sort by source node ID
                                if sort_col.direction == "desc" {
                                    b.source.cmp(&a.source)
                                } else {
                                    a.source.cmp(&b.source)
                                }
                            } else if sort_col.column == "Target" {
                                // Sort by target node ID
                                if sort_col.direction == "desc" {
                                    b.target.cmp(&a.target)
                                } else {
                                    a.target.cmp(&b.target)
                                }
                            } else {
                                // Sort by attribute
                                let a_val = a.attributes.get(&sort_col.column);
                                let b_val = b.attributes.get(&sort_col.column);

                                let cmp = match (a_val, b_val) {
                                    (Some(av), Some(bv)) => {
                                        av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (Some(_), None) => std::cmp::Ordering::Less,
                                    (None, Some(_)) => std::cmp::Ordering::Greater,
                                    (None, None) => std::cmp::Ordering::Equal,
                                };

                                if sort_col.direction == "desc" {
                                    cmp.reverse()
                                } else {
                                    cmp
                                }
                            };

                            if ordering != std::cmp::Ordering::Equal {
                                return ordering;
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                }

                let total_rows = edges.len();
                let end = std::cmp::min(offset + window_size, total_rows);
                let edges_window = &edges[offset..end];

                // Build headers from all unique attributes across all edges in window
                let mut headers =
                    vec!["ID".to_string(), "Source".to_string(), "Target".to_string()];
                let mut attr_keys = std::collections::HashSet::new();
                for edge in edges_window {
                    for key in edge.attributes.keys() {
                        attr_keys.insert(key.clone());
                    }
                }
                let mut sorted_keys: Vec<_> = attr_keys.into_iter().collect();
                sorted_keys.sort();
                headers.extend(sorted_keys);

                // Build rows
                let mut rows = Vec::new();
                for edge in edges_window {
                    let mut row = vec![
                        serde_json::Value::Number(edge.id.into()),
                        serde_json::Value::Number(edge.source.into()),
                        serde_json::Value::Number(edge.target.into()),
                    ];
                    for key in headers.iter().skip(3) {
                        let value = edge
                            .attributes
                            .get(key)
                            .map(attr_value_to_json)
                            .unwrap_or(serde_json::Value::Null);
                        row.push(value);
                    }
                    rows.push(row);
                }

                Ok(TableDataWindow {
                    headers,
                    rows,
                    total_rows,
                    start_offset: offset,
                    data_type: "edges".to_string(),
                })
            }
        }
    }
}

/// Convert AttrValue to JSON for table display
fn attr_value_to_json(attr: &AttrValue) -> serde_json::Value {
    match attr {
        AttrValue::Float(f) => serde_json::Value::Number(
            serde_json::Number::from_f64(*f as f64).unwrap_or_else(|| serde_json::Number::from(0)),
        ),
        AttrValue::Int(i) => serde_json::Value::Number((*i).into()),
        AttrValue::Text(s) => serde_json::Value::String(s.clone()),
        AttrValue::Bool(b) => serde_json::Value::Bool(*b),
        AttrValue::SmallInt(i) => serde_json::Value::Number((*i as i64).into()),
        AttrValue::CompactText(s) => serde_json::Value::String(s.as_str().to_string()),
        AttrValue::Null => serde_json::Value::Null,
        _ => serde_json::Value::String(format!("{:?}", attr)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::display::DataType;
    use crate::viz::streaming::data_source::{
        DataSchema, DataWindow, GraphEdge, GraphMetadata, GraphNode,
    };
    use std::collections::HashMap;

    /// Mock DataSource for testing
    #[derive(Debug)]
    struct MockDataSource {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
    }

    impl MockDataSource {
        fn new() -> Self {
            let mut node1_attrs = HashMap::new();
            node1_attrs.insert("label".to_string(), AttrValue::Text("Node 1".to_string()));

            let mut node2_attrs = HashMap::new();
            node2_attrs.insert("label".to_string(), AttrValue::Text("Node 2".to_string()));

            let mut edge_attrs = HashMap::new();
            edge_attrs.insert("weight".to_string(), AttrValue::Float(1.0));

            Self {
                nodes: vec![
                    GraphNode {
                        id: "0".to_string(),
                        label: Some("Node 1".to_string()),
                        attributes: node1_attrs,
                        position: None,
                    },
                    GraphNode {
                        id: "1".to_string(),
                        label: Some("Node 2".to_string()),
                        attributes: node2_attrs,
                        position: None,
                    },
                ],
                edges: vec![GraphEdge {
                    id: "0".to_string(),
                    source: "0".to_string(),
                    target: "1".to_string(),
                    label: Some("Edge 1".to_string()),
                    weight: Some(1.0),
                    attributes: edge_attrs,
                }],
            }
        }
    }

    impl DataSource for MockDataSource {
        fn total_rows(&self) -> usize {
            self.nodes.len()
        }
        fn total_cols(&self) -> usize {
            3
        }
        fn supports_streaming(&self) -> bool {
            true
        }
        fn supports_graph_view(&self) -> bool {
            true
        }

        fn get_window(&self, _start: usize, _count: usize) -> DataWindow {
            DataWindow::new(
                vec!["id".to_string(), "label".to_string()],
                vec![],
                DataSchema {
                    columns: vec![],
                    primary_key: Some("id".to_string()),
                    source_type: "mock".to_string(),
                },
                self.nodes.len(),
                0,
            )
        }

        fn get_schema(&self) -> DataSchema {
            DataSchema {
                columns: vec![],
                primary_key: Some("id".to_string()),
                source_type: "mock".to_string(),
            }
        }

        fn get_column_types(&self) -> Vec<DataType> {
            vec![DataType::Integer, DataType::String]
        }

        fn get_column_names(&self) -> Vec<String> {
            vec!["id".to_string(), "label".to_string()]
        }

        fn get_source_id(&self) -> String {
            "mock_source".to_string()
        }
        fn get_version(&self) -> u64 {
            1
        }

        fn get_graph_nodes(&self) -> Vec<GraphNode> {
            self.nodes.clone()
        }

        fn get_graph_edges(&self) -> Vec<GraphEdge> {
            self.edges.clone()
        }

        fn get_graph_metadata(&self) -> GraphMetadata {
            GraphMetadata {
                node_count: self.nodes.len(),
                edge_count: self.edges.len(),
                is_directed: false,
                has_weights: true,
                attribute_types: HashMap::new(),
            }
        }

        fn compute_layout(
            &self,
            _algorithm: LayoutAlgorithm,
        ) -> Vec<crate::viz::streaming::data_source::NodePosition> {
            vec![]
        }
    }

    #[test]
    fn test_initial_snapshot() {
        let mock_ds = Arc::new(MockDataSource::new());
        let accessor = DataSourceRealtimeAccessor::new(mock_ds);

        let snapshot = accessor.initial_snapshot().unwrap();

        assert_eq!(snapshot.nodes.len(), 2);
        assert_eq!(snapshot.edges.len(), 1);
        assert_eq!(snapshot.positions.len(), 2); // Default positions generated
        assert_eq!(snapshot.meta.node_count, 2);
        assert_eq!(snapshot.meta.edge_count, 1);
    }

    #[test]
    fn test_control_messages() {
        let mock_ds = Arc::new(MockDataSource::new());
        let accessor = DataSourceRealtimeAccessor::new(mock_ds);

        // Test embedding change
        let control = ControlMsg::ChangeEmbedding {
            method: "pca".to_string(),
            k: 3,
            params: HashMap::new(),
        };

        accessor.apply_control(control).unwrap();
        assert_eq!(*accessor.embedding_dimensions.read().unwrap(), 3);
    }
}
