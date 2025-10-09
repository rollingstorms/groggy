//! Phase 3: Real-time Streaming and Interactive Visualization System
//!
//! This module provides a complete real-time visualization pipeline that combines:
//! - Phase 1: Multi-dimensional embeddings with energy-based optimization
//! - Phase 2: Projection to 2D honeycomb coordinates with quality metrics
//! - Phase 3: Real-time streaming, interaction, and adaptive performance
//!
//! The system supports:
//! - Live graph updates with incremental embedding computation
//! - Real-time parameter adjustment and smooth transitions
//! - WebSocket streaming of position updates
//! - Interactive controls for zooming, filtering, and exploration
//! - Adaptive quality controls for performance optimization
//! - Comprehensive monitoring and debugging capabilities

use crate::storage::matrix::GraphMatrix;
use crate::viz::embeddings::EmbeddingConfig;
use crate::viz::projection::{
    HoneycombConfig, InterpolationConfig, ProjectionConfig, ProjectionMethod, QualityConfig,
};
use crate::viz::streaming::data_source::Position;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

pub mod accessor;
pub mod engine;
pub mod engine_sync;
pub mod interaction;
pub mod server;

pub use accessor::*;
pub use engine::*;
pub use engine_sync::*;
pub use server::*;

/// Supported layout algorithms for the realtime engine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutKind {
    Honeycomb,
    ForceDirected,
    Circular,
    Grid,
}

impl LayoutKind {
    /// Canonical lowercase name used across engine/server/UI boundaries
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutKind::Honeycomb => "honeycomb",
            LayoutKind::ForceDirected => "force_directed",
            LayoutKind::Circular => "circular",
            LayoutKind::Grid => "grid",
        }
    }

    /// Variants that should bypass control-layer debounce
    pub fn is_layout_parameter(name: &str) -> bool {
        name.starts_with("layout.")
    }
}

impl fmt::Display for LayoutKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct LayoutKindParseError {
    raw: String,
}

impl fmt::Display for LayoutKindParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown layout algorithm '{}'", self.raw)
    }
}

impl std::error::Error for LayoutKindParseError {}

impl FromStr for LayoutKind {
    type Err = LayoutKindParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalized = s.trim().to_lowercase().replace(['-', ' '], "_");

        let kind = match normalized.as_str() {
            "honeycomb" => LayoutKind::Honeycomb,
            "force_directed" | "force" | "force_directed_layout" => LayoutKind::ForceDirected,
            "circular" | "circle" => LayoutKind::Circular,
            "grid" | "matrix" => LayoutKind::Grid,
            _ => {
                return Err(LayoutKindParseError { raw: s.to_string() });
            }
        };

        Ok(kind)
    }
}

/// Configuration for the real-time visualization system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeVizConfig {
    /// Embedding configuration for Phase 1
    pub embedding_config: EmbeddingConfig,

    /// Projection configuration for Phase 2
    pub projection_config: ProjectionConfig,

    /// Real-time specific settings
    pub realtime_config: RealTimeConfig,

    /// Performance and quality settings
    pub performance_config: PerformanceConfig,

    /// Interactive control settings
    pub interaction_config: InteractionConfig,

    /// Streaming server configuration
    pub streaming_config: StreamingConfig,
}

/// Real-time visualization specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeConfig {
    /// Target frames per second for smooth animation
    pub target_fps: f64,

    /// Whether to enable incremental updates for dynamic graphs
    pub enable_incremental_updates: bool,

    /// Maximum time budget per frame (milliseconds)
    pub frame_time_budget_ms: f64,

    /// Whether to enable adaptive quality scaling
    pub enable_adaptive_quality: bool,

    /// Minimum quality threshold for adaptive scaling
    pub min_quality_threshold: f64,

    /// Whether to enable position prediction for smooth motion
    pub enable_position_prediction: bool,

    /// Number of frames to look ahead for prediction
    pub prediction_lookahead_frames: usize,
}

/// Performance monitoring and adaptive quality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,

    /// Sampling interval for performance metrics (milliseconds)
    pub monitoring_interval_ms: u64,

    /// Frame time history window size for smoothing
    pub frame_time_history_size: usize,

    /// Memory usage monitoring threshold (MB)
    pub memory_threshold_mb: usize,

    /// Whether to enable automatic quality adaptation
    pub enable_auto_quality_adaptation: bool,

    /// Quality adaptation sensitivity (0.0 = none, 1.0 = aggressive)
    pub quality_adaptation_sensitivity: f64,

    /// Whether to enable debug performance overlays
    pub enable_debug_overlay: bool,
}

/// Interactive control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    /// Whether to enable real-time parameter adjustment
    pub enable_parameter_controls: bool,

    /// Whether to enable node selection and highlighting
    pub enable_node_selection: bool,

    /// Whether to enable real-time filtering
    pub enable_realtime_filtering: bool,

    /// Whether to enable zoom and pan controls
    pub enable_zoom_pan: bool,

    /// Zoom sensitivity factor
    pub zoom_sensitivity: f64,

    /// Pan sensitivity factor
    pub pan_sensitivity: f64,

    /// Selection highlighting configuration
    pub selection_config: SelectionConfig,

    /// Filter control configuration
    pub filter_config: FilterConfig,
}

/// Node selection and highlighting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Whether to highlight neighboring nodes
    pub highlight_neighbors: bool,

    /// Number of neighbor hops to highlight
    pub neighbor_hop_count: usize,

    /// Selection highlight color (hex)
    pub highlight_color: String,

    /// Selection highlight intensity (0.0 - 1.0)
    pub highlight_intensity: f64,

    /// Whether to show selection analytics
    pub show_selection_analytics: bool,
}

/// Real-time filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Whether to enable attribute-based filtering
    pub enable_attribute_filters: bool,

    /// Whether to enable degree-based filtering
    pub enable_degree_filters: bool,

    /// Whether to enable community-based filtering
    pub enable_community_filters: bool,

    /// Filter update delay to prevent too frequent updates (ms)
    pub filter_debounce_ms: u64,

    /// Whether to use smooth filter transitions
    pub enable_smooth_transitions: bool,
}

/// Streaming server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// WebSocket server port
    pub server_port: u16,

    /// Maximum concurrent WebSocket connections
    pub max_connections: usize,

    /// Update broadcast interval (milliseconds)
    pub broadcast_interval_ms: u64,

    /// Whether to enable position compression for bandwidth optimization
    pub enable_position_compression: bool,

    /// Position precision for compression (decimal places)
    pub position_precision: usize,

    /// Whether to enable update batching
    pub enable_update_batching: bool,

    /// Maximum batch size for updates
    pub max_batch_size: usize,
}

/// Real-time visualization state
#[derive(Debug)]
pub struct RealTimeVizState {
    /// Current node positions
    pub positions: Vec<Position>,

    /// Fast lookup from node_id to position index
    pub node_index: HashMap<crate::types::NodeId, usize>,

    /// Current embedding matrix
    pub embedding: Option<GraphMatrix>,

    /// Animation state for smooth transitions
    pub animation_state: AnimationState,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Selection state
    pub selection: SelectionState,

    /// Filter state
    pub filters: FilterState,

    /// Last update timestamp
    pub last_update: Instant,

    /// Current layout algorithm being used
    pub current_layout: LayoutKind,

    /// Parameters for the current layout algorithm
    pub current_layout_params: std::collections::HashMap<String, String>,

    /// Cached parameters per layout kind to preserve user tweaks when toggling algorithms
    pub layout_param_cache:
        std::collections::HashMap<LayoutKind, std::collections::HashMap<String, String>>,
}

/// Animation state for smooth transitions
#[derive(Debug, Clone)]
pub struct AnimationState {
    /// Whether animation is currently active
    pub is_animating: bool,

    /// Animation start time
    pub start_time: Instant,

    /// Animation duration
    pub duration: Duration,

    /// Source positions for interpolation
    pub source_positions: Vec<Position>,

    /// Target positions for interpolation
    pub target_positions: Vec<Position>,

    /// Current interpolation progress (0.0 - 1.0)
    pub progress: f64,

    /// Easing function for smooth animation
    pub easing_function: String,
}

/// Performance monitoring metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Current frame rate
    pub current_fps: f64,

    /// Average frame time (milliseconds)
    pub average_frame_time_ms: f64,

    /// Current memory usage (MB)
    pub memory_usage_mb: f64,

    /// Number of active nodes being rendered
    pub active_node_count: usize,

    /// Number of active edges being rendered
    pub active_edge_count: usize,

    /// Current quality level (0.0 - 1.0)
    pub current_quality_level: f64,

    /// Number of WebSocket connections
    pub websocket_connections: usize,

    /// Update broadcast rate
    pub broadcast_rate_hz: f64,

    /// Last embedding computation time (ms)
    pub last_embedding_time_ms: f64,

    /// Last projection computation time (ms)
    pub last_projection_time_ms: f64,
}

/// Node selection state
#[derive(Debug, Clone)]
pub struct SelectionState {
    /// Currently selected node IDs
    pub selected_nodes: Vec<usize>,

    /// Highlighted neighbor nodes
    pub highlighted_neighbors: Vec<usize>,

    /// Selection bounding box
    pub selection_bounds: Option<BoundingBox>,

    /// Selection analytics data
    pub analytics: SelectionAnalytics,
}

/// Selection analytics information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionAnalytics {
    /// Number of selected nodes
    pub node_count: usize,

    /// Number of edges within selection
    pub internal_edge_count: usize,

    /// Number of edges crossing selection boundary
    pub boundary_edge_count: usize,

    /// Average degree of selected nodes
    pub average_degree: f64,

    /// Selection density (edges / possible edges)
    pub selection_density: f64,
}

/// Bounding box for selections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

/// Real-time filter state
#[derive(Debug, Clone)]
pub struct FilterState {
    /// Attribute-based filters
    pub attribute_filters: HashMap<String, AttributeFilter>,

    /// Degree range filter
    pub degree_filter: Option<DegreeFilter>,

    /// Community-based filters
    pub community_filters: Vec<CommunityFilter>,

    /// Currently visible node IDs
    pub visible_nodes: Vec<usize>,

    /// Filter transition state
    pub transition_state: FilterTransitionState,
}

/// Attribute-based filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeFilter {
    /// Attribute name
    pub attribute: String,

    /// Filter type (range, categorical, text)
    pub filter_type: AttributeFilterType,

    /// Filter value(s)
    pub values: Vec<String>,

    /// Whether filter is currently active
    pub is_active: bool,
}

/// Attribute filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeFilterType {
    /// Numeric range filter
    Range { min: f64, max: f64 },
    /// Categorical inclusion filter
    Categorical { included_values: Vec<String> },
    /// Text search filter
    TextSearch {
        pattern: String,
        case_sensitive: bool,
    },
    /// Boolean filter
    Boolean { value: bool },
}

/// Degree-based filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegreeFilter {
    /// Minimum degree threshold
    pub min_degree: usize,

    /// Maximum degree threshold
    pub max_degree: usize,

    /// Whether to include isolated nodes
    pub include_isolated: bool,
}

/// Community-based filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityFilter {
    /// Community detection algorithm used
    pub algorithm: String,

    /// Selected community IDs to show
    pub selected_communities: Vec<usize>,

    /// Minimum community size threshold
    pub min_community_size: usize,
}

/// Filter transition state for smooth animations
#[derive(Debug, Clone)]
pub struct FilterTransitionState {
    /// Whether filter transition is active
    pub is_transitioning: bool,

    /// Transition start time
    pub start_time: Instant,

    /// Transition duration
    pub duration: Duration,

    /// Nodes being faded in
    pub fading_in: Vec<usize>,

    /// Nodes being faded out
    pub fading_out: Vec<usize>,

    /// Current transition progress (0.0 - 1.0)
    pub progress: f64,
}

impl Default for RealTimeVizConfig {
    fn default() -> Self {
        Self {
            embedding_config: EmbeddingConfig::default(),
            projection_config: ProjectionConfig::default(),
            realtime_config: RealTimeConfig::default(),
            performance_config: PerformanceConfig::default(),
            interaction_config: InteractionConfig::default(),
            streaming_config: StreamingConfig::default(),
        }
    }
}

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            enable_incremental_updates: true,
            frame_time_budget_ms: 16.67, // ~60 FPS
            enable_adaptive_quality: true,
            min_quality_threshold: 0.3,
            enable_position_prediction: true,
            prediction_lookahead_frames: 3,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval_ms: 100, // 10 Hz monitoring
            frame_time_history_size: 60, // 1 second at 60 FPS
            memory_threshold_mb: 512,
            enable_auto_quality_adaptation: true,
            quality_adaptation_sensitivity: 0.5,
            enable_debug_overlay: false,
        }
    }
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            enable_parameter_controls: true,
            enable_node_selection: true,
            enable_realtime_filtering: true,
            enable_zoom_pan: true,
            zoom_sensitivity: 1.0,
            pan_sensitivity: 1.0,
            selection_config: SelectionConfig::default(),
            filter_config: FilterConfig::default(),
        }
    }
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            highlight_neighbors: true,
            neighbor_hop_count: 2,
            highlight_color: "#ff6b6b".to_string(),
            highlight_intensity: 0.8,
            show_selection_analytics: true,
        }
    }
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            enable_attribute_filters: true,
            enable_degree_filters: true,
            enable_community_filters: true,
            filter_debounce_ms: 300,
            enable_smooth_transitions: true,
        }
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            server_port: 8080,
            max_connections: 100,
            broadcast_interval_ms: 33, // ~30 FPS for updates
            enable_position_compression: true,
            position_precision: 2,
            enable_update_batching: true,
            max_batch_size: 50,
        }
    }
}

impl Default for AnimationState {
    fn default() -> Self {
        Self {
            is_animating: false,
            start_time: Instant::now(),
            duration: Duration::from_millis(1000),
            source_positions: Vec::new(),
            target_positions: Vec::new(),
            progress: 0.0,
            easing_function: "ease-in-out".to_string(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            current_fps: 0.0,
            average_frame_time_ms: 0.0,
            memory_usage_mb: 0.0,
            active_node_count: 0,
            active_edge_count: 0,
            current_quality_level: 1.0,
            websocket_connections: 0,
            broadcast_rate_hz: 0.0,
            last_embedding_time_ms: 0.0,
            last_projection_time_ms: 0.0,
        }
    }
}

impl Default for SelectionState {
    fn default() -> Self {
        Self {
            selected_nodes: Vec::new(),
            highlighted_neighbors: Vec::new(),
            selection_bounds: None,
            analytics: SelectionAnalytics::default(),
        }
    }
}

impl Default for SelectionAnalytics {
    fn default() -> Self {
        Self {
            node_count: 0,
            internal_edge_count: 0,
            boundary_edge_count: 0,
            average_degree: 0.0,
            selection_density: 0.0,
        }
    }
}

impl Default for FilterState {
    fn default() -> Self {
        Self {
            attribute_filters: HashMap::new(),
            degree_filter: None,
            community_filters: Vec::new(),
            visible_nodes: Vec::new(),
            transition_state: FilterTransitionState::default(),
        }
    }
}

impl Default for FilterTransitionState {
    fn default() -> Self {
        Self {
            is_transitioning: false,
            start_time: Instant::now(),
            duration: Duration::from_millis(500),
            fading_in: Vec::new(),
            fading_out: Vec::new(),
            progress: 0.0,
        }
    }
}

/// Parameter value that can be either an array of values or a column name
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VizParameter<T> {
    /// Direct array of values - must match the number of nodes/edges
    Array(Vec<T>),
    /// Column name to extract values from
    Column(String),
    /// Single value to apply to all elements
    Value(T),
    /// No value specified (use defaults)
    None,
}

impl<T> Default for VizParameter<T> {
    fn default() -> Self {
        VizParameter::None
    }
}

impl<T> VizParameter<T> {
    /// Check if this parameter has a value
    pub fn is_some(&self) -> bool {
        !matches!(self, VizParameter::None)
    }

    /// Get the column name if this is a column parameter
    pub fn as_column(&self) -> Option<&str> {
        match self {
            VizParameter::Column(name) => Some(name),
            _ => None,
        }
    }

    /// Get the array if this is an array parameter
    pub fn as_array(&self) -> Option<&Vec<T>> {
        match self {
            VizParameter::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Get the single value if this is a value parameter
    pub fn as_value(&self) -> Option<&T> {
        match self {
            VizParameter::Value(val) => Some(val),
            _ => None,
        }
    }
}

/// Comprehensive visualization configuration supporting all styling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizConfig {
    // === Node Styling ===
    /// Node colors - can be array of colors, column name, or single color
    pub node_color: VizParameter<String>,
    /// Node sizes - can be array of sizes, column name, or single size
    pub node_size: VizParameter<f64>,
    /// Node shapes - can be array of shapes, column name, or single shape
    pub node_shape: VizParameter<String>,
    /// Node opacity values (0.0-1.0)
    pub node_opacity: VizParameter<f64>,
    /// Node border colors
    pub node_border_color: VizParameter<String>,
    /// Node border widths
    pub node_border_width: VizParameter<f64>,

    // === Edge Styling ===
    /// Edge colors - can be array of colors, column name, or single color
    pub edge_color: VizParameter<String>,
    /// Edge widths - can be array of widths, column name, or single width
    pub edge_width: VizParameter<f64>,
    /// Edge opacity values (0.0-1.0)
    pub edge_opacity: VizParameter<f64>,
    /// Edge styles (solid, dashed, dotted)
    pub edge_style: VizParameter<String>,

    // === Labels & Text ===
    /// Node labels - can be array of labels, column name, or single label
    pub node_label: VizParameter<String>,
    /// Edge labels - can be array of labels, column name, or single label
    pub edge_label: VizParameter<String>,
    /// Label font sizes
    pub label_size: VizParameter<f64>,
    /// Label colors
    pub label_color: VizParameter<String>,
    /// Edge label font sizes
    pub edge_label_size: VizParameter<f64>,
    /// Edge label colors
    pub edge_label_color: VizParameter<String>,
    /// Columns to show in tooltips (list of column names)
    pub tooltip_columns: Vec<String>,

    // === Positioning ===
    /// X coordinates for fixed positioning
    pub x: VizParameter<f64>,
    /// Y coordinates for fixed positioning
    pub y: VizParameter<f64>,
    /// Z coordinates for 3D positioning
    pub z: VizParameter<f64>,

    // === Filtering & Visibility ===
    /// Query expression for showing nodes (e.g., "degree > 3")
    pub show_nodes_where: Option<String>,
    /// Query expression for showing edges
    pub show_edges_where: Option<String>,
    /// Query expression for highlighting nodes
    pub highlight_nodes_where: Option<String>,
    /// Query expression for highlighting edges
    pub highlight_edges_where: Option<String>,

    // === Scaling & Ranges ===
    /// Range for mapping values to node sizes [min_size, max_size]
    pub node_size_range: Option<(f64, f64)>,
    /// Range for mapping values to edge widths [min_width, max_width]
    pub edge_width_range: Option<(f64, f64)>,
    /// Color palette for categorical data or gradient endpoints
    pub color_palette: Option<Vec<String>>,
    /// Color scale type ("linear", "log", "categorical")
    pub color_scale_type: Option<String>,

    // === Layout & Algorithm ===
    /// Layout algorithm to use
    pub layout_algorithm: Option<String>,
    /// Additional layout parameters
    pub layout_params: HashMap<String, String>,

    // === Interaction ===
    /// Click behavior ("select", "info", "none")
    pub click_behavior: Option<String>,
    /// Hover behavior ("highlight", "tooltip", "none")
    pub hover_behavior: Option<String>,
    /// Selection mode ("single", "multiple", "none")
    pub selection_mode: Option<String>,
    /// Zoom behavior ("enabled", "disabled", "focus")
    pub zoom_behavior: Option<String>,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            // Node styling
            node_color: VizParameter::default(),
            node_size: VizParameter::default(),
            node_shape: VizParameter::default(),
            node_opacity: VizParameter::default(),
            node_border_color: VizParameter::default(),
            node_border_width: VizParameter::default(),

            // Edge styling
            edge_color: VizParameter::default(),
            edge_width: VizParameter::default(),
            edge_opacity: VizParameter::default(),
            edge_style: VizParameter::default(),

            // Labels & text
            node_label: VizParameter::default(),
            edge_label: VizParameter::default(),
            label_size: VizParameter::default(),
            label_color: VizParameter::default(),
            edge_label_size: VizParameter::default(),
            edge_label_color: VizParameter::default(),
            tooltip_columns: Vec::new(),

            // Positioning
            x: VizParameter::default(),
            y: VizParameter::default(),
            z: VizParameter::default(),

            // Filtering
            show_nodes_where: None,
            show_edges_where: None,
            highlight_nodes_where: None,
            highlight_edges_where: None,

            // Scaling
            node_size_range: None,
            edge_width_range: None,
            color_palette: None,
            color_scale_type: None,

            // Layout
            layout_algorithm: None,
            layout_params: HashMap::new(),

            // Interaction
            click_behavior: None,
            hover_behavior: None,
            selection_mode: None,
            zoom_behavior: None,
        }
    }
}

impl VizConfig {
    /// Create a new VizConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set node color parameter
    pub fn with_node_color(mut self, color: VizParameter<String>) -> Self {
        self.node_color = color;
        self
    }

    /// Set node size parameter
    pub fn with_node_size(mut self, size: VizParameter<f64>) -> Self {
        self.node_size = size;
        self
    }

    /// Set edge color parameter
    pub fn with_edge_color(mut self, color: VizParameter<String>) -> Self {
        self.edge_color = color;
        self
    }

    /// Set layout algorithm
    pub fn with_layout_algorithm(mut self, algorithm: String) -> Self {
        self.layout_algorithm = Some(algorithm);
        self
    }

    /// Add a layout parameter
    pub fn with_layout_param(mut self, key: String, value: String) -> Self {
        self.layout_params.insert(key, value);
        self
    }

    /// Set tooltip columns
    pub fn with_tooltip_columns(mut self, columns: Vec<String>) -> Self {
        self.tooltip_columns = columns;
        self
    }

    /// Validate that array parameters have the correct length
    pub fn validate_array_lengths(
        &self,
        node_count: usize,
        edge_count: usize,
    ) -> Result<(), String> {
        // Validate node-related arrays
        if let VizParameter::Array(arr) = &self.node_color {
            if arr.len() != node_count {
                return Err(format!(
                    "node_color array length {} doesn't match node count {}",
                    arr.len(),
                    node_count
                ));
            }
        }
        if let VizParameter::Array(arr) = &self.node_size {
            if arr.len() != node_count {
                return Err(format!(
                    "node_size array length {} doesn't match node count {}",
                    arr.len(),
                    node_count
                ));
            }
        }
        if let VizParameter::Array(arr) = &self.x {
            if arr.len() != node_count {
                return Err(format!(
                    "x array length {} doesn't match node count {}",
                    arr.len(),
                    node_count
                ));
            }
        }
        if let VizParameter::Array(arr) = &self.y {
            if arr.len() != node_count {
                return Err(format!(
                    "y array length {} doesn't match node count {}",
                    arr.len(),
                    node_count
                ));
            }
        }

        // Validate edge-related arrays
        if let VizParameter::Array(arr) = &self.edge_color {
            if arr.len() != edge_count {
                return Err(format!(
                    "edge_color array length {} doesn't match edge count {}",
                    arr.len(),
                    edge_count
                ));
            }
        }
        if let VizParameter::Array(arr) = &self.edge_width {
            if arr.len() != edge_count {
                return Err(format!(
                    "edge_width array length {} doesn't match edge count {}",
                    arr.len(),
                    edge_count
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_viz_config_defaults() {
        let config = RealTimeVizConfig::default();
        assert_eq!(config.realtime_config.target_fps, 60.0);
        assert!(config.realtime_config.enable_incremental_updates);
        assert!(config.performance_config.enable_monitoring);
        assert!(config.interaction_config.enable_parameter_controls);
        assert_eq!(config.streaming_config.server_port, 8080);
    }

    #[test]
    fn test_performance_metrics_initialization() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.current_fps, 0.0);
        assert_eq!(metrics.current_quality_level, 1.0);
        assert_eq!(metrics.active_node_count, 0);
    }

    #[test]
    fn test_animation_state_initialization() {
        let state = AnimationState::default();
        assert!(!state.is_animating);
        assert_eq!(state.progress, 0.0);
        assert_eq!(state.easing_function, "ease-in-out");
    }

    #[test]
    fn test_filter_state_initialization() {
        let filters = FilterState::default();
        assert!(filters.attribute_filters.is_empty());
        assert!(filters.degree_filter.is_none());
        assert!(filters.community_filters.is_empty());
        assert!(!filters.transition_state.is_transitioning);
    }
}
