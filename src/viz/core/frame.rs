//! Frame data structures for visualization output
//!
//! Defines the unified frame format that all visualization backends consume.
//! This provides a consistent interface between the core engine and adapters.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::viz::streaming::data_source::{Position, GraphNode as VizNode, GraphEdge as VizEdge};

/// Unified frame data that all visualization backends consume
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizFrame {
    /// Frame metadata
    pub metadata: FrameMetadata,
    
    /// Node data with current positions
    pub nodes: Vec<FrameNode>,
    
    /// Edge data 
    pub edges: Vec<FrameEdge>,
    
    /// Animation/transition data
    pub animation: AnimationData,
    
    /// Styling information
    pub style: FrameStyle,
    
    /// Interaction state
    pub interaction: InteractionData,
}

/// Frame metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// Unique frame identifier
    pub frame_id: String,
    
    /// Timestamp when frame was generated
    pub timestamp: u64,
    
    /// Physics simulation state
    pub simulation_state: SimulationFrameState,
    
    /// Frame dimensions
    pub dimensions: FrameDimensions,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
}

/// Node data in frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameNode {
    /// Node ID
    pub id: String,
    
    /// Current position
    pub position: Position,
    
    /// Previous position (for smooth transitions)
    pub previous_position: Option<Position>,
    
    /// Visual properties
    pub visual: NodeVisual,
    
    /// Interaction state
    pub interaction_state: NodeInteractionState,
    
    /// Node attributes/data
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Edge data in frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameEdge {
    /// Edge ID
    pub id: String,
    
    /// Source node ID
    pub source: String,
    
    /// Target node ID
    pub target: String,
    
    /// Visual properties
    pub visual: EdgeVisual,
    
    /// Edge attributes/data
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Visual properties for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVisual {
    /// Node radius/size
    pub radius: f64,
    
    /// Fill color
    pub fill_color: String,
    
    /// Stroke color
    pub stroke_color: String,
    
    /// Stroke width
    pub stroke_width: f64,
    
    /// Opacity (0.0 - 1.0)
    pub opacity: f64,
    
    /// Label text
    pub label: Option<String>,
    
    /// Label styling
    pub label_style: LabelStyle,
    
    /// Shape type
    pub shape: NodeShape,
}

/// Visual properties for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeVisual {
    /// Stroke color
    pub stroke_color: String,
    
    /// Stroke width
    pub stroke_width: f64,
    
    /// Opacity (0.0 - 1.0)
    pub opacity: f64,
    
    /// Line style
    pub line_style: LineStyle,
    
    /// Arrow/marker configuration
    pub markers: EdgeMarkers,
    
    /// Label text
    pub label: Option<String>,
    
    /// Label styling
    pub label_style: LabelStyle,
}

/// Node interaction state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInteractionState {
    /// Whether node is currently hovered
    pub is_hovered: bool,
    
    /// Whether node is currently selected
    pub is_selected: bool,
    
    /// Whether node is currently being dragged
    pub is_dragged: bool,
    
    /// Whether node is pinned (fixed position)
    pub is_pinned: bool,
    
    /// Highlight level (0.0 = none, 1.0 = full)
    pub highlight: f64,
}

/// Physics simulation state in frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationFrameState {
    /// Current alpha (cooling) value
    pub alpha: f64,
    
    /// Current iteration number
    pub iteration: usize,
    
    /// Total system energy
    pub energy: f64,
    
    /// Whether simulation is still running
    pub is_running: bool,
    
    /// Whether simulation has converged
    pub has_converged: bool,
}

/// Frame dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameDimensions {
    /// Width of the visualization area
    pub width: f64,
    
    /// Height of the visualization area
    pub height: f64,
    
    /// Bounding box of all nodes
    pub bounds: BoundingBox,
    
    /// Zoom level
    pub zoom: f64,
    
    /// Pan offset
    pub pan: Position,
}

/// Bounding box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

/// Performance metrics for frame generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time to generate frame (milliseconds)
    pub generation_time_ms: f64,
    
    /// Physics simulation time (milliseconds)
    pub physics_time_ms: f64,
    
    /// Rendering time (milliseconds)
    pub rendering_time_ms: f64,
    
    /// Number of nodes processed
    pub node_count: usize,
    
    /// Number of edges processed
    pub edge_count: usize,
}

/// Animation/transition data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationData {
    /// Duration of transitions (milliseconds)
    pub transition_duration: u64,
    
    /// Easing function for transitions
    pub easing: EasingFunction,
    
    /// Whether to animate position changes
    pub animate_positions: bool,
    
    /// Whether to animate style changes
    pub animate_styles: bool,
}

/// Frame-level styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameStyle {
    /// Background color
    pub background_color: String,
    
    /// Grid configuration
    pub grid: Option<GridStyle>,
    
    /// Theme information
    pub theme: String,
    
    /// Custom CSS/styling
    pub custom_styles: HashMap<String, String>,
}

/// Interaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionData {
    /// Whether interactions are enabled
    pub enabled: bool,
    
    /// Drag configuration
    pub drag: DragConfig,
    
    /// Zoom configuration
    pub zoom: ZoomConfig,
    
    /// Selection configuration
    pub selection: SelectionConfig,
}

/// Label styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelStyle {
    /// Font family
    pub font_family: String,
    
    /// Font size
    pub font_size: f64,
    
    /// Font weight
    pub font_weight: String,
    
    /// Text color
    pub color: String,
    
    /// Background color (if any)
    pub background_color: Option<String>,
    
    /// Text alignment
    pub alignment: TextAlignment,
    
    /// Whether to show label
    pub visible: bool,
}

/// Node shape types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Pentagon,
    Hexagon,
    Star,
    Custom(String), // SVG path or custom shape identifier
}

/// Line style for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineStyle {
    Solid,
    Dashed,
    Dotted,
    DashDot,
}

/// Edge markers (arrows, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMarkers {
    /// Start marker
    pub start: Option<MarkerType>,
    
    /// End marker
    pub end: Option<MarkerType>,
    
    /// Marker size
    pub size: f64,
}

/// Marker types for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarkerType {
    Arrow,
    Circle,
    Square,
    Diamond,
}

/// Text alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextAlignment {
    Left,
    Center,
    Right,
}

/// Easing functions for animations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
}

/// Grid styling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridStyle {
    /// Whether to show grid
    pub visible: bool,
    
    /// Grid line color
    pub color: String,
    
    /// Grid spacing
    pub spacing: f64,
    
    /// Grid opacity
    pub opacity: f64,
}

/// Drag interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragConfig {
    /// Whether dragging is enabled
    pub enabled: bool,
    
    /// Whether to fix dragged nodes
    pub fix_on_drag: bool,
    
    /// Drag sensitivity
    pub sensitivity: f64,
}

/// Zoom interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoomConfig {
    /// Whether zooming is enabled
    pub enabled: bool,
    
    /// Minimum zoom level
    pub min: f64,
    
    /// Maximum zoom level
    pub max: f64,
    
    /// Zoom sensitivity
    pub sensitivity: f64,
}

/// Selection interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Whether selection is enabled
    pub enabled: bool,
    
    /// Multiple selection mode
    pub multiple: bool,
    
    /// Selection highlight style
    pub highlight_style: String,
}

impl Default for NodeVisual {
    fn default() -> Self {
        Self {
            radius: 20.0,
            fill_color: "#4CAF50".to_string(),
            stroke_color: "#333333".to_string(),
            stroke_width: 2.0,
            opacity: 1.0,
            label: None,
            label_style: LabelStyle::default(),
            shape: NodeShape::Circle,
        }
    }
}

impl Default for EdgeVisual {
    fn default() -> Self {
        Self {
            stroke_color: "#999999".to_string(),
            stroke_width: 2.0,
            opacity: 0.6,
            line_style: LineStyle::Solid,
            markers: EdgeMarkers {
                start: None,
                end: None,
                size: 8.0,
            },
            label: None,
            label_style: LabelStyle::default(),
        }
    }
}

impl Default for LabelStyle {
    fn default() -> Self {
        Self {
            font_family: "Arial, sans-serif".to_string(),
            font_size: 12.0,
            font_weight: "normal".to_string(),
            color: "#333333".to_string(),
            background_color: None,
            alignment: TextAlignment::Center,
            visible: true,
        }
    }
}

impl Default for NodeInteractionState {
    fn default() -> Self {
        Self {
            is_hovered: false,
            is_selected: false,
            is_dragged: false,
            is_pinned: false,
            highlight: 0.0,
        }
    }
}

impl Default for AnimationData {
    fn default() -> Self {
        Self {
            transition_duration: 300,
            easing: EasingFunction::EaseInOut,
            animate_positions: true,
            animate_styles: true,
        }
    }
}

impl Default for FrameStyle {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            grid: None,
            theme: "default".to_string(),
            custom_styles: HashMap::new(),
        }
    }
}

impl Default for InteractionData {
    fn default() -> Self {
        Self {
            enabled: true,
            drag: DragConfig {
                enabled: true,
                fix_on_drag: false,
                sensitivity: 1.0,
            },
            zoom: ZoomConfig {
                enabled: true,
                min: 0.1,
                max: 10.0,
                sensitivity: 1.0,
            },
            selection: SelectionConfig {
                enabled: true,
                multiple: true,
                highlight_style: "outline".to_string(),
            },
        }
    }
}

impl VizFrame {
    /// Create a new frame from nodes and edges
    pub fn new(
        nodes: &[VizNode], 
        edges: &[VizEdge], 
        positions: &HashMap<String, Position>
    ) -> Self {
        let frame_nodes: Vec<FrameNode> = nodes
            .iter()
            .map(|node| FrameNode {
                id: node.id.clone(),
                position: positions.get(&node.id).cloned().unwrap_or(Position { x: 0.0, y: 0.0 }),
                previous_position: None,
                visual: NodeVisual::default(),
                interaction_state: NodeInteractionState::default(),
                attributes: node.attributes.iter()
                    .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or(serde_json::Value::Null)))
                    .collect(),
            })
            .collect();
        
        let frame_edges: Vec<FrameEdge> = edges
            .iter()
            .map(|edge| FrameEdge {
                id: edge.id.clone(),
                source: edge.source.clone(),
                target: edge.target.clone(),
                visual: EdgeVisual::default(),
                attributes: edge.attributes.iter()
                    .map(|(k, v)| (k.clone(), serde_json::to_value(v).unwrap_or(serde_json::Value::Null)))
                    .collect(),
            })
            .collect();
        
        // Calculate bounding box
        let bounds = Self::calculate_bounds(&frame_nodes);
        
        Self {
            metadata: FrameMetadata {
                frame_id: format!("frame_{}", fastrand::u64(..)),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                simulation_state: SimulationFrameState {
                    alpha: 1.0,
                    iteration: 0,
                    energy: 0.0,
                    is_running: false,
                    has_converged: false,
                },
                dimensions: FrameDimensions {
                    width: 800.0,
                    height: 600.0,
                    bounds,
                    zoom: 1.0,
                    pan: Position { x: 0.0, y: 0.0 },
                },
                performance: PerformanceMetrics {
                    generation_time_ms: 0.0,
                    physics_time_ms: 0.0,
                    rendering_time_ms: 0.0,
                    node_count: frame_nodes.len(),
                    edge_count: frame_edges.len(),
                },
            },
            nodes: frame_nodes,
            edges: frame_edges,
            animation: AnimationData::default(),
            style: FrameStyle::default(),
            interaction: InteractionData::default(),
        }
    }
    
    /// Calculate bounding box for all nodes
    fn calculate_bounds(nodes: &[FrameNode]) -> BoundingBox {
        if nodes.is_empty() {
            return BoundingBox {
                min_x: 0.0,
                max_x: 0.0,
                min_y: 0.0,
                max_y: 0.0,
            };
        }
        
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        
        for node in nodes {
            min_x = min_x.min(node.position.x);
            max_x = max_x.max(node.position.x);
            min_y = min_y.min(node.position.y);
            max_y = max_y.max(node.position.y);
        }
        
        BoundingBox { min_x, max_x, min_y, max_y }
    }
    
    /// Update frame with new positions
    pub fn update_positions(&mut self, positions: &HashMap<String, Position>) {
        for node in &mut self.nodes {
            if let Some(new_pos) = positions.get(&node.id) {
                node.previous_position = Some(node.position.clone());
                node.position = new_pos.clone();
            }
        }
        
        // Recalculate bounds
        self.metadata.dimensions.bounds = Self::calculate_bounds(&self.nodes);
    }
    
    /// Update simulation state
    pub fn update_simulation_state(&mut self, alpha: f64, iteration: usize, energy: f64, is_running: bool) {
        self.metadata.simulation_state = SimulationFrameState {
            alpha,
            iteration,
            energy,
            is_running,
            has_converged: alpha < 0.001,
        };
    }
    
    /// Convert to JSON for serialization
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Create from JSON
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}