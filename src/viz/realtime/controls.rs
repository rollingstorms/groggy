//! Interactive controls for real-time parameter adjustment
//!
//! Provides UI controls and parameter manipulation for live visualization tuning,
//! including sliders, dropdowns, and real-time feedback mechanisms.

use super::*;
use crate::storage::matrix::GraphMatrix;
use crate::viz::embeddings::{EmbeddingMethod, EnergyFunction};
use crate::viz::projection::{
    EasingFunction, HoneycombLayoutStrategy, InterpolationMethod, ProjectionMethod,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Interactive control panel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPanelConfig {
    /// Embedding controls configuration
    pub embedding_controls: EmbeddingControlsConfig,

    /// Projection controls configuration
    pub projection_controls: ProjectionControlsConfig,

    /// Animation controls configuration
    pub animation_controls: AnimationControlsConfig,

    /// Filter controls configuration
    pub filter_controls: FilterControlsConfig,

    /// Visual style controls
    pub style_controls: StyleControlsConfig,

    /// Performance controls
    pub performance_controls: PerformanceControlsConfig,

    /// Whether to show advanced controls
    pub show_advanced_controls: bool,

    /// Control panel layout
    pub layout: ControlPanelLayout,
}

/// Embedding parameter controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingControlsConfig {
    /// Enable embedding method selection
    pub enable_method_selection: bool,

    /// Enable dimension slider
    pub enable_dimension_control: bool,

    /// Dimension range (min, max, default)
    pub dimension_range: (usize, usize, usize),

    /// Enable energy function controls
    pub enable_energy_controls: bool,

    /// Energy function parameters
    pub energy_params: EnergyParameterControls,

    /// Enable spectral controls
    pub enable_spectral_controls: bool,

    /// Spectral parameter controls
    pub spectral_params: SpectralParameterControls,

    /// Real-time update mode
    pub realtime_updates: bool,

    /// Update debounce delay (ms)
    pub update_debounce_ms: u64,
}

/// Energy function parameter controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyParameterControls {
    /// Attraction strength control
    pub attraction_strength: ParameterControl<f64>,

    /// Repulsion strength control
    pub repulsion_strength: ParameterControl<f64>,

    /// Damping factor control
    pub damping_factor: ParameterControl<f64>,

    /// Spring constant control
    pub spring_constant: ParameterControl<f64>,

    /// Number of iterations control
    pub iterations: ParameterControl<usize>,

    /// Learning rate control
    pub learning_rate: ParameterControl<f64>,
}

/// Spectral embedding parameter controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralParameterControls {
    /// Whether to skip constant eigenvectors
    pub skip_constant: ParameterControl<bool>,

    /// Whether to use normalized Laplacian
    pub normalized: ParameterControl<bool>,

    /// Eigenvalue threshold
    pub eigenvalue_threshold: ParameterControl<f64>,
}

/// Projection parameter controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionControlsConfig {
    /// Enable projection method selection
    pub enable_method_selection: bool,

    /// Enable honeycomb controls
    pub enable_honeycomb_controls: bool,

    /// Honeycomb parameter controls
    pub honeycomb_params: HoneycombParameterControls,

    /// Enable quality controls
    pub enable_quality_controls: bool,

    /// Quality parameter controls
    pub quality_params: QualityParameterControls,

    /// Real-time updates
    pub realtime_updates: bool,

    /// Update debounce delay (ms)
    pub update_debounce_ms: u64,
}

/// Honeycomb layout parameter controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoneycombParameterControls {
    /// Cell size control
    pub cell_size: ParameterControl<f64>,

    /// Layout strategy selection
    pub layout_strategy: ParameterControl<HoneycombLayoutStrategy>,

    /// Grid padding control
    pub grid_padding: ParameterControl<f64>,

    /// Snap to centers toggle
    pub snap_to_centers: ParameterControl<bool>,
}

/// Quality parameter controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityParameterControls {
    /// Neighborhood preservation weight
    pub neighborhood_weight: ParameterControl<f64>,

    /// Distance preservation weight
    pub distance_weight: ParameterControl<f64>,

    /// Clustering preservation weight
    pub clustering_weight: ParameterControl<f64>,

    /// Number of neighbors for quality calculation
    pub k_neighbors: ParameterControl<usize>,

    /// Quality optimization toggle
    pub optimize_for_quality: ParameterControl<bool>,
}

/// Animation and interpolation controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationControlsConfig {
    /// Enable animation controls
    pub enable_animation_controls: bool,

    /// Animation speed control
    pub animation_speed: ParameterControl<f64>,

    /// Interpolation method selection
    pub interpolation_method: ParameterControl<InterpolationMethod>,

    /// Easing function selection
    pub easing_function: ParameterControl<EasingFunction>,

    /// Number of interpolation steps
    pub interpolation_steps: ParameterControl<usize>,

    /// Enable smooth transitions
    pub enable_smooth_transitions: ParameterControl<bool>,

    /// Transition duration control
    pub transition_duration: ParameterControl<f64>,
}

/// Filter controls configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterControlsConfig {
    /// Enable attribute filters
    pub enable_attribute_filters: bool,

    /// Available attribute filters
    pub available_attributes: Vec<AttributeFilterControl>,

    /// Enable degree filters
    pub enable_degree_filters: bool,

    /// Degree filter controls
    pub degree_filter: DegreeFilterControl,

    /// Enable community filters
    pub enable_community_filters: bool,

    /// Community filter controls
    pub community_filter: CommunityFilterControl,

    /// Enable spatial filters
    pub enable_spatial_filters: bool,

    /// Spatial filter controls
    pub spatial_filter: SpatialFilterControl,

    /// Filter combination mode
    pub combination_mode: FilterCombinationMode,
}

/// Visual style controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleControlsConfig {
    /// Node size control
    pub node_size: ParameterControl<f64>,

    /// Node color scheme
    pub node_color_scheme: ParameterControl<String>,

    /// Edge thickness control
    pub edge_thickness: ParameterControl<f64>,

    /// Edge opacity control
    pub edge_opacity: ParameterControl<f64>,

    /// Background color
    pub background_color: ParameterControl<String>,

    /// Grid visibility
    pub show_grid: ParameterControl<bool>,

    /// Label visibility
    pub show_labels: ParameterControl<bool>,

    /// Selection highlight style
    pub selection_style: SelectionStyleControl,
}

/// Performance tuning controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceControlsConfig {
    /// Target FPS control
    pub target_fps: ParameterControl<f64>,

    /// Quality level control
    pub quality_level: ParameterControl<f64>,

    /// Adaptive quality toggle
    pub adaptive_quality: ParameterControl<bool>,

    /// Level of detail controls
    pub level_of_detail: LevelOfDetailControls,

    /// Culling controls
    pub culling_controls: CullingControls,

    /// Memory management
    pub memory_controls: MemoryControls,
}

/// Generic parameter control definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterControl<T> {
    /// Current value
    pub value: T,

    /// Minimum value (for numeric types)
    pub min: Option<T>,

    /// Maximum value (for numeric types)
    pub max: Option<T>,

    /// Step size (for numeric types)
    pub step: Option<T>,

    /// Available options (for enum types)
    pub options: Option<Vec<T>>,

    /// Control type (slider, dropdown, toggle, etc.)
    pub control_type: ControlType,

    /// Display label
    pub label: String,

    /// Help text
    pub help_text: Option<String>,

    /// Whether control is enabled
    pub enabled: bool,

    /// Whether changes trigger immediate updates
    pub realtime_update: bool,
}

/// Control UI types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlType {
    /// Numeric slider
    Slider,
    /// Text input
    TextInput,
    /// Dropdown selection
    Dropdown,
    /// Toggle/checkbox
    Toggle,
    /// Color picker
    ColorPicker,
    /// Range slider (min/max)
    RangeSlider,
    /// Multi-select list
    MultiSelect,
    /// Button
    Button,
}

/// Attribute filter control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeFilterControl {
    /// Attribute name
    pub attribute_name: String,

    /// Attribute data type
    pub data_type: AttributeDataType,

    /// Filter control
    pub control: AttributeFilterType,

    /// Whether filter is currently active
    pub active: bool,
}

/// Attribute data types for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeDataType {
    Numeric,
    Categorical,
    Boolean,
    Text,
    Date,
}

/// Degree filter control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegreeFilterControl {
    /// Minimum degree control
    pub min_degree: ParameterControl<usize>,

    /// Maximum degree control
    pub max_degree: ParameterControl<usize>,

    /// Include isolated nodes
    pub include_isolated: ParameterControl<bool>,

    /// Active state
    pub active: bool,
}

/// Community filter control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityFilterControl {
    /// Community detection algorithm
    pub algorithm: ParameterControl<String>,

    /// Minimum community size
    pub min_community_size: ParameterControl<usize>,

    /// Selected communities
    pub selected_communities: Vec<usize>,

    /// Active state
    pub active: bool,
}

/// Spatial filter control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialFilterControl {
    /// Bounding box filter
    pub bounding_box: Option<BoundingBox>,

    /// Circular region filter
    pub circular_region: Option<CircularRegion>,

    /// Polygon filter
    pub polygon_region: Option<Vec<Position>>,

    /// Active state
    pub active: bool,
}

/// Circular filter region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularRegion {
    pub center: Position,
    pub radius: f64,
}

/// Filter combination modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterCombinationMode {
    And,
    Or,
    Custom,
}

/// Selection highlighting style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionStyleControl {
    /// Highlight color
    pub highlight_color: ParameterControl<String>,

    /// Highlight intensity
    pub highlight_intensity: ParameterControl<f64>,

    /// Neighbor highlighting
    pub highlight_neighbors: ParameterControl<bool>,

    /// Number of neighbor hops
    pub neighbor_hops: ParameterControl<usize>,

    /// Selection outline style
    pub outline_style: ParameterControl<String>,
}

/// Level of detail controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelOfDetailControls {
    /// Enable LOD
    pub enable_lod: ParameterControl<bool>,

    /// Distance thresholds for LOD levels
    pub distance_thresholds: Vec<f64>,

    /// Node detail levels
    pub node_detail_levels: Vec<NodeDetailLevel>,

    /// Edge detail levels
    pub edge_detail_levels: Vec<EdgeDetailLevel>,
}

/// Node detail levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeDetailLevel {
    Full,
    Simplified,
    IconOnly,
    Point,
    Hidden,
}

/// Edge detail levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeDetailLevel {
    Full,
    Simplified,
    Straight,
    Hidden,
}

/// Culling controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CullingControls {
    /// Enable frustum culling
    pub frustum_culling: ParameterControl<bool>,

    /// Enable occlusion culling
    pub occlusion_culling: ParameterControl<bool>,

    /// Enable distance culling
    pub distance_culling: ParameterControl<bool>,

    /// Culling distance threshold
    pub culling_distance: ParameterControl<f64>,
}

/// Memory management controls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryControls {
    /// Enable memory monitoring
    pub memory_monitoring: ParameterControl<bool>,

    /// Memory usage threshold
    pub memory_threshold: ParameterControl<f64>,

    /// Automatic garbage collection
    pub auto_gc: ParameterControl<bool>,

    /// Cache size limits
    pub cache_limits: CacheLimits,
}

/// Cache size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLimits {
    /// Position cache size
    pub position_cache_mb: f64,

    /// Embedding cache size
    pub embedding_cache_mb: f64,

    /// Texture cache size
    pub texture_cache_mb: f64,
}

/// Control panel layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPanelLayout {
    /// Panel position
    pub position: PanelPosition,

    /// Panel size
    pub size: PanelSize,

    /// Collapsible sections
    pub collapsible_sections: bool,

    /// Tab organization
    pub tab_organization: bool,

    /// Available tabs
    pub tabs: Vec<ControlTab>,
}

/// Panel position options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelPosition {
    Left,
    Right,
    Top,
    Bottom,
    Floating,
    Overlay,
}

/// Panel size configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelSize {
    pub width: f64,
    pub height: f64,
    pub resizable: bool,
    pub min_width: f64,
    pub min_height: f64,
}

/// Control panel tabs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlTab {
    pub id: String,
    pub title: String,
    pub icon: Option<String>,
    pub controls: Vec<String>,
    pub enabled: bool,
}

/// Interactive control manager
pub struct InteractiveControlManager {
    /// Control panel configuration
    config: ControlPanelConfig,

    /// Current parameter values
    parameters: HashMap<String, ParameterValue>,

    /// Parameter change listeners
    change_listeners: Vec<Box<dyn ParameterChangeListener>>,

    /// Debounce timers for real-time updates
    debounce_timers: HashMap<String, Instant>,

    /// Control state history for undo/redo
    history: ParameterHistory,
}

/// Parameter value wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    UInteger(usize),
    Boolean(bool),
    String(String),
    Array(Vec<ParameterValue>),
}

/// Parameter change listener trait
pub trait ParameterChangeListener: Send + Sync {
    fn on_parameter_changed(
        &self,
        name: &str,
        old_value: &ParameterValue,
        new_value: &ParameterValue,
    );
    fn on_batch_changed(&self, changes: &[(String, ParameterValue, ParameterValue)]);
}

/// Parameter change history for undo/redo
#[derive(Debug, Clone)]
pub struct ParameterHistory {
    /// History stack
    history: Vec<ParameterSnapshot>,

    /// Current position in history
    current_position: usize,

    /// Maximum history size
    max_history_size: usize,
}

/// Parameter snapshot for history
#[derive(Debug, Clone)]
pub struct ParameterSnapshot {
    /// Timestamp
    pub timestamp: Instant,

    /// Parameter values at this point
    pub parameters: HashMap<String, ParameterValue>,

    /// Description of change
    pub description: String,
}

impl InteractiveControlManager {
    /// Create new control manager
    pub fn new(config: ControlPanelConfig) -> Self {
        Self {
            config,
            parameters: HashMap::new(),
            change_listeners: Vec::new(),
            debounce_timers: HashMap::new(),
            history: ParameterHistory::new(),
        }
    }

    /// Initialize parameters from configuration
    pub fn initialize(&mut self) -> GraphResult<()> {
        // Initialize embedding parameters
        if self.config.embedding_controls.enable_dimension_control {
            let (_, _, default) = self.config.embedding_controls.dimension_range;
            self.set_parameter("embedding.dimensions", ParameterValue::UInteger(default))?;
        }

        // Initialize projection parameters
        if self.config.projection_controls.enable_honeycomb_controls {
            self.set_parameter(
                "projection.cell_size",
                ParameterValue::Float(
                    self.config
                        .projection_controls
                        .honeycomb_params
                        .cell_size
                        .value,
                ),
            )?;
        }

        // Initialize animation parameters
        if self.config.animation_controls.enable_animation_controls {
            self.set_parameter(
                "animation.speed",
                ParameterValue::Float(self.config.animation_controls.animation_speed.value),
            )?;
        }

        Ok(())
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: &str, value: ParameterValue) -> GraphResult<()> {
        let old_value = self.parameters.get(name).cloned();

        // Validate parameter value
        self.validate_parameter(name, &value)?;

        // Store new value
        self.parameters.insert(name.to_string(), value.clone());

        // Notify listeners
        if let Some(old_val) = old_value {
            for listener in &self.change_listeners {
                listener.on_parameter_changed(name, &old_val, &value);
            }
        }

        // Handle debounced updates
        if self.should_debounce_parameter(name) {
            self.debounce_timers
                .insert(name.to_string(), Instant::now());
        }

        Ok(())
    }

    /// Get a parameter value
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters.get(name)
    }

    /// Add parameter change listener
    pub fn add_change_listener(&mut self, listener: Box<dyn ParameterChangeListener>) {
        self.change_listeners.push(listener);
    }

    /// Apply a batch of parameter changes
    pub fn apply_batch_changes(
        &mut self,
        changes: Vec<(String, ParameterValue)>,
    ) -> GraphResult<()> {
        let mut change_records = Vec::new();

        for (name, value) in changes {
            let old_value = self.parameters.get(&name).cloned();
            self.validate_parameter(&name, &value)?;
            self.parameters.insert(name.clone(), value.clone());

            if let Some(old_val) = old_value {
                change_records.push((name, old_val, value));
            }
        }

        // Notify listeners of batch change
        for listener in &self.change_listeners {
            listener.on_batch_changed(&change_records);
        }

        Ok(())
    }

    /// Create parameter snapshot for history
    pub fn create_snapshot(&mut self, description: String) {
        self.history.add_snapshot(ParameterSnapshot {
            timestamp: Instant::now(),
            parameters: self.parameters.clone(),
            description,
        });
    }

    /// Undo last parameter change
    pub fn undo(&mut self) -> GraphResult<bool> {
        if let Some(snapshot) = self.history.undo() {
            self.parameters = snapshot.parameters.clone();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Redo last undone change
    pub fn redo(&mut self) -> GraphResult<bool> {
        if let Some(snapshot) = self.history.redo() {
            self.parameters = snapshot.parameters.clone();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Validate parameter value
    fn validate_parameter(&self, name: &str, value: &ParameterValue) -> GraphResult<()> {
        // TODO: Implement parameter validation based on config
        Ok(())
    }

    /// Check if parameter should be debounced
    fn should_debounce_parameter(&self, name: &str) -> bool {
        // Check configuration for debounce settings
        name.starts_with("embedding.") || name.starts_with("projection.")
    }
}

impl ParameterHistory {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            current_position: 0,
            max_history_size: 100,
        }
    }

    pub fn add_snapshot(&mut self, snapshot: ParameterSnapshot) {
        // Remove any history after current position
        self.history.truncate(self.current_position + 1);

        // Add new snapshot
        self.history.push(snapshot);
        self.current_position = self.history.len() - 1;

        // Limit history size
        if self.history.len() > self.max_history_size {
            self.history.remove(0);
            self.current_position -= 1;
        }
    }

    pub fn undo(&mut self) -> Option<&ParameterSnapshot> {
        if self.current_position > 0 {
            self.current_position -= 1;
            self.history.get(self.current_position)
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<&ParameterSnapshot> {
        if self.current_position < self.history.len() - 1 {
            self.current_position += 1;
            self.history.get(self.current_position)
        } else {
            None
        }
    }
}

// Default implementations for control configurations
impl Default for ControlPanelConfig {
    fn default() -> Self {
        Self {
            embedding_controls: EmbeddingControlsConfig::default(),
            projection_controls: ProjectionControlsConfig::default(),
            animation_controls: AnimationControlsConfig::default(),
            filter_controls: FilterControlsConfig::default(),
            style_controls: StyleControlsConfig::default(),
            performance_controls: PerformanceControlsConfig::default(),
            show_advanced_controls: false,
            layout: ControlPanelLayout::default(),
        }
    }
}

impl Default for EmbeddingControlsConfig {
    fn default() -> Self {
        Self {
            enable_method_selection: true,
            enable_dimension_control: true,
            dimension_range: (2, 20, 5),
            enable_energy_controls: true,
            energy_params: EnergyParameterControls::default(),
            enable_spectral_controls: true,
            spectral_params: SpectralParameterControls::default(),
            realtime_updates: false,
            update_debounce_ms: 500,
        }
    }
}

impl Default for EnergyParameterControls {
    fn default() -> Self {
        Self {
            attraction_strength: ParameterControl {
                value: 1.0,
                min: Some(0.1),
                max: Some(10.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Attraction Strength".to_string(),
                help_text: Some(
                    "Controls how strongly connected nodes attract each other".to_string(),
                ),
                enabled: true,
                realtime_update: false,
            },
            repulsion_strength: ParameterControl {
                value: 1.0,
                min: Some(0.1),
                max: Some(10.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Repulsion Strength".to_string(),
                help_text: Some("Controls how strongly nodes repel each other".to_string()),
                enabled: true,
                realtime_update: false,
            },
            damping_factor: ParameterControl {
                value: 0.9,
                min: Some(0.1),
                max: Some(1.0),
                step: Some(0.01),
                options: None,
                control_type: ControlType::Slider,
                label: "Damping Factor".to_string(),
                help_text: Some("Controls energy dissipation in the system".to_string()),
                enabled: true,
                realtime_update: true,
            },
            spring_constant: ParameterControl {
                value: 0.1,
                min: Some(0.01),
                max: Some(1.0),
                step: Some(0.01),
                options: None,
                control_type: ControlType::Slider,
                label: "Spring Constant".to_string(),
                help_text: Some("Spring stiffness for edge constraints".to_string()),
                enabled: true,
                realtime_update: false,
            },
            iterations: ParameterControl {
                value: 1000,
                min: Some(100),
                max: Some(10000),
                step: Some(100),
                options: None,
                control_type: ControlType::Slider,
                label: "Iterations".to_string(),
                help_text: Some("Number of optimization iterations".to_string()),
                enabled: true,
                realtime_update: false,
            },
            learning_rate: ParameterControl {
                value: 0.01,
                min: Some(0.001),
                max: Some(0.1),
                step: Some(0.001),
                options: None,
                control_type: ControlType::Slider,
                label: "Learning Rate".to_string(),
                help_text: Some("Step size for optimization updates".to_string()),
                enabled: true,
                realtime_update: false,
            },
        }
    }
}

// Additional default implementations would continue here...
// For brevity, implementing key ones:

impl Default for ControlPanelLayout {
    fn default() -> Self {
        Self {
            position: PanelPosition::Right,
            size: PanelSize {
                width: 300.0,
                height: 600.0,
                resizable: true,
                min_width: 250.0,
                min_height: 400.0,
            },
            collapsible_sections: true,
            tab_organization: true,
            tabs: vec![
                ControlTab {
                    id: "embedding".to_string(),
                    title: "Embedding".to_string(),
                    icon: Some("layers".to_string()),
                    controls: vec!["embedding.*".to_string()],
                    enabled: true,
                },
                ControlTab {
                    id: "projection".to_string(),
                    title: "Projection".to_string(),
                    icon: Some("map".to_string()),
                    controls: vec!["projection.*".to_string()],
                    enabled: true,
                },
                ControlTab {
                    id: "animation".to_string(),
                    title: "Animation".to_string(),
                    icon: Some("play".to_string()),
                    controls: vec!["animation.*".to_string()],
                    enabled: true,
                },
                ControlTab {
                    id: "filters".to_string(),
                    title: "Filters".to_string(),
                    icon: Some("filter".to_string()),
                    controls: vec!["filter.*".to_string()],
                    enabled: true,
                },
                ControlTab {
                    id: "style".to_string(),
                    title: "Style".to_string(),
                    icon: Some("palette".to_string()),
                    controls: vec!["style.*".to_string()],
                    enabled: true,
                },
                ControlTab {
                    id: "performance".to_string(),
                    title: "Performance".to_string(),
                    icon: Some("speed".to_string()),
                    controls: vec!["performance.*".to_string()],
                    enabled: true,
                },
            ],
        }
    }
}

// Implement more default traits as needed...
impl Default for SpectralParameterControls {
    fn default() -> Self {
        Self {
            skip_constant: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Skip Constant Eigenvectors".to_string(),
                help_text: Some("Skip eigenvalues near zero (constant eigenvectors)".to_string()),
                enabled: true,
                realtime_update: false,
            },
            normalized: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Normalized Laplacian".to_string(),
                help_text: Some("Use normalized Laplacian matrix".to_string()),
                enabled: true,
                realtime_update: false,
            },
            eigenvalue_threshold: ParameterControl {
                value: 1e-6,
                min: Some(1e-10),
                max: Some(1e-3),
                step: Some(1e-7),
                options: None,
                control_type: ControlType::Slider,
                label: "Eigenvalue Threshold".to_string(),
                help_text: Some("Threshold for filtering small eigenvalues".to_string()),
                enabled: true,
                realtime_update: false,
            },
        }
    }
}

impl Default for ProjectionControlsConfig {
    fn default() -> Self {
        Self {
            enable_method_selection: true,
            enable_honeycomb_controls: true,
            honeycomb_params: HoneycombParameterControls::default(),
            enable_quality_controls: true,
            quality_params: QualityParameterControls::default(),
            realtime_updates: true,
            update_debounce_ms: 300,
        }
    }
}

impl Default for HoneycombParameterControls {
    fn default() -> Self {
        Self {
            cell_size: ParameterControl {
                value: 40.0,
                min: Some(10.0),
                max: Some(200.0),
                step: Some(5.0),
                options: None,
                control_type: ControlType::Slider,
                label: "Cell Size".to_string(),
                help_text: Some("Size of hexagonal cells in pixels".to_string()),
                enabled: true,
                realtime_update: true,
            },
            layout_strategy: ParameterControl {
                value: HoneycombLayoutStrategy::Spiral,
                min: None,
                max: None,
                step: None,
                options: Some(vec![
                    HoneycombLayoutStrategy::Spiral,
                    HoneycombLayoutStrategy::DensityBased,
                    HoneycombLayoutStrategy::DistancePreserving,
                ]),
                control_type: ControlType::Dropdown,
                label: "Layout Strategy".to_string(),
                help_text: Some("Method for placing nodes on honeycomb grid".to_string()),
                enabled: true,
                realtime_update: false,
            },
            grid_padding: ParameterControl {
                value: 20.0,
                min: Some(0.0),
                max: Some(100.0),
                step: Some(5.0),
                options: None,
                control_type: ControlType::Slider,
                label: "Grid Padding".to_string(),
                help_text: Some("Padding around the honeycomb grid".to_string()),
                enabled: true,
                realtime_update: true,
            },
            snap_to_centers: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Snap to Centers".to_string(),
                help_text: Some("Snap nodes to exact hexagon centers".to_string()),
                enabled: true,
                realtime_update: true,
            },
        }
    }
}

impl Default for QualityParameterControls {
    fn default() -> Self {
        Self {
            neighborhood_weight: ParameterControl {
                value: 0.4,
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Neighborhood Weight".to_string(),
                help_text: Some("Weight for neighborhood preservation metric".to_string()),
                enabled: true,
                realtime_update: true,
            },
            distance_weight: ParameterControl {
                value: 0.4,
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Distance Weight".to_string(),
                help_text: Some("Weight for distance preservation metric".to_string()),
                enabled: true,
                realtime_update: true,
            },
            clustering_weight: ParameterControl {
                value: 0.2,
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Clustering Weight".to_string(),
                help_text: Some("Weight for clustering preservation metric".to_string()),
                enabled: true,
                realtime_update: true,
            },
            k_neighbors: ParameterControl {
                value: 10,
                min: Some(3),
                max: Some(50),
                step: Some(1),
                options: None,
                control_type: ControlType::Slider,
                label: "K Neighbors".to_string(),
                help_text: Some("Number of neighbors for quality calculations".to_string()),
                enabled: true,
                realtime_update: false,
            },
            optimize_for_quality: ParameterControl {
                value: false,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Optimize for Quality".to_string(),
                help_text: Some("Enable quality-based optimization".to_string()),
                enabled: true,
                realtime_update: false,
            },
        }
    }
}

// Continue with other default implementations as needed...

impl Default for AnimationControlsConfig {
    fn default() -> Self {
        Self {
            enable_animation_controls: true,
            animation_speed: ParameterControl {
                value: 1.0,
                min: Some(0.1),
                max: Some(5.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Animation Speed".to_string(),
                help_text: Some("Speed multiplier for animations".to_string()),
                enabled: true,
                realtime_update: true,
            },
            interpolation_method: ParameterControl {
                value: InterpolationMethod::Linear,
                min: None,
                max: None,
                step: None,
                options: Some(vec![
                    InterpolationMethod::Linear,
                    InterpolationMethod::Spline,
                ]),
                control_type: ControlType::Dropdown,
                label: "Interpolation Method".to_string(),
                help_text: Some("Method for interpolating between positions".to_string()),
                enabled: true,
                realtime_update: false,
            },
            easing_function: ParameterControl {
                value: EasingFunction::EaseInOut,
                min: None,
                max: None,
                step: None,
                options: Some(vec![
                    EasingFunction::Linear,
                    EasingFunction::EaseIn,
                    EasingFunction::EaseOut,
                    EasingFunction::EaseInOut,
                    EasingFunction::Bounce,
                    EasingFunction::Elastic,
                ]),
                control_type: ControlType::Dropdown,
                label: "Easing Function".to_string(),
                help_text: Some("Easing function for smooth animations".to_string()),
                enabled: true,
                realtime_update: true,
            },
            interpolation_steps: ParameterControl {
                value: 30,
                min: Some(5),
                max: Some(100),
                step: Some(5),
                options: None,
                control_type: ControlType::Slider,
                label: "Interpolation Steps".to_string(),
                help_text: Some("Number of interpolation steps for smooth transitions".to_string()),
                enabled: true,
                realtime_update: false,
            },
            enable_smooth_transitions: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Smooth Transitions".to_string(),
                help_text: Some("Enable smooth transitions between states".to_string()),
                enabled: true,
                realtime_update: true,
            },
            transition_duration: ParameterControl {
                value: 1.0,
                min: Some(0.1),
                max: Some(5.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Transition Duration".to_string(),
                help_text: Some("Duration of transitions in seconds".to_string()),
                enabled: true,
                realtime_update: true,
            },
        }
    }
}

impl Default for FilterControlsConfig {
    fn default() -> Self {
        Self {
            enable_attribute_filters: true,
            available_attributes: vec![],
            enable_degree_filters: true,
            degree_filter: DegreeFilterControl::default(),
            enable_community_filters: true,
            community_filter: CommunityFilterControl::default(),
            enable_spatial_filters: true,
            spatial_filter: SpatialFilterControl::default(),
            combination_mode: FilterCombinationMode::And,
        }
    }
}

impl Default for DegreeFilterControl {
    fn default() -> Self {
        Self {
            min_degree: ParameterControl {
                value: 0,
                min: Some(0),
                max: Some(100),
                step: Some(1),
                options: None,
                control_type: ControlType::Slider,
                label: "Minimum Degree".to_string(),
                help_text: Some("Minimum node degree to display".to_string()),
                enabled: true,
                realtime_update: true,
            },
            max_degree: ParameterControl {
                value: 100,
                min: Some(1),
                max: Some(1000),
                step: Some(1),
                options: None,
                control_type: ControlType::Slider,
                label: "Maximum Degree".to_string(),
                help_text: Some("Maximum node degree to display".to_string()),
                enabled: true,
                realtime_update: true,
            },
            include_isolated: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Include Isolated Nodes".to_string(),
                help_text: Some("Show nodes with no connections".to_string()),
                enabled: true,
                realtime_update: true,
            },
            active: false,
        }
    }
}

impl Default for CommunityFilterControl {
    fn default() -> Self {
        Self {
            algorithm: ParameterControl {
                value: "louvain".to_string(),
                min: None,
                max: None,
                step: None,
                options: Some(vec![
                    "louvain".to_string(),
                    "modularity".to_string(),
                    "leiden".to_string(),
                ]),
                control_type: ControlType::Dropdown,
                label: "Detection Algorithm".to_string(),
                help_text: Some("Community detection algorithm to use".to_string()),
                enabled: true,
                realtime_update: false,
            },
            min_community_size: ParameterControl {
                value: 3,
                min: Some(1),
                max: Some(100),
                step: Some(1),
                options: None,
                control_type: ControlType::Slider,
                label: "Minimum Community Size".to_string(),
                help_text: Some("Minimum size for communities to display".to_string()),
                enabled: true,
                realtime_update: true,
            },
            selected_communities: vec![],
            active: false,
        }
    }
}

impl Default for SpatialFilterControl {
    fn default() -> Self {
        Self {
            bounding_box: None,
            circular_region: None,
            polygon_region: None,
            active: false,
        }
    }
}

impl Default for StyleControlsConfig {
    fn default() -> Self {
        Self {
            node_size: ParameterControl {
                value: 8.0,
                min: Some(2.0),
                max: Some(50.0),
                step: Some(1.0),
                options: None,
                control_type: ControlType::Slider,
                label: "Node Size".to_string(),
                help_text: Some("Size of nodes in pixels".to_string()),
                enabled: true,
                realtime_update: true,
            },
            node_color_scheme: ParameterControl {
                value: "default".to_string(),
                min: None,
                max: None,
                step: None,
                options: Some(vec![
                    "default".to_string(),
                    "viridis".to_string(),
                    "plasma".to_string(),
                    "community".to_string(),
                    "degree".to_string(),
                ]),
                control_type: ControlType::Dropdown,
                label: "Color Scheme".to_string(),
                help_text: Some("Color scheme for nodes".to_string()),
                enabled: true,
                realtime_update: true,
            },
            edge_thickness: ParameterControl {
                value: 1.0,
                min: Some(0.1),
                max: Some(10.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Edge Thickness".to_string(),
                help_text: Some("Thickness of edges".to_string()),
                enabled: true,
                realtime_update: true,
            },
            edge_opacity: ParameterControl {
                value: 0.7,
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Edge Opacity".to_string(),
                help_text: Some("Opacity of edges".to_string()),
                enabled: true,
                realtime_update: true,
            },
            background_color: ParameterControl {
                value: "#ffffff".to_string(),
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::ColorPicker,
                label: "Background Color".to_string(),
                help_text: Some("Background color of the visualization".to_string()),
                enabled: true,
                realtime_update: true,
            },
            show_grid: ParameterControl {
                value: false,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Show Grid".to_string(),
                help_text: Some("Display honeycomb grid lines".to_string()),
                enabled: true,
                realtime_update: true,
            },
            show_labels: ParameterControl {
                value: false,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Show Labels".to_string(),
                help_text: Some("Display node labels".to_string()),
                enabled: true,
                realtime_update: true,
            },
            selection_style: SelectionStyleControl::default(),
        }
    }
}

impl Default for SelectionStyleControl {
    fn default() -> Self {
        Self {
            highlight_color: ParameterControl {
                value: "#ff6b6b".to_string(),
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::ColorPicker,
                label: "Highlight Color".to_string(),
                help_text: Some("Color for highlighting selected nodes".to_string()),
                enabled: true,
                realtime_update: true,
            },
            highlight_intensity: ParameterControl {
                value: 0.8,
                min: Some(0.0),
                max: Some(1.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Highlight Intensity".to_string(),
                help_text: Some("Intensity of selection highlighting".to_string()),
                enabled: true,
                realtime_update: true,
            },
            highlight_neighbors: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Highlight Neighbors".to_string(),
                help_text: Some("Highlight neighboring nodes of selection".to_string()),
                enabled: true,
                realtime_update: true,
            },
            neighbor_hops: ParameterControl {
                value: 2,
                min: Some(1),
                max: Some(5),
                step: Some(1),
                options: None,
                control_type: ControlType::Slider,
                label: "Neighbor Hops".to_string(),
                help_text: Some("Number of hops for neighbor highlighting".to_string()),
                enabled: true,
                realtime_update: true,
            },
            outline_style: ParameterControl {
                value: "solid".to_string(),
                min: None,
                max: None,
                step: None,
                options: Some(vec![
                    "solid".to_string(),
                    "dashed".to_string(),
                    "dotted".to_string(),
                    "glow".to_string(),
                ]),
                control_type: ControlType::Dropdown,
                label: "Outline Style".to_string(),
                help_text: Some("Style of selection outline".to_string()),
                enabled: true,
                realtime_update: true,
            },
        }
    }
}

impl Default for PerformanceControlsConfig {
    fn default() -> Self {
        Self {
            target_fps: ParameterControl {
                value: 60.0,
                min: Some(15.0),
                max: Some(120.0),
                step: Some(15.0),
                options: None,
                control_type: ControlType::Slider,
                label: "Target FPS".to_string(),
                help_text: Some("Target frame rate for visualization".to_string()),
                enabled: true,
                realtime_update: true,
            },
            quality_level: ParameterControl {
                value: 1.0,
                min: Some(0.1),
                max: Some(1.0),
                step: Some(0.1),
                options: None,
                control_type: ControlType::Slider,
                label: "Quality Level".to_string(),
                help_text: Some("Overall quality level (affects detail)".to_string()),
                enabled: true,
                realtime_update: true,
            },
            adaptive_quality: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Adaptive Quality".to_string(),
                help_text: Some("Automatically adjust quality for performance".to_string()),
                enabled: true,
                realtime_update: true,
            },
            level_of_detail: LevelOfDetailControls::default(),
            culling_controls: CullingControls::default(),
            memory_controls: MemoryControls::default(),
        }
    }
}

impl Default for LevelOfDetailControls {
    fn default() -> Self {
        Self {
            enable_lod: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Enable LOD".to_string(),
                help_text: Some("Enable level of detail optimization".to_string()),
                enabled: true,
                realtime_update: true,
            },
            distance_thresholds: vec![100.0, 500.0, 1000.0],
            node_detail_levels: vec![
                NodeDetailLevel::Full,
                NodeDetailLevel::Simplified,
                NodeDetailLevel::IconOnly,
                NodeDetailLevel::Point,
            ],
            edge_detail_levels: vec![
                EdgeDetailLevel::Full,
                EdgeDetailLevel::Simplified,
                EdgeDetailLevel::Straight,
                EdgeDetailLevel::Hidden,
            ],
        }
    }
}

impl Default for CullingControls {
    fn default() -> Self {
        Self {
            frustum_culling: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Frustum Culling".to_string(),
                help_text: Some("Cull objects outside view frustum".to_string()),
                enabled: true,
                realtime_update: true,
            },
            occlusion_culling: ParameterControl {
                value: false,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Occlusion Culling".to_string(),
                help_text: Some("Cull objects hidden behind others".to_string()),
                enabled: true,
                realtime_update: true,
            },
            distance_culling: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Distance Culling".to_string(),
                help_text: Some("Cull objects beyond distance threshold".to_string()),
                enabled: true,
                realtime_update: true,
            },
            culling_distance: ParameterControl {
                value: 1000.0,
                min: Some(100.0),
                max: Some(5000.0),
                step: Some(100.0),
                options: None,
                control_type: ControlType::Slider,
                label: "Culling Distance".to_string(),
                help_text: Some("Distance threshold for culling".to_string()),
                enabled: true,
                realtime_update: true,
            },
        }
    }
}

impl Default for MemoryControls {
    fn default() -> Self {
        Self {
            memory_monitoring: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Memory Monitoring".to_string(),
                help_text: Some("Monitor memory usage".to_string()),
                enabled: true,
                realtime_update: true,
            },
            memory_threshold: ParameterControl {
                value: 512.0,
                min: Some(128.0),
                max: Some(2048.0),
                step: Some(64.0),
                options: None,
                control_type: ControlType::Slider,
                label: "Memory Threshold (MB)".to_string(),
                help_text: Some("Memory usage threshold for optimization".to_string()),
                enabled: true,
                realtime_update: true,
            },
            auto_gc: ParameterControl {
                value: true,
                min: None,
                max: None,
                step: None,
                options: None,
                control_type: ControlType::Toggle,
                label: "Auto Garbage Collection".to_string(),
                help_text: Some("Automatically trigger garbage collection".to_string()),
                enabled: true,
                realtime_update: true,
            },
            cache_limits: CacheLimits {
                position_cache_mb: 64.0,
                embedding_cache_mb: 128.0,
                texture_cache_mb: 256.0,
            },
        }
    }
}

/// Honeycomb-specific n-dimensional rotation controls via canvas and node dragging
#[derive(Debug, Clone)]
pub struct HoneycombInteractionController {
    /// Current rotation matrix for n-dimensional space
    rotation_matrix: GraphMatrix<f64>,

    /// Dimensions of the embedding space
    embedding_dimensions: usize,

    /// Canvas drag state
    canvas_drag_state: CanvasDragState,

    /// Node drag state
    node_drag_state: NodeDragState,

    /// Sensitivity settings
    rotation_sensitivity: f64,

    /// Whether to enable smooth rotation interpolation
    smooth_rotation: bool,

    /// Rotation accumulation for momentum
    rotation_momentum: RotationMomentum,
}

/// Canvas dragging state for n-dimensional rotation
#[derive(Debug, Clone)]
pub struct CanvasDragState {
    /// Whether canvas is currently being dragged
    is_dragging: bool,

    /// Start position of drag (screen coordinates)
    drag_start: (f64, f64),

    /// Current drag position (screen coordinates)
    current_position: (f64, f64),

    /// Previous drag position for delta calculation
    previous_position: (f64, f64),

    /// Which mouse button is being used
    mouse_button: MouseButton,

    /// Modifier keys pressed during drag
    modifiers: ModifierKeys,

    /// Rotation axes being manipulated
    active_rotation_axes: Vec<(usize, usize)>, // Pairs of dimension indices
}

/// Node dragging state for direct embedding manipulation
#[derive(Debug, Clone)]
pub struct NodeDragState {
    /// Currently dragged node ID
    dragged_node: Option<usize>,

    /// Original position in n-dimensional space
    original_position: Option<Vec<f64>>,

    /// Current position during drag
    current_position: Option<Vec<f64>>,

    /// Drag start position (screen coordinates)
    drag_start_screen: (f64, f64),

    /// Current screen position
    current_screen_position: (f64, f64),

    /// Whether to constrain to current hyperplane
    constrain_to_hyperplane: bool,

    /// Influence radius for neighboring nodes
    influence_radius: f64,

    /// Neighboring nodes affected by drag
    influenced_nodes: HashMap<usize, f64>, // node_id -> influence_weight
}

/// Rotation momentum for smooth interactions
#[derive(Debug, Clone)]
pub struct RotationMomentum {
    /// Angular velocity for each rotation axis pair
    angular_velocities: HashMap<(usize, usize), f64>,

    /// Decay factor for momentum
    decay_factor: f64,

    /// Minimum velocity threshold
    min_velocity_threshold: f64,

    /// Last update timestamp
    last_update: Instant,
}

/// Mouse button identification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Other(u8),
}

/// Modifier keys state
#[derive(Debug, Clone)]
pub struct ModifierKeys {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

impl HoneycombInteractionController {
    /// Create new honeycomb interaction controller
    pub fn new(embedding_dimensions: usize) -> GraphResult<Self> {
        let rotation_matrix = GraphMatrix::<f64>::identity(embedding_dimensions)?;

        Ok(Self {
            rotation_matrix,
            embedding_dimensions,
            canvas_drag_state: CanvasDragState::new(),
            node_drag_state: NodeDragState::new(),
            rotation_sensitivity: 0.01,
            smooth_rotation: true,
            rotation_momentum: RotationMomentum::new(),
        })
    }

    /// Handle canvas drag start
    pub fn start_canvas_drag(
        &mut self,
        screen_pos: (f64, f64),
        button: MouseButton,
        modifiers: ModifierKeys,
    ) {
        self.canvas_drag_state.is_dragging = true;
        self.canvas_drag_state.drag_start = screen_pos;
        self.canvas_drag_state.current_position = screen_pos;
        self.canvas_drag_state.previous_position = screen_pos;
        self.canvas_drag_state.mouse_button = button;
        self.canvas_drag_state.modifiers = modifiers.clone();

        // Determine rotation axes based on modifiers and button
        self.canvas_drag_state.active_rotation_axes = self.get_rotation_axes(&button, &modifiers);
    }

    /// Handle canvas drag update
    pub fn update_canvas_drag(&mut self, screen_pos: (f64, f64)) -> Option<GraphMatrix<f64>> {
        if !self.canvas_drag_state.is_dragging {
            return None;
        }

        let delta_x = screen_pos.0 - self.canvas_drag_state.previous_position.0;
        let delta_y = screen_pos.1 - self.canvas_drag_state.previous_position.1;

        self.canvas_drag_state.previous_position = self.canvas_drag_state.current_position;
        self.canvas_drag_state.current_position = screen_pos;

        // Apply rotation based on drag delta
        self.apply_rotation_delta(delta_x, delta_y)
    }

    /// Handle canvas drag end
    pub fn end_canvas_drag(&mut self) {
        if self.canvas_drag_state.is_dragging {
            // Calculate final momentum
            let total_delta_x =
                self.canvas_drag_state.current_position.0 - self.canvas_drag_state.drag_start.0;
            let total_delta_y =
                self.canvas_drag_state.current_position.1 - self.canvas_drag_state.drag_start.1;

            self.update_momentum(total_delta_x, total_delta_y);
        }

        self.canvas_drag_state.is_dragging = false;
        self.canvas_drag_state.active_rotation_axes.clear();
    }

    /// Handle node drag start
    pub fn start_node_drag(
        &mut self,
        node_id: usize,
        node_position: Vec<f64>,
        screen_pos: (f64, f64),
    ) -> GraphResult<()> {
        if node_position.len() != self.embedding_dimensions {
            return Err(GraphError::InvalidInput(format!(
                "Node position dimensions ({}) don't match embedding dimensions ({})",
                node_position.len(),
                self.embedding_dimensions
            )));
        }

        self.node_drag_state.dragged_node = Some(node_id);
        self.node_drag_state.original_position = Some(node_position.clone());
        self.node_drag_state.current_position = Some(node_position);
        self.node_drag_state.drag_start_screen = screen_pos;
        self.node_drag_state.current_screen_position = screen_pos;

        Ok(())
    }

    /// Handle node drag update
    pub fn update_node_drag(&mut self, screen_pos: (f64, f64)) -> Option<(usize, Vec<f64>)> {
        if let Some(node_id) = self.node_drag_state.dragged_node {
            self.node_drag_state.current_screen_position = screen_pos;

            // Convert screen delta to n-dimensional position change
            if let Some(new_position) = self.screen_delta_to_nd_position(screen_pos) {
                self.node_drag_state.current_position = Some(new_position.clone());
                return Some((node_id, new_position));
            }
        }
        None
    }

    /// Handle node drag end
    pub fn end_node_drag(&mut self) {
        self.node_drag_state.dragged_node = None;
        self.node_drag_state.original_position = None;
        self.node_drag_state.current_position = None;
        self.node_drag_state.influenced_nodes.clear();
    }

    /// Get current rotation matrix
    pub fn get_rotation_matrix(&self) -> &GraphMatrix<f64> {
        &self.rotation_matrix
    }

    /// Update momentum and apply continuous rotation
    pub fn update_momentum(&mut self, delta_x: f64, delta_y: f64) {
        let now = Instant::now();
        let dt = if self.rotation_momentum.last_update == Instant::now() {
            Duration::from_millis(16) // Assume 60 FPS
        } else {
            now.duration_since(self.rotation_momentum.last_update)
        };

        let dt_secs = dt.as_secs_f64();

        // Update angular velocities for active rotation axes
        for &(dim1, dim2) in &self.canvas_drag_state.active_rotation_axes {
            let angular_velocity = (delta_x + delta_y) * self.rotation_sensitivity / dt_secs;
            self.rotation_momentum
                .angular_velocities
                .insert((dim1, dim2), angular_velocity);
        }

        self.rotation_momentum.last_update = now;
    }

    /// Apply continuous momentum rotation
    pub fn apply_momentum(&mut self, dt: Duration) -> Option<GraphMatrix<f64>> {
        let dt_secs = dt.as_secs_f64();
        let mut has_momentum = false;
        let mut rotations_to_apply = Vec::new();

        // Collect rotations to apply while iterating mutably
        for (&(dim1, dim2), velocity) in &mut self.rotation_momentum.angular_velocities {
            if velocity.abs() > self.rotation_momentum.min_velocity_threshold {
                // Calculate rotation for this axis pair
                let angle = *velocity * dt_secs;
                rotations_to_apply.push((dim1, dim2, angle));
                has_momentum = true;

                // Apply decay
                *velocity *= self.rotation_momentum.decay_factor;
            }
        }

        // Apply collected rotations
        for (dim1, dim2, angle) in rotations_to_apply {
            if let Ok(rotation) = self.create_rotation_matrix(dim1, dim2, angle) {
                self.rotation_matrix = self
                    .rotation_matrix
                    .multiply(&rotation)
                    .unwrap_or(self.rotation_matrix.clone());
            }
        }

        if has_momentum {
            Some(self.rotation_matrix.clone())
        } else {
            None
        }
    }

    /// Create rotation matrix for specific dimension pair
    fn create_rotation_matrix(
        &self,
        dim1: usize,
        dim2: usize,
        angle: f64,
    ) -> GraphResult<GraphMatrix<f64>> {
        let mut rotation = GraphMatrix::<f64>::identity(self.embedding_dimensions)?;

        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        // Set rotation elements for the specified plane
        rotation.set(dim1, dim1, cos_angle)?;
        rotation.set(dim1, dim2, -sin_angle)?;
        rotation.set(dim2, dim1, sin_angle)?;
        rotation.set(dim2, dim2, cos_angle)?;

        Ok(rotation)
    }

    /// Determine rotation axes based on mouse button and modifiers
    fn get_rotation_axes(
        &self,
        button: &MouseButton,
        modifiers: &ModifierKeys,
    ) -> Vec<(usize, usize)> {
        match button {
            MouseButton::Left => {
                if modifiers.shift {
                    // Rotate in dimensions 0-1
                    vec![(0, 1)]
                } else if modifiers.ctrl {
                    // Rotate in dimensions 2-3 if available
                    if self.embedding_dimensions > 3 {
                        vec![(2, 3)]
                    } else {
                        vec![(0, 2)]
                    }
                } else {
                    // Default rotation in first two dimensions
                    vec![(0, 1)]
                }
            }
            MouseButton::Right => {
                // Rotate in higher dimensions
                if self.embedding_dimensions > 2 {
                    vec![(0, 2), (1, 3)]
                } else {
                    vec![(0, 1)]
                }
            }
            MouseButton::Middle => {
                // Multi-dimensional rotation
                let mut axes = Vec::new();
                for i in 0..self.embedding_dimensions {
                    for j in (i + 1)..self.embedding_dimensions {
                        if axes.len() < 2 {
                            // Limit to 2 simultaneous rotations
                            axes.push((i, j));
                        }
                    }
                }
                axes
            }
            _ => vec![(0, 1)],
        }
    }

    /// Apply rotation delta from canvas drag
    fn apply_rotation_delta(&mut self, delta_x: f64, delta_y: f64) -> Option<GraphMatrix<f64>> {
        let mut updated = false;

        for &(dim1, dim2) in &self.canvas_drag_state.active_rotation_axes {
            let angle_x = delta_x * self.rotation_sensitivity;
            let angle_y = delta_y * self.rotation_sensitivity;

            // Apply rotation for X movement
            if let Ok(rotation_x) = self.create_rotation_matrix(dim1, dim2, angle_x) {
                self.rotation_matrix = self
                    .rotation_matrix
                    .multiply(&rotation_x)
                    .unwrap_or(self.rotation_matrix.clone());
                updated = true;
            }

            // Apply rotation for Y movement (different axis if available)
            let alt_dim2 = if dim2 + 1 < self.embedding_dimensions {
                dim2 + 1
            } else {
                (dim2 + 1) % self.embedding_dimensions
            };
            if alt_dim2 != dim1 && alt_dim2 != dim2 {
                if let Ok(rotation_y) = self.create_rotation_matrix(dim1, alt_dim2, angle_y) {
                    self.rotation_matrix = self
                        .rotation_matrix
                        .multiply(&rotation_y)
                        .unwrap_or(self.rotation_matrix.clone());
                    updated = true;
                }
            }
        }

        if updated {
            Some(self.rotation_matrix.clone())
        } else {
            None
        }
    }

    /// Convert screen delta to n-dimensional position change
    fn screen_delta_to_nd_position(&self, screen_pos: (f64, f64)) -> Option<Vec<f64>> {
        if let Some(original_pos) = &self.node_drag_state.original_position {
            let screen_delta_x = screen_pos.0 - self.node_drag_state.drag_start_screen.0;
            let screen_delta_y = screen_pos.1 - self.node_drag_state.drag_start_screen.1;

            // Convert screen delta to n-dimensional change
            // For now, map screen X to first two dimensions alternately, Y to remaining dimensions
            let mut new_position = original_pos.clone();

            if new_position.len() >= 2 {
                new_position[0] += screen_delta_x * 0.01; // Scale factor
                new_position[1] += screen_delta_y * 0.01;

                // Optionally affect higher dimensions based on modifiers
                if self.node_drag_state.constrain_to_hyperplane {
                    // Keep higher dimensions constant
                } else if new_position.len() > 2 {
                    // Distribute change across higher dimensions
                    let higher_dim_factor = 0.005;
                    for i in 2..new_position.len() {
                        new_position[i] += (screen_delta_x + screen_delta_y) * higher_dim_factor
                            / (i as f64 - 1.0);
                    }
                }
            }

            Some(new_position)
        } else {
            None
        }
    }
}

impl CanvasDragState {
    fn new() -> Self {
        Self {
            is_dragging: false,
            drag_start: (0.0, 0.0),
            current_position: (0.0, 0.0),
            previous_position: (0.0, 0.0),
            mouse_button: MouseButton::Left,
            modifiers: ModifierKeys {
                shift: false,
                ctrl: false,
                alt: false,
                meta: false,
            },
            active_rotation_axes: Vec::new(),
        }
    }
}

impl NodeDragState {
    fn new() -> Self {
        Self {
            dragged_node: None,
            original_position: None,
            current_position: None,
            drag_start_screen: (0.0, 0.0),
            current_screen_position: (0.0, 0.0),
            constrain_to_hyperplane: false,
            influence_radius: 50.0,
            influenced_nodes: HashMap::new(),
        }
    }
}

impl RotationMomentum {
    fn new() -> Self {
        Self {
            angular_velocities: HashMap::new(),
            decay_factor: 0.95,
            min_velocity_threshold: 0.001,
            last_update: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_panel_config_creation() {
        let config = ControlPanelConfig::default();
        assert!(config.embedding_controls.enable_method_selection);
        assert_eq!(config.embedding_controls.dimension_range.2, 5); // default dimensions
    }

    #[test]
    fn test_interactive_control_manager() {
        let config = ControlPanelConfig::default();
        let mut manager = InteractiveControlManager::new(config);

        assert!(manager.initialize().is_ok());

        // Test parameter setting
        let result = manager.set_parameter("test.param", ParameterValue::Float(42.0));
        assert!(result.is_ok());

        // Test parameter retrieval
        if let Some(ParameterValue::Float(value)) = manager.get_parameter("test.param") {
            assert_eq!(*value, 42.0);
        } else {
            panic!("Parameter not found or wrong type");
        }
    }

    #[test]
    fn test_parameter_history() {
        let mut history = ParameterHistory::new();

        let snapshot1 = ParameterSnapshot {
            timestamp: Instant::now(),
            parameters: HashMap::new(),
            description: "Initial state".to_string(),
        };

        history.add_snapshot(snapshot1);
        assert_eq!(history.current_position, 0);

        // Test that undo/redo work correctly
        assert!(history.undo().is_none()); // Can't undo from initial position
    }

    #[test]
    fn test_parameter_control_defaults() {
        let control = ParameterControl {
            value: 1.0f64,
            min: Some(0.0),
            max: Some(10.0),
            step: Some(0.1),
            options: None,
            control_type: ControlType::Slider,
            label: "Test Control".to_string(),
            help_text: None,
            enabled: true,
            realtime_update: false,
        };

        assert_eq!(control.value, 1.0);
        assert_eq!(control.min, Some(0.0));
        assert!(control.enabled);
    }
}
