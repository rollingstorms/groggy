//! Real-time visualization engine
//!
//! The main engine that orchestrates the complete real-time visualization pipeline,
//! combining Phase 1 (embeddings), Phase 2 (projections), and Phase 3 (streaming).

use super::*;
use crate::api::graph::Graph;
use crate::errors::{GraphResult, GraphError};
use crate::storage::matrix::GraphMatrix;
use crate::viz::embeddings::{EmbeddingEngine, EmbeddingMethod, GraphEmbeddingExt};
use crate::viz::projection::{ProjectionEngine, ProjectionEngineFactory, GraphProjectionExt};
use crate::viz::streaming::data_source::Position;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::{interval, sleep};

/// Main real-time visualization engine
#[derive(Debug)]
pub struct RealTimeVizEngine {
    /// Configuration for the visualization
    config: RealTimeVizConfig,

    /// Current visualization state
    state: Arc<Mutex<RealTimeVizState>>,

    /// Reference to the graph being visualized
    graph: Arc<Mutex<Graph>>,

    /// Channel for sending position updates
    position_sender: Option<mpsc::UnboundedSender<PositionUpdate>>,

    /// Channel for receiving control commands
    control_receiver: Option<mpsc::UnboundedReceiver<ControlCommand>>,

    /// Performance monitor
    performance_monitor: PerformanceMonitor,

    /// Incremental update manager
    incremental_manager: IncrementalUpdateManager,
    /// Honeycomb-specific interaction controller (for n-dimensional rotation)
    honeycomb_controller: Option<crate::viz::realtime::controls::HoneycombInteractionController>,

    /// Active animation controllers
    animation_controllers: Vec<AnimationController>,

    /// Frame timing history for FPS calculation
    frame_times: Vec<Duration>,

    /// Whether the engine is currently running
    is_running: bool,
}

/// Position update message for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    /// Node ID
    pub node_id: usize,

    /// New position
    pub position: Position,

    /// Update timestamp
    pub timestamp: u64,

    /// Update type (full, incremental, interpolated)
    pub update_type: PositionUpdateType,

    /// Quality metrics for this update
    pub quality: Option<PositionQuality>,
}

/// Types of position updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionUpdateType {
    /// Full recomputation from scratch
    Full,
    /// Incremental update from graph changes
    Incremental,
    /// Interpolated position during animation
    Interpolated,
    /// Predicted position for smooth motion
    Predicted,
}

/// Quality metrics for position updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionQuality {
    /// Local neighborhood preservation (0.0 - 1.0)
    pub neighborhood_preservation: f64,

    /// Distance preservation from original embedding
    pub distance_preservation: f64,

    /// Stress metric for this position
    pub stress: f64,

    /// Confidence in this position (0.0 - 1.0)
    pub confidence: f64,
}

/// Control commands for interactive manipulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    /// Update embedding parameters
    UpdateEmbeddingParams {
        method: EmbeddingMethod,
        dimensions: usize,
    },

    /// Update projection parameters
    UpdateProjectionParams {
        method: ProjectionMethod,
        honeycomb_config: HoneycombConfig,
    },

    /// Update quality settings
    UpdateQualitySettings {
        config: QualityConfig,
    },

    /// Update animation settings
    UpdateAnimationSettings {
        config: InterpolationConfig,
    },

    /// Apply filters
    ApplyFilter {
        filter_type: FilterType,
        parameters: HashMap<String, serde_json::Value>,
    },

    /// Select nodes
    SelectNodes {
        node_ids: Vec<usize>,
        selection_mode: SelectionMode,
    },

    /// Zoom to region
    ZoomToRegion {
        bounds: BoundingBox,
        animation_duration: Duration,
    },

    /// Pan view
    PanView {
        delta_x: f64,
        delta_y: f64,
    },

    /// Pause/resume animation
    ToggleAnimation {
        pause: bool,
    },

    /// Reset view to default
    ResetView,

    /// Add nodes to graph
    AddNodes {
        node_data: Vec<HashMap<String, serde_json::Value>>,
    },

    /// Add edges to graph
    AddEdges {
        edge_data: Vec<(usize, usize, HashMap<String, serde_json::Value>)>,
    },

    /// Remove nodes from graph
    RemoveNodes {
        node_ids: Vec<usize>,
    },

    /// Remove edges from graph
    RemoveEdges {
        edge_ids: Vec<usize>,
    },
}

/// Filter types for real-time filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    Attribute,
    Degree,
    Community,
    Spatial,
    Temporal,
}

/// Node selection modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMode {
    Replace,
    Add,
    Remove,
    Toggle,
}

impl RealTimeVizEngine {
    /// Create a new real-time visualization engine
    pub fn new(graph: Graph, config: RealTimeVizConfig) -> Self {
        let state = Arc::new(Mutex::new(RealTimeVizState {
            positions: Vec::new(),
            embedding: None,
            animation_state: AnimationState::default(),
            performance: PerformanceMetrics::default(),
            selection: SelectionState::default(),
            filters: FilterState::default(),
            last_update: Instant::now(),
        }));

        Self {
            config,
            state,
            graph: Arc::new(Mutex::new(graph)),
            position_sender: None,
            control_receiver: None,
            performance_monitor: PerformanceMonitor::new(),
            incremental_manager: IncrementalUpdateManager::new(),
            honeycomb_controller: None, // Will be initialized in initialize() if needed
            animation_controllers: Vec::new(),
            frame_times: Vec::new(),
            is_running: false,
        }
    }

    /// Initialize the visualization engine
    pub async fn initialize(&mut self) -> GraphResult<()> {
        // Perform initial embedding and projection
        self.compute_initial_layout().await?;

        // Initialize performance monitoring
        self.performance_monitor.start()?;

        // Setup communication channels
        let (position_tx, position_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();

        self.position_sender = Some(position_tx);
        self.control_receiver = Some(control_rx);

        // Initialize incremental update manager
        let graph = self.graph.lock().unwrap();
        self.incremental_manager.initialize(&*graph)?;

        // Initialize honeycomb interaction controller if using honeycomb projection
        if matches!(self.config.projection_config.method, crate::viz::projection::ProjectionMethod::UMAP { .. }) {
            // For honeycomb layouts, we use multi-dimensional embeddings
            let embedding_dimensions = self.config.embedding_config.dimensions;
            if embedding_dimensions > 2 {
                let honeycomb_controller = crate::viz::realtime::controls::HoneycombInteractionController::new(embedding_dimensions)?;
                self.honeycomb_controller = Some(honeycomb_controller);
            }
        }

        Ok(())
    }

    /// Start the real-time visualization loop
    pub async fn start(&mut self) -> GraphResult<()> {
        if self.is_running {
            return Err(GraphError::InvalidInput(
                "Engine is already running".to_string()
            ));
        }

        self.is_running = true;

        // Create frame timer based on target FPS
        let frame_duration = Duration::from_secs_f64(1.0 / self.config.realtime_config.target_fps);
        let mut frame_timer = interval(frame_duration);

        // Main visualization loop
        while self.is_running {
            let frame_start = Instant::now();

            // Process control commands
            let mut commands_to_process = Vec::new();
            if let Some(ref mut receiver) = self.control_receiver {
                while let Ok(command) = receiver.try_recv() {
                    commands_to_process.push(command);
                }
            }

            for command in commands_to_process {
                self.process_control_command(command).await?;
            }

            // Update visualization if needed
            if self.needs_update()? {
                self.update_visualization().await?;
            }

            // Update animations
            self.update_animations().await?;

            // Update performance metrics
            let frame_time = frame_start.elapsed();
            self.update_performance_metrics(frame_time)?;

            // Adaptive quality control
            if self.config.performance_config.enable_auto_quality_adaptation {
                self.adapt_quality_settings(frame_time)?;
            }

            // Send position updates if needed
            self.broadcast_position_updates().await?;

            // Wait for next frame
            frame_timer.tick().await;
        }

        Ok(())
    }

    /// Stop the visualization engine
    pub fn stop(&mut self) {
        self.is_running = false;
        self.performance_monitor.stop();
    }

    /// Get a reference to the graph
    pub fn graph(&self) -> &Arc<Mutex<Graph>> {
        &self.graph
    }

    /// Compute initial layout (embeddings + projections)
    async fn compute_initial_layout(&mut self) -> GraphResult<()> {
        let embedding_start = Instant::now();

        // Phase 1: Compute embeddings
        let graph = self.graph.lock().unwrap();
        let embedding = graph.compute_embedding(&self.config.embedding_config)?;
        drop(graph);

        let embedding_time = embedding_start.elapsed();

        let projection_start = Instant::now();

        // Phase 2: Project to 2D honeycomb coordinates
        let graph = self.graph.lock().unwrap();
        let positions = graph.project_to_honeycomb(&embedding, &self.config.projection_config)?;
        drop(graph);

        let projection_time = projection_start.elapsed();

        // Update state
        {
            let mut state = self.state.lock().unwrap();
            state.embedding = Some(embedding);
            state.positions = positions;
            state.performance.last_embedding_time_ms = embedding_time.as_secs_f64() * 1000.0;
            state.performance.last_projection_time_ms = projection_time.as_secs_f64() * 1000.0;
            state.last_update = Instant::now();
        }

        Ok(())
    }

    /// Process a control command
    async fn process_control_command(&mut self, command: ControlCommand) -> GraphResult<()> {
        match command {
            ControlCommand::UpdateEmbeddingParams { method, dimensions } => {
                self.config.embedding_config.method = method;
                self.config.embedding_config.dimensions = dimensions;
                self.trigger_full_recomputation().await?;
            }

            ControlCommand::UpdateProjectionParams { method, honeycomb_config } => {
                self.config.projection_config.method = method;
                self.config.projection_config.honeycomb_config = honeycomb_config;
                self.trigger_projection_recomputation().await?;
            }

            ControlCommand::UpdateQualitySettings { config } => {
                self.config.projection_config.quality_config = config;
                self.trigger_quality_recomputation().await?;
            }

            ControlCommand::ApplyFilter { filter_type, parameters } => {
                self.apply_realtime_filter(filter_type, parameters).await?;
            }

            ControlCommand::SelectNodes { node_ids, selection_mode } => {
                self.update_node_selection(node_ids, selection_mode).await?;
            }

            ControlCommand::ZoomToRegion { bounds, animation_duration } => {
                self.animate_zoom_to_region(bounds, animation_duration).await?;
            }

            ControlCommand::PanView { delta_x, delta_y } => {
                self.pan_view(delta_x, delta_y).await?;
            }

            ControlCommand::AddNodes { node_data } => {
                self.add_nodes_incrementally(node_data).await?;
            }

            ControlCommand::AddEdges { edge_data } => {
                self.add_edges_incrementally(edge_data).await?;
            }

            ControlCommand::RemoveNodes { node_ids } => {
                self.remove_nodes_incrementally(node_ids).await?;
            }

            ControlCommand::ResetView => {
                self.reset_view_to_default().await?;
            }

            _ => {
                // Handle other commands...
            }
        }

        Ok(())
    }

    /// Check if visualization needs updating
    fn needs_update(&self) -> GraphResult<bool> {
        let state = self.state.lock().unwrap();

        // Check if incremental updates are pending
        if self.incremental_manager.has_pending_updates() {
            return Ok(true);
        }

        // Check if animation is active
        if state.animation_state.is_animating {
            return Ok(true);
        }

        // Check if filters changed
        if state.filters.transition_state.is_transitioning {
            return Ok(true);
        }

        Ok(false)
    }

    /// Update visualization (main update logic)
    async fn update_visualization(&mut self) -> GraphResult<()> {
        // Process incremental updates if enabled
        if self.config.realtime_config.enable_incremental_updates &&
           self.incremental_manager.has_pending_updates() {
            self.process_incremental_updates().await?;
        }

        // Update filter transitions
        self.update_filter_transitions().await?;

        // Update any other dynamic aspects
        self.update_dynamic_aspects().await?;

        Ok(())
    }

    /// Update animations
    async fn update_animations(&mut self) -> GraphResult<()> {
        let mut state = self.state.lock().unwrap();

        if state.animation_state.is_animating {
            let elapsed = state.animation_state.start_time.elapsed();
            let progress = elapsed.as_secs_f64() / state.animation_state.duration.as_secs_f64();

            if progress >= 1.0 {
                // Animation complete
                state.positions = state.animation_state.target_positions.clone();
                state.animation_state.is_animating = false;
                state.animation_state.progress = 1.0;
            } else {
                // Interpolate positions
                state.animation_state.progress = progress;
                state.positions = self.interpolate_positions(
                    &state.animation_state.source_positions,
                    &state.animation_state.target_positions,
                    progress,
                    &state.animation_state.easing_function,
                )?;
            }
        }

        Ok(())
    }

    /// Interpolate between two sets of positions
    fn interpolate_positions(
        &self,
        source: &[Position],
        target: &[Position],
        progress: f64,
        easing_function: &str,
    ) -> GraphResult<Vec<Position>> {
        if source.len() != target.len() {
            return Err(GraphError::InvalidInput(
                "Source and target position arrays must have same length".to_string()
            ));
        }

        // Apply easing function
        let eased_progress = self.apply_easing_function(progress, easing_function)?;

        let mut interpolated = Vec::with_capacity(source.len());
        for (src, tgt) in source.iter().zip(target.iter()) {
            let x = src.x + (tgt.x - src.x) * eased_progress;
            let y = src.y + (tgt.y - src.y) * eased_progress;
            interpolated.push(Position { x, y });
        }

        Ok(interpolated)
    }

    /// Apply easing function to progress value
    fn apply_easing_function(&self, progress: f64, function: &str) -> GraphResult<f64> {
        match function {
            "linear" => Ok(progress),
            "ease-in" => Ok(progress * progress),
            "ease-out" => Ok(1.0 - (1.0 - progress) * (1.0 - progress)),
            "ease-in-out" => {
                if progress < 0.5 {
                    Ok(2.0 * progress * progress)
                } else {
                    Ok(1.0 - 2.0 * (1.0 - progress) * (1.0 - progress))
                }
            }
            "bounce" => {
                // Simplified bounce easing
                let n1 = 7.5625;
                let d1 = 2.75;
                let p = progress;

                if p < 1.0 / d1 {
                    Ok(n1 * p * p)
                } else if p < 2.0 / d1 {
                    let p2 = p - 1.5 / d1;
                    Ok(n1 * p2 * p2 + 0.75)
                } else if p < 2.5 / d1 {
                    let p2 = p - 2.25 / d1;
                    Ok(n1 * p2 * p2 + 0.9375)
                } else {
                    let p2 = p - 2.625 / d1;
                    Ok(n1 * p2 * p2 + 0.984375)
                }
            }
            _ => Err(GraphError::InvalidInput(
                format!("Unknown easing function: {}", function)
            )),
        }
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, frame_time: Duration) -> GraphResult<()> {
        // Add frame time to history
        self.frame_times.push(frame_time);

        // Keep only recent frame times
        let max_history = self.config.performance_config.frame_time_history_size;
        if self.frame_times.len() > max_history {
            self.frame_times.drain(0..self.frame_times.len() - max_history);
        }

        // Calculate metrics
        let average_frame_time = self.frame_times.iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>() / self.frame_times.len() as f64;

        let current_fps = if average_frame_time > 0.0 {
            1.0 / average_frame_time
        } else {
            0.0
        };

        // Update state
        {
            let mut state = self.state.lock().unwrap();
            state.performance.current_fps = current_fps;
            state.performance.average_frame_time_ms = average_frame_time * 1000.0;

            // Update node/edge counts
            let graph = self.graph.lock().unwrap();
            state.performance.active_node_count = graph.space().node_count();
            state.performance.active_edge_count = graph.space().edge_count();
        }

        Ok(())
    }

    /// Adapt quality settings based on performance
    fn adapt_quality_settings(&mut self, frame_time: Duration) -> GraphResult<()> {
        let target_frame_time = Duration::from_secs_f64(
            1.0 / self.config.realtime_config.target_fps
        );

        let performance_ratio = frame_time.as_secs_f64() / target_frame_time.as_secs_f64();
        let sensitivity = self.config.performance_config.quality_adaptation_sensitivity;

        if performance_ratio > 1.2 {
            // Performance is poor, reduce quality
            let quality_reduction = (performance_ratio - 1.0) * sensitivity;
            self.reduce_quality_settings(quality_reduction)?;
        } else if performance_ratio < 0.8 {
            // Performance is good, can increase quality
            let quality_increase = (1.0 - performance_ratio) * sensitivity * 0.5;
            self.increase_quality_settings(quality_increase)?;
        }

        Ok(())
    }

    /// Reduce quality settings for better performance
    fn reduce_quality_settings(&mut self, reduction_factor: f64) -> GraphResult<()> {
        // Reduce honeycomb cell density
        self.config.projection_config.honeycomb_config.cell_size *= 1.0 + reduction_factor * 0.1;

        // Reduce interpolation steps
        let current_steps = self.config.projection_config.interpolation_config.steps;
        let new_steps = ((current_steps as f64) * (1.0 - reduction_factor * 0.2)).max(5.0) as usize;
        self.config.projection_config.interpolation_config.steps = new_steps;

        // Update quality level
        {
            let mut state = self.state.lock().unwrap();
            state.performance.current_quality_level =
                (state.performance.current_quality_level - reduction_factor * 0.1).max(0.1);
        }

        Ok(())
    }

    /// Increase quality settings when performance allows
    fn increase_quality_settings(&mut self, increase_factor: f64) -> GraphResult<()> {
        // Increase honeycomb cell density (reduce cell size)
        self.config.projection_config.honeycomb_config.cell_size *= 1.0 - increase_factor * 0.05;

        // Increase interpolation steps
        let current_steps = self.config.projection_config.interpolation_config.steps;
        let new_steps = ((current_steps as f64) * (1.0 + increase_factor * 0.1)).min(100.0) as usize;
        self.config.projection_config.interpolation_config.steps = new_steps;

        // Update quality level
        {
            let mut state = self.state.lock().unwrap();
            state.performance.current_quality_level =
                (state.performance.current_quality_level + increase_factor * 0.05).min(1.0);
        }

        Ok(())
    }

    /// Broadcast position updates to connected clients
    async fn broadcast_position_updates(&self) -> GraphResult<()> {
        if let Some(ref sender) = self.position_sender {
            let state = self.state.lock().unwrap();

            // Create position updates
            for (node_id, position) in state.positions.iter().enumerate() {
                let update = PositionUpdate {
                    node_id,
                    position: position.clone(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    update_type: if state.animation_state.is_animating {
                        PositionUpdateType::Interpolated
                    } else {
                        PositionUpdateType::Full
                    },
                    quality: Some(PositionQuality {
                        neighborhood_preservation: 0.8, // TODO: Compute actual metrics
                        distance_preservation: 0.7,
                        stress: 0.3,
                        confidence: state.performance.current_quality_level,
                    }),
                };

                if let Err(_) = sender.send(update) {
                    // Client disconnected, continue with other updates
                }
            }
        }

        Ok(())
    }

    // Placeholder methods for incremental updates and other features
    async fn trigger_full_recomputation(&mut self) -> GraphResult<()> {
        self.compute_initial_layout().await
    }

    async fn trigger_projection_recomputation(&mut self) -> GraphResult<()> {
        // TODO: Implement projection-only recomputation
        Ok(())
    }

    async fn trigger_quality_recomputation(&mut self) -> GraphResult<()> {
        // TODO: Implement quality-only recomputation
        Ok(())
    }

    async fn apply_realtime_filter(&mut self, _filter_type: FilterType, _parameters: HashMap<String, serde_json::Value>) -> GraphResult<()> {
        // TODO: Implement real-time filtering
        Ok(())
    }

    async fn update_node_selection(&mut self, _node_ids: Vec<usize>, _mode: SelectionMode) -> GraphResult<()> {
        // TODO: Implement node selection
        Ok(())
    }

    async fn animate_zoom_to_region(&mut self, _bounds: BoundingBox, _duration: Duration) -> GraphResult<()> {
        // TODO: Implement zoom animation
        Ok(())
    }

    async fn pan_view(&mut self, _delta_x: f64, _delta_y: f64) -> GraphResult<()> {
        // TODO: Implement view panning
        Ok(())
    }

    async fn add_nodes_incrementally(&mut self, _node_data: Vec<HashMap<String, serde_json::Value>>) -> GraphResult<()> {
        // TODO: Implement incremental node addition
        Ok(())
    }

    async fn add_edges_incrementally(&mut self, _edge_data: Vec<(usize, usize, HashMap<String, serde_json::Value>)>) -> GraphResult<()> {
        // TODO: Implement incremental edge addition
        Ok(())
    }

    async fn remove_nodes_incrementally(&mut self, _node_ids: Vec<usize>) -> GraphResult<()> {
        // TODO: Implement incremental node removal
        Ok(())
    }

    async fn reset_view_to_default(&mut self) -> GraphResult<()> {
        // TODO: Implement view reset
        Ok(())
    }

    async fn process_incremental_updates(&mut self) -> GraphResult<()> {
        // TODO: Implement incremental update processing
        Ok(())
    }

    async fn update_filter_transitions(&mut self) -> GraphResult<()> {
        // TODO: Implement filter transition updates
        Ok(())
    }

    async fn update_dynamic_aspects(&mut self) -> GraphResult<()> {
        // TODO: Implement other dynamic aspect updates
        Ok(())
    }
}

/// Performance monitoring helper
#[derive(Debug)]
pub struct PerformanceMonitor {
    start_time: Option<Instant>,
    monitoring_enabled: bool,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: None,
            monitoring_enabled: false,
        }
    }

    pub fn start(&mut self) -> GraphResult<()> {
        self.start_time = Some(Instant::now());
        self.monitoring_enabled = true;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.monitoring_enabled = false;
        self.start_time = None;
    }
}

/// Incremental update manager
#[derive(Debug)]
pub struct IncrementalUpdateManager {
    initialized: bool,
    pending_updates: Vec<GraphChange>,
}

#[derive(Debug, Clone)]
pub struct GraphChange {
    pub change_type: GraphChangeType,
    pub node_id: Option<usize>,
    pub edge_id: Option<usize>,
    pub timestamp: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GraphChangeType {
    NodeAdded,
    NodeRemoved,
    NodeAttributeChanged,
    EdgeAdded,
    EdgeRemoved,
    EdgeAttributeChanged,
}

impl IncrementalUpdateManager {
    pub fn new() -> Self {
        Self {
            initialized: false,
            pending_updates: Vec::new(),
        }
    }

    pub fn initialize(&mut self, _graph: &Graph) -> GraphResult<()> {
        self.initialized = true;
        Ok(())
    }

    pub fn has_pending_updates(&self) -> bool {
        !self.pending_updates.is_empty()
    }

}

/// Animation controller for smooth transitions
#[derive(Debug)]
pub struct AnimationController {
    pub animation_type: AnimationType,
    pub start_time: Instant,
    pub duration: Duration,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub enum AnimationType {
    PositionTransition,
    ZoomTransition,
    FilterTransition,
    SelectionTransition,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let graph = Graph::new();
        let config = RealTimeVizConfig::default();
        let engine = RealTimeVizEngine::new(graph, config);
        assert!(!engine.is_running);
    }

    #[test]
    fn test_easing_functions() {
        let graph = Graph::new();
        let config = RealTimeVizConfig::default();
        let engine = RealTimeVizEngine::new(graph, config);

        assert_eq!(engine.apply_easing_function(0.0, "linear").unwrap(), 0.0);
        assert_eq!(engine.apply_easing_function(1.0, "linear").unwrap(), 1.0);
        assert_eq!(engine.apply_easing_function(0.5, "linear").unwrap(), 0.5);

        assert_eq!(engine.apply_easing_function(0.0, "ease-in").unwrap(), 0.0);
        assert_eq!(engine.apply_easing_function(1.0, "ease-in").unwrap(), 1.0);
    }

    #[test]
    fn test_position_interpolation() {
        let graph = Graph::new();
        let config = RealTimeVizConfig::default();
        let engine = RealTimeVizEngine::new(graph, config);

        let source = vec![Position { x: 0.0, y: 0.0 }, Position { x: 10.0, y: 10.0 }];
        let target = vec![Position { x: 20.0, y: 20.0 }, Position { x: 30.0, y: 30.0 }];

        let interpolated = engine.interpolate_positions(&source, &target, 0.5, "linear").unwrap();
        assert_eq!(interpolated[0].x, 10.0);
        assert_eq!(interpolated[0].y, 10.0);
        assert_eq!(interpolated[1].x, 20.0);
        assert_eq!(interpolated[1].y, 20.0);
    }
}