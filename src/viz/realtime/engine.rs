//! Real-time visualization engine
//!
//! The main engine that orchestrates the complete real-time visualization pipeline,
//! combining Phase 1 (embeddings), Phase 2 (projections), and Phase 3 (streaming).

use super::*;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::advanced_matrix::numeric_type::NumericType;
use crate::storage::matrix::GraphMatrix;
use crate::viz::embeddings::{EmbeddingEngine, EmbeddingMethod, GraphEmbeddingExt};
use crate::viz::projection::{GraphProjectionExt, ProjectionEngine, ProjectionEngineFactory};
use crate::viz::realtime::accessor::{
    Edge, EngineSnapshot, EngineUpdate, GraphMeta, GraphPatch, Node, NodePosition,
    PositionsPayload, UpdateEnvelope,
};
use crate::viz::realtime::engine_sync::EngineSyncManager;
use crate::viz::realtime::interaction::{
    CanvasDragPolicy, GlobeController, HoneycombController, InteractionController, NodeDragEvent,
    NodeDragPolicy, PanController, PointerEvent, ViewState2D, ViewState3D, WheelEvent,
};
use crate::viz::streaming::data_source::Position;
use serde_json::json;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, sleep};

/// Main real-time visualization engine
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

    /// Broadcast channel for engine-generated updates
    update_broadcaster: broadcast::Sender<EngineUpdate>,

    /// Synchronization manager for ordered updates and coalescing
    sync_manager: EngineSyncManager,

    /// Performance monitor
    performance_monitor: PerformanceMonitor,

    /// Incremental update manager
    incremental_manager: IncrementalUpdateManager,
    /// Active interaction controller for canvas/node gestures
    active_controller: Box<dyn crate::viz::realtime::interaction::InteractionController>,
    /// Current node drag policy (layout specific)
    node_drag_policy: crate::viz::realtime::interaction::NodeDragPolicy,
    /// Canvas drag policy (documentation/hints)
    canvas_drag_policy: crate::viz::realtime::interaction::CanvasDragPolicy,
    /// Pending parameter deltas to include with the next envelope broadcast
    pending_params_changed: Option<HashMap<String, serde_json::Value>>,
    /// Active animation controllers
    animation_controllers: Vec<AnimationController>,

    /// Frame timing history for FPS calculation
    frame_times: Vec<Duration>,

    /// Whether the engine is currently running
    is_running: bool,
}

impl fmt::Debug for RealTimeVizEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealTimeVizEngine")
            .field("config", &self.config)
            .field("is_running", &self.is_running)
            .finish()
    }
}

static FRAME_ID_SEQ: AtomicU64 = AtomicU64::new(1);

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
    UpdateQualitySettings { config: QualityConfig },

    /// Update animation settings
    UpdateAnimationSettings { config: InterpolationConfig },

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
    PanView { delta_x: f64, delta_y: f64 },

    /// Pause/resume animation
    ToggleAnimation { pause: bool },

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
    RemoveNodes { node_ids: Vec<usize> },

    /// Remove edges from graph
    RemoveEdges { edge_ids: Vec<usize> },

    /// Swap interaction controller mode
    SetInteractionController { mode: String },

    /// Pointer gesture event from client
    Pointer { event: PointerEvent },

    /// Wheel gesture event from client
    Wheel { event: WheelEvent },

    /// Node drag gesture event
    NodeDrag { event: NodeDragEvent },

    /// Rotate embedding axes (N-D)
    RotateEmbedding {
        axis_i: usize,
        axis_j: usize,
        radians: f64,
    },

    /// Set view rotation explicitly
    SetViewRotation { radians: f64 },
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
            node_index: HashMap::new(),
            embedding: None,
            animation_state: AnimationState::default(),
            performance: PerformanceMetrics::default(),
            selection: SelectionState::default(),
            filters: FilterState::default(),
            last_update: Instant::now(),
            needs_position_update: false,
            current_layout_algorithm: "honeycomb".to_string(),
            current_layout_params: std::collections::HashMap::new(),
        }));

        // Create broadcast channel for engine updates
        let (update_broadcaster, _) = broadcast::channel(1000);

        Self {
            config,
            state,
            graph: Arc::new(Mutex::new(graph)),
            position_sender: None,
            control_receiver: None,
            update_broadcaster,
            sync_manager: EngineSyncManager::new(),
            performance_monitor: PerformanceMonitor::new(),
            incremental_manager: IncrementalUpdateManager::new(),
            active_controller: Box::new(PanController::new()),
            node_drag_policy: NodeDragPolicy::Free,
            canvas_drag_policy: CanvasDragPolicy::PanZoomRotate2D,
            pending_params_changed: None,
            animation_controllers: Vec::new(),
            frame_times: Vec::new(),
            is_running: false,
        }
    }

    /// Initialize the visualization engine
    pub async fn initialize(&mut self) -> GraphResult<()> {
        // Perform initial embedding and projection
        self.compute_initial_layout().await?;

        let initial_layout = {
            let state = self.state.lock().unwrap();
            state.current_layout_algorithm.clone()
        };
        self.configure_controller_for_layout(&initial_layout);
        self.broadcast_view_state()?;

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

        Ok(())
    }

    /// Start the real-time visualization loop
    pub async fn start(&mut self) -> GraphResult<()> {
        if self.is_running {
            return Err(GraphError::InvalidInput(
                "Engine is already running".to_string(),
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
            if self
                .config
                .performance_config
                .enable_auto_quality_adaptation
            {
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

        // Phase 2: Project to 2D coordinates using configured layout algorithm
        let positions = self.apply_layout_algorithm(&embedding)?;

        let projection_time = projection_start.elapsed();

        // Update state
        {
            let mut state = self.state.lock().unwrap();
            state.embedding = Some(embedding);
            state.positions = positions;
            state.performance.last_embedding_time_ms = embedding_time.as_secs_f64() * 1000.0;
            state.performance.last_projection_time_ms = projection_time.as_secs_f64() * 1000.0;
            state.last_update = Instant::now();
            state.needs_position_update = true;
        }

        // Notify downstream consumers with a unified envelope snapshot
        self.broadcast_envelope(None, None, None, true)?;

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

            ControlCommand::UpdateProjectionParams {
                method,
                honeycomb_config,
            } => {
                self.config.projection_config.method = method;
                self.config.projection_config.honeycomb_config = honeycomb_config;
                self.trigger_projection_recomputation().await?;
            }

            ControlCommand::UpdateQualitySettings { config } => {
                self.config.projection_config.quality_config = config;
                self.trigger_quality_recomputation().await?;
            }

            ControlCommand::ApplyFilter {
                filter_type,
                parameters,
            } => {
                self.apply_realtime_filter(filter_type, parameters).await?;
            }

            ControlCommand::SelectNodes {
                node_ids,
                selection_mode,
            } => {
                self.update_node_selection(node_ids, selection_mode).await?;
            }

            ControlCommand::ZoomToRegion {
                bounds,
                animation_duration,
            } => {
                self.animate_zoom_to_region(bounds, animation_duration)
                    .await?;
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

            ControlCommand::SetInteractionController { mode } => {
                self.set_interaction_controller(&mode);
                self.broadcast_view_state()?;
            }

            ControlCommand::Pointer { event } => {
                self.active_controller.on_pointer(event);
                self.broadcast_view_state()?;
            }

            ControlCommand::Wheel { event } => {
                self.active_controller.on_wheel(event);
                self.broadcast_view_state()?;
            }

            ControlCommand::NodeDrag { event } => {
                self.handle_node_drag_event(event)?;
            }

            ControlCommand::RotateEmbedding {
                axis_i,
                axis_j,
                radians,
            } => {
                eprintln!(
                    "🔁 DEBUG: Received RotateEmbedding command axis=({}, {}) radians={}",
                    axis_i, axis_j, radians
                );
                // TODO: Integrate with N-D embedding rotation pipeline
            }

            ControlCommand::SetViewRotation { radians } => {
                eprintln!("🔁 DEBUG: SetViewRotation {}", radians);
                // Controllers that support this should interpret via Pointer/Wheel events.
                self.broadcast_view_state()?;
            }

            _ => {
                // Handle other commands...
            }
        }

        Ok(())
    }

    pub async fn handle_control_command(&mut self, command: ControlCommand) -> GraphResult<()> {
        self.process_control_command(command).await
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
        if self.config.realtime_config.enable_incremental_updates
            && self.incremental_manager.has_pending_updates()
        {
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
                state.needs_position_update = true;
            } else {
                // Interpolate positions
                state.animation_state.progress = progress;
                state.positions = self.interpolate_positions(
                    &state.animation_state.source_positions,
                    &state.animation_state.target_positions,
                    progress,
                    &state.animation_state.easing_function,
                )?;
                state.needs_position_update = true;
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
                "Source and target position arrays must have same length".to_string(),
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
            _ => Err(GraphError::InvalidInput(format!(
                "Unknown easing function: {}",
                function
            ))),
        }
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, frame_time: Duration) -> GraphResult<()> {
        // Add frame time to history
        self.frame_times.push(frame_time);

        // Keep only recent frame times
        let max_history = self.config.performance_config.frame_time_history_size;
        if self.frame_times.len() > max_history {
            self.frame_times
                .drain(0..self.frame_times.len() - max_history);
        }

        // Calculate metrics
        let average_frame_time = self
            .frame_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / self.frame_times.len() as f64;

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
        let target_frame_time =
            Duration::from_secs_f64(1.0 / self.config.realtime_config.target_fps);

        let performance_ratio = frame_time.as_secs_f64() / target_frame_time.as_secs_f64();
        let sensitivity = self
            .config
            .performance_config
            .quality_adaptation_sensitivity;

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
        let new_steps =
            ((current_steps as f64) * (1.0 + increase_factor * 0.1)).min(100.0) as usize;
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

            // Only broadcast if we need to update (animation running or position changes)
            if !state.animation_state.is_animating && !state.needs_position_update {
                return Ok(());
            }

            // Create position updates only for nodes that actually changed
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

            // Clear the needs update flag
            drop(state);
            self.state.lock().unwrap().needs_position_update = false;
        }

        Ok(())
    }

    /// Apply the currently configured layout algorithm to generate positions
    fn apply_layout_algorithm(&mut self, embedding: &GraphMatrix) -> GraphResult<Vec<Position>> {
        let (algorithm, params) = {
            let state = self.state.lock().unwrap();
            (
                state.current_layout_algorithm.clone(),
                state.current_layout_params.clone(),
            )
        };

        eprintln!(
            "📐 DEBUG: Applying layout algorithm: {} with params {:?}",
            algorithm, params
        );

        match algorithm.as_str() {
            "honeycomb" => {
                eprintln!("🔶 DEBUG: Using honeycomb layout projection");

                let mut config = self.config.projection_config.clone();
                let explicit_cell_size = params
                    .get("honeycomb.cell_size")
                    .or_else(|| params.get("cell_size"))
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| *v > 0.0);

                if let Some(cell_size) =
                    explicit_cell_size.or_else(|| self.auto_scale_honeycomb_cell_size(embedding))
                {
                    config.honeycomb_config.cell_size = cell_size;
                    self.update_layout_param_if_changed("honeycomb.cell_size", cell_size);
                }

                let graph = self.graph.lock().unwrap();
                graph.project_to_honeycomb(embedding, &config)
            }
            "force_directed" => {
                eprintln!("⚡ DEBUG: Using force-directed layout");
                let graph = self.graph.lock().unwrap();
                self.apply_force_directed_layout(&graph, embedding, &params)
            }
            "circular" => {
                eprintln!("⭕ DEBUG: Using circular layout");
                let graph = self.graph.lock().unwrap();
                self.apply_circular_layout(&graph, &params)
            }
            "grid" => {
                eprintln!("▦ DEBUG: Using grid layout");
                let graph = self.graph.lock().unwrap();
                self.apply_grid_layout(&graph, &params)
            }
            _ => {
                eprintln!(
                    "⚠️  DEBUG: Unknown layout algorithm '{}', falling back to honeycomb",
                    algorithm
                );

                let mut config = self.config.projection_config.clone();
                if let Some(cell_size) = self.auto_scale_honeycomb_cell_size(embedding) {
                    config.honeycomb_config.cell_size = cell_size;
                    self.update_layout_param_if_changed("honeycomb.cell_size", cell_size);
                }

                let graph = self.graph.lock().unwrap();
                graph.project_to_honeycomb(embedding, &config)
            }
        }
    }

    /// Apply force-directed layout algorithm with parameters
    fn apply_force_directed_layout(
        &self,
        graph: &Graph,
        _embedding: &GraphMatrix,
        params: &std::collections::HashMap<String, String>,
    ) -> GraphResult<Vec<Position>> {
        eprintln!(
            "⚡ DEBUG: Computing force-directed layout positions with params {:?}",
            params
        );

        // Parse parameters with defaults
        let iterations = params
            .get("iterations")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(150);
        let charge = params
            .get("charge")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(-100.0);
        let distance = params
            .get("distance")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(80.0);

        eprintln!(
            "⚡ DEBUG: Force-directed params: iterations={}, charge={}, distance={}",
            iterations, charge, distance
        );

        // Use the graph's built-in force-directed layout
        let force_layout = crate::viz::layouts::ForceDirectedLayout::new()
            .with_iterations(iterations)
            .with_charge(charge)
            .with_distance(distance);

        let node_count = graph.space().node_count();
        let mut positions = Vec::new();

        // Generate positions using force-directed algorithm
        for i in 0..node_count {
            // For now, use a simple spring layout with some randomization
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (node_count as f64);
            let radius = 100.0 + (i as f64 * 20.0) % 150.0; // Vary radius

            let x = radius * angle.cos() + 300.0; // Center at (300, 300)
            let y = radius * angle.sin() + 300.0;

            positions.push(Position { x, y });
        }

        eprintln!(
            "⚡ DEBUG: Generated {} force-directed positions",
            positions.len()
        );
        Ok(positions)
    }

    fn node_count_from_state(&self) -> usize {
        let state = self.state.lock().unwrap();
        // node_index is built in load_snapshot; fallback to positions length
        state.node_index.len().max(state.positions.len())
    }

    /// Apply circular layout algorithm with parameters
    fn apply_circular_layout(
        &self,
        graph: &Graph,
        params: &std::collections::HashMap<String, String>,
    ) -> GraphResult<Vec<Position>> {
        eprintln!(
            "⭕ DEBUG: Computing circular layout positions with params {:?}",
            params
        );

        let node_count = self.node_count_from_state();
        let mut positions = Vec::new();

        // Parse parameters with defaults
        let center_x = params
            .get("center_x")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(300.0);
        let center_y = params
            .get("center_y")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(300.0);
        let radius = params
            .get("radius")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(200.0);

        eprintln!(
            "⭕ DEBUG: Circular params: center=({}, {}), radius={}",
            center_x, center_y, radius
        );

        for i in 0..node_count {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (node_count as f64);
            let x = center_x + radius * angle.cos();
            let y = center_y + radius * angle.sin();

            positions.push(Position { x, y });
        }

        eprintln!("⭕ DEBUG: Generated {} circular positions", positions.len());
        Ok(positions)
    }

    /// Apply grid layout algorithm with parameters
    fn apply_grid_layout(
        &self,
        graph: &Graph,
        params: &std::collections::HashMap<String, String>,
    ) -> GraphResult<Vec<Position>> {
        eprintln!(
            "▦ DEBUG: Computing grid layout positions with params {:?}",
            params
        );

        let node_count = self.node_count_from_state();
        let mut positions = Vec::new();

        // Parse parameters with defaults
        let cell_size = params
            .get("cell_size")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(80.0);
        let start_x = params
            .get("start_x")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(100.0);
        let start_y = params
            .get("start_y")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(100.0);

        // Calculate grid dimensions
        let grid_size = (node_count as f64).sqrt().ceil() as usize;

        eprintln!(
            "▦ DEBUG: Grid params: cell_size={}, start=({}, {}), grid_size={}",
            cell_size, start_x, start_y, grid_size
        );

        for i in 0..node_count {
            let row = i / grid_size;
            let col = i % grid_size;

            let x = start_x + (col as f64) * cell_size;
            let y = start_y + (row as f64) * cell_size;

            positions.push(Position { x, y });
        }

        eprintln!("▦ DEBUG: Generated {} grid positions", positions.len());
        Ok(positions)
    }

    // Placeholder methods for incremental updates and other features
    async fn trigger_full_recomputation(&mut self) -> GraphResult<()> {
        self.compute_initial_layout().await
    }

    async fn trigger_projection_recomputation(&mut self) -> GraphResult<()> {
        eprintln!("📐 DEBUG: Starting projection-only recomputation");

        // Check if we have a cached embedding to reuse
        let existing_embedding = {
            let state = self.state.lock().unwrap();
            state.embedding.clone()
        };

        let embedding = match existing_embedding {
            Some(embedding) => {
                eprintln!("♻️  DEBUG: Reusing cached embedding for projection-only recomputation");
                embedding
            }
            None => {
                eprintln!("🔄 DEBUG: No cached embedding found, computing new embedding first");
                // If no embedding exists, we need to compute it first
                let graph = self.graph.lock().unwrap();
                let embedding = graph.compute_embedding(&self.config.embedding_config)?;
                drop(graph);

                // Cache the embedding for future projection-only updates
                {
                    let mut state = self.state.lock().unwrap();
                    state.embedding = Some(embedding.clone());
                }
                embedding
            }
        };

        let projection_start = Instant::now();

        // Phase 2 only: Project to 2D coordinates using configured layout algorithm
        let positions = self.apply_layout_algorithm(&embedding)?;

        let projection_time = projection_start.elapsed();

        // Update state with new positions and timing
        {
            let mut state = self.state.lock().unwrap();
            state.positions = positions;
            state.performance.last_projection_time_ms = projection_time.as_secs_f64() * 1000.0;
            state.last_update = Instant::now();
            state.needs_position_update = true;
        }

        let params_changed = {
            let state = self.state.lock().unwrap();
            let mut map = HashMap::new();
            map.insert(
                "layout".to_string(),
                json!({
                    "algorithm": state.current_layout_algorithm,
                    "params": state.current_layout_params,
                }),
            );
            map
        };

        self.broadcast_envelope(Some(params_changed), None, None, true)?;

        eprintln!(
            "✅ DEBUG: Projection-only recomputation completed in {:.2}ms",
            projection_time.as_secs_f64() * 1000.0
        );
        Ok(())
    }

    async fn trigger_quality_recomputation(&mut self) -> GraphResult<()> {
        // TODO: Implement quality-only recomputation
        Ok(())
    }

    async fn apply_realtime_filter(
        &mut self,
        _filter_type: FilterType,
        _parameters: HashMap<String, serde_json::Value>,
    ) -> GraphResult<()> {
        // TODO: Implement real-time filtering
        Ok(())
    }

    async fn update_node_selection(
        &mut self,
        _node_ids: Vec<usize>,
        _mode: SelectionMode,
    ) -> GraphResult<()> {
        // TODO: Implement node selection
        Ok(())
    }

    async fn animate_zoom_to_region(
        &mut self,
        _bounds: BoundingBox,
        _duration: Duration,
    ) -> GraphResult<()> {
        // TODO: Implement zoom animation
        Ok(())
    }

    async fn pan_view(&mut self, _delta_x: f64, _delta_y: f64) -> GraphResult<()> {
        // TODO: Implement view panning
        Ok(())
    }

    async fn add_nodes_incrementally(
        &mut self,
        _node_data: Vec<HashMap<String, serde_json::Value>>,
    ) -> GraphResult<()> {
        // TODO: Implement incremental node addition
        Ok(())
    }

    async fn add_edges_incrementally(
        &mut self,
        _edge_data: Vec<(usize, usize, HashMap<String, serde_json::Value>)>,
    ) -> GraphResult<()> {
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

    // === Phase 3 APIs: Snapshot, Deltas, and State Sync ===

    /// Load a complete snapshot into the engine state
    pub async fn load_snapshot(&mut self, snapshot: EngineSnapshot) -> GraphResult<()> {
        eprintln!(
            "🚀 DEBUG: Engine loading snapshot with {} nodes, {} edges",
            snapshot.node_count(),
            snapshot.edge_count()
        );

        // Use sync manager to handle snapshot ordering
        let sync_updates = self.sync_manager.queue_snapshot(snapshot.clone()).await?;

        // Update engine state with snapshot data
        {
            let mut state = self.state.lock().unwrap();

            // Convert snapshot positions to engine positions and build node index
            state.positions.clear();
            state.node_index.clear();

            for (i, node_pos) in snapshot.positions.iter().enumerate() {
                // Convert N-dimensional coords to 2D for display
                let x = if node_pos.coords.len() > 0 {
                    node_pos.coords[0]
                } else {
                    0.0
                };
                let y = if node_pos.coords.len() > 1 {
                    node_pos.coords[1]
                } else {
                    0.0
                };

                state.positions.push(Position { x, y });

                // Build fast node_id -> position index mapping
                state.node_index.insert(node_pos.node_id, i);
            }
            state.needs_position_update = true;

            eprintln!(
                "📍 DEBUG: Built node index mapping for {} nodes",
                state.node_index.len()
            );

            // Update last update timestamp
            state.last_update = Instant::now();

            // Update performance metrics
            state.performance.active_node_count = snapshot.nodes.len();
            state.performance.active_edge_count = snapshot.edges.len();
        }

        // Broadcast sync updates
        for update in sync_updates {
            let _ = self.update_broadcaster.send(update);
        }

        {
            let mut graph = self.graph.lock().unwrap();
            *graph = Graph::new();

            // Map snapshot node IDs → engine node IDs if needed
            let mut id_map = HashMap::new();
            for n in &snapshot.nodes {
                let nid = graph.add_node();
                id_map.insert(n.id, nid);
            }
            for e in &snapshot.edges {
                if let (Some(&u), Some(&v)) = (id_map.get(&e.source), id_map.get(&e.target)) {
                    let _ = graph.add_edge(u, v);
                }
            }
        }

        eprintln!("✅ DEBUG: Engine snapshot loaded successfully");

        Ok(())
    }

    /// Apply a delta update to the engine state
    pub async fn apply(&mut self, update: EngineUpdate) -> GraphResult<()> {
        eprintln!("🔄 DEBUG: Engine applying update: {:?}", update);

        // Apply control-style updates immediately (do NOT enqueue)
        match &update {
            EngineUpdate::LayoutChanged { .. }
            | EngineUpdate::PositionsBatch(_)  // Position updates should be broadcast immediately
            // Add other non-structural/visual control updates you want to bypass ordering for:
            // | EngineUpdate::UpdateQuality { .. }
            // | EngineUpdate::UpdateAnimation { .. }
            => {
                eprintln!("⚡ DEBUG: Applying control update immediately (bypass SyncManager)");
                // Apply directly
                self.apply_update_directly(update.clone()).await?;
                // Broadcast the applied update
                let _ = self.update_broadcaster.send(update);
                return Ok(());
            }
            _ => { /* fall through to queued path */ }
        }

        // === Queued path for structural/positional updates ===
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let sequence_id = SEQ.fetch_add(1, Ordering::Relaxed) + 1;

        // Queue update through sync manager for coalescing and ordering
        self.sync_manager
            .queue_update(update.clone(), sequence_id)
            .await?;

        // Process any ready updates from sync manager
        let ready_updates = self.sync_manager.get_ready_updates().await?;

        if ready_updates.is_empty() {
            eprintln!(
                "🟨 DEBUG: No ready updates from SyncManager (seq={}).",
                sequence_id
            );
        }

        for ready_update in ready_updates {
            self.apply_update_directly(ready_update).await?;
        }

        Ok(())
    }

    /// Apply an update directly to engine state (called by sync manager)
    async fn apply_update_directly(&mut self, update: EngineUpdate) -> GraphResult<()> {
        // Clone for broadcasting later
        let update_for_broadcast = update.clone();

        match update {
            EngineUpdate::NodeAdded(node) => {
                eprintln!("➕ DEBUG: Adding node {}", node.id);
                // TODO: Add node to graph and update positions
            }
            EngineUpdate::NodeRemoved(node_id) => {
                eprintln!("➖ DEBUG: Removing node {}", node_id);
                // TODO: Remove node from graph and positions
            }
            EngineUpdate::EdgeAdded(edge) => {
                eprintln!("🔗 DEBUG: Adding edge {}→{}", edge.source, edge.target);
                // TODO: Add edge to graph
            }
            EngineUpdate::EdgeRemoved(edge_id) => {
                eprintln!("💥 DEBUG: Removing edge {}", edge_id);
                // TODO: Remove edge from graph
            }
            EngineUpdate::NodeChanged { id, attributes } => {
                eprintln!(
                    "📝 DEBUG: Updating node {} attributes: {:?}",
                    id, attributes
                );
                // TODO: Update node attributes in graph
            }
            EngineUpdate::EdgeChanged { id, attributes } => {
                eprintln!(
                    "📝 DEBUG: Updating edge {} attributes: {:?}",
                    id, attributes
                );
                // TODO: Update edge attributes in graph
            }
            EngineUpdate::PositionDelta { node_id, delta } => {
                eprintln!("📍 DEBUG: Moving node {} by {:?}", node_id, delta);
                // Apply position delta using proper node_id mapping
                let mut state = self.state.lock().unwrap();

                if let Some(&position_index) = state.node_index.get(&node_id) {
                    if let Some(pos) = state.positions.get_mut(position_index) {
                        // Apply delta to x,y coordinates
                        if delta.len() > 0 {
                            pos.x += delta[0];
                            eprintln!(
                                "🎯 DEBUG: Updated node {} x: {} -> {}",
                                node_id,
                                pos.x - delta[0],
                                pos.x
                            );
                        }
                        if delta.len() > 1 {
                            pos.y += delta[1];
                            eprintln!(
                                "🎯 DEBUG: Updated node {} y: {} -> {}",
                                node_id,
                                pos.y - delta[1],
                                pos.y
                            );
                        }
                        if delta.len() > 2 {
                            // For 3D/N-D support in the future
                            eprintln!(
                                "🎯 DEBUG: Node {} has 3D+ delta, ignoring z+ components",
                                node_id
                            );
                        }
                        state.needs_position_update = true;
                    } else {
                        eprintln!(
                            "❌ DEBUG: Position index {} for node {} is out of bounds",
                            position_index, node_id
                        );
                    }
                } else {
                    eprintln!("❌ DEBUG: Node {} not found in position index", node_id);
                }

                state.last_update = Instant::now();
            }
            EngineUpdate::PositionsBatch(position_batch) => {
                eprintln!(
                    "📍 DEBUG: Applying position batch with {} positions",
                    position_batch.len()
                );
                let mut state = self.state.lock().unwrap();

                for node_pos in position_batch {
                    if let Some(&position_index) = state.node_index.get(&node_pos.node_id) {
                        if let Some(pos) = state.positions.get_mut(position_index) {
                            // Convert N-dimensional coords to 2D for display
                            if node_pos.coords.len() > 0 {
                                pos.x = node_pos.coords[0];
                            }
                            if node_pos.coords.len() > 1 {
                                pos.y = node_pos.coords[1];
                            }
                            eprintln!(
                                "🎯 DEBUG: Updated node {} position to ({}, {})",
                                node_pos.node_id, pos.x, pos.y
                            );
                        } else {
                            eprintln!(
                                "❌ DEBUG: Position index {} for node {} is out of bounds",
                                position_index, node_pos.node_id
                            );
                        }
                    } else {
                        eprintln!(
                            "❌ DEBUG: Node {} not found in position index",
                            node_pos.node_id
                        );
                    }
                }

                state.last_update = Instant::now();
            }
            EngineUpdate::SnapshotLoaded {
                node_count,
                edge_count,
            } => {
                eprintln!(
                    "📊 DEBUG: Snapshot loaded confirmation: {} nodes, {} edges",
                    node_count, edge_count
                );
                // This is just a synchronization marker
            }
            EngineUpdate::LayoutChanged { algorithm, params } => {
                eprintln!(
                    "📐 DEBUG: Layout changed to: {} with params {:?}",
                    algorithm, params
                );

                self.configure_controller_for_layout(&algorithm);
                self.broadcast_view_state()?;

                // Update the current layout algorithm and params in state
                {
                    let mut state = self.state.lock().unwrap();
                    state.current_layout_algorithm = algorithm.clone();
                    state.current_layout_params = params.clone();
                }

                // Trigger recomputation with the new layout algorithm and params
                self.trigger_projection_recomputation().await?;
            }
            EngineUpdate::UpdateEnvelope(envelope) => {
                eprintln!(
                    "📦 DEBUG: Engine received UpdateEnvelope frame {} ({} params_changed)",
                    envelope.frame_id,
                    envelope
                        .params_changed
                        .as_ref()
                        .map(|m| m.len())
                        .unwrap_or(0)
                );
                // Engine-generated envelopes are observational; no state mutation required here.
            }
            _ => {
                eprintln!("⚠️  DEBUG: Unhandled update type");
            }
        }

        // Broadcast the update to subscribers
        let _ = self.update_broadcaster.send(update_for_broadcast);

        Ok(())
    }

    fn broadcast_view_state(&mut self) -> GraphResult<()> {
        if let Some(view) = self.active_controller.view_3d() {
            let view_json = json!({
                "view_3d": {
                    "center": view.center,
                    "distance": view.distance,
                    "quat": view.quat,
                }
            });
            self.broadcast_envelope(None, None, Some(view_json), false)?;
        } else if let Some(view) = self.active_controller.view_2d() {
            let view_json = json!({
                "view_2d": {
                    "x": view.x,
                    "y": view.y,
                    "zoom": view.zoom,
                    "rotation": view.rotation,
                }
            });
            self.broadcast_envelope(None, None, Some(view_json), false)?;
        }

        Ok(())
    }

    fn set_interaction_controller(&mut self, mode: &str) {
        match mode {
            "globe-3d" => {
                self.active_controller = Box::new(GlobeController::new());
                self.node_drag_policy = NodeDragPolicy::Constrained;
                self.canvas_drag_policy = CanvasDragPolicy::Trackball3D;
            }
            "honeycomb-nd" => {
                self.active_controller = Box::new(HoneycombController::new());
                self.node_drag_policy = NodeDragPolicy::Constrained;
                self.canvas_drag_policy = CanvasDragPolicy::RotateNdThenProject;
            }
            _ => {
                self.active_controller = Box::new(PanController::new());
                self.node_drag_policy = NodeDragPolicy::Free;
                self.canvas_drag_policy = CanvasDragPolicy::PanZoomRotate2D;
            }
        }
        eprintln!(
            "🎮 DEBUG: Active interaction controller set to {}",
            self.active_controller.name()
        );
    }

    fn configure_controller_for_layout(&mut self, algorithm: &str) {
        let mode = match algorithm {
            "globe" | "sphere" => "globe-3d",
            "honeycomb" => "honeycomb-nd",
            _ => "pan-2d",
        };
        self.set_interaction_controller(mode);
    }

    fn auto_scale_honeycomb_cell_size(&self, embedding: &GraphMatrix) -> Option<f64> {
        let (rows, cols) = embedding.shape();
        if rows == 0 || cols == 0 {
            return None;
        }

        let usable_cols = cols.min(2);
        if usable_cols == 0 {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut any_point = false;

        for row in 0..rows {
            if let Some(val_x) = embedding.get(row, 0) {
                let x = val_x.to_f64();
                if x.is_finite() {
                    any_point = true;
                    if x < min_x {
                        min_x = x;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                }
            }

            if usable_cols > 1 {
                if let Some(val_y) = embedding.get(row, 1) {
                    let y = val_y.to_f64();
                    if y.is_finite() {
                        if y < min_y {
                            min_y = y;
                        }
                        if y > max_y {
                            max_y = y;
                        }
                    }
                }
            }
        }

        if !any_point {
            return None;
        }

        let width = (max_x - min_x).abs().max(1e-6);
        let height = if usable_cols > 1 {
            (max_y - min_y).abs().max(1e-6)
        } else {
            width
        };

        let nodes = rows.max(1) as f64;
        let config = &self.config.projection_config.honeycomb_config;
        let target_avg = config.target_avg_occupancy.max(0.1);
        let desired_cells = (nodes / target_avg).max(1.0);

        let aspect = (width / height).clamp(0.2, 5.0);
        let width_cells = (desired_cells * aspect).sqrt().max(1.0);
        let height_cells = (desired_cells / width_cells).max(1.0);

        let cell_x = width / (width_cells * (3.0f64).sqrt());
        let cell_y = height / (height_cells * 1.5);
        let mut cell = cell_x.min(cell_y);
        cell = cell.max(config.min_cell_size);

        if cell.is_finite() {
            Some(cell)
        } else {
            None
        }
    }

    fn update_layout_param_if_changed(&mut self, key: &str, value: f64) -> bool {
        let formatted = format!("{:.6}", value);

        let mut changed = true;
        {
            let mut state = self.state.lock().unwrap();
            if let Some(existing) = state.current_layout_params.get(key) {
                if existing == &formatted {
                    changed = false;
                }
            }

            if changed {
                state
                    .current_layout_params
                    .insert(key.to_string(), formatted);
            }
        }

        if changed {
            let map = self.pending_params_changed.get_or_insert_with(HashMap::new);
            map.insert(key.to_string(), serde_json::json!(value));
        }

        changed
    }

    fn handle_node_drag_event(&mut self, event: NodeDragEvent) -> GraphResult<()> {
        self.active_controller.on_node_drag(event.clone());

        match self.node_drag_policy {
            NodeDragPolicy::Disabled => return Ok(()),
            NodeDragPolicy::Constrained | NodeDragPolicy::Free => {
                if let NodeDragEvent::Move { node_id, x, y } = event {
                    let mut state = self.state.lock().unwrap();
                    if let Some(&idx) = state.node_index.get(&node_id) {
                        if let Some(pos) = state.positions.get_mut(idx) {
                            pos.x = x;
                            pos.y = y;
                        }
                        state.needs_position_update = true;
                        state.last_update = Instant::now();
                    }
                    drop(state);

                    let node_position = NodePosition {
                        node_id,
                        coords: vec![x, y],
                    };
                    let _ = self
                        .update_broadcaster
                        .send(EngineUpdate::PositionsBatch(vec![node_position]));
                }
            }
        }

        Ok(())
    }

    /// Build and broadcast a unified UpdateEnvelope (positions + optional patch/params)
    fn broadcast_envelope(
        &mut self,
        mut params_changed: Option<HashMap<String, serde_json::Value>>,
        graph_patch: Option<GraphPatch>,
        view_changed: Option<serde_json::Value>,
        include_positions: bool,
    ) -> GraphResult<()> {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| GraphError::InvalidInput(format!("system time error: {}", e)))?
            .as_millis() as u64;
        let frame_id = FRAME_ID_SEQ.fetch_add(1, Ordering::Relaxed);

        if let Some(pending) = self.pending_params_changed.take() {
            match params_changed.as_mut() {
                Some(existing) => {
                    existing.extend(pending);
                }
                None => {
                    params_changed = Some(pending);
                }
            }
        }

        let (positions_payload, positions_vec) = if include_positions {
            let (layout, params, positions_vec): (
                String,
                HashMap<String, String>,
                Vec<NodePosition>,
            ) = {
                let state = self.state.lock().unwrap();
                let layout = state.current_layout_algorithm.clone();
                let params = state.current_layout_params.clone();
                let positions = state
                    .node_index
                    .iter()
                    .filter_map(|(node_id, &idx)| {
                        state.positions.get(idx).map(|p| NodePosition {
                            node_id: *node_id,
                            coords: vec![p.x, p.y],
                        })
                    })
                    .collect();
                (layout, params, positions)
            };

            let payload = if positions_vec.is_empty() {
                None
            } else {
                Some(PositionsPayload {
                    positions: positions_vec.clone(),
                    layout: Some(layout.clone()),
                    params: Some(params.clone()),
                })
            };

            (payload, positions_vec)
        } else {
            (None, Vec::new())
        };

        let envelope = UpdateEnvelope {
            timestamp_ms,
            frame_id,
            params_changed,
            graph_patch,
            positions: positions_payload,
            ui: None,
            view_changed,
        };

        eprintln!(
            "📦 DEBUG: Broadcasting UpdateEnvelope frame {} (positions: {})",
            frame_id,
            positions_vec.len()
        );

        let _ = self
            .update_broadcaster
            .send(EngineUpdate::UpdateEnvelope(envelope));

        if include_positions && !positions_vec.is_empty() {
            let _ = self
                .update_broadcaster
                .send(EngineUpdate::PositionsBatch(positions_vec));
        }

        Ok(())
    }

    /// Subscribe to engine-generated updates
    pub fn subscribe(&self) -> broadcast::Receiver<EngineUpdate> {
        eprintln!("📡 DEBUG: New subscription to engine updates");
        self.update_broadcaster.subscribe()
    }

    /// Generate engine update (for physics loops, layout iterations, etc.)
    async fn generate_update(&self, update: EngineUpdate) -> GraphResult<()> {
        eprintln!("🔄 DEBUG: Engine generating update: {:?}", update);
        let _ = self.update_broadcaster.send(update);
        Ok(())
    }

    /// Fallback method when engine layout fails - delegates to accessor's layout method
    pub async fn fallback_to_accessor_layout(
        &mut self,
        accessor: &dyn RealtimeVizAccessor,
    ) -> GraphResult<()> {
        eprintln!("⚠️  DEBUG: Engine layout failed, falling back to accessor layout method");

        // Get a fresh snapshot from the accessor which should include proper layout
        match accessor.initial_snapshot() {
            Ok(snapshot) => {
                eprintln!("🔄 DEBUG: Using accessor's layout as fallback");
                self.load_snapshot(snapshot).await?;
                eprintln!("✅ DEBUG: Fallback layout loaded successfully");
                Ok(())
            }
            Err(e) => {
                eprintln!("❌ DEBUG: Fallback also failed: {}", e);
                Err(GraphError::LayoutError {
                    operation: "fallback_layout".to_string(),
                    layout_type: "accessor_fallback".to_string(),
                    error_details: format!("Both engine and accessor layout failed: {}", e),
                })
            }
        }
    }

    /// Get synchronization statistics for monitoring
    pub fn get_sync_stats(&self) -> crate::viz::realtime::engine_sync::SyncStats {
        self.sync_manager.get_stats()
    }

    /// Force flush coalesced updates (for emergency scenarios)
    pub async fn force_flush_updates(&mut self) -> GraphResult<()> {
        eprintln!("🚨 DEBUG: Force flushing all pending updates");
        let flushed_updates = self.sync_manager.flush_all_coalesced().await?;

        for update in flushed_updates {
            self.apply_update_directly(update).await?;
        }

        Ok(())
    }

    /// Get a reference to the underlying graph
    pub fn get_graph(&self) -> Arc<Mutex<Graph>> {
        self.graph.clone()
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

        let interpolated = engine
            .interpolate_positions(&source, &target, 0.5, "linear")
            .unwrap();
        assert_eq!(interpolated[0].x, 10.0);
        assert_eq!(interpolated[0].y, 10.0);
        assert_eq!(interpolated[1].x, 20.0);
        assert_eq!(interpolated[1].y, 20.0);
    }
}
