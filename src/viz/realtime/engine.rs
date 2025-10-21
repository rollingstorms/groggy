//! Real-time visualization engine
//!
//! The main engine that orchestrates the complete real-time visualization pipeline,
//! combining Phase 1 (embeddings), Phase 2 (projections), and Phase 3 (streaming).

#![allow(clippy::arc_with_non_send_sync)]

use super::*;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::advanced_matrix::numeric_type::NumericType;
use crate::storage::matrix::GraphMatrix;
use crate::viz::embeddings::flat_embedding::{compute_flat_embedding, FlatEmbedConfig};
use crate::viz::embeddings::{EmbeddingMethod, GraphEmbeddingExt};
use crate::viz::projection::GraphProjectionExt;
use crate::viz::realtime::accessor::{
    EngineSnapshot, EngineUpdate, PositionsPayload, UpdateEnvelope,
};
use crate::viz::realtime::engine_sync::EngineSyncManager;
use crate::viz::realtime::interaction::{
    CanvasDragPolicy, GlobeController, HoneycombController, InteractionCommand, NodeDragEvent,
    NodeDragPolicy, PanController, PointerEvent, ViewState3D, WheelEvent,
};
use crate::viz::streaming::data_source::Position;
use serde_json::json;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc};
use tokio::time::interval;

/// Main real-time visualization engine
pub struct RealTimeVizEngine {
    /// Configuration for the visualization
    config: RealTimeVizConfig,

    /// Current visualization state
    state: Arc<Mutex<RealTimeVizState>>,

    /// Reference to the graph being visualized
    graph: Arc<Mutex<Graph>>,

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
    #[allow(dead_code)]
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
            current_layout: LayoutKind::Honeycomb,
            current_layout_params: std::collections::HashMap::new(),
            layout_param_cache: std::collections::HashMap::new(),
        }));

        // Create broadcast channel for engine updates
        let (update_broadcaster, _) = broadcast::channel(1000);

        Self {
            config,
            state,
            graph: Arc::new(Mutex::new(graph)),
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
            state.current_layout
        };
        self.configure_controller_for_layout(initial_layout);
        self.broadcast_view_state().await?;

        // Initialize performance monitoring
        self.performance_monitor.start()?;

        // Setup communication channels
        let (_control_tx, control_rx) = mpsc::unbounded_channel();

        self.control_receiver = Some(control_rx);

        // Initialize incremental update manager
        let graph = self.graph.lock().unwrap();
        self.incremental_manager.initialize(&graph)?;

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
        let (embedding, graph_was_empty) = {
            let graph = self.graph.lock().unwrap();
            let node_count = graph.space().node_count();

            if node_count == 0 {
                // Graph is empty, check if we have state positions to work from
                let state_node_count = self.node_count_from_state();
                if state_node_count > 0 {
                    // Generate a fallback embedding from current positions
                    let positions = {
                        let state = self.state.lock().unwrap();
                        state.positions.clone()
                    };

                    // Create embedding matrix from current positions
                    let mut embedding_data = Vec::with_capacity(state_node_count * 2);
                    for pos in &positions {
                        embedding_data.push(pos.x);
                        embedding_data.push(pos.y);
                    }

                    let embedding = GraphMatrix::from_row_major_data(
                        embedding_data,
                        state_node_count,
                        2,
                        None,
                    )?;

                    (embedding, true)
                } else {
                    return Err(GraphError::InvalidInput(
                        "No graph data and no state positions available".to_string(),
                    ));
                }
            } else {
                let embedding = graph.compute_embedding(&self.config.embedding_config)?;
                (embedding, false)
            }
        };

        let embedding_time = embedding_start.elapsed();

        if graph_was_empty {
            // Used fallback embedding from state positions
        }

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
                self.broadcast_view_state().await?;
            }

            ControlCommand::Pointer { event } => {
                let commands = self.active_controller.on_pointer(event);
                self.process_interaction_commands(commands).await?;
                self.broadcast_view_state().await?;
            }

            ControlCommand::Wheel { event } => {
                let commands = self.active_controller.on_wheel(event);
                self.process_interaction_commands(commands).await?;
                self.broadcast_view_state().await?;
            }

            ControlCommand::NodeDrag { event } => {
                self.handle_node_drag_event(event)?;
            }

            ControlCommand::RotateEmbedding {
                axis_i,
                axis_j,
                radians,
            } => {
                self.apply_nd_rotation(axis_i, axis_j, radians).await?;
            }

            ControlCommand::SetViewRotation { radians: _ } => {
                // Controllers that support this should interpret via Pointer/Wheel events.
                self.broadcast_view_state().await?;
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

    /// Translate positions so their bbox center is at (0,0).
    /// If `target_radius_px` is Some(r), also scale uniformly so
    /// the bbox half-diagonal ≈ r pixels (simple fit-to-view).
    fn center_and_fit_positions(
        mut positions: Vec<Position>,
        target_radius_px: Option<f64>,
    ) -> Vec<Position> {
        if positions.is_empty() {
            return positions;
        }

        // bbox
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for p in &positions {
            if p.x.is_finite() {
                xmin = xmin.min(p.x);
                xmax = xmax.max(p.x);
            }
            if p.y.is_finite() {
                ymin = ymin.min(p.y);
                ymax = ymax.max(p.y);
            }
        }
        let cx = (xmin + xmax) * 0.5;
        let cy = (ymin + ymax) * 0.5;

        // translate to origin
        for p in &mut positions {
            p.x -= cx;
            p.y -= cy;
        }

        if let Some(target) = target_radius_px {
            // half-diagonal of bbox (farthest corner from origin)
            let hw = (xmax - xmin) * 0.5;
            let hh = (ymax - ymin) * 0.5;
            let half_diag = (hw * hw + hh * hh).sqrt().max(1e-9);
            let s = target / half_diag;
            for p in &mut positions {
                p.x *= s;
                p.y *= s;
            }
        }

        positions
    }

    /// Apply the currently configured layout algorithm to generate positions
    fn apply_layout_algorithm(&mut self, embedding: &GraphMatrix) -> GraphResult<Vec<Position>> {
        let (layout_kind, params) = {
            let state = self.state.lock().unwrap();
            (state.current_layout, state.current_layout_params.clone())
        };

        let mut positions = match layout_kind {
            LayoutKind::Honeycomb => {
                let mut config = self.config.projection_config.clone();

                // Check for flat embedding flag
                let use_flat_embed = params
                    .get("flat_embed")
                    .map(|s| s == "true" || s == "1")
                    .unwrap_or(false);

                // Apply flat embedding preprocessing if requested
                let flat_embedding_matrix;
                let embedding_to_use = if use_flat_embed {
                    // Applying flat embedding preprocessing

                    // Create flat embedding configuration
                    let flat_config = FlatEmbedConfig {
                        iterations: params
                            .get("flat_embed.iterations")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(800),
                        learning_rate: params
                            .get("flat_embed.learning_rate")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.03),
                        edge_cohesion_weight: params
                            .get("flat_embed.edge_cohesion_weight")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1.0),
                        repulsion_weight: params
                            .get("flat_embed.repulsion_weight")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.1),
                        spread_weight: params
                            .get("flat_embed.spread_weight")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.05),
                        ..FlatEmbedConfig::default()
                    };

                    match compute_flat_embedding(
                        embedding,
                        &self.graph.lock().unwrap(),
                        &flat_config,
                    ) {
                        Ok(flat_positions) => {
                            // Flat embedding computed successfully

                            // Convert positions back to GraphMatrix for honeycomb projection
                            let flat_data: Vec<f64> = flat_positions
                                .iter()
                                .flat_map(|pos| vec![pos.x, pos.y])
                                .collect();

                            match crate::storage::advanced_matrix::unified_matrix::UnifiedMatrix::from_data(flat_data, flat_positions.len(), 2) {
                                Ok(unified) => {
                                    flat_embedding_matrix = crate::storage::matrix::GraphMatrix::from_storage(unified);
                                    &flat_embedding_matrix
                                }
                                Err(_e) => {
                                    // Failed to create matrix from flat positions, using original embedding
                                    embedding
                                }
                            }
                        }
                        Err(_e) => {
                            // Flat embedding failed, using original embedding
                            embedding
                        }
                    }
                } else {
                    embedding
                };

                // Check for explicit cell size first
                let explicit_cell_size = params
                    .get("honeycomb.cell_size")
                    .or_else(|| params.get("cell_size"))
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| *v > 0.0);

                if let Some(cell_size) = explicit_cell_size {
                    config.honeycomb_config.cell_size = cell_size;
                    self.update_layout_param_if_changed("honeycomb.cell_size", cell_size);
                } else {
                    // Apply bin density calculation as per fix plan
                    let target_bins = params
                        .get("bins")
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(64.0);

                    if let Some(cell_size) =
                        self.calculate_honeycomb_cell_size_from_bins(embedding_to_use, target_bins)
                    {
                        config.honeycomb_config.cell_size = cell_size;
                        self.update_layout_param_if_changed("honeycomb.cell_size", cell_size);
                        eprintln!(
                            "🔶 DEBUG: Auto-scaled honeycomb cell size to {:.2} for {} target bins",
                            cell_size, target_bins
                        );
                    } else if let Some(cell_size) =
                        self.auto_scale_honeycomb_cell_size(embedding_to_use)
                    {
                        config.honeycomb_config.cell_size = cell_size;
                        self.update_layout_param_if_changed("honeycomb.cell_size", cell_size);
                    }
                }

                let graph = self.graph.lock().unwrap();
                graph.project_to_honeycomb(embedding_to_use, &config)?
            }
            LayoutKind::ForceDirected => {
                // Using force-directed layout
                let graph = self.graph.lock().unwrap();
                self.apply_force_directed_layout(&graph, embedding, &params)?
            }
            LayoutKind::Circular => {
                // Using circular layout
                let graph = self.graph.lock().unwrap();
                self.apply_circular_layout(&graph, embedding, &params)?
            }
            LayoutKind::Grid => {
                // Using grid layout
                let graph = self.graph.lock().unwrap();
                self.apply_grid_layout(&graph, embedding, &params)?
            }
        };

        // Center & fit to a sensible radius (~350 px)
        positions = Self::center_and_fit_positions(positions, Some(350.0));

        Ok(positions)
    }

    /// Apply force-directed layout algorithm with parameters
    fn apply_force_directed_layout(
        &self,
        _graph: &Graph,
        embedding: &GraphMatrix,
        params: &std::collections::HashMap<String, String>,
    ) -> GraphResult<Vec<Position>> {
        // Computing force-directed layout positions

        // Parse parameters with defaults
        let _iterations = params
            .get("iterations")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(150);
        let _charge = params
            .get("charge")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(-100.0);
        let _distance = params
            .get("distance")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(80.0);

        let node_count = self.node_count_from_state();
        let mut positions = Vec::new();

        // Initialize positions using first 2 dimensions of embedding
        for i in 0..node_count {
            let (x, y) = if embedding.shape().1 >= 2 {
                // Use first 2 dimensions of embedding as initial positions
                let x_val = embedding.get(i, 0).unwrap_or(0.0);
                let y_val = embedding.get(i, 1).unwrap_or(0.0);
                (x_val, y_val)
            } else {
                // Fallback to circular arrangement if embedding has <2 dimensions
                let angle = (i as f64) * 2.0 * std::f64::consts::PI / (node_count as f64);
                let radius = 100.0;
                (radius * angle.cos(), radius * angle.sin())
            };

            positions.push(Position { x, y });
        }

        Ok(positions)
    }

    fn node_count_from_state(&self) -> usize {
        let state = self.state.lock().unwrap();
        // node_index is built in load_snapshot; fallback to positions length
        state.node_index.len().max(state.positions.len())
    }

    /// Apply circular layout algorithm with parameters using embedding for ordering
    fn apply_circular_layout(
        &self,
        _graph: &Graph,
        embedding: &GraphMatrix,
        params: &std::collections::HashMap<String, String>,
    ) -> GraphResult<Vec<Position>> {
        let node_count = self.node_count_from_state();
        let mut positions = Vec::new();

        // Parse parameters with defaults (center around origin)
        let center_x = params
            .get("center_x")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let center_y = params
            .get("center_y")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let radius = params
            .get("radius")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(200.0);

        if embedding.shape().1 >= 2 {
            // Use embedding to determine angular ordering - sort by first principal component
            let mut node_angles: Vec<(usize, f64)> = (0..node_count)
                .map(|i| {
                    let x_val = embedding.get(i, 0).unwrap_or(0.0);
                    let y_val = embedding.get(i, 1).unwrap_or(0.0);
                    let angle = y_val.atan2(x_val);
                    (i, angle)
                })
                .collect();

            // Sort by angle to preserve embedding neighborhood structure
            node_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Place nodes on circle in embedding-determined order
            for (circle_pos, &(node_idx, _)) in node_angles.iter().enumerate() {
                let angle = (circle_pos as f64) * 2.0 * std::f64::consts::PI / (node_count as f64);
                let x = center_x + radius * angle.cos();
                let y = center_y + radius * angle.sin();

                // Ensure positions vector has enough capacity
                if node_idx >= positions.len() {
                    positions.resize(node_idx + 1, Position { x: 0.0, y: 0.0 });
                }
                positions[node_idx] = Position { x, y };
            }
        } else {
            // Fallback to simple circular arrangement
            for i in 0..node_count {
                let angle = (i as f64) * 2.0 * std::f64::consts::PI / (node_count as f64);
                let x = center_x + radius * angle.cos();
                let y = center_y + radius * angle.sin();
                positions.push(Position { x, y });
            }
        }

        Ok(positions)
    }

    /// Apply grid layout algorithm with parameters
    fn apply_grid_layout(
        &self,
        _graph: &Graph,
        embedding: &GraphMatrix,
        params: &std::collections::HashMap<String, String>,
    ) -> GraphResult<Vec<Position>> {
        let node_count = self.node_count_from_state();
        let mut positions = Vec::new();

        // Calculate grid dimensions first
        let grid_size = (node_count as f64).sqrt().ceil() as usize;

        // Parse parameters with defaults (center grid around origin)
        let cell_size = params
            .get("cell_size")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(40.0);
        let start_x = params
            .get("start_x")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(-((grid_size as f64 - 1.0) * cell_size) * 0.5);
        let start_y = params
            .get("start_y")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(-((grid_size as f64 - 1.0) * cell_size) * 0.5);

        if embedding.shape().1 >= 2 {
            // Use embedding coordinates to determine grid ordering
            // Sort nodes by distance from origin to create a more sensible grid ordering
            let mut node_distances: Vec<(usize, f64)> = (0..node_count)
                .map(|i| {
                    let x_val = embedding.get(i, 0).unwrap_or(0.0);
                    let y_val = embedding.get(i, 1).unwrap_or(0.0);
                    let distance = (x_val * x_val + y_val * y_val).sqrt();
                    (i, distance)
                })
                .collect();

            // Sort by distance to create radial ordering in grid
            node_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Create positions vector with correct size
            positions.resize(node_count, Position { x: 0.0, y: 0.0 });

            // Place nodes in grid based on embedding-determined ordering
            for (grid_pos, &(node_idx, _)) in node_distances.iter().enumerate() {
                let row = grid_pos / grid_size;
                let col = grid_pos % grid_size;

                let x = start_x + (col as f64) * cell_size;
                let y = start_y + (row as f64) * cell_size;

                positions[node_idx] = Position { x, y };
            }
        } else {
            // Fallback to simple grid arrangement
            for i in 0..node_count {
                let row = i / grid_size;
                let col = i % grid_size;

                let x = start_x + (col as f64) * cell_size;
                let y = start_y + (row as f64) * cell_size;

                positions.push(Position { x, y });
            }
        }

        Ok(positions)
    }

    // Placeholder methods for incremental updates and other features
    async fn trigger_full_recomputation(&mut self) -> GraphResult<()> {
        self.compute_initial_layout().await
    }

    async fn trigger_projection_recomputation(&mut self) -> GraphResult<()> {
        // Check if we have a cached embedding to reuse
        let existing_embedding = {
            let state = self.state.lock().unwrap();
            state.embedding.clone()
        };

        let embedding = match existing_embedding {
            Some(embedding) => embedding,
            None => {
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
        }

        let params_changed = {
            let state = self.state.lock().unwrap();
            let mut map = HashMap::new();
            map.insert(
                "layout".to_string(),
                json!({
                    "algorithm": state.current_layout.to_string(),
                    "params": state.current_layout_params.clone(),
                }),
            );
            map
        };

        // Broadcast recomputation result
        self.broadcast_envelope(Some(params_changed), None, None, true)?;

        Ok(())
    }

    async fn trigger_view_aware_projection(&mut self, view: ViewState3D) -> GraphResult<()> {
        // Check if we have a cached embedding to reuse
        let existing_embedding = {
            let state = self.state.lock().unwrap();
            state.embedding.clone()
        };

        let embedding = match existing_embedding {
            Some(embedding) => embedding,
            None => {
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

        // Apply 3D view transform to embedding before 2D projection
        let transformed_embedding = self.apply_3d_view_transform(&embedding, &view)?;

        // Phase 2: Project transformed 3D points to 2D coordinates
        let positions = self.apply_layout_algorithm(&transformed_embedding)?;

        let projection_time = projection_start.elapsed();

        // Update state with new positions and timing
        {
            let mut state = self.state.lock().unwrap();
            state.positions = positions;
            state.performance.last_projection_time_ms = projection_time.as_secs_f64() * 1000.0;
            state.last_update = Instant::now();
        }

        // Broadcast the updated positions and view state
        let view_json = json!({
            "view_3d": {
                "center": view.center,
                "distance": view.distance,
                "quat": view.quat,
            }
        });
        self.broadcast_envelope(None, None, Some(view_json), true)?;

        Ok(())
    }

    fn apply_3d_view_transform(
        &self,
        embedding: &GraphMatrix,
        view: &ViewState3D,
    ) -> GraphResult<GraphMatrix> {
        use crate::viz::realtime::interaction::math::{Quat, Vec3};

        // Create quaternion from view state
        let quat = Quat {
            w: view.quat[0],
            x: view.quat[1],
            y: view.quat[2],
            z: view.quat[3],
        };

        // Get embedding dimensions and data
        let (n_nodes, n_dims) = embedding.shape();
        // Create a vector to hold the embedding data
        let mut embedding_data = Vec::with_capacity(n_nodes * n_dims);
        for row in 0..n_nodes {
            for col in 0..n_dims {
                embedding_data.push(embedding.get(row, col).unwrap_or(0.0));
            }
        }

        // For 3D transformations, we need at least 3 dimensions
        // If embedding has fewer than 3 dims, pad with zeros
        // If it has more, we'll transform the first 3 dimensions
        let dims_to_transform = std::cmp::min(n_dims, 3);

        let mut transformed_data = embedding_data.clone();

        // Transform each point
        for node_idx in 0..n_nodes {
            // Extract 3D point (pad with zeros if needed)
            let mut point = Vec3::new(0.0, 0.0, 0.0);

            if dims_to_transform >= 1 {
                point.x = embedding_data[node_idx * n_dims];
            }
            if dims_to_transform >= 2 {
                point.y = embedding_data[node_idx * n_dims + 1];
            }
            if dims_to_transform >= 3 {
                point.z = embedding_data[node_idx * n_dims + 2];
            }

            // Apply 3D transformation:
            // 1. Translate by view center
            point = point.add(Vec3::new(view.center[0], view.center[1], view.center[2]));

            // 2. Apply quaternion rotation
            point = quat.rotate_vec3(point);

            // 3. Scale by distance (for zoom effect)
            let distance_scale = view.distance / 600.0; // normalize around default distance
            point = point.mul(distance_scale);

            // Store transformed coordinates back
            if dims_to_transform >= 1 {
                transformed_data[node_idx * n_dims] = point.x;
            }
            if dims_to_transform >= 2 {
                transformed_data[node_idx * n_dims + 1] = point.y;
            }
            if dims_to_transform >= 3 {
                transformed_data[node_idx * n_dims + 2] = point.z;
            }
        }

        // Create new GraphMatrix with transformed data
        GraphMatrix::from_row_major_data(transformed_data, n_nodes, n_dims, None)
    }

    async fn apply_nd_rotation(
        &mut self,
        axis_i: usize,
        axis_j: usize,
        radians: f64,
    ) -> GraphResult<()> {
        // Get the current embedding
        let existing_embedding = {
            let state = self.state.lock().unwrap();
            state.embedding.clone()
        };

        let embedding = match existing_embedding {
            Some(embedding) => embedding,
            None => {
                let graph = self.graph.lock().unwrap();
                let embedding = graph.compute_embedding(&self.config.embedding_config)?;
                drop(graph);

                // Cache the embedding for future operations
                {
                    let mut state = self.state.lock().unwrap();
                    state.embedding = Some(embedding.clone());
                }
                embedding
            }
        };

        // Apply N-D rotation to embedding
        let rotated_embedding = self.rotate_embedding_nd(&embedding, axis_i, axis_j, radians)?;

        // Update the cached embedding with rotated version
        {
            let mut state = self.state.lock().unwrap();
            state.embedding = Some(rotated_embedding.clone());
        }

        // Trigger projection recomputation with the rotated embedding
        self.trigger_projection_recomputation().await?;

        Ok(())
    }

    fn rotate_embedding_nd(
        &self,
        embedding: &GraphMatrix,
        axis_i: usize,
        axis_j: usize,
        radians: f64,
    ) -> GraphResult<GraphMatrix> {
        let (n_nodes, n_dims) = embedding.shape();

        // Validate axes
        if axis_i >= n_dims || axis_j >= n_dims || axis_i == axis_j {
            return Err(GraphError::InvalidInput(format!(
                "Invalid rotation axes: axis_i={}, axis_j={}, dims={}",
                axis_i, axis_j, n_dims
            )));
        }

        // Create a vector to hold the embedding data
        let mut embedding_data = Vec::with_capacity(n_nodes * n_dims);
        for row in 0..n_nodes {
            for col in 0..n_dims {
                embedding_data.push(embedding.get(row, col).unwrap_or(0.0));
            }
        }
        let mut rotated_data = embedding_data.clone();

        // Precompute rotation values
        let cos_theta = radians.cos();
        let sin_theta = radians.sin();

        // Apply 2D rotation in the specified plane for each point
        for node_idx in 0..n_nodes {
            let base_idx = node_idx * n_dims;

            // Get coordinates for the two axes
            let xi = embedding_data[base_idx + axis_i];
            let xj = embedding_data[base_idx + axis_j];

            // Apply 2D rotation matrix
            let new_xi = xi * cos_theta - xj * sin_theta;
            let new_xj = xi * sin_theta + xj * cos_theta;

            // Store rotated coordinates
            rotated_data[base_idx + axis_i] = new_xi;
            rotated_data[base_idx + axis_j] = new_xj;
        }

        GraphMatrix::from_row_major_data(rotated_data, n_nodes, n_dims, None)
    }

    async fn process_interaction_commands(
        &mut self,
        commands: Vec<InteractionCommand>,
    ) -> GraphResult<()> {
        for command in commands {
            match command {
                InteractionCommand::RotateEmbedding {
                    axis_i,
                    axis_j,
                    radians,
                } => {
                    self.apply_nd_rotation(axis_i, axis_j, radians).await?;
                }
                InteractionCommand::TriggerRecomputation => {
                    self.trigger_projection_recomputation().await?;
                }
                InteractionCommand::UpdateAutoScale {
                    target_occupancy,
                    min_cell_size,
                } => {
                    // Update honeycomb controller configuration
                    if let Some(honeycomb) = self
                        .active_controller
                        .as_any()
                        .downcast_mut::<HoneycombController>()
                    {
                        honeycomb.configure_auto_scaling(target_occupancy, min_cell_size, true);
                    }
                }
                InteractionCommand::ExposeAutoScaleControls {
                    target_occupancy,
                    min_cell_size,
                    enabled,
                } => {
                    // Create UI control exposure message
                    let controls_json = json!({
                        "auto_scale_controls": {
                            "target_occupancy": {
                                "value": target_occupancy,
                                "min": 0.5,
                                "max": 2.0,
                                "step": 0.1,
                                "label": "Target Occupancy",
                                "description": "Target nodes per honeycomb cell (1.0 = one node per cell, optimal)"
                            },
                            "min_cell_size": {
                                "value": min_cell_size,
                                "min": 8.0,
                                "max": 60.0,
                                "step": 2.0,
                                "label": "Min Cell Size",
                                "description": "Minimum size of honeycomb cells in pixels"
                            },
                            "enabled": enabled
                        }
                    });

                    // Broadcast the control exposure to clients
                    self.broadcast_envelope(None, None, Some(controls_json), false)?;
                }
            }
        }
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
                let x = if !node_pos.coords.is_empty() {
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

        Ok(())
    }

    /// Apply a delta update to the engine state
    pub async fn apply(&mut self, update: EngineUpdate) -> GraphResult<()> {
        // Apply control-style updates immediately (do NOT enqueue)
        match &update {
            EngineUpdate::LayoutChanged { .. }
            | EngineUpdate::PositionsBatch(_)  // Position updates should be broadcast immediately
            // Add other non-structural/visual control updates you want to bypass ordering for:
            // | EngineUpdate::UpdateQuality { .. }
            // | EngineUpdate::UpdateAnimation { .. }
            => {
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
            EngineUpdate::NodeAdded(_node) => {
                // TODO: Add node to graph and update positions
            }
            EngineUpdate::NodeRemoved(_node_id) => {
                // TODO: Remove node from graph and positions
            }
            EngineUpdate::EdgeAdded(_edge) => {
                // TODO: Add edge to graph
            }
            EngineUpdate::EdgeRemoved(_edge_id) => {
                // TODO: Remove edge from graph
            }
            EngineUpdate::NodeChanged {
                id: _id,
                attributes: _attributes,
            } => {
                // TODO: Update node attributes in graph
            }
            EngineUpdate::EdgeChanged {
                id: _id,
                attributes: _attributes,
            } => {
                // TODO: Update edge attributes in graph
            }
            EngineUpdate::PositionDelta { node_id, delta } => {
                // Apply position delta using proper node_id mapping
                let mut state = self.state.lock().unwrap();

                if let Some(&position_index) = state.node_index.get(&node_id) {
                    if let Some(pos) = state.positions.get_mut(position_index) {
                        // Apply delta to x,y coordinates
                        if !delta.is_empty() {
                            pos.x += delta[0];
                        }
                        if delta.len() > 1 {
                            pos.y += delta[1];
                        }
                        if delta.len() > 2 {
                            // For 3D/N-D support in the future
                        }
                    }
                }

                state.last_update = Instant::now();
            }
            EngineUpdate::PositionsBatch(position_batch) => {
                let mut state = self.state.lock().unwrap();

                for node_pos in position_batch {
                    if let Some(&position_index) = state.node_index.get(&node_pos.node_id) {
                        if let Some(pos) = state.positions.get_mut(position_index) {
                            // Convert N-dimensional coords to 2D for display
                            if !node_pos.coords.is_empty() {
                                pos.x = node_pos.coords[0];
                            }
                            if node_pos.coords.len() > 1 {
                                pos.y = node_pos.coords[1];
                            }
                        }
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
                match algorithm.parse::<LayoutKind>() {
                    Ok(layout_kind) => {
                        self.apply_layout_change(layout_kind, params).await?;
                    }
                    Err(err) => {
                        eprintln!(
                            "⚠️  WARNING: {} – ignoring layout change and keeping current layout",
                            err
                        );
                    }
                }
            }
            EngineUpdate::EmbeddingChanged { method, dimensions } => {
                // Map method string to EmbeddingMethod enum
                let embedding_method = match method.as_str() {
                    "spectral" => EmbeddingMethod::Spectral {
                        normalized: true,
                        eigenvalue_threshold: 1e-6,
                    },
                    "random" => EmbeddingMethod::RandomND {
                        distribution: crate::viz::embeddings::RandomDistribution::Gaussian {
                            mean: 0.0,
                            stddev: 1.0,
                        },
                        normalize: true,
                    },
                    "energy" => EmbeddingMethod::EnergyND {
                        iterations: 1000,
                        learning_rate: 0.01,
                        annealing: true,
                    },
                    "force_directed" => EmbeddingMethod::ForceDirectedND {
                        spring_constant: 1.0,
                        repulsion_strength: 100.0,
                        iterations: 1000,
                    },
                    // Fallback to spectral for unknown methods
                    _ => {
                        eprintln!(
                            "⚠️  WARNING: Unknown embedding method '{}', falling back to spectral",
                            method
                        );
                        EmbeddingMethod::Spectral {
                            normalized: true,
                            eigenvalue_threshold: 1e-6,
                        }
                    }
                };

                // Update the current embedding configuration in state
                self.config.embedding_config.dimensions = dimensions;
                self.config.embedding_config.method = embedding_method;

                eprintln!(
                    "🧠 DEBUG: Mapped method '{}' to enum variant, updated config",
                    method
                );

                // Trigger full recomputation (embedding + projection)
                self.trigger_full_recomputation().await?;
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
                // Unhandled update type
            }
        }

        // Broadcast the update to subscribers
        let _ = self.update_broadcaster.send(update_for_broadcast);

        Ok(())
    }

    async fn apply_layout_change(
        &mut self,
        layout_kind: LayoutKind,
        mut incoming_params: HashMap<String, String>,
    ) -> GraphResult<()> {
        eprintln!(
            "📐 DEBUG: Applying layout change to {} with params {:?}",
            layout_kind, incoming_params
        );

        if incoming_params.is_empty() {
            if let Some(cached) = {
                let state = self.state.lock().unwrap();
                state.layout_param_cache.get(&layout_kind).cloned()
            } {
                eprintln!(
                    "📐 DEBUG: Restoring cached params for {}: {:?}",
                    layout_kind, cached
                );
                incoming_params = cached;
            }
        }

        self.configure_controller_for_layout(layout_kind);
        self.broadcast_view_state().await?;

        self.cancel_interpolation();
        self.reset_layout_config_for(layout_kind);

        {
            let mut state = self.state.lock().unwrap();
            let previous_layout = state.current_layout;
            let previous_snapshot = state.current_layout_params.clone();
            state
                .layout_param_cache
                .insert(previous_layout, previous_snapshot);
            state.current_layout = layout_kind;
            state.current_layout_params = incoming_params.clone();
            state
                .layout_param_cache
                .insert(layout_kind, incoming_params.clone());
        }

        {
            let map = self.pending_params_changed.get_or_insert_with(HashMap::new);
            map.insert(
                "layout.algorithm".to_string(),
                json!(layout_kind.to_string()),
            );
        }

        let flushed_updates = self.sync_manager.flush_all_coalesced().await?;
        if !flushed_updates.is_empty() {
            eprintln!(
                "🚨 DEBUG: Discarded {} coalesced updates prior to layout recompute",
                flushed_updates.len()
            );
        }

        self.trigger_projection_recomputation().await?;

        Ok(())
    }

    fn cancel_interpolation(&mut self) {
        let mut state = self.state.lock().unwrap();
        if state.animation_state.is_animating {
            // Cancelling active interpolation before layout switch
        }
        state.animation_state = AnimationState::default();
    }

    fn reset_layout_config_for(&mut self, layout_kind: LayoutKind) {
        if let LayoutKind::Honeycomb = layout_kind {
            // Resetting honeycomb projection config to defaults
            self.config.projection_config.honeycomb_config =
                crate::viz::projection::HoneycombConfig::default();
        }
    }

    async fn broadcast_view_state(&mut self) -> GraphResult<()> {
        if let Some(view) = self.active_controller.view_3d() {
            let view_json = json!({
                "view_3d": {
                    "center": view.center,
                    "distance": view.distance,
                    "quat": view.quat,
                }
            });

            // For 3D controllers, re-project embedding with view transform to show visible orbiting
            if self.active_controller.name() == "globe-3d" {
                // Trigger view-aware projection that applies 3D transform before 2D projection
                self.trigger_view_aware_projection(view.clone()).await?;
            } else {
                // For non-globe 3D controllers, just broadcast view state
                self.broadcast_envelope(None, None, Some(view_json), false)?;
            }
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

        // Activate the controller and get any commands it wants to send
        let embedding_dims = {
            let state = self.state.lock().unwrap();
            state.embedding.as_ref().map(|emb| emb.shape().1)
        };

        let activation_commands = self.active_controller.on_activate(embedding_dims);

        // Process activation commands (like exposing auto-scale controls)
        if !activation_commands.is_empty() {
            // We need to spawn a task to process async commands, but for now just log them
            for cmd in activation_commands {
                if let InteractionCommand::ExposeAutoScaleControls {
                    target_occupancy: _,
                    min_cell_size: _,
                    enabled: _,
                } = cmd
                {}
            }
        }
    }

    fn configure_controller_for_layout(&mut self, algorithm: LayoutKind) {
        let mode = match algorithm {
            LayoutKind::Honeycomb => "honeycomb-nd",
            LayoutKind::ForceDirected | LayoutKind::Circular | LayoutKind::Grid => "pan-2d",
        };
        self.set_interaction_controller(mode);
    }

    /// Calculate honeycomb cell size based on embedding spread and target bin count
    /// This implements the bin density calculation from the fix plan
    fn calculate_honeycomb_cell_size_from_bins(
        &self,
        embedding: &GraphMatrix,
        target_bins: f64,
    ) -> Option<f64> {
        let (rows, cols) = embedding.shape();
        if rows == 0 || cols < 2 {
            return None;
        }

        // Compute rough spread on first 2 dimensions
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );

        for i in 0..rows {
            if let Some(x_val) = embedding.get(i, 0) {
                let x = x_val.to_f64();
                if x.is_finite() {
                    xmin = xmin.min(x);
                    xmax = xmax.max(x);
                }
            }
            if let Some(y_val) = embedding.get(i, 1) {
                let y = y_val.to_f64();
                if y.is_finite() {
                    ymin = ymin.min(y);
                    ymax = ymax.max(y);
                }
            }
        }

        if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
            return None;
        }

        let dx = (xmax - xmin).max(1e-9);
        let dy = (ymax - ymin).max(1e-9);

        // Pick cell size so we get ~target_bins across the larger span
        let span = dx.max(dy);
        let cell_size = span / target_bins.max(1.0);

        if cell_size.is_finite() && cell_size > 0.0 {
            // Apply reasonable bounds
            Some(cell_size.clamp(1.0, 1000.0))
        } else {
            None
        }
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
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }

            if usable_cols > 1 {
                if let Some(val_y) = embedding.get(row, 1) {
                    let y = val_y.to_f64();
                    if y.is_finite() {
                        min_y = min_y.min(y);
                        max_y = max_y.max(y);
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
        let cfg = &self.config.projection_config.honeycomb_config;

        if cfg.auto_cell_size && cfg.target_cols > 0 && cfg.target_rows > 0 {
            let cell_x = width / ((cfg.target_cols as f64).max(1.0) * 1.5);
            let cell_y = height / ((cfg.target_rows as f64).max(1.0) * (3.0f64).sqrt());
            let mut cell = cell_x.min(cell_y) * cfg.scale_multiplier.max(0.1);
            if cfg.min_cell_size > 0.0 {
                cell = cell.max(cfg.min_cell_size);
            }
            return if cell.is_finite() {
                Some(cell.clamp(4.0, 400.0))
            } else {
                None
            };
        }

        // Legacy fallback: approximate number of cells from target occupancy
        let target_avg = cfg.target_avg_occupancy.max(0.1);
        let desired_cells = (nodes / target_avg).max(1.0);

        let aspect = (width / height).clamp(0.2, 5.0);
        let width_cells = (desired_cells * aspect).sqrt().max(1.0);
        let height_cells = (desired_cells / width_cells).max(1.0);

        let cell_x = width / (width_cells * (3.0f64).sqrt());
        let cell_y = height / (height_cells * 1.5);
        let mut cell = cell_x.min(cell_y) * cfg.scale_multiplier.max(0.1);
        if cfg.min_cell_size > 0.0 {
            cell = cell.max(cfg.min_cell_size);
        }

        if cell.is_finite() {
            Some(cell.clamp(4.0, 400.0))
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
                let layout_kind = state.current_layout;
                state
                    .current_layout_params
                    .insert(key.to_string(), formatted.clone());

                let snapshot = state.current_layout_params.clone();
                state.layout_param_cache.insert(layout_kind, snapshot);
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
                LayoutKind,
                HashMap<String, String>,
                Vec<NodePosition>,
            ) = {
                let state = self.state.lock().unwrap();
                let layout = state.current_layout;
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

            let layout_string = layout.to_string();

            let payload = if positions_vec.is_empty() {
                None
            } else {
                Some(PositionsPayload {
                    positions: positions_vec.clone(),
                    layout: Some(layout_string.clone()),
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
        self.update_broadcaster.subscribe()
    }

    /// Fallback method when engine layout fails - delegates to accessor's layout method
    pub async fn fallback_to_accessor_layout(
        &mut self,
        accessor: &dyn RealtimeVizAccessor,
    ) -> GraphResult<()> {
        // Get a fresh snapshot from the accessor which should include proper layout
        match accessor.initial_snapshot() {
            Ok(snapshot) => {
                self.load_snapshot(snapshot).await?;
                // Fallback layout loaded successfully
                Ok(())
            }
            Err(e) => {
                // Fallback also failed
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

    #[test]
    fn center_and_fit_positions_centers_and_scales() {
        let positions = vec![
            Position { x: 100.0, y: 50.0 },
            Position { x: 200.0, y: 150.0 },
            Position { x: 300.0, y: -50.0 },
        ];
        let target = 250.0;
        let out = RealTimeVizEngine::center_and_fit_positions(positions, Some(target));

        // centroid ~ (0,0)
        let (mut sx, mut sy) = (0.0, 0.0);
        for p in &out {
            sx += p.x;
            sy += p.y;
        }
        let n = out.len() as f64;
        let (mx, my) = (sx / n, sy / n);
        assert!(mx.abs() < 1e-6, "mx={mx}");
        assert!(my.abs() < 1e-6, "my={my}");

        // half-diagonal ≤ target + epsilon
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for p in &out {
            if p.x.is_finite() {
                xmin = xmin.min(p.x);
                xmax = xmax.max(p.x);
            }
            if p.y.is_finite() {
                ymin = ymin.min(p.y);
                ymax = ymax.max(p.y);
            }
        }
        let hw = (xmax - xmin) * 0.5;
        let hh = (ymax - ymin) * 0.5;
        let half_diag = (hw * hw + hh * hh).sqrt();
        assert!(
            half_diag <= target * 1.0001,
            "half_diag={half_diag} > target={target}"
        );
    }
}
