//! Main VizEngine that coordinates physics, rendering, and interaction
//!
//! This is the core engine that all visualization backends use. It provides
//! a single update() method that produces VizFrames for consumption by adapters.

use std::collections::HashMap;
use crate::errors::GraphResult;
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge, Position};
// use crate::viz::layouts::LayoutEngine; // TODO: Will be used for custom layout engines
use super::{
    physics::{PhysicsEngine, PhysicsState},
    rendering::{RenderingEngine, RenderingConfig, RenderOutput},
    interaction::InteractionState,
    frame::{VizFrame, FrameMetadata, SimulationFrameState, FrameDimensions, PerformanceMetrics, BoundingBox},
};

/// Main visualization engine that coordinates all components
pub struct VizEngine {
    // Core state
    nodes: Vec<VizNode>,
    edges: Vec<VizEdge>,
    
    // Engines
    physics: PhysicsEngine,
    renderer: RenderingEngine,
    
    // State
    positions: HashMap<String, Position>,
    physics_state: Option<PhysicsState>,
    interaction_state: InteractionState,
    
    // Configuration
    config: VizConfig,
    
    // Performance tracking
    last_update_time: Option<std::time::Instant>,
    frame_count: u64,
}

/// Configuration for the visualization engine
#[derive(Debug, Clone)]
pub struct VizConfig {
    /// Canvas dimensions
    pub width: f64,
    pub height: f64,
    
    /// Whether physics simulation is enabled
    pub physics_enabled: bool,
    
    /// Whether to run physics continuously or just once
    pub continuous_physics: bool,
    
    /// Target frame rate for continuous updates
    pub target_fps: f64,
    
    /// Whether interactions are enabled
    pub interactions_enabled: bool,
    
    /// Auto-fit graph to canvas
    pub auto_fit: bool,
    
    /// Padding around graph when auto-fitting
    pub fit_padding: f64,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            physics_enabled: true,
            continuous_physics: false,
            target_fps: 60.0,
            interactions_enabled: true,
            auto_fit: true,
            fit_padding: 50.0,
        }
    }
}

impl VizEngine {
    /// Create a new visualization engine
    pub fn new(config: VizConfig) -> Self {
        let physics = PhysicsEngine::new();
        
        let render_config = RenderingConfig {
            width: config.width,
            height: config.height,
            ..Default::default()
        };
        let renderer = RenderingEngine::new(render_config);
        
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            physics,
            renderer,
            positions: HashMap::new(),
            physics_state: None,
            interaction_state: InteractionState::new(),
            config,
            last_update_time: None,
            frame_count: 0,
        }
    }
    
    /// Set nodes and edges for visualization
    pub fn set_data(&mut self, nodes: Vec<VizNode>, edges: Vec<VizEdge>) -> GraphResult<()> {
        self.nodes = nodes;
        self.edges = edges;
        
        // Initialize physics if enabled
        if self.config.physics_enabled {
            self.initialize_physics()?;
        } else {
            // Use simple layout for initial positions
            self.initialize_simple_layout()?;
        }
        
        Ok(())
    }
    
    /// Update the visualization and return a new frame
    /// This is the main method used by ALL adapters
    pub fn update(&mut self) -> GraphResult<VizFrame> {
        let start_time = std::time::Instant::now();
        
        // Run physics simulation step if enabled and needed
        if self.config.physics_enabled {
            self.update_physics()?;
        }
        
        // Auto-fit if needed
        if self.config.auto_fit {
            self.apply_auto_fit();
        }
        
        // Create frame
        let mut frame = VizFrame::new(&self.nodes, &self.edges, &self.positions);
        
        // Update frame with current state
        self.update_frame_metadata(&mut frame, start_time);
        self.apply_interaction_state(&mut frame);
        
        self.frame_count += 1;
        self.last_update_time = Some(start_time);
        
        Ok(frame)
    }
    
    /// Force a complete physics simulation (useful for static layouts)
    pub fn simulate_to_completion(&mut self) -> GraphResult<VizFrame> {
        if !self.config.physics_enabled {
            return self.update(); // Just return current frame
        }
        
        let start_time = std::time::Instant::now();
        
        // Run complete simulation
        let final_positions = self.physics.simulate(&self.nodes, &self.edges)?;
        self.positions = final_positions;
        
        // Create final frame
        let mut frame = VizFrame::new(&self.nodes, &self.edges, &self.positions);
        self.update_frame_metadata(&mut frame, start_time);
        self.apply_interaction_state(&mut frame);
        
        // Mark simulation as completed
        frame.metadata.simulation_state.is_running = false;
        frame.metadata.simulation_state.has_converged = true;
        frame.metadata.simulation_state.alpha = 0.0;
        
        Ok(frame)
    }
    
    /// Initialize physics simulation
    fn initialize_physics(&mut self) -> GraphResult<()> {
        if !self.nodes.is_empty() {
            let state = self.physics.initialize_simulation(&self.nodes);
            
            // Extract initial positions
            for (i, node) in self.nodes.iter().enumerate() {
                self.positions.insert(node.id.clone(), state.positions[i].clone());
            }
            
            self.physics_state = Some(state);
        }
        
        Ok(())
    }
    
    /// Update physics simulation by one step
    fn update_physics(&mut self) -> GraphResult<()> {
        if let Some(ref mut state) = self.physics_state {
            // Check if we should continue simulating
            let should_continue = self.config.continuous_physics || 
                                state.alpha > self.physics.alpha_min;
            
            if should_continue {
                // Run one simulation step
                self.physics.step(state, &self.nodes, &self.edges);
                
                // Update positions from physics state
                for (i, node) in self.nodes.iter().enumerate() {
                    self.positions.insert(node.id.clone(), state.positions[i].clone());
                }
            }
        }
        
        Ok(())
    }
    
    /// Initialize simple layout (for when physics is disabled)
    fn initialize_simple_layout(&mut self) -> GraphResult<()> {
        if self.nodes.is_empty() {
            return Ok(());
        }
        
        // Use circular layout as default
        let radius = 200.0;
        let angle_step = 2.0 * std::f64::consts::PI / self.nodes.len() as f64;
        
        for (i, node) in self.nodes.iter().enumerate() {
            let angle = i as f64 * angle_step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            self.positions.insert(node.id.clone(), Position { x, y });
        }
        
        Ok(())
    }
    
    /// Apply auto-fit to center and scale the graph
    fn apply_auto_fit(&mut self) {
        if self.positions.is_empty() {
            return;
        }
        
        // Calculate current bounds
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        
        for pos in self.positions.values() {
            min_x = min_x.min(pos.x);
            max_x = max_x.max(pos.x);
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
        }
        
        let graph_width = max_x - min_x;
        let graph_height = max_y - min_y;
        
        if graph_width == 0.0 || graph_height == 0.0 {
            return; // Avoid division by zero
        }
        
        // Calculate scale to fit with padding
        let available_width = self.config.width - 2.0 * self.config.fit_padding;
        let available_height = self.config.height - 2.0 * self.config.fit_padding;
        
        let scale_x = available_width / graph_width;
        let scale_y = available_height / graph_height;
        let scale = scale_x.min(scale_y); // Use smaller scale to fit both dimensions
        
        // Calculate center offsets
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;
        let target_center_x = 0.0; // Center at origin
        let target_center_y = 0.0;
        
        // Apply transformation to all positions
        for pos in self.positions.values_mut() {
            // Translate to origin, scale, then translate to target center
            pos.x = (pos.x - center_x) * scale + target_center_x;
            pos.y = (pos.y - center_y) * scale + target_center_y;
        }
        
        // Update physics state positions if active
        if let Some(ref mut state) = self.physics_state {
            for (i, node) in self.nodes.iter().enumerate() {
                if let Some(pos) = self.positions.get(&node.id) {
                    state.positions[i] = pos.clone();
                }
            }
        }
    }
    
    /// Update frame metadata
    fn update_frame_metadata(&self, frame: &mut VizFrame, start_time: std::time::Instant) {
        let generation_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        frame.metadata = FrameMetadata {
            frame_id: format!("frame_{}", self.frame_count),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            simulation_state: if let Some(ref state) = self.physics_state {
                SimulationFrameState {
                    alpha: state.alpha,
                    iteration: state.iteration,
                    energy: state.energy,
                    is_running: self.config.physics_enabled && state.alpha > self.physics.alpha_min,
                    has_converged: state.alpha <= self.physics.alpha_min,
                }
            } else {
                SimulationFrameState {
                    alpha: 0.0,
                    iteration: 0,
                    energy: 0.0,
                    is_running: false,
                    has_converged: true,
                }
            },
            dimensions: FrameDimensions {
                width: self.config.width,
                height: self.config.height,
                bounds: self.calculate_bounds(),
                zoom: self.interaction_state.camera.zoom,
                pan: self.interaction_state.camera.pan.clone(),
            },
            performance: PerformanceMetrics {
                generation_time_ms: generation_time,
                physics_time_ms: 0.0, // Would be measured separately in real implementation
                rendering_time_ms: 0.0,
                node_count: self.nodes.len(),
                edge_count: self.edges.len(),
            },
        };
    }
    
    /// Calculate bounding box of current positions
    fn calculate_bounds(&self) -> BoundingBox {
        if self.positions.is_empty() {
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
        
        for pos in self.positions.values() {
            min_x = min_x.min(pos.x);
            max_x = max_x.max(pos.x);
            min_y = min_y.min(pos.y);
            max_y = max_y.max(pos.y);
        }
        
        BoundingBox { min_x, max_x, min_y, max_y }
    }
    
    /// Apply interaction state to frame nodes
    fn apply_interaction_state(&self, frame: &mut VizFrame) {
        for node in &mut frame.nodes {
            node.interaction_state = self.interaction_state.get_node_interaction_state(&node.id);
        }
    }
    
    /// Render current frame to specified format
    pub fn render(&self, frame: &VizFrame) -> GraphResult<RenderOutput> {
        self.renderer.render(frame)
    }
    
    // === Interaction Methods ===
    
    /// Start dragging a node
    pub fn start_drag(&mut self, node_id: String, position: Position) {
        self.interaction_state.start_drag(node_id, position);
    }
    
    /// Update drag position
    pub fn update_drag(&mut self, node_id: &str, new_position: Position) {
        self.interaction_state.update_drag(node_id, new_position);
        
        // Update actual node position
        if let Some(current_pos) = self.positions.get_mut(node_id) {
            *current_pos = new_position;
            
            // Update physics state if active
            if let Some(ref mut state) = self.physics_state {
                if let Some(index) = self.nodes.iter().position(|n| n.id == node_id) {
                    state.positions[index] = new_position;
                    // Reset velocity for dragged node
                    state.velocities[index] = Position { x: 0.0, y: 0.0 };
                }
            }
        }
    }
    
    /// End dragging a node
    pub fn end_drag(&mut self, node_id: &str, final_position: Position) {
        self.interaction_state.end_drag(node_id, final_position);
    }
    
    /// Select a node
    pub fn select_node(&mut self, node_id: String) {
        self.interaction_state.select_node(node_id);
    }
    
    /// Deselect a node
    pub fn deselect_node(&mut self, node_id: &str) {
        self.interaction_state.deselect_node(node_id);
    }
    
    /// Set hovered node
    pub fn set_hover(&mut self, node_id: Option<String>) {
        self.interaction_state.set_hover(node_id);
    }
    
    /// Pin a node (fix its position)
    pub fn pin_node(&mut self, node_id: String) {
        self.interaction_state.pin_node(node_id);
    }
    
    /// Unpin a node
    pub fn unpin_node(&mut self, node_id: &str) {
        self.interaction_state.unpin_node(node_id);
    }
    
    /// Set camera zoom
    pub fn set_zoom(&mut self, zoom: f64, center: Position) {
        self.interaction_state.set_zoom(zoom, center);
    }
    
    /// Set camera pan
    pub fn set_pan(&mut self, pan: Position) {
        self.interaction_state.set_pan(pan);
    }
    
    // === Configuration Methods ===
    
    /// Update engine configuration
    pub fn set_config(&mut self, config: VizConfig) -> GraphResult<()> {
        let old_physics_enabled = self.config.physics_enabled;
        self.config = config;
        
        // Reinitialize physics if it was enabled/disabled
        if self.config.physics_enabled != old_physics_enabled {
            if self.config.physics_enabled {
                self.initialize_physics()?;
            } else {
                self.physics_state = None;
            }
        }
        
        // Update renderer dimensions
        self.renderer.config.width = self.config.width;
        self.renderer.config.height = self.config.height;
        
        Ok(())
    }
    
    /// Update physics engine configuration
    pub fn set_physics_config(&mut self, physics: PhysicsEngine) -> GraphResult<()> {
        self.physics = physics;
        
        // Reinitialize if physics is active
        if self.config.physics_enabled {
            self.initialize_physics()?;
        }
        
        Ok(())
    }
    
    /// Get current physics state (for debugging/monitoring)
    pub fn get_physics_state(&self) -> Option<&PhysicsState> {
        self.physics_state.as_ref()
    }
    
    /// Get current interaction state
    pub fn get_interaction_state(&self) -> &InteractionState {
        &self.interaction_state
    }
    
    /// Get current positions
    pub fn get_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }
    
    /// Check if simulation is running
    pub fn is_simulation_running(&self) -> bool {
        self.config.physics_enabled && 
        self.physics_state.as_ref()
            .map(|state| state.alpha > self.physics.alpha_min)
            .unwrap_or(false)
    }
    
    /// Get frame rate information
    pub fn get_frame_info(&self) -> FrameInfo {
        FrameInfo {
            frame_count: self.frame_count,
            target_fps: self.config.target_fps,
            last_update: self.last_update_time,
        }
    }
}

/// Frame rate information
#[derive(Debug, Clone)]
pub struct FrameInfo {
    pub frame_count: u64,
    pub target_fps: f64,
    pub last_update: Option<std::time::Instant>,
}

/// Builder for VizEngine
pub struct VizEngineBuilder {
    config: VizConfig,
    physics: Option<PhysicsEngine>,
    rendering_config: Option<RenderingConfig>,
}

impl VizEngineBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: VizConfig::default(),
            physics: None,
            rendering_config: None,
        }
    }
    
    /// Set dimensions
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }
    
    /// Enable/disable physics
    pub fn with_physics(mut self, enabled: bool) -> Self {
        self.config.physics_enabled = enabled;
        self
    }
    
    /// Set continuous physics
    pub fn with_continuous_physics(mut self, continuous: bool) -> Self {
        self.config.continuous_physics = continuous;
        self
    }
    
    /// Set target FPS
    pub fn with_target_fps(mut self, fps: f64) -> Self {
        self.config.target_fps = fps;
        self
    }
    
    /// Set custom physics engine
    pub fn with_physics_engine(mut self, physics: PhysicsEngine) -> Self {
        self.physics = Some(physics);
        self
    }
    
    /// Set rendering configuration
    pub fn with_rendering_config(mut self, config: RenderingConfig) -> Self {
        self.rendering_config = Some(config);
        self
    }
    
    /// Build the engine
    pub fn build(self) -> VizEngine {
        let mut engine = VizEngine::new(self.config);
        
        if let Some(physics) = self.physics {
            engine.physics = physics;
        }
        
        if let Some(render_config) = self.rendering_config {
            engine.renderer = RenderingEngine::new(render_config);
        }
        
        engine
    }
}

impl Default for VizEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}