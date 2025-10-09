//! Graph layout algorithms for visualization

use crate::errors::GraphResult;
use crate::viz::streaming::data_source::{GraphEdge as VizEdge, GraphNode as VizNode, Position};

/// Trait for graph layout algorithms
pub trait LayoutEngine {
    /// Compute positions for nodes given the graph structure
    fn compute_layout(
        &self,
        nodes: &[VizNode],
        edges: &[VizEdge],
    ) -> GraphResult<Vec<(String, Position)>>;

    /// Get the name of this layout algorithm
    fn name(&self) -> &str;

    /// Check if this layout supports incremental updates
    fn supports_incremental(&self) -> bool {
        false
    }
}

/// Advanced Force-directed layout with comprehensive physics simulation
/// Based on Fruchterman-Reingold and Barnes-Hut optimizations
#[derive(Clone)]
pub struct ForceDirectedLayout {
    // Core physics parameters
    pub charge: f64,       // Node repulsion strength (negative = repulsive)
    pub distance: f64,     // Ideal edge length
    pub iterations: usize, // Maximum simulation steps

    // Advanced physics parameters
    pub gravity: f64,     // Central gravitational force (0-1)
    pub friction: f64,    // Velocity damping (0-1)
    pub theta: f64,       // Barnes-Hut approximation parameter (0-1)
    pub alpha: f64,       // Cooling factor for simulation
    pub alpha_min: f64,   // Minimum alpha before stopping
    pub alpha_decay: f64, // Alpha decay rate per iteration

    // Force strength parameters
    pub link_strength: f64,    // Spring force strength
    pub charge_strength: f64,  // Coulomb force multiplier
    pub center_strength: f64,  // Centering force strength
    pub collision_radius: f64, // Node collision detection radius

    // Simulation bounds and optimization
    pub bounds: Option<(f64, f64, f64, f64)>, // (min_x, max_x, min_y, max_y)
    pub enable_barnes_hut: bool,              // Use Barnes-Hut optimization for large graphs
    pub enable_collision: bool,               // Enable node collision detection
    pub adaptive_cooling: bool,               // Adjust cooling based on system energy

    // Velocity and position constraints
    pub max_velocity: Option<f64>, // Maximum node velocity per iteration
    pub position_constraints: Vec<NodeConstraint>, // Fixed/constrained nodes
}

/// Node positioning constraints
#[derive(Clone, Debug)]
pub struct NodeConstraint {
    pub node_id: String,
    pub constraint_type: ConstraintType,
}

#[derive(Clone, Debug)]
pub enum ConstraintType {
    Fixed(Position),                       // Node fixed at specific position
    CircularBounds(f64),                   // Node constrained to circular area
    RectangularBounds(f64, f64, f64, f64), // Node constrained to rectangle
    AttractedTo(Position, f64),            // Node attracted to position with strength
}

/// Internal simulation state
struct SimulationState {
    positions: Vec<Position>,
    velocities: Vec<Position>,
    forces: Vec<Position>,
    alpha: f64,
    #[allow(dead_code)]
    energy: f64,
    iteration: usize,
}

impl Default for ForceDirectedLayout {
    fn default() -> Self {
        Self {
            // Core parameters
            charge: -300.0,
            distance: 50.0,
            iterations: 300,

            // Physics parameters
            gravity: 0.1,
            friction: 0.9,
            theta: 0.9,
            alpha: 1.0,
            alpha_min: 0.001,
            alpha_decay: 0.99,

            // Force strengths
            link_strength: 0.1,
            charge_strength: 1.0,
            center_strength: 0.1,
            collision_radius: 5.0,

            // Optimization settings
            bounds: None,
            enable_barnes_hut: true,
            enable_collision: true,
            adaptive_cooling: true,
            max_velocity: Some(100.0),
            position_constraints: Vec::new(),
        }
    }
}

impl ForceDirectedLayout {
    /// Create a new force-directed layout with custom parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder pattern for configuring physics parameters
    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_distance(mut self, distance: f64) -> Self {
        self.distance = distance;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_gravity(mut self, gravity: f64) -> Self {
        self.gravity = gravity.clamp(0.0, 1.0);
        self
    }

    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction.clamp(0.0, 1.0);
        self
    }

    pub fn with_alpha_decay(mut self, alpha_decay: f64) -> Self {
        self.alpha_decay = alpha_decay.clamp(0.1, 1.0);
        self
    }

    pub fn with_bounds(mut self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        self.bounds = Some((min_x, max_x, min_y, max_y));
        self
    }

    pub fn enable_optimization(mut self, barnes_hut: bool, collision: bool) -> Self {
        self.enable_barnes_hut = barnes_hut;
        self.enable_collision = collision;
        self
    }

    pub fn add_constraint(mut self, constraint: NodeConstraint) -> Self {
        self.position_constraints.push(constraint);
        self
    }

    /// Initialize random positions for nodes
    fn initialize_positions(&self, nodes: &[VizNode]) -> Vec<Position> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut positions = Vec::with_capacity(nodes.len());

        for node in nodes {
            // Use node ID for deterministic randomness
            let mut hasher = DefaultHasher::new();
            node.id.hash(&mut hasher);
            let seed = hasher.finish();

            // Generate position based on seed
            let x = ((seed & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu32 as f64 - 0.5) * 200.0;
            let y = (((seed >> 32) & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu32 as f64 - 0.5) * 200.0;

            positions.push(Position { x, y });
        }

        positions
    }

    /// Initialize simulation state
    fn initialize_simulation(&self, nodes: &[VizNode]) -> SimulationState {
        let positions = self.initialize_positions(nodes);
        let velocities = vec![Position { x: 0.0, y: 0.0 }; nodes.len()];
        let forces = vec![Position { x: 0.0, y: 0.0 }; nodes.len()];

        SimulationState {
            positions,
            velocities,
            forces,
            alpha: self.alpha,
            energy: 0.0,
            iteration: 0,
        }
    }

    /// Calculate repulsive forces between nodes
    fn calculate_repulsive_forces(&self, state: &mut SimulationState, nodes: &[VizNode]) {
        let n = nodes.len();

        if self.enable_barnes_hut && n > 100 {
            // Use Barnes-Hut approximation for large graphs
            self.barnes_hut_forces(state, nodes);
        } else {
            // Direct N² calculation for smaller graphs
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = state.positions[j].x - state.positions[i].x;
                    let dy = state.positions[j].y - state.positions[i].y;
                    let distance = (dx * dx + dy * dy).sqrt().max(1.0);

                    // Coulomb's law: F = k * q1 * q2 / r²
                    let force_magnitude =
                        self.charge * self.charge_strength * state.alpha / (distance * distance);
                    let force_x = force_magnitude * dx / distance;
                    let force_y = force_magnitude * dy / distance;

                    state.forces[i].x -= force_x;
                    state.forces[i].y -= force_y;
                    state.forces[j].x += force_x;
                    state.forces[j].y += force_y;
                }
            }
        }
    }

    /// Barnes-Hut algorithm for efficient force calculation
    fn barnes_hut_forces(&self, state: &mut SimulationState, _nodes: &[VizNode]) {
        // Simplified Barnes-Hut - in production would use quadtree
        // For now, implement a spatial grid approximation
        let grid_size = 50.0;
        let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> =
            std::collections::HashMap::new();

        // Build spatial grid
        for (i, pos) in state.positions.iter().enumerate() {
            let gx = (pos.x / grid_size).floor() as i32;
            let gy = (pos.y / grid_size).floor() as i32;
            grid.entry((gx, gy)).or_insert_with(Vec::new).push(i);
        }

        // Calculate forces using grid approximation
        for (i, pos) in state.positions.iter().enumerate() {
            let gx = (pos.x / grid_size).floor() as i32;
            let gy = (pos.y / grid_size).floor() as i32;

            // Check neighboring grid cells
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(neighbors) = grid.get(&(gx + dx, gy + dy)) {
                        for &j in neighbors {
                            if i != j {
                                let dx = state.positions[j].x - state.positions[i].x;
                                let dy = state.positions[j].y - state.positions[i].y;
                                let distance = (dx * dx + dy * dy).sqrt().max(1.0);

                                let force_magnitude =
                                    self.charge * self.charge_strength * state.alpha
                                        / (distance * distance);
                                let force_x = force_magnitude * dx / distance;
                                let force_y = force_magnitude * dy / distance;

                                state.forces[i].x -= force_x;
                                state.forces[i].y -= force_y;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Calculate attractive forces from edges
    fn calculate_attractive_forces(
        &self,
        state: &mut SimulationState,
        edges: &[VizEdge],
        node_indices: &std::collections::HashMap<String, usize>,
    ) {
        for edge in edges {
            if let (Some(&source_idx), Some(&target_idx)) = (
                node_indices.get(&edge.source),
                node_indices.get(&edge.target),
            ) {
                let dx = state.positions[target_idx].x - state.positions[source_idx].x;
                let dy = state.positions[target_idx].y - state.positions[source_idx].y;
                let distance = (dx * dx + dy * dy).sqrt().max(1.0);

                // Hooke's law: F = k * (distance - ideal_length)
                let displacement = distance - self.distance;
                let force_magnitude = self.link_strength * displacement * state.alpha;
                let force_x = force_magnitude * dx / distance;
                let force_y = force_magnitude * dy / distance;

                state.forces[source_idx].x += force_x;
                state.forces[source_idx].y += force_y;
                state.forces[target_idx].x -= force_x;
                state.forces[target_idx].y -= force_y;
            }
        }
    }

    /// Apply gravitational force toward center
    fn apply_gravity(&self, state: &mut SimulationState) {
        if self.gravity > 0.0 {
            // Calculate center of mass
            let mut center_x = 0.0;
            let mut center_y = 0.0;
            for pos in &state.positions {
                center_x += pos.x;
                center_y += pos.y;
            }
            center_x /= state.positions.len() as f64;
            center_y /= state.positions.len() as f64;

            // Apply gravitational force
            for (i, pos) in state.positions.iter().enumerate() {
                let dx = center_x - pos.x;
                let dy = center_y - pos.y;
                let force_magnitude = self.gravity * self.center_strength * state.alpha;

                state.forces[i].x += force_magnitude * dx;
                state.forces[i].y += force_magnitude * dy;
            }
        }
    }

    /// Apply collision detection forces
    fn apply_collision_forces(&self, state: &mut SimulationState) {
        if !self.enable_collision {
            return;
        }

        let n = state.positions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = state.positions[j].x - state.positions[i].x;
                let dy = state.positions[j].y - state.positions[i].y;
                let distance = (dx * dx + dy * dy).sqrt();
                let min_distance = self.collision_radius * 2.0;

                if distance < min_distance && distance > 0.0 {
                    let overlap = min_distance - distance;
                    let force_magnitude = overlap * 0.5 * state.alpha;
                    let force_x = force_magnitude * dx / distance;
                    let force_y = force_magnitude * dy / distance;

                    state.forces[i].x -= force_x;
                    state.forces[i].y -= force_y;
                    state.forces[j].x += force_x;
                    state.forces[j].y += force_y;
                }
            }
        }
    }

    /// Apply position constraints
    fn apply_constraints(
        &self,
        state: &mut SimulationState,
        node_indices: &std::collections::HashMap<String, usize>,
    ) {
        for constraint in &self.position_constraints {
            if let Some(&idx) = node_indices.get(&constraint.node_id) {
                match &constraint.constraint_type {
                    ConstraintType::Fixed(pos) => {
                        state.positions[idx] = *pos;
                        state.velocities[idx] = Position { x: 0.0, y: 0.0 };
                    }
                    ConstraintType::CircularBounds(radius) => {
                        let distance = (state.positions[idx].x * state.positions[idx].x
                            + state.positions[idx].y * state.positions[idx].y)
                            .sqrt();
                        if distance > *radius {
                            let scale = radius / distance;
                            state.positions[idx].x *= scale;
                            state.positions[idx].y *= scale;
                        }
                    }
                    ConstraintType::RectangularBounds(min_x, max_x, min_y, max_y) => {
                        state.positions[idx].x = state.positions[idx].x.clamp(*min_x, *max_x);
                        state.positions[idx].y = state.positions[idx].y.clamp(*min_y, *max_y);
                    }
                    ConstraintType::AttractedTo(target, strength) => {
                        let dx = target.x - state.positions[idx].x;
                        let dy = target.y - state.positions[idx].y;
                        state.forces[idx].x += dx * strength * state.alpha;
                        state.forces[idx].y += dy * strength * state.alpha;
                    }
                }
            }
        }
    }

    /// Update positions and velocities based on forces
    fn update_physics(&self, state: &mut SimulationState) {
        for i in 0..state.positions.len() {
            // Update velocity with friction
            state.velocities[i].x = (state.velocities[i].x + state.forces[i].x) * self.friction;
            state.velocities[i].y = (state.velocities[i].y + state.forces[i].y) * self.friction;

            // Apply velocity limits
            if let Some(max_vel) = self.max_velocity {
                let vel_magnitude = (state.velocities[i].x * state.velocities[i].x
                    + state.velocities[i].y * state.velocities[i].y)
                    .sqrt();
                if vel_magnitude > max_vel {
                    let scale = max_vel / vel_magnitude;
                    state.velocities[i].x *= scale;
                    state.velocities[i].y *= scale;
                }
            }

            // Update position
            state.positions[i].x += state.velocities[i].x;
            state.positions[i].y += state.velocities[i].y;

            // Apply bounds if specified
            if let Some((min_x, max_x, min_y, max_y)) = self.bounds {
                state.positions[i].x = state.positions[i].x.clamp(min_x, max_x);
                state.positions[i].y = state.positions[i].y.clamp(min_y, max_y);
            }

            // Reset forces for next iteration
            state.forces[i].x = 0.0;
            state.forces[i].y = 0.0;
        }
    }

    /// Calculate system energy for adaptive cooling
    fn calculate_energy(&self, state: &SimulationState) -> f64 {
        let mut energy = 0.0;
        for velocity in &state.velocities {
            energy += velocity.x * velocity.x + velocity.y * velocity.y;
        }
        energy.sqrt()
    }

    /// Update alpha (temperature) for simulated annealing
    fn update_alpha(&self, state: &mut SimulationState) {
        if self.adaptive_cooling {
            // Adjust cooling based on system energy
            let energy = self.calculate_energy(state);
            let energy_factor = if energy > 10.0 { 0.99 } else { 0.95 };
            state.alpha *= energy_factor;
        } else {
            state.alpha *= self.alpha_decay;
        }

        state.alpha = state.alpha.max(self.alpha_min);
    }
}

impl LayoutEngine for ForceDirectedLayout {
    fn compute_layout(
        &self,
        nodes: &[VizNode],
        edges: &[VizEdge],
    ) -> GraphResult<Vec<(String, Position)>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Create node index mapping
        let node_indices: std::collections::HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();

        // Initialize simulation
        let mut state = self.initialize_simulation(nodes);

        // Main simulation loop
        for iteration in 0..self.iterations {
            state.iteration = iteration;

            // Calculate all forces
            self.calculate_repulsive_forces(&mut state, nodes);
            self.calculate_attractive_forces(&mut state, edges, &node_indices);
            self.apply_gravity(&mut state);
            self.apply_collision_forces(&mut state);
            self.apply_constraints(&mut state, &node_indices);

            // Update physics
            self.update_physics(&mut state);
            self.update_alpha(&mut state);

            // Check convergence
            if state.alpha < self.alpha_min {
                break;
            }
        }

        // Return final positions
        let positions = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), state.positions[i]))
            .collect();

        Ok(positions)
    }

    fn name(&self) -> &str {
        "force-directed"
    }

    fn supports_incremental(&self) -> bool {
        true
    }
}

/// Circular layout implementation
pub struct CircularLayout {
    pub radius: Option<f64>,
}

impl Default for CircularLayout {
    fn default() -> Self {
        Self { radius: None }
    }
}

impl LayoutEngine for CircularLayout {
    fn compute_layout(
        &self,
        nodes: &[VizNode],
        _edges: &[VizEdge],
    ) -> GraphResult<Vec<(String, Position)>> {
        let mut positions = Vec::new();
        let radius = self.radius.unwrap_or(200.0);
        let angle_step = 2.0 * std::f64::consts::PI / nodes.len() as f64;

        for (i, node) in nodes.iter().enumerate() {
            let angle = i as f64 * angle_step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            positions.push((node.id.clone(), Position { x, y }));
        }

        Ok(positions)
    }

    fn name(&self) -> &str {
        "circular"
    }
}

/// Custom Layout Plugin System
/// Allows users to define and register their own layout algorithms
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Dynamic layout function type
pub type LayoutFunction =
    dyn Fn(&[VizNode], &[VizEdge]) -> GraphResult<Vec<(String, Position)>> + Send + Sync;

/// Custom layout configuration and metadata
#[derive(Clone, Debug)]
pub struct LayoutPlugin {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, ParameterSpec>,
    pub author: Option<String>,
    pub version: Option<String>,
    pub tags: Vec<String>,
}

/// Parameter specification for custom layouts
#[derive(Clone, Debug)]
pub struct ParameterSpec {
    pub name: String,
    pub description: String,
    pub parameter_type: ParameterType,
    pub default_value: ParameterValue,
    pub constraints: Option<ParameterConstraints>,
}

#[derive(Clone, Debug)]
pub enum ParameterType {
    Float,
    Integer,
    Boolean,
    String,
    Choice(Vec<String>),
}

#[derive(Clone, Debug)]
pub enum ParameterValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
}

#[derive(Clone, Debug)]
pub struct ParameterConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub allowed_values: Option<Vec<String>>,
}

/// Custom layout implementation with dynamic parameters
pub struct CustomLayout {
    pub plugin: LayoutPlugin,
    pub parameters: HashMap<String, ParameterValue>,
    pub layout_fn: Arc<LayoutFunction>,
}

impl CustomLayout {
    /// Create a new custom layout
    pub fn new(
        plugin: LayoutPlugin,
        layout_fn: Arc<LayoutFunction>,
        parameters: Option<HashMap<String, ParameterValue>>,
    ) -> Self {
        let mut layout_params = HashMap::new();

        // Initialize with default values
        for (key, spec) in &plugin.parameters {
            layout_params.insert(key.clone(), spec.default_value.clone());
        }

        // Override with provided parameters
        if let Some(params) = parameters {
            for (key, value) in params {
                if plugin.parameters.contains_key(&key) {
                    layout_params.insert(key, value);
                }
            }
        }

        Self {
            plugin,
            parameters: layout_params,
            layout_fn,
        }
    }

    /// Set a parameter value with validation
    pub fn set_parameter(&mut self, name: &str, value: ParameterValue) -> Result<(), String> {
        if let Some(spec) = self.plugin.parameters.get(name) {
            // Validate parameter type
            if !self.validate_parameter_type(&value, &spec.parameter_type) {
                return Err(format!(
                    "Invalid type for parameter '{}': expected {:?}",
                    name, spec.parameter_type
                ));
            }

            // Validate constraints
            if let Some(constraints) = &spec.constraints {
                if !self.validate_parameter_constraints(&value, constraints) {
                    return Err(format!("Parameter '{}' violates constraints", name));
                }
            }

            self.parameters.insert(name.to_string(), value);
            Ok(())
        } else {
            Err(format!("Unknown parameter: {}", name))
        }
    }

    /// Get a parameter value
    pub fn get_parameter(&self, name: &str) -> Option<&ParameterValue> {
        self.parameters.get(name)
    }

    /// Validate parameter type
    fn validate_parameter_type(
        &self,
        value: &ParameterValue,
        expected_type: &ParameterType,
    ) -> bool {
        match (value, expected_type) {
            (ParameterValue::Float(_), ParameterType::Float) => true,
            (ParameterValue::Integer(_), ParameterType::Integer) => true,
            (ParameterValue::Boolean(_), ParameterType::Boolean) => true,
            (ParameterValue::String(_), ParameterType::String) => true,
            (ParameterValue::String(s), ParameterType::Choice(choices)) => choices.contains(s),
            _ => false,
        }
    }

    /// Validate parameter constraints
    fn validate_parameter_constraints(
        &self,
        value: &ParameterValue,
        constraints: &ParameterConstraints,
    ) -> bool {
        match value {
            ParameterValue::Float(f) => {
                if let Some(min) = constraints.min_value {
                    if *f < min {
                        return false;
                    }
                }
                if let Some(max) = constraints.max_value {
                    if *f > max {
                        return false;
                    }
                }
            }
            ParameterValue::Integer(i) => {
                if let Some(min) = constraints.min_value {
                    if (*i as f64) < min {
                        return false;
                    }
                }
                if let Some(max) = constraints.max_value {
                    if (*i as f64) > max {
                        return false;
                    }
                }
            }
            ParameterValue::String(s) => {
                if let Some(allowed) = &constraints.allowed_values {
                    if !allowed.contains(s) {
                        return false;
                    }
                }
            }
            _ => {}
        }
        true
    }
}

impl LayoutEngine for CustomLayout {
    fn compute_layout(
        &self,
        nodes: &[VizNode],
        edges: &[VizEdge],
    ) -> GraphResult<Vec<(String, Position)>> {
        (self.layout_fn)(nodes, edges)
    }

    fn name(&self) -> &str {
        &self.plugin.name
    }

    fn supports_incremental(&self) -> bool {
        // Custom layouts can specify this in their plugin metadata
        self.plugin.tags.contains(&"incremental".to_string())
    }
}

impl Clone for CustomLayout {
    fn clone(&self) -> Self {
        Self {
            plugin: self.plugin.clone(),
            parameters: self.parameters.clone(),
            layout_fn: Arc::clone(&self.layout_fn),
        }
    }
}

/// Global layout plugin registry
pub struct LayoutRegistry {
    layouts: Arc<Mutex<HashMap<String, CustomLayout>>>,
    plugins: Arc<Mutex<HashMap<String, LayoutPlugin>>>,
}

impl LayoutRegistry {
    /// Create a new layout registry
    pub fn new() -> Self {
        let registry = Self {
            layouts: Arc::new(Mutex::new(HashMap::new())),
            plugins: Arc::new(Mutex::new(HashMap::new())),
        };

        // Register built-in layouts
        registry.register_builtin_layouts();
        registry
    }

    /// Register a custom layout plugin
    pub fn register_layout<F>(&self, plugin: LayoutPlugin, layout_fn: F) -> Result<(), String>
    where
        F: Fn(&[VizNode], &[VizEdge]) -> GraphResult<Vec<(String, Position)>>
            + Send
            + Sync
            + 'static,
    {
        let name = plugin.name.clone();

        // Check for name conflicts
        {
            let layouts = self.layouts.lock().unwrap();
            if layouts.contains_key(&name) {
                return Err(format!("Layout '{}' already registered", name));
            }
        }

        // Create custom layout
        let custom_layout = CustomLayout::new(plugin.clone(), Arc::new(layout_fn), None);

        // Register both plugin and layout
        {
            let mut plugins = self.plugins.lock().unwrap();
            plugins.insert(name.clone(), plugin);
        }
        {
            let mut layouts = self.layouts.lock().unwrap();
            layouts.insert(name, custom_layout);
        }

        Ok(())
    }

    /// Get a layout by name
    pub fn get_layout(&self, name: &str) -> Option<CustomLayout> {
        let layouts = self.layouts.lock().unwrap();
        layouts.get(name).cloned()
    }

    /// Get all available layout names
    pub fn list_layouts(&self) -> Vec<String> {
        let layouts = self.layouts.lock().unwrap();
        layouts.keys().cloned().collect()
    }

    /// Get plugin information
    pub fn get_plugin_info(&self, name: &str) -> Option<LayoutPlugin> {
        let plugins = self.plugins.lock().unwrap();
        plugins.get(name).cloned()
    }

    /// Search layouts by tags
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<String> {
        let plugins = self.plugins.lock().unwrap();
        plugins
            .iter()
            .filter(|(_, plugin)| tags.iter().any(|tag| plugin.tags.contains(tag)))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Create a layout instance with custom parameters
    pub fn create_layout_with_params(
        &self,
        name: &str,
        parameters: HashMap<String, ParameterValue>,
    ) -> Result<CustomLayout, String> {
        let layouts = self.layouts.lock().unwrap();
        if let Some(base_layout) = layouts.get(name) {
            let mut custom_layout = base_layout.clone();

            // Apply custom parameters
            for (param_name, param_value) in parameters {
                custom_layout.set_parameter(&param_name, param_value)?;
            }

            Ok(custom_layout)
        } else {
            Err(format!("Layout '{}' not found", name))
        }
    }

    /// Register built-in layouts as plugins
    fn register_builtin_layouts(&self) {
        // Force-directed layout plugin
        let force_directed_plugin = LayoutPlugin {
            name: "force-directed".to_string(),
            description: "Physics-based force-directed layout with comprehensive simulation"
                .to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "charge".to_string(),
                    ParameterSpec {
                        name: "charge".to_string(),
                        description: "Node repulsion strength (negative values)".to_string(),
                        parameter_type: ParameterType::Float,
                        default_value: ParameterValue::Float(-300.0),
                        constraints: Some(ParameterConstraints {
                            min_value: Some(-10000.0),
                            max_value: Some(0.0),
                            allowed_values: None,
                        }),
                    },
                );
                params.insert(
                    "distance".to_string(),
                    ParameterSpec {
                        name: "distance".to_string(),
                        description: "Ideal edge length".to_string(),
                        parameter_type: ParameterType::Float,
                        default_value: ParameterValue::Float(50.0),
                        constraints: Some(ParameterConstraints {
                            min_value: Some(1.0),
                            max_value: Some(1000.0),
                            allowed_values: None,
                        }),
                    },
                );
                params.insert(
                    "iterations".to_string(),
                    ParameterSpec {
                        name: "iterations".to_string(),
                        description: "Maximum simulation iterations".to_string(),
                        parameter_type: ParameterType::Integer,
                        default_value: ParameterValue::Integer(300),
                        constraints: Some(ParameterConstraints {
                            min_value: Some(1.0),
                            max_value: Some(10000.0),
                            allowed_values: None,
                        }),
                    },
                );
                params.insert(
                    "gravity".to_string(),
                    ParameterSpec {
                        name: "gravity".to_string(),
                        description: "Central gravitational force (0-1)".to_string(),
                        parameter_type: ParameterType::Float,
                        default_value: ParameterValue::Float(0.1),
                        constraints: Some(ParameterConstraints {
                            min_value: Some(0.0),
                            max_value: Some(1.0),
                            allowed_values: None,
                        }),
                    },
                );
                params
            },
            author: Some("Groggy Team".to_string()),
            version: Some("1.0.0".to_string()),
            tags: vec![
                "physics".to_string(),
                "force-directed".to_string(),
                "incremental".to_string(),
            ],
        };

        // Circular layout plugin
        let circular_plugin = LayoutPlugin {
            name: "circular".to_string(),
            description: "Arrange nodes in a circle".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "radius".to_string(),
                    ParameterSpec {
                        name: "radius".to_string(),
                        description: "Circle radius (auto if not specified)".to_string(),
                        parameter_type: ParameterType::Float,
                        default_value: ParameterValue::Float(200.0),
                        constraints: Some(ParameterConstraints {
                            min_value: Some(10.0),
                            max_value: Some(5000.0),
                            allowed_values: None,
                        }),
                    },
                );
                params
            },
            author: Some("Groggy Team".to_string()),
            version: Some("1.0.0".to_string()),
            tags: vec![
                "geometric".to_string(),
                "circular".to_string(),
                "simple".to_string(),
            ],
        };

        // Register as custom layouts
        let force_layout_fn =
            |nodes: &[VizNode], edges: &[VizEdge]| -> GraphResult<Vec<(String, Position)>> {
                let layout = ForceDirectedLayout::default();
                layout.compute_layout(nodes, edges)
            };

        let circular_layout_fn =
            |nodes: &[VizNode], edges: &[VizEdge]| -> GraphResult<Vec<(String, Position)>> {
                let layout = CircularLayout::default();
                layout.compute_layout(nodes, edges)
            };

        // Register layouts (ignore errors since they're built-in)
        let _ = self.register_layout(force_directed_plugin, force_layout_fn);
        let _ = self.register_layout(circular_plugin, circular_layout_fn);
    }
}

// Global layout registry instance
lazy_static::lazy_static! {
    pub static ref LAYOUT_REGISTRY: LayoutRegistry = LayoutRegistry::new();
}

/// Convenience functions for common layout operations
impl LayoutRegistry {
    /// Get the global registry instance
    pub fn global() -> &'static LayoutRegistry {
        &LAYOUT_REGISTRY
    }

    /// Register a simple layout function
    pub fn register_simple_layout<F>(
        name: &str,
        description: &str,
        layout_fn: F,
    ) -> Result<(), String>
    where
        F: Fn(&[VizNode], &[VizEdge]) -> GraphResult<Vec<(String, Position)>>
            + Send
            + Sync
            + 'static,
    {
        let plugin = LayoutPlugin {
            name: name.to_string(),
            description: description.to_string(),
            parameters: HashMap::new(),
            author: None,
            version: None,
            tags: vec!["custom".to_string()],
        };

        Self::global().register_layout(plugin, layout_fn)
    }
}

/// Example custom layout implementations
/// Grid layout implementation
pub fn create_grid_layout_plugin() -> (LayoutPlugin, Arc<LayoutFunction>) {
    let plugin = LayoutPlugin {
        name: "grid".to_string(),
        description: "Arrange nodes in a regular grid pattern".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert(
                "columns".to_string(),
                ParameterSpec {
                    name: "columns".to_string(),
                    description: "Number of columns (auto if not specified)".to_string(),
                    parameter_type: ParameterType::Integer,
                    default_value: ParameterValue::Integer(0), // 0 means auto
                    constraints: Some(ParameterConstraints {
                        min_value: Some(0.0),
                        max_value: Some(1000.0),
                        allowed_values: None,
                    }),
                },
            );
            params.insert(
                "cell_size".to_string(),
                ParameterSpec {
                    name: "cell_size".to_string(),
                    description: "Size of each grid cell".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(50.0),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(1.0),
                        max_value: Some(1000.0),
                        allowed_values: None,
                    }),
                },
            );
            params
        },
        author: Some("Groggy Team".to_string()),
        version: Some("1.0.0".to_string()),
        tags: vec![
            "geometric".to_string(),
            "grid".to_string(),
            "ordered".to_string(),
        ],
    };

    let layout_fn: Arc<LayoutFunction> = Arc::new(|nodes: &[VizNode], _edges: &[VizEdge]| {
        let mut positions = Vec::new();
        let cell_size = 50.0; // Would get from parameters in real implementation
        let columns = if nodes.len() < 10 {
            nodes.len()
        } else {
            (nodes.len() as f64).sqrt().ceil() as usize
        };

        for (i, node) in nodes.iter().enumerate() {
            let row = i / columns;
            let col = i % columns;
            let x = col as f64 * cell_size;
            let y = row as f64 * cell_size;
            positions.push((node.id.clone(), Position { x, y }));
        }

        Ok(positions)
    });

    (plugin, layout_fn)
}

/// Random layout implementation
pub fn create_random_layout_plugin() -> (LayoutPlugin, Arc<LayoutFunction>) {
    let plugin = LayoutPlugin {
        name: "random".to_string(),
        description: "Place nodes at random positions".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert(
                "width".to_string(),
                ParameterSpec {
                    name: "width".to_string(),
                    description: "Width of the random area".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(400.0),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(10.0),
                        max_value: Some(10000.0),
                        allowed_values: None,
                    }),
                },
            );
            params.insert(
                "height".to_string(),
                ParameterSpec {
                    name: "height".to_string(),
                    description: "Height of the random area".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(400.0),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(10.0),
                        max_value: Some(10000.0),
                        allowed_values: None,
                    }),
                },
            );
            params
        },
        author: Some("Groggy Team".to_string()),
        version: Some("1.0.0".to_string()),
        tags: vec!["random".to_string(), "simple".to_string()],
    };

    let layout_fn: Arc<LayoutFunction> = Arc::new(|nodes: &[VizNode], _edges: &[VizEdge]| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut positions = Vec::new();
        let width = 400.0; // Would get from parameters
        let height = 400.0;

        for node in nodes {
            // Deterministic randomness based on node ID
            let mut hasher = DefaultHasher::new();
            node.id.hash(&mut hasher);
            let seed = hasher.finish();

            let x = ((seed & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu32 as f64) * width - width / 2.0;
            let y =
                (((seed >> 32) & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu32 as f64) * height - height / 2.0;

            positions.push((node.id.clone(), Position { x, y }));
        }

        Ok(positions)
    });

    (plugin, layout_fn)
}

/// Honeycomb layout implementation
/// Creates a hexagonal grid pattern that efficiently packs nodes
pub struct HoneycombLayout {
    pub cell_size: f64,
    pub energy_optimization: bool,
    pub iterations: usize,
    pub spring_constant: f64,
    pub damping: f64,
}

impl Default for HoneycombLayout {
    fn default() -> Self {
        Self {
            cell_size: 40.0,
            energy_optimization: true,
            iterations: 100,
            spring_constant: 0.1,
            damping: 0.9,
        }
    }
}

impl HoneycombLayout {
    /// Calculate hexagonal grid coordinates
    fn hex_to_pixel(&self, q: i32, r: i32) -> Position {
        let x = self.cell_size * (3.0f64.sqrt() * q as f64 + 3.0f64.sqrt() / 2.0 * r as f64);
        let y = self.cell_size * (3.0 / 2.0 * r as f64);
        Position { x, y }
    }

    /// Generate spiral hex coordinates
    fn spiral_hex_coordinates(&self, count: usize) -> Vec<(i32, i32)> {
        let mut coords = Vec::with_capacity(count);

        if count == 0 {
            return coords;
        }

        // Start with center
        coords.push((0, 0));
        if count == 1 {
            return coords;
        }

        // Generate spiral outward
        let mut ring = 1;
        while coords.len() < count {
            // Each ring has 6 * ring positions
            for side in 0..6 {
                for i in 0..ring {
                    if coords.len() >= count {
                        break;
                    }

                    let angle = side as f64 * std::f64::consts::PI / 3.0;
                    let q = (ring as f64 * angle.cos()
                        + i as f64 * (angle + std::f64::consts::PI / 3.0).cos())
                    .round() as i32;
                    let r = (ring as f64 * angle.sin()
                        + i as f64 * (angle + std::f64::consts::PI / 3.0).sin())
                    .round() as i32;

                    coords.push((q, r));
                }
                if coords.len() >= count {
                    break;
                }
            }
            ring += 1;
        }

        coords.truncate(count);
        coords
    }

    /// Optimize positions using energy minimization
    fn energy_optimize(
        &self,
        positions: &mut [Position],
        edges: &[VizEdge],
        node_indices: &std::collections::HashMap<String, usize>,
    ) {
        if !self.energy_optimization {
            return;
        }

        let mut velocities = vec![Position { x: 0.0, y: 0.0 }; positions.len()];

        for _iteration in 0..self.iterations {
            // Calculate forces
            let mut forces = vec![Position { x: 0.0, y: 0.0 }; positions.len()];

            // Spring forces from edges
            for edge in edges {
                if let (Some(&source_idx), Some(&target_idx)) = (
                    node_indices.get(&edge.source),
                    node_indices.get(&edge.target),
                ) {
                    let dx = positions[target_idx].x - positions[source_idx].x;
                    let dy = positions[target_idx].y - positions[source_idx].y;
                    let distance = (dx * dx + dy * dy).sqrt().max(1.0);

                    // Hooke's law with ideal distance being the hex cell size
                    let ideal_distance = self.cell_size * 1.5; // Slightly larger than cell size
                    let displacement = distance - ideal_distance;
                    let force_magnitude = self.spring_constant * displacement;

                    let force_x = force_magnitude * dx / distance;
                    let force_y = force_magnitude * dy / distance;

                    forces[source_idx].x += force_x;
                    forces[source_idx].y += force_y;
                    forces[target_idx].x -= force_x;
                    forces[target_idx].y -= force_y;
                }
            }

            // Update positions using Verlet integration
            for i in 0..positions.len() {
                velocities[i].x = (velocities[i].x + forces[i].x) * self.damping;
                velocities[i].y = (velocities[i].y + forces[i].y) * self.damping;

                positions[i].x += velocities[i].x;
                positions[i].y += velocities[i].y;
            }
        }
    }
}

impl LayoutEngine for HoneycombLayout {
    fn compute_layout(
        &self,
        nodes: &[VizNode],
        edges: &[VizEdge],
    ) -> GraphResult<Vec<(String, Position)>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Generate hexagonal coordinates
        let hex_coords = self.spiral_hex_coordinates(nodes.len());

        // Convert to pixel positions
        let mut positions: Vec<Position> = hex_coords
            .iter()
            .map(|&(q, r)| self.hex_to_pixel(q, r))
            .collect();

        // Create node index mapping for energy optimization
        let node_indices: std::collections::HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();

        // Apply energy optimization if enabled
        self.energy_optimize(&mut positions, edges, &node_indices);

        // Return final positions
        let result = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), positions[i]))
            .collect();

        Ok(result)
    }

    fn name(&self) -> &str {
        "honeycomb"
    }

    fn supports_incremental(&self) -> bool {
        self.energy_optimization
    }
}

/// Energy-based layout using simulated annealing and global energy minimization
/// Based on the Kamada-Kawai algorithm with modern energy functions
pub struct EnergyBasedLayout {
    pub iterations: usize,
    pub cooling_rate: f64,
    pub initial_temperature: f64,
    pub min_temperature: f64,
    pub energy_function: EnergyFunction,
    pub perturbation_strength: f64,
    pub convergence_threshold: f64,
    pub use_global_optimization: bool,
}

#[derive(Clone, Debug)]
pub enum EnergyFunction {
    /// Standard spring energy (quadratic)
    Spring {
        spring_constant: f64,
        rest_length: f64,
    },
    /// Lennard-Jones potential (attractive/repulsive)
    LennardJones { epsilon: f64, sigma: f64 },
    /// Custom energy function with attractions and repulsions
    Custom {
        attraction: f64,
        repulsion: f64,
        ideal_distance: f64,
    },
    /// Multi-level energy with different forces at different scales
    MultiLevel { levels: Vec<EnergyLevel> },
}

#[derive(Clone, Debug)]
pub struct EnergyLevel {
    pub scale: f64,
    pub attraction: f64,
    pub repulsion: f64,
    pub range: (f64, f64), // (min_distance, max_distance) where this level applies
}

impl Default for EnergyBasedLayout {
    fn default() -> Self {
        Self {
            iterations: 1000,
            cooling_rate: 0.995,
            initial_temperature: 100.0,
            min_temperature: 0.01,
            energy_function: EnergyFunction::Custom {
                attraction: 0.1,
                repulsion: 1000.0,
                ideal_distance: 50.0,
            },
            perturbation_strength: 10.0,
            convergence_threshold: 0.001,
            use_global_optimization: true,
        }
    }
}

impl EnergyBasedLayout {
    /// Create energy-based layout with custom parameters
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_energy_function(mut self, energy_function: EnergyFunction) -> Self {
        self.energy_function = energy_function;
        self
    }

    pub fn with_cooling(mut self, cooling_rate: f64, initial_temp: f64) -> Self {
        self.cooling_rate = cooling_rate.clamp(0.1, 1.0);
        self.initial_temperature = initial_temp.max(0.1);
        self
    }

    /// Initialize random positions for energy minimization
    fn initialize_random_positions(&self, nodes: &[VizNode]) -> Vec<Position> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut positions = Vec::with_capacity(nodes.len());
        let radius = 100.0;

        for node in nodes {
            let mut hasher = DefaultHasher::new();
            node.id.hash(&mut hasher);
            let seed = hasher.finish();

            let angle =
                ((seed & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu32 as f64) * 2.0 * std::f64::consts::PI;
            let r = (((seed >> 32) & 0xFFFFFFFF) as f64 / 0xFFFFFFFFu32 as f64) * radius;

            let x = r * angle.cos();
            let y = r * angle.sin();

            positions.push(Position { x, y });
        }

        positions
    }

    /// Calculate total system energy
    fn calculate_energy(
        &self,
        positions: &[Position],
        edges: &[VizEdge],
        node_indices: &std::collections::HashMap<String, usize>,
    ) -> f64 {
        let mut total_energy = 0.0;

        match &self.energy_function {
            EnergyFunction::Spring {
                spring_constant,
                rest_length,
            } => {
                // Spring energy from edges
                for edge in edges {
                    if let (Some(&i), Some(&j)) = (
                        node_indices.get(&edge.source),
                        node_indices.get(&edge.target),
                    ) {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let displacement = distance - rest_length;
                        total_energy += 0.5 * spring_constant * displacement * displacement;
                    }
                }

                // Repulsive energy between all pairs
                for i in 0..positions.len() {
                    for j in (i + 1)..positions.len() {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                        total_energy += 100.0 / distance; // Coulomb-like repulsion
                    }
                }
            }

            EnergyFunction::LennardJones { epsilon, sigma } => {
                // Lennard-Jones potential
                for i in 0..positions.len() {
                    for j in (i + 1)..positions.len() {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt().max(0.1);

                        let r6 = (sigma / distance).powi(6);
                        let r12 = r6 * r6;
                        total_energy += 4.0 * epsilon * (r12 - r6);
                    }
                }

                // Additional spring energy for connected nodes
                for edge in edges {
                    if let (Some(&i), Some(&j)) = (
                        node_indices.get(&edge.source),
                        node_indices.get(&edge.target),
                    ) {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        total_energy -= epsilon * 0.5 * distance; // Attractive term for connected nodes
                    }
                }
            }

            EnergyFunction::Custom {
                attraction,
                repulsion,
                ideal_distance,
            } => {
                // Attractive forces from edges
                for edge in edges {
                    if let (Some(&i), Some(&j)) = (
                        node_indices.get(&edge.source),
                        node_indices.get(&edge.target),
                    ) {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let displacement = distance - ideal_distance;
                        total_energy += 0.5 * attraction * displacement * displacement;
                    }
                }

                // Repulsive forces between all pairs
                for i in 0..positions.len() {
                    for j in (i + 1)..positions.len() {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                        total_energy += repulsion / (distance * distance);
                    }
                }
            }

            EnergyFunction::MultiLevel { levels } => {
                // Multi-level energy calculation
                for level in levels {
                    for i in 0..positions.len() {
                        for j in (i + 1)..positions.len() {
                            let dx = positions[j].x - positions[i].x;
                            let dy = positions[j].y - positions[i].y;
                            let distance = (dx * dx + dy * dy).sqrt().max(0.1);

                            if distance >= level.range.0 && distance <= level.range.1 {
                                // Apply this level's forces
                                total_energy += level.repulsion / (distance * distance);

                                // Attraction for connected nodes
                                if edges.iter().any(|edge| {
                                    (node_indices.get(&edge.source) == Some(&i)
                                        && node_indices.get(&edge.target) == Some(&j))
                                        || (node_indices.get(&edge.source) == Some(&j)
                                            && node_indices.get(&edge.target) == Some(&i))
                                }) {
                                    total_energy -= level.attraction * distance * level.scale;
                                }
                            }
                        }
                    }
                }
            }
        }

        total_energy
    }

    /// Perform simulated annealing optimization
    fn simulated_annealing(
        &self,
        positions: &mut [Position],
        edges: &[VizEdge],
        node_indices: &std::collections::HashMap<String, usize>,
    ) {
        let mut temperature = self.initial_temperature;
        let mut current_energy = self.calculate_energy(positions, edges, node_indices);
        let mut best_positions = positions.to_vec();
        let mut best_energy = current_energy;

        for iteration in 0..self.iterations {
            // Cool down
            temperature *= self.cooling_rate;
            if temperature < self.min_temperature {
                break;
            }

            // Try random perturbation
            let node_idx = fastrand::usize(..positions.len());
            let old_position = positions[node_idx];

            // Generate random perturbation
            let dx = (fastrand::f64() - 0.5) * self.perturbation_strength * temperature
                / self.initial_temperature;
            let dy = (fastrand::f64() - 0.5) * self.perturbation_strength * temperature
                / self.initial_temperature;

            positions[node_idx].x += dx;
            positions[node_idx].y += dy;

            // Calculate new energy
            let new_energy = self.calculate_energy(positions, edges, node_indices);
            let energy_delta = new_energy - current_energy;

            // Accept or reject the move
            if energy_delta < 0.0 || fastrand::f64() < (-energy_delta / temperature).exp() {
                // Accept the move
                current_energy = new_energy;

                if new_energy < best_energy {
                    best_energy = new_energy;
                    best_positions = positions.to_vec();
                }
            } else {
                // Reject the move
                positions[node_idx] = old_position;
            }

            // Check convergence
            if iteration > 100 && (iteration % 100 == 0) {
                let energy_change =
                    (best_energy - current_energy).abs() / best_energy.abs().max(1.0);
                if energy_change < self.convergence_threshold {
                    break;
                }
            }
        }

        // Restore best configuration
        positions.copy_from_slice(&best_positions);
    }

    /// Perform global optimization using multiple starting points
    fn global_optimization(&self, nodes: &[VizNode], edges: &[VizEdge]) -> Vec<Position> {
        let node_indices: std::collections::HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();

        if !self.use_global_optimization {
            let mut positions = self.initialize_random_positions(nodes);
            self.simulated_annealing(&mut positions, edges, &node_indices);
            return positions;
        }

        // Try multiple starting configurations
        let num_trials = 5;
        let mut best_positions = Vec::new();
        let mut best_energy = f64::INFINITY;

        for _trial in 0..num_trials {
            let mut positions = self.initialize_random_positions(nodes);
            self.simulated_annealing(&mut positions, edges, &node_indices);

            let energy = self.calculate_energy(&positions, edges, &node_indices);
            if energy < best_energy {
                best_energy = energy;
                best_positions = positions;
            }
        }

        best_positions
    }
}

impl LayoutEngine for EnergyBasedLayout {
    fn compute_layout(
        &self,
        nodes: &[VizNode],
        edges: &[VizEdge],
    ) -> GraphResult<Vec<(String, Position)>> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Perform global energy optimization
        let positions = self.global_optimization(nodes, edges);

        // Return final positions
        let result = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), positions[i]))
            .collect();

        Ok(result)
    }

    fn name(&self) -> &str {
        "energy-based"
    }

    fn supports_incremental(&self) -> bool {
        true
    }
}

/// Helper function to register example layouts
pub fn register_example_layouts() {
    let registry = LayoutRegistry::global();

    // Register grid layout
    let (grid_plugin, grid_fn) = create_grid_layout_plugin();
    let _ = registry.register_layout(grid_plugin, move |nodes, edges| grid_fn(nodes, edges));

    // Register random layout
    let (random_plugin, random_fn) = create_random_layout_plugin();
    let _ = registry.register_layout(random_plugin, move |nodes, edges| random_fn(nodes, edges));
}

/// Register advanced layouts (honeycomb and energy-based)
pub fn register_advanced_layouts() {
    let registry = LayoutRegistry::global();

    // Register honeycomb layout
    let honeycomb_plugin = LayoutPlugin {
        name: "honeycomb".to_string(),
        description: "Hexagonal grid layout with optional energy optimization".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert(
                "cell_size".to_string(),
                ParameterSpec {
                    name: "cell_size".to_string(),
                    description: "Size of hexagonal cells".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(40.0),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(5.0),
                        max_value: Some(1000.0),
                        allowed_values: None,
                    }),
                },
            );
            params.insert(
                "energy_optimization".to_string(),
                ParameterSpec {
                    name: "energy_optimization".to_string(),
                    description: "Enable energy-based position optimization".to_string(),
                    parameter_type: ParameterType::Boolean,
                    default_value: ParameterValue::Boolean(true),
                    constraints: None,
                },
            );
            params.insert(
                "iterations".to_string(),
                ParameterSpec {
                    name: "iterations".to_string(),
                    description: "Number of optimization iterations".to_string(),
                    parameter_type: ParameterType::Integer,
                    default_value: ParameterValue::Integer(100),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(1.0),
                        max_value: Some(1000.0),
                        allowed_values: None,
                    }),
                },
            );
            params
        },
        author: Some("Groggy Team".to_string()),
        version: Some("1.0.0".to_string()),
        tags: vec![
            "geometric".to_string(),
            "honeycomb".to_string(),
            "energy".to_string(),
            "incremental".to_string(),
        ],
    };

    let honeycomb_fn =
        |nodes: &[VizNode], edges: &[VizEdge]| -> GraphResult<Vec<(String, Position)>> {
            let layout = HoneycombLayout::default();
            layout.compute_layout(nodes, edges)
        };

    let _ = registry.register_layout(honeycomb_plugin, honeycomb_fn);

    // Register energy-based layout
    let energy_plugin = LayoutPlugin {
        name: "energy-based".to_string(),
        description: "Global energy minimization using simulated annealing".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert(
                "iterations".to_string(),
                ParameterSpec {
                    name: "iterations".to_string(),
                    description: "Maximum optimization iterations".to_string(),
                    parameter_type: ParameterType::Integer,
                    default_value: ParameterValue::Integer(1000),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(10.0),
                        max_value: Some(10000.0),
                        allowed_values: None,
                    }),
                },
            );
            params.insert(
                "cooling_rate".to_string(),
                ParameterSpec {
                    name: "cooling_rate".to_string(),
                    description: "Temperature cooling rate (0.9-1.0)".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(0.995),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(0.9),
                        max_value: Some(1.0),
                        allowed_values: None,
                    }),
                },
            );
            params.insert(
                "initial_temperature".to_string(),
                ParameterSpec {
                    name: "initial_temperature".to_string(),
                    description: "Starting temperature for annealing".to_string(),
                    parameter_type: ParameterType::Float,
                    default_value: ParameterValue::Float(100.0),
                    constraints: Some(ParameterConstraints {
                        min_value: Some(1.0),
                        max_value: Some(1000.0),
                        allowed_values: None,
                    }),
                },
            );
            params.insert(
                "energy_function".to_string(),
                ParameterSpec {
                    name: "energy_function".to_string(),
                    description: "Type of energy function to use".to_string(),
                    parameter_type: ParameterType::Choice(vec![
                        "spring".to_string(),
                        "lennard_jones".to_string(),
                        "custom".to_string(),
                        "multi_level".to_string(),
                    ]),
                    default_value: ParameterValue::String("custom".to_string()),
                    constraints: None,
                },
            );
            params
        },
        author: Some("Groggy Team".to_string()),
        version: Some("1.0.0".to_string()),
        tags: vec![
            "energy".to_string(),
            "optimization".to_string(),
            "annealing".to_string(),
            "incremental".to_string(),
        ],
    };

    let energy_fn =
        |nodes: &[VizNode], edges: &[VizEdge]| -> GraphResult<Vec<(String, Position)>> {
            let layout = EnergyBasedLayout::default();
            layout.compute_layout(nodes, edges)
        };

    let _ = registry.register_layout(energy_plugin, energy_fn);
}
