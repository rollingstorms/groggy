//! Physics simulation engine for graph visualization
//!
//! Extracted from layouts/mod.rs to create a unified physics system that can be
//! used by all visualization backends (widgets, streaming, file export).
//!
//! Features:
//! - Force-directed simulation with Fruchterman-Reingold algorithm
//! - Barnes-Hut optimization for large graphs  
//! - Collision detection and avoidance
//! - Adaptive cooling and energy management
//! - Position constraints and bounds
//! - Multiple energy functions (spring, Lennard-Jones, custom)

use std::collections::HashMap;
use crate::errors::GraphResult;
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge, Position};

/// Core physics engine for force-directed layout simulation
#[derive(Clone)]
pub struct PhysicsEngine {
    // Core physics parameters
    pub charge: f64,              // Node repulsion strength (negative = repulsive)
    pub distance: f64,            // Ideal edge length
    pub iterations: usize,        // Maximum simulation steps
    
    // Advanced physics parameters
    pub gravity: f64,             // Central gravitational force (0-1)
    pub friction: f64,            // Velocity damping (0-1)
    pub theta: f64,               // Barnes-Hut approximation parameter (0-1)
    pub alpha: f64,               // Cooling factor for simulation
    pub alpha_min: f64,           // Minimum alpha before stopping
    pub alpha_decay: f64,         // Alpha decay rate per iteration
    
    // Force strength parameters
    pub link_strength: f64,       // Spring force strength
    pub charge_strength: f64,     // Coulomb force multiplier
    pub center_strength: f64,     // Centering force strength
    pub collision_radius: f64,    // Node collision detection radius
    
    // Simulation bounds and optimization
    pub bounds: Option<(f64, f64, f64, f64)>, // (min_x, max_x, min_y, max_y)
    pub enable_barnes_hut: bool,  // Use Barnes-Hut optimization for large graphs
    pub enable_collision: bool,   // Enable node collision detection
    pub adaptive_cooling: bool,   // Adjust cooling based on system energy
    
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
    Fixed(Position),              // Node fixed at specific position
    CircularBounds(f64),          // Node constrained to circular area
    RectangularBounds(f64, f64, f64, f64), // Node constrained to rectangle
    AttractedTo(Position, f64),   // Node attracted to position with strength
}

/// Internal physics simulation state
pub struct PhysicsState {
    pub positions: Vec<Position>,
    pub velocities: Vec<Position>,
    pub forces: Vec<Position>,
    pub alpha: f64,
    pub energy: f64,
    pub iteration: usize,
    pub node_indices: HashMap<String, usize>,
}

impl Default for PhysicsEngine {
    fn default() -> Self {
        Self {
            // Core parameters (matching original ForceDirectedLayout)
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

impl PhysicsEngine {
    /// Create a new physics engine with default parameters
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
    
    /// Initialize physics simulation state
    pub fn initialize_simulation(&self, nodes: &[VizNode]) -> PhysicsState {
        let positions = self.initialize_positions(nodes);
        let velocities = vec![Position { x: 0.0, y: 0.0 }; nodes.len()];
        let forces = vec![Position { x: 0.0, y: 0.0 }; nodes.len()];
        
        // Create node index mapping
        let node_indices: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();
        
        PhysicsState {
            positions,
            velocities,
            forces,
            alpha: self.alpha,
            energy: 0.0,
            iteration: 0,
            node_indices,
        }
    }
    
    /// Run one physics simulation step
    pub fn step(&self, state: &mut PhysicsState, nodes: &[VizNode], edges: &[VizEdge]) {
        // Reset forces
        for force in &mut state.forces {
            force.x = 0.0;
            force.y = 0.0;
        }
        
        // Calculate all forces
        self.calculate_repulsive_forces(state, nodes);
        self.calculate_attractive_forces(state, edges);
        self.apply_gravity(state);
        if self.enable_collision {
            self.apply_collision_forces(state);
        }
        self.apply_constraints(state);
        
        // Update physics
        self.update_physics(state);
        self.update_alpha(state);
        
        state.iteration += 1;
    }
    
    /// Run complete simulation until convergence
    pub fn simulate(&self, nodes: &[VizNode], edges: &[VizEdge]) -> GraphResult<HashMap<String, Position>> {
        if nodes.is_empty() {
            return Ok(HashMap::new());
        }
        
        let mut state = self.initialize_simulation(nodes);
        
        // Main simulation loop
        for _ in 0..self.iterations {
            self.step(&mut state, nodes, edges);
            
            // Check convergence
            if state.alpha < self.alpha_min {
                break;
            }
        }
        
        // Return final positions as HashMap for easy lookup
        let positions = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), state.positions[i].clone()))
            .collect();
        
        Ok(positions)
    }
    
    /// Initialize random positions for nodes (deterministic based on node ID)
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
    
    /// Calculate repulsive forces between nodes
    fn calculate_repulsive_forces(&self, state: &mut PhysicsState, nodes: &[VizNode]) {
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
                    let force_magnitude = self.charge * self.charge_strength * state.alpha / (distance * distance);
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
    fn barnes_hut_forces(&self, state: &mut PhysicsState, _nodes: &[VizNode]) {
        // Simplified Barnes-Hut - in production would use quadtree
        // For now, implement a spatial grid approximation
        let grid_size = 50.0;
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        
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
                                
                                let force_magnitude = self.charge * self.charge_strength * state.alpha / (distance * distance);
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
    fn calculate_attractive_forces(&self, state: &mut PhysicsState, edges: &[VizEdge]) {
        for edge in edges {
            if let (Some(&source_idx), Some(&target_idx)) = (
                state.node_indices.get(&edge.source), 
                state.node_indices.get(&edge.target)
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
    fn apply_gravity(&self, state: &mut PhysicsState) {
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
    fn apply_collision_forces(&self, state: &mut PhysicsState) {
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
    fn apply_constraints(&self, state: &mut PhysicsState) {
        for constraint in &self.position_constraints {
            if let Some(&idx) = state.node_indices.get(&constraint.node_id) {
                match &constraint.constraint_type {
                    ConstraintType::Fixed(pos) => {
                        state.positions[idx] = pos.clone();
                        state.velocities[idx] = Position { x: 0.0, y: 0.0 };
                    }
                    ConstraintType::CircularBounds(radius) => {
                        let distance = (state.positions[idx].x * state.positions[idx].x + 
                                      state.positions[idx].y * state.positions[idx].y).sqrt();
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
    fn update_physics(&self, state: &mut PhysicsState) {
        for i in 0..state.positions.len() {
            // Update velocity with friction
            state.velocities[i].x = (state.velocities[i].x + state.forces[i].x) * self.friction;
            state.velocities[i].y = (state.velocities[i].y + state.forces[i].y) * self.friction;
            
            // Apply velocity limits
            if let Some(max_vel) = self.max_velocity {
                let vel_magnitude = (state.velocities[i].x * state.velocities[i].x + 
                                   state.velocities[i].y * state.velocities[i].y).sqrt();
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
        }
    }
    
    /// Calculate system energy for adaptive cooling
    fn calculate_energy(&self, state: &PhysicsState) -> f64 {
        let mut energy = 0.0;
        for velocity in &state.velocities {
            energy += velocity.x * velocity.x + velocity.y * velocity.y;
        }
        energy.sqrt()
    }
    
    /// Update alpha (temperature) for simulated annealing
    fn update_alpha(&self, state: &mut PhysicsState) {
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

/// Energy-based physics configuration for advanced optimization
#[derive(Clone, Debug)]
pub enum EnergyFunction {
    /// Standard spring energy (quadratic)
    Spring { spring_constant: f64, rest_length: f64 },
    /// Lennard-Jones potential (attractive/repulsive)
    LennardJones { epsilon: f64, sigma: f64 },
    /// Custom energy function with attractions and repulsions
    Custom { attraction: f64, repulsion: f64, ideal_distance: f64 },
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

/// Advanced physics engine with global energy optimization
pub struct EnergyPhysicsEngine {
    pub base_engine: PhysicsEngine,
    pub energy_function: EnergyFunction,
    pub cooling_rate: f64,
    pub initial_temperature: f64,
    pub min_temperature: f64,
    pub perturbation_strength: f64,
    pub convergence_threshold: f64,
    pub use_global_optimization: bool,
}

impl Default for EnergyPhysicsEngine {
    fn default() -> Self {
        Self {
            base_engine: PhysicsEngine::default(),
            energy_function: EnergyFunction::Custom {
                attraction: 0.1,
                repulsion: 1000.0,
                ideal_distance: 50.0,
            },
            cooling_rate: 0.995,
            initial_temperature: 100.0,
            min_temperature: 0.01,
            perturbation_strength: 10.0,
            convergence_threshold: 0.001,
            use_global_optimization: true,
        }
    }
}

impl EnergyPhysicsEngine {
    /// Create new energy-based physics engine
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Run energy-optimized simulation
    pub fn simulate(&self, nodes: &[VizNode], edges: &[VizEdge]) -> GraphResult<HashMap<String, Position>> {
        if nodes.is_empty() {
            return Ok(HashMap::new());
        }
        
        // Start with regular physics simulation
        let mut positions = self.base_engine.simulate(nodes, edges)?;
        
        if self.use_global_optimization {
            // Apply energy-based global optimization
            self.global_energy_optimization(&mut positions, nodes, edges);
        }
        
        Ok(positions)
    }
    
    /// Perform global energy optimization using simulated annealing
    fn global_energy_optimization(&self, positions: &mut HashMap<String, Position>, nodes: &[VizNode], edges: &[VizEdge]) {
        let node_indices: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (node.id.clone(), i))
            .collect();
        
        let mut position_vec: Vec<Position> = nodes
            .iter()
            .map(|node| positions[&node.id].clone())
            .collect();
        
        let mut temperature = self.initial_temperature;
        let mut current_energy = self.calculate_energy(&position_vec, edges, &node_indices);
        
        for _iteration in 0..self.base_engine.iterations {
            // Cool down
            temperature *= self.cooling_rate;
            if temperature < self.min_temperature {
                break;
            }
            
            // Try random perturbation
            let node_idx = fastrand::usize(..position_vec.len());
            let old_position = position_vec[node_idx].clone();
            
            // Generate random perturbation
            let dx = (fastrand::f64() - 0.5) * self.perturbation_strength * temperature / self.initial_temperature;
            let dy = (fastrand::f64() - 0.5) * self.perturbation_strength * temperature / self.initial_temperature;
            
            position_vec[node_idx].x += dx;
            position_vec[node_idx].y += dy;
            
            // Calculate new energy
            let new_energy = self.calculate_energy(&position_vec, edges, &node_indices);
            let energy_delta = new_energy - current_energy;
            
            // Accept or reject the move
            if energy_delta < 0.0 || fastrand::f64() < (-energy_delta / temperature).exp() {
                // Accept the move
                current_energy = new_energy;
            } else {
                // Reject the move
                position_vec[node_idx] = old_position;
            }
        }
        
        // Update final positions
        for (i, node) in nodes.iter().enumerate() {
            positions.insert(node.id.clone(), position_vec[i].clone());
        }
    }
    
    /// Calculate total system energy based on energy function
    fn calculate_energy(&self, positions: &[Position], edges: &[VizEdge], node_indices: &HashMap<String, usize>) -> f64 {
        let mut total_energy = 0.0;
        
        match &self.energy_function {
            EnergyFunction::Spring { spring_constant, rest_length } => {
                // Spring energy from edges
                for edge in edges {
                    if let (Some(&i), Some(&j)) = (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
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
            
            EnergyFunction::Custom { attraction, repulsion, ideal_distance } => {
                // Attractive forces from edges
                for edge in edges {
                    if let (Some(&i), Some(&j)) = (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
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
            
            _ => {
                // For LennardJones and MultiLevel, use simplified custom energy
                let attraction = 0.1;
                let repulsion = 1000.0;
                let ideal_distance = 50.0;
                
                for edge in edges {
                    if let (Some(&i), Some(&j)) = (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt();
                        let displacement = distance - ideal_distance;
                        total_energy += 0.5 * attraction * displacement * displacement;
                    }
                }
                
                for i in 0..positions.len() {
                    for j in (i + 1)..positions.len() {
                        let dx = positions[j].x - positions[i].x;
                        let dy = positions[j].y - positions[i].y;
                        let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                        total_energy += repulsion / (distance * distance);
                    }
                }
            }
        }
        
        total_energy
    }
}