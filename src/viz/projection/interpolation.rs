//! Smooth interpolation system for real-time projection transitions

use super::{EasingFunction, InterpolationConfig, InterpolationMethod};
use crate::errors::{GraphError, GraphResult};
use crate::viz::projection::honeycomb::HoneycombGrid;
use crate::viz::streaming::data_source::Position;
use std::collections::HashMap;

/// Interpolation engine for smooth transitions between projections
#[derive(Debug)]
pub struct InterpolationEngine {
    config: InterpolationConfig,
}

impl InterpolationEngine {
    /// Create a new interpolation engine
    pub fn new(config: InterpolationConfig) -> Self {
        Self { config }
    }

    /// Generate smooth interpolation between two sets of positions
    pub fn interpolate_positions(
        &self,
        start_positions: &[Position],
        end_positions: &[Position],
    ) -> GraphResult<Vec<Vec<Position>>> {
        if start_positions.len() != end_positions.len() {
            return Err(GraphError::InvalidInput(
                "Start and end position arrays must have the same length".to_string(),
            ));
        }

        if !self.config.enable_interpolation {
            return Ok(vec![end_positions.to_vec()]);
        }

        match &self.config.method {
            InterpolationMethod::Linear => {
                self.linear_interpolation(start_positions, end_positions)
            }
            InterpolationMethod::Bezier { control_points } => {
                self.bezier_interpolation(start_positions, end_positions, control_points)
            }
            InterpolationMethod::Spline => {
                self.spline_interpolation(start_positions, end_positions)
            }
            InterpolationMethod::SpringPhysics { damping, stiffness } => self
                .spring_physics_interpolation(start_positions, end_positions, *damping, *stiffness),
        }
    }

    /// Interpolate with honeycomb constraint preservation
    pub fn interpolate_with_honeycomb(
        &self,
        start_positions: &[Position],
        end_positions: &[Position],
        honeycomb_grid: &HoneycombGrid,
    ) -> GraphResult<Vec<Vec<Position>>> {
        if !self.config.preserve_honeycomb {
            return self.interpolate_positions(start_positions, end_positions);
        }

        // Generate base interpolation
        let base_interpolation = self.interpolate_positions(start_positions, end_positions)?;

        // Apply honeycomb constraints to each frame
        let mut constrained_interpolation = Vec::new();

        for frame in base_interpolation {
            let constrained_frame = self.apply_honeycomb_constraints(&frame, honeycomb_grid)?;
            constrained_interpolation.push(constrained_frame);
        }

        Ok(constrained_interpolation)
    }

    /// Linear interpolation between positions
    fn linear_interpolation(
        &self,
        start_positions: &[Position],
        end_positions: &[Position],
    ) -> GraphResult<Vec<Vec<Position>>> {
        let mut frames = Vec::with_capacity(self.config.steps);

        for step in 0..self.config.steps {
            let t = step as f64 / (self.config.steps - 1) as f64;
            let eased_t = self.apply_easing(t);

            let mut frame = Vec::with_capacity(start_positions.len());
            for (start, end) in start_positions.iter().zip(end_positions.iter()) {
                let x = start.x + eased_t * (end.x - start.x);
                let y = start.y + eased_t * (end.y - start.y);
                frame.push(Position { x, y });
            }
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Bezier curve interpolation
    fn bezier_interpolation(
        &self,
        start_positions: &[Position],
        end_positions: &[Position],
        control_points: &[Position],
    ) -> GraphResult<Vec<Vec<Position>>> {
        let mut frames = Vec::with_capacity(self.config.steps);

        for step in 0..self.config.steps {
            let t = step as f64 / (self.config.steps - 1) as f64;
            let eased_t = self.apply_easing(t);

            let mut frame = Vec::with_capacity(start_positions.len());

            for (i, (start, end)) in start_positions.iter().zip(end_positions.iter()).enumerate() {
                // Use control points if available, otherwise create default ones
                let default_control1 = Position {
                    x: start.x + (end.x - start.x) * 0.25,
                    y: start.y + (end.y - start.y) * 0.25,
                };
                let default_control2 = Position {
                    x: start.x + (end.x - start.x) * 0.75,
                    y: start.y + (end.y - start.y) * 0.75,
                };

                let control1 = control_points.get(i * 2).unwrap_or(&default_control1);
                let control2 = control_points.get(i * 2 + 1).unwrap_or(&default_control2);

                let pos = self.cubic_bezier(start, control1, control2, end, eased_t);
                frame.push(pos);
            }
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Spline interpolation for smooth curves
    fn spline_interpolation(
        &self,
        start_positions: &[Position],
        end_positions: &[Position],
    ) -> GraphResult<Vec<Vec<Position>>> {
        // For simplicity, use Catmull-Rom spline with generated intermediate points
        let mut frames = Vec::with_capacity(self.config.steps);

        for step in 0..self.config.steps {
            let t = step as f64 / (self.config.steps - 1) as f64;
            let eased_t = self.apply_easing(t);

            let mut frame = Vec::with_capacity(start_positions.len());

            for (start, end) in start_positions.iter().zip(end_positions.iter()) {
                // Generate control points for Catmull-Rom spline
                let p0 = Position {
                    x: start.x - (end.x - start.x) * 0.2,
                    y: start.y - (end.y - start.y) * 0.2,
                };
                let p1 = *start;
                let p2 = *end;
                let p3 = Position {
                    x: end.x + (end.x - start.x) * 0.2,
                    y: end.y + (end.y - start.y) * 0.2,
                };

                let pos = self.catmull_rom_spline(&p0, &p1, &p2, &p3, eased_t);
                frame.push(pos);
            }
            frames.push(frame);
        }

        Ok(frames)
    }

    /// Spring physics interpolation for natural motion
    fn spring_physics_interpolation(
        &self,
        start_positions: &[Position],
        end_positions: &[Position],
        damping: f64,
        stiffness: f64,
    ) -> GraphResult<Vec<Vec<Position>>> {
        let mut frames = Vec::with_capacity(self.config.steps);
        let dt = 1.0 / self.config.steps as f64;

        // Initialize physics state for each node
        let mut positions = start_positions.to_vec();
        let mut velocities = vec![Position { x: 0.0, y: 0.0 }; start_positions.len()];

        for _ in 0..self.config.steps {
            // Update physics for each node
            for i in 0..positions.len() {
                let target = end_positions[i];
                let current = positions[i];

                // Spring force toward target
                let spring_force_x = stiffness * (target.x - current.x);
                let spring_force_y = stiffness * (target.y - current.y);

                // Damping force
                let damping_force_x = -damping * velocities[i].x;
                let damping_force_y = -damping * velocities[i].y;

                // Total force
                let total_force_x = spring_force_x + damping_force_x;
                let total_force_y = spring_force_y + damping_force_y;

                // Update velocity (assuming unit mass)
                velocities[i].x += total_force_x * dt;
                velocities[i].y += total_force_y * dt;

                // Update position
                positions[i].x += velocities[i].x * dt;
                positions[i].y += velocities[i].y * dt;
            }

            frames.push(positions.clone());
        }

        Ok(frames)
    }

    /// Apply easing function to interpolation parameter
    fn apply_easing(&self, t: f64) -> f64 {
        match &self.config.easing {
            EasingFunction::Linear => t,
            EasingFunction::EaseIn => t * t,
            EasingFunction::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
            EasingFunction::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t) * (1.0 - t)
                }
            }
            EasingFunction::Bounce => {
                if t < 1.0 / 2.75 {
                    7.5625 * t * t
                } else if t < 2.0 / 2.75 {
                    let t = t - 1.5 / 2.75;
                    7.5625 * t * t + 0.75
                } else if t < 2.5 / 2.75 {
                    let t = t - 2.25 / 2.75;
                    7.5625 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / 2.75;
                    7.5625 * t * t + 0.984375
                }
            }
            EasingFunction::Elastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    let p = 0.3;
                    let s = p / 4.0;
                    -(2.0_f64.powf(10.0 * (t - 1.0))
                        * ((t - 1.0 - s) * (2.0 * std::f64::consts::PI) / p).sin())
                }
            }
            EasingFunction::Custom { function: _ } => {
                // For now, fallback to ease-in-out
                // In a full implementation, this would parse and execute the custom function
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - 2.0 * (1.0 - t) * (1.0 - t)
                }
            }
        }
    }

    /// Cubic Bezier curve evaluation
    fn cubic_bezier(
        &self,
        p0: &Position,
        p1: &Position,
        p2: &Position,
        p3: &Position,
        t: f64,
    ) -> Position {
        let u = 1.0 - t;
        let tt = t * t;
        let uu = u * u;
        let uuu = uu * u;
        let ttt = tt * t;

        let x = uuu * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + ttt * p3.x;
        let y = uuu * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + ttt * p3.y;

        Position { x, y }
    }

    /// Catmull-Rom spline evaluation
    fn catmull_rom_spline(
        &self,
        p0: &Position,
        p1: &Position,
        p2: &Position,
        p3: &Position,
        t: f64,
    ) -> Position {
        let t2 = t * t;
        let t3 = t2 * t;

        let x = 0.5
            * ((2.0 * p1.x)
                + (-p0.x + p2.x) * t
                + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3);

        let y = 0.5
            * ((2.0 * p1.y)
                + (-p0.y + p2.y) * t
                + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3);

        Position { x, y }
    }

    /// Apply honeycomb grid constraints to maintain hex alignment
    fn apply_honeycomb_constraints(
        &self,
        positions: &[Position],
        honeycomb_grid: &HoneycombGrid,
    ) -> GraphResult<Vec<Position>> {
        let mut constrained_positions = Vec::with_capacity(positions.len());

        for (i, pos) in positions.iter().enumerate() {
            if let Some(hex_coord) = honeycomb_grid.get_hex_coord(i) {
                // Blend between free position and hex center
                let hex_center = honeycomb_grid.hex_to_pixel(&hex_coord);
                let blend_factor = 0.7; // How strongly to enforce hex alignment

                let constrained_pos = Position {
                    x: blend_factor * hex_center.x + (1.0 - blend_factor) * pos.x,
                    y: blend_factor * hex_center.y + (1.0 - blend_factor) * pos.y,
                };

                constrained_positions.push(constrained_pos);
            } else {
                constrained_positions.push(*pos);
            }
        }

        Ok(constrained_positions)
    }
}

/// Animation state for managing ongoing interpolations
#[derive(Debug)]
pub struct AnimationState {
    /// Current animation frames
    frames: Vec<Vec<Position>>,
    /// Current frame index
    current_frame: usize,
    /// Whether animation is active
    is_active: bool,
    /// Animation start time (for timing calculations)
    start_time: Option<std::time::Instant>,
    /// Animation duration in milliseconds
    duration_ms: u64,
}

impl AnimationState {
    /// Create a new animation state
    pub fn new(frames: Vec<Vec<Position>>, duration_ms: u64) -> Self {
        Self {
            frames,
            current_frame: 0,
            is_active: false,
            start_time: None,
            duration_ms,
        }
    }

    /// Start the animation
    pub fn start(&mut self) {
        self.is_active = true;
        self.current_frame = 0;
        self.start_time = Some(std::time::Instant::now());
    }

    /// Stop the animation
    pub fn stop(&mut self) {
        self.is_active = false;
        self.start_time = None;
    }

    /// Update animation state and return current positions
    pub fn update(&mut self) -> Option<&Vec<Position>> {
        if !self.is_active || self.frames.is_empty() {
            return None;
        }

        if let Some(start_time) = self.start_time {
            let elapsed = start_time.elapsed().as_millis() as u64;
            let progress = elapsed as f64 / self.duration_ms as f64;

            if progress >= 1.0 {
                // Animation complete
                self.current_frame = self.frames.len() - 1;
                self.is_active = false;
            } else {
                // Calculate current frame based on elapsed time
                self.current_frame =
                    ((progress * self.frames.len() as f64) as usize).min(self.frames.len() - 1);
            }

            Some(&self.frames[self.current_frame])
        } else {
            None
        }
    }

    /// Get current positions without updating
    pub fn current_positions(&self) -> Option<&Vec<Position>> {
        if self.current_frame < self.frames.len() {
            Some(&self.frames[self.current_frame])
        } else {
            None
        }
    }

    /// Check if animation is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Get animation progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.frames.is_empty() {
            return 1.0;
        }
        self.current_frame as f64 / (self.frames.len() - 1) as f64
    }

    /// Set animation to a specific progress
    pub fn set_progress(&mut self, progress: f64) {
        let progress = progress.clamp(0.0, 1.0);
        self.current_frame =
            ((progress * (self.frames.len() - 1) as f64) as usize).min(self.frames.len() - 1);
    }
}

/// Animation manager for handling multiple concurrent animations
#[derive(Debug)]
pub struct AnimationManager {
    animations: HashMap<String, AnimationState>,
}

impl AnimationManager {
    /// Create a new animation manager
    pub fn new() -> Self {
        Self {
            animations: HashMap::new(),
        }
    }

    /// Add a new animation
    pub fn add_animation(&mut self, id: String, animation: AnimationState) {
        self.animations.insert(id, animation);
    }

    /// Start an animation
    pub fn start_animation(&mut self, id: &str) -> bool {
        if let Some(animation) = self.animations.get_mut(id) {
            animation.start();
            true
        } else {
            false
        }
    }

    /// Stop an animation
    pub fn stop_animation(&mut self, id: &str) -> bool {
        if let Some(animation) = self.animations.get_mut(id) {
            animation.stop();
            true
        } else {
            false
        }
    }

    /// Update all active animations
    pub fn update_all(&mut self) -> HashMap<String, Vec<Position>> {
        let mut current_positions = HashMap::new();

        for (id, animation) in &mut self.animations {
            if let Some(positions) = animation.update() {
                current_positions.insert(id.clone(), positions.clone());
            }
        }

        current_positions
    }

    /// Get current positions for an animation
    pub fn get_current_positions(&self, id: &str) -> Option<&Vec<Position>> {
        self.animations.get(id)?.current_positions()
    }

    /// Remove completed animations
    pub fn cleanup_completed(&mut self) {
        self.animations.retain(|_, animation| animation.is_active());
    }

    /// Check if any animations are active
    pub fn has_active_animations(&self) -> bool {
        self.animations.values().any(|a| a.is_active())
    }
}

impl Default for AnimationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let config = InterpolationConfig {
            enable_interpolation: true,
            method: InterpolationMethod::Linear,
            steps: 5,
            easing: EasingFunction::Linear,
            preserve_honeycomb: false,
        };

        let engine = InterpolationEngine::new(config);

        let start = vec![Position { x: 0.0, y: 0.0 }];
        let end = vec![Position { x: 10.0, y: 10.0 }];

        let frames = engine.interpolate_positions(&start, &end).unwrap();
        assert_eq!(frames.len(), 5);

        // Check first and last frames
        assert_eq!(frames[0][0].x, 0.0);
        assert_eq!(frames[0][0].y, 0.0);
        assert_eq!(frames[4][0].x, 10.0);
        assert_eq!(frames[4][0].y, 10.0);

        // Check middle frame
        assert!((frames[2][0].x - 5.0).abs() < 1e-10);
        assert!((frames[2][0].y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_easing_functions() {
        let config = InterpolationConfig::default();
        let engine = InterpolationEngine::new(config);

        // Test various easing functions
        assert_eq!(engine.apply_easing(0.0), 0.0);
        assert_eq!(engine.apply_easing(1.0), 1.0);

        // Ease-in should be slower at start
        let ease_in_mid = {
            let config = InterpolationConfig {
                easing: EasingFunction::EaseIn,
                ..Default::default()
            };
            let engine = InterpolationEngine::new(config);
            engine.apply_easing(0.5)
        };
        assert!(ease_in_mid < 0.5);

        // Ease-out should be faster at start
        let ease_out_mid = {
            let config = InterpolationConfig {
                easing: EasingFunction::EaseOut,
                ..Default::default()
            };
            let engine = InterpolationEngine::new(config);
            engine.apply_easing(0.5)
        };
        assert!(ease_out_mid > 0.5);
    }

    #[test]
    fn test_animation_state() {
        let frames = vec![
            vec![Position { x: 0.0, y: 0.0 }],
            vec![Position { x: 5.0, y: 5.0 }],
            vec![Position { x: 10.0, y: 10.0 }],
        ];

        let mut animation = AnimationState::new(frames, 1000);
        assert!(!animation.is_active());

        animation.start();
        assert!(animation.is_active());
        assert_eq!(animation.progress(), 0.0);

        animation.set_progress(0.5);
        assert_eq!(animation.current_frame, 1);

        animation.set_progress(1.0);
        assert_eq!(animation.current_frame, 2);
    }

    #[test]
    fn test_animation_manager() {
        let mut manager = AnimationManager::new();

        let frames = vec![vec![Position { x: 0.0, y: 0.0 }]];
        let animation = AnimationState::new(frames, 1000);

        manager.add_animation("test".to_string(), animation);
        assert!(manager.start_animation("test"));
        assert!(manager.has_active_animations());

        assert!(manager.stop_animation("test"));
        assert!(!manager.has_active_animations());
    }

    #[test]
    fn test_spring_physics() {
        let config = InterpolationConfig {
            enable_interpolation: true,
            method: InterpolationMethod::SpringPhysics {
                damping: 0.8,
                stiffness: 0.2,
            },
            steps: 10,
            easing: EasingFunction::Linear,
            preserve_honeycomb: false,
        };

        let engine = InterpolationEngine::new(config);

        let start = vec![Position { x: 0.0, y: 0.0 }];
        let end = vec![Position { x: 10.0, y: 0.0 }];

        let frames = engine.interpolate_positions(&start, &end).unwrap();
        assert_eq!(frames.len(), 10);

        // Spring physics should create smooth motion toward target
        for frame in frames {
            assert!(frame[0].x >= 0.0);
            assert!(frame[0].x <= 10.0);
        }
    }
}
