use serde::{Deserialize, Serialize};
use std::any::Any;

pub mod globe_controller;
pub mod math;
pub mod pan_controller;

pub use globe_controller::GlobeController;
pub use math::{Quat, Vec3};
pub use pan_controller::PanController;

/// Phases for pointer interactions (mouse/touch)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PointerPhase {
    Start,
    Move,
    End,
}

/// Normalized pointer event delivered from clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerEvent {
    pub phase: PointerPhase,
    pub dx: f64,
    pub dy: f64,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

/// High level wheel events (zoom/rotate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WheelEvent {
    Zoom { delta: f64 },
    Rotate { delta: f64 },
}

/// Node drag gesture events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeDragEvent {
    Start { node_id: usize, x: f64, y: f64 },
    Move { node_id: usize, x: f64, y: f64 },
    End { node_id: usize },
}

/// Canonical 2D view state (pan/zoom/rotation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewState2D {
    pub x: f64,
    pub y: f64,
    pub zoom: f64,
    pub rotation: f64,
}

/// Canonical 3D view state (orbit camera)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewState3D {
    pub center: [f64; 3],
    pub distance: f64,
    pub quat: [f64; 4],
}

/// Command that interaction controllers can send back to the engine
#[derive(Debug, Clone)]
pub enum InteractionCommand {
    RotateEmbedding {
        axis_i: usize,
        axis_j: usize,
        radians: f64,
    },
    TriggerRecomputation,
    UpdateAutoScale {
        target_occupancy: f64,
        min_cell_size: f64,
    },
    ExposeAutoScaleControls {
        target_occupancy: f64,
        min_cell_size: f64,
        enabled: bool,
    },
}

/// Unified interaction controller interface
pub trait InteractionController: Send {
    fn name(&self) -> &str;

    fn on_pointer(&mut self, _ev: PointerEvent) -> Vec<InteractionCommand> {
        Vec::new()
    }

    fn on_wheel(&mut self, _ev: WheelEvent) -> Vec<InteractionCommand> {
        Vec::new()
    }

    fn on_node_drag(&mut self, _ev: NodeDragEvent) -> Vec<InteractionCommand> {
        Vec::new()
    }

    fn view_2d(&self) -> Option<ViewState2D> {
        None
    }

    fn view_3d(&self) -> Option<ViewState3D> {
        None
    }

    /// Called when controller is activated to set up context
    fn on_activate(&mut self, _embedding_dims: Option<usize>) -> Vec<InteractionCommand> {
        Vec::new()
    }

    /// Support for downcasting to specific controller types
    fn as_any(&mut self) -> &mut dyn Any;
}

/// Policies for node dragging based on layout
#[derive(Debug, Clone, Copy)]
pub enum NodeDragPolicy {
    Disabled,
    Free,
    Constrained,
}

/// Canvas drag policies (for documentation and future branching)
#[derive(Debug, Clone, Copy)]
pub enum CanvasDragPolicy {
    PanZoomRotate2D,
    Trackball3D,
    RotateNdThenProject,
}

/// Honeycomb controller for N-dimensional embeddings with drag constraints and rotation
pub struct HoneycombController {
    /// Current N-D rotation state as axis pairs and angles
    rotations: Vec<(usize, usize, f64)>, // (axis_i, axis_j, radians)

    /// Node drag constraints policy
    drag_policy: NodeDragPolicy,

    /// Active node being dragged
    dragging_node: Option<usize>,

    /// Starting position of current drag
    drag_start: Option<(f64, f64)>,

    /// Sensitivity settings for different interactions
    rotation_sensitivity: f64,
    #[allow(dead_code)]
    drag_sensitivity: f64,

    /// Embedding dimensions count (determined at runtime)
    embedding_dims: Option<usize>,

    /// Auto-scaling parameters (will be configurable via param plumbing)
    target_occupancy: f64,
    min_cell_size: f64,
    auto_scale_enabled: bool,
}

impl HoneycombController {
    pub fn new() -> Self {
        Self {
            rotations: Vec::new(),
            drag_policy: NodeDragPolicy::Constrained,
            dragging_node: None,
            drag_start: None,
            rotation_sensitivity: 0.01,
            drag_sensitivity: 1.0,
            embedding_dims: None,
            target_occupancy: 1.0, // One node per hexagonal cell for optimal clarity
            min_cell_size: 10.0,
            auto_scale_enabled: true,
        }
    }

    /// Update embedding dimensions when layout changes
    pub fn set_embedding_dims(&mut self, dims: usize) {
        self.embedding_dims = Some(dims);
        // Clear rotations that reference dimensions beyond the new count
        self.rotations.retain(|(i, j, _)| *i < dims && *j < dims);
    }

    /// Configure auto-scaling parameters
    pub fn configure_auto_scaling(
        &mut self,
        target_occupancy: f64,
        min_cell_size: f64,
        enabled: bool,
    ) {
        self.target_occupancy = target_occupancy;
        self.min_cell_size = min_cell_size;
        self.auto_scale_enabled = enabled;
    }

    /// Add or update N-D rotation between two axes
    pub fn rotate_embedding_axes(&mut self, axis_i: usize, axis_j: usize, radians: f64) {
        // Check if embedding dimensions are set and axes are valid
        if let Some(dims) = self.embedding_dims {
            if axis_i >= dims || axis_j >= dims || axis_i == axis_j {
                eprintln!(
                    "🔴 WARNING: Invalid axis rotation request: axis_i={}, axis_j={}, dims={}",
                    axis_i, axis_j, dims
                );
                return;
            }
        }

        // Find existing rotation for this axis pair or add new one
        if let Some(existing) = self
            .rotations
            .iter_mut()
            .find(|(i, j, _)| (*i == axis_i && *j == axis_j) || (*i == axis_j && *j == axis_i))
        {
            existing.2 += radians;
        } else {
            self.rotations.push((axis_i, axis_j, radians));
        }

        eprintln!(
            "🔄 DEBUG: Added N-D rotation {} radians between axes {} and {}",
            radians, axis_i, axis_j
        );
    }

    /// Get current rotation transformations for broadcasting to engine
    pub fn get_rotations(&self) -> &Vec<(usize, usize, f64)> {
        &self.rotations
    }

    /// Apply drag constraints based on policy
    fn constrain_drag_position(&self, node_id: usize, x: f64, y: f64) -> (f64, f64) {
        match self.drag_policy {
            NodeDragPolicy::Disabled => {
                // Don't allow any movement
                if let Some((start_x, start_y)) = self.drag_start {
                    (start_x, start_y)
                } else {
                    (x, y)
                }
            }
            NodeDragPolicy::Free => {
                // Allow unlimited movement
                (x, y)
            }
            NodeDragPolicy::Constrained => {
                // Apply honeycomb-specific constraints
                self.apply_honeycomb_constraints(node_id, x, y)
            }
        }
    }

    /// Apply honeycomb-specific drag constraints
    fn apply_honeycomb_constraints(&self, _node_id: usize, x: f64, y: f64) -> (f64, f64) {
        // For honeycomb layout, we might want to:
        // 1. Snap to hex grid positions
        // 2. Maintain minimum cell spacing
        // 3. Preserve hexagonal neighborhood relationships

        // For now, just apply minimum cell size constraint
        let constrained_x = x.clamp(-1000.0, 1000.0); // Reasonable bounds
        let constrained_y = y.clamp(-1000.0, 1000.0);

        (constrained_x, constrained_y)
    }
}

impl InteractionController for HoneycombController {
    fn name(&self) -> &str {
        "honeycomb-nd"
    }

    fn on_pointer(&mut self, ev: PointerEvent) -> Vec<InteractionCommand> {
        let mut commands = Vec::new();
        match ev.phase {
            PointerPhase::Start => {
                self.drag_start = Some((0.0, 0.0)); // Will be updated with actual position
                                                    // 🫸 DEBUG: Honeycomb pointer start,
            }
            PointerPhase::Move => {
                if self.drag_start.is_some() {
                    if ev.shift && ev.ctrl {
                        // Shift+Ctrl: Trigger N-D rotation on first two available axes
                        if let Some(dims) = self.embedding_dims {
                            if dims >= 2 {
                                let rotation_amount = ev.dx * self.rotation_sensitivity;
                                self.rotate_embedding_axes(0, 1, rotation_amount);
                                commands.push(InteractionCommand::RotateEmbedding {
                                    axis_i: 0,
                                    axis_j: 1,
                                    radians: rotation_amount,
                                });
                            }
                        }
                    } else if ev.shift {
                        // Shift: Rotate different axis pair
                        if let Some(dims) = self.embedding_dims {
                            if dims >= 3 {
                                let rotation_amount = ev.dx * self.rotation_sensitivity;
                                self.rotate_embedding_axes(1, 2, rotation_amount);
                                commands.push(InteractionCommand::RotateEmbedding {
                                    axis_i: 1,
                                    axis_j: 2,
                                    radians: rotation_amount,
                                });
                            }
                        }
                    } else {
                        // Regular drag (no modifiers): Default N-D rotation on primary axes
                        if let Some(dims) = self.embedding_dims {
                            if dims >= 2 {
                                let rotation_amount = ev.dx * self.rotation_sensitivity * 0.5; // Slower for regular drag
                                self.rotate_embedding_axes(0, 1, rotation_amount);
                                commands.push(InteractionCommand::RotateEmbedding {
                                    axis_i: 0,
                                    axis_j: 1,
                                    radians: rotation_amount,
                                });
                                // 🔄 DEBUG: Honeycomb regular drag rotation: axes (0,1) by {:.4} radians", rotation_amount);
                            }
                        }
                    }
                }
            }
            PointerPhase::End => {
                self.drag_start = None;
                self.dragging_node = None;
                // 🫸 DEBUG: Honeycomb pointer end,
            }
        }
        commands
    }

    fn on_wheel(&mut self, ev: WheelEvent) -> Vec<InteractionCommand> {
        let mut commands = Vec::new();
        match ev {
            WheelEvent::Zoom { delta: _ } => {
                // Zoom could trigger auto-scaling adjustments
                if self.auto_scale_enabled {
                    // 🔍 DEBUG: Honeycomb zoom - considering auto-scale adjustments,
                }
            }
            WheelEvent::Rotate { delta } => {
                // Wheel rotation could trigger N-D axis rotations
                if let Some(dims) = self.embedding_dims {
                    if dims >= 2 {
                        let rotation_amount = delta * self.rotation_sensitivity * 0.1;
                        self.rotate_embedding_axes(0, 1, rotation_amount);
                        commands.push(InteractionCommand::RotateEmbedding {
                            axis_i: 0,
                            axis_j: 1,
                            radians: rotation_amount,
                        });
                    }
                }
            }
        }
        commands
    }

    fn on_node_drag(&mut self, ev: NodeDragEvent) -> Vec<InteractionCommand> {
        let commands = Vec::new();
        match ev {
            NodeDragEvent::Start { node_id, x, y } => {
                self.dragging_node = Some(node_id);
                self.drag_start = Some((x, y));
                eprintln!(
                    "🎯 DEBUG: Honeycomb node drag start: node={}, pos=({:.1}, {:.1})",
                    node_id, x, y
                );
            }
            NodeDragEvent::Move { node_id, x, y } => {
                if self.dragging_node == Some(node_id) {
                    let (constrained_x, constrained_y) =
                        self.constrain_drag_position(node_id, x, y);
                    if (constrained_x - x).abs() > 0.1 || (constrained_y - y).abs() > 0.1 {
                        // Honeycomb drag constrained
                    }
                    // The engine would use the constrained position for actual updates
                }
            }
            NodeDragEvent::End { node_id } => {
                if self.dragging_node == Some(node_id) {
                    self.dragging_node = None;
                    self.drag_start = None;
                    // Honeycomb node drag end
                }
            }
        }
        commands
    }

    fn on_activate(&mut self, embedding_dims: Option<usize>) -> Vec<InteractionCommand> {
        let mut commands = Vec::new();

        if let Some(dims) = embedding_dims {
            self.set_embedding_dims(dims);
            eprintln!(
                "🔧 DEBUG: Honeycomb controller activated with {} dimensions",
                dims
            );
        }

        // When honeycomb controller is activated, expose auto-scaling controls in UI
        commands.push(InteractionCommand::ExposeAutoScaleControls {
            target_occupancy: self.target_occupancy,
            min_cell_size: self.min_cell_size,
            enabled: self.auto_scale_enabled,
        });

        // Honeycomb auto-scale controls exposed

        commands
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
