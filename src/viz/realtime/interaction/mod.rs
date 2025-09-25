use serde::{Deserialize, Serialize};

pub mod globe_controller;
mod math;
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

/// Unified interaction controller interface
pub trait InteractionController: Send {
    fn name(&self) -> &str;

    fn on_pointer(&mut self, _ev: PointerEvent) {}
    fn on_wheel(&mut self, _ev: WheelEvent) {}
    fn on_node_drag(&mut self, _ev: NodeDragEvent) {}

    fn view_2d(&self) -> Option<ViewState2D> {
        None
    }

    fn view_3d(&self) -> Option<ViewState3D> {
        None
    }
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

/// Default honeycomb placeholder controller (reuses legacy logic incrementally)
pub struct HoneycombController;

impl HoneycombController {
    pub fn new() -> Self {
        Self
    }
}

impl InteractionController for HoneycombController {
    fn name(&self) -> &str {
        "honeycomb-nd"
    }
}
