use super::math::{clamp, Quat, Vec3};
use super::{
    InteractionCommand, InteractionController, NodeDragEvent, PointerEvent, PointerPhase,
    ViewState3D, WheelEvent,
};
use std::any::Any;

pub struct GlobeController {
    view: ViewState3D,
    dragging: bool,
    rotation_sensitivity: f64,
    pan_sensitivity: f64,
    zoom_sensitivity: f64,
    allow_pan: bool,
}

impl GlobeController {
    pub fn new() -> Self {
        Self {
            view: ViewState3D {
                center: [0.0, 0.0, 0.0],
                distance: 600.0,
                quat: [1.0, 0.0, 0.0, 0.0],
            },
            dragging: false,
            rotation_sensitivity: 0.005,
            pan_sensitivity: 1.0,
            zoom_sensitivity: 1.1,
            allow_pan: true,
        }
    }

    fn quat(&self) -> Quat {
        Quat {
            w: self.view.quat[0],
            x: self.view.quat[1],
            y: self.view.quat[2],
            z: self.view.quat[3],
        }
    }

    fn store_quat(&mut self, q: Quat) {
        self.view.quat = [q.w, q.x, q.y, q.z];
    }

    fn orbit(&mut self, dx: f64, dy: f64, slow: bool) {
        let scale = if slow { 0.5 } else { 1.0 };
        let yaw = -dx * self.rotation_sensitivity * scale;
        let pitch = -dy * self.rotation_sensitivity * scale;

        let current = self.quat();
        let world_up = Vec3::new(0.0, 1.0, 0.0);
        let yaw_q = Quat::from_axis_angle(world_up, yaw);

        let world_right = Vec3::new(1.0, 0.0, 0.0);
        let camera_right = current.rotate_vec3(world_right);
        let pitch_q = Quat::from_axis_angle(camera_right, pitch);

        let updated = pitch_q.mul(yaw_q.mul(current));
        self.store_quat(updated);
    }

    fn pan(&mut self, dx: f64, dy: f64) {
        if !self.allow_pan {
            return;
        }
        let q = self.quat();
        let right = q.rotate_vec3(Vec3::new(1.0, 0.0, 0.0));
        let up = q.rotate_vec3(Vec3::new(0.0, 1.0, 0.0));

        let scale = self.pan_sensitivity * (self.view.distance / 600.0);
        let delta = right.mul(-dx * scale).add(up.mul(dy * scale));
        self.view.center[0] += delta.x;
        self.view.center[1] += delta.y;
        self.view.center[2] += delta.z;
    }

    fn zoom(&mut self, delta: f64) {
        let factor = if delta < 0.0 {
            self.zoom_sensitivity
        } else {
            1.0 / self.zoom_sensitivity
        };
        self.view.distance = clamp(self.view.distance * factor, 50.0, 5000.0);
    }

    fn roll(&mut self, delta: f64) {
        let forward = self.quat().rotate_vec3(Vec3::new(0.0, 0.0, -1.0));
        let roll_q = Quat::from_axis_angle(forward, delta * 0.002);
        let updated = roll_q.mul(self.quat());
        self.store_quat(updated);
    }
}

impl InteractionController for GlobeController {
    fn name(&self) -> &str {
        "globe-3d"
    }

    fn on_pointer(&mut self, ev: PointerEvent) -> Vec<InteractionCommand> {
        match ev.phase {
            PointerPhase::Start => {
                self.dragging = true;
            }
            PointerPhase::Move if self.dragging => {
                if ev.shift {
                    self.pan(ev.dx, ev.dy);
                } else {
                    let slow = ev.ctrl || ev.alt;
                    self.orbit(ev.dx, ev.dy, slow);
                }
            }
            PointerPhase::End => self.dragging = false,
            _ => {}
        }
        Vec::new()
    }

    fn on_wheel(&mut self, ev: WheelEvent) -> Vec<InteractionCommand> {
        match ev {
            WheelEvent::Zoom { delta } => self.zoom(delta),
            WheelEvent::Rotate { delta } => self.roll(delta),
        }
        Vec::new()
    }

    fn on_node_drag(&mut self, _ev: NodeDragEvent) -> Vec<InteractionCommand> {
        Vec::new()
    }

    fn view_3d(&self) -> Option<ViewState3D> {
        Some(self.view.clone())
    }

    fn on_activate(&mut self, _embedding_dims: Option<usize>) -> Vec<InteractionCommand> {
        Vec::new()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
