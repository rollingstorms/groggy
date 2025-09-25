use super::{
    InteractionController, NodeDragEvent, PointerEvent, PointerPhase, ViewState2D, WheelEvent,
};

pub struct PanController {
    view: ViewState2D,
    dragging: bool,
}

impl PanController {
    pub fn new() -> Self {
        Self {
            view: ViewState2D {
                x: 0.0,
                y: 0.0,
                zoom: 1.0,
                rotation: 0.0,
            },
            dragging: false,
        }
    }

    fn clamp_zoom(&self, zoom: f64) -> f64 {
        zoom.clamp(0.05, 10.0)
    }
}

impl InteractionController for PanController {
    fn name(&self) -> &str {
        "pan-2d"
    }

    fn on_pointer(&mut self, ev: PointerEvent) {
        match ev.phase {
            PointerPhase::Start => self.dragging = true,
            PointerPhase::Move => {
                if self.dragging {
                    if ev.shift {
                        self.view.rotation += ev.dx * 0.005;
                    } else {
                        self.view.x -= ev.dx / self.view.zoom;
                        self.view.y -= ev.dy / self.view.zoom;
                    }
                }
            }
            PointerPhase::End => self.dragging = false,
        }
    }

    fn on_wheel(&mut self, ev: WheelEvent) {
        match ev {
            WheelEvent::Zoom { delta } => {
                let factor = if delta < 0.0 { 1.1 } else { 0.9 };
                self.view.zoom = self.clamp_zoom(self.view.zoom * factor);
            }
            WheelEvent::Rotate { delta } => {
                self.view.rotation += delta * 0.01;
            }
        }
    }

    fn on_node_drag(&mut self, _ev: NodeDragEvent) {}

    fn view_2d(&self) -> Option<ViewState2D> {
        Some(self.view.clone())
    }
}
