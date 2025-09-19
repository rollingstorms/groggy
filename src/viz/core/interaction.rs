//! Interaction state management for visualization
//!
//! Handles user interactions like dragging, zooming, selection across all backends

use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use crate::viz::streaming::data_source::Position;

/// Unified interaction state manager
#[derive(Debug, Clone)]
pub struct InteractionState {
    /// Currently selected nodes
    pub selected_nodes: HashSet<String>,
    
    /// Currently hovered node (if any)
    pub hovered_node: Option<String>,
    
    /// Nodes being dragged
    pub dragged_nodes: HashMap<String, DragState>,
    
    /// Pinned nodes (fixed positions)
    pub pinned_nodes: HashSet<String>,
    
    /// Camera state (pan/zoom)
    pub camera: CameraState,
    
    /// Interaction configuration
    pub config: InteractionConfig,
    
    /// Event history for gesture recognition
    pub event_history: Vec<InteractionEvent>,
}

/// Drag state for a node
#[derive(Debug, Clone)]
pub struct DragState {
    /// Node ID being dragged
    pub node_id: String,
    
    /// Starting position when drag began
    pub start_position: Position,
    
    /// Current offset from original position
    pub offset: Position,
    
    /// Whether node was pinned when drag started
    pub was_pinned: bool,
    
    /// Timestamp when drag started
    pub start_time: u64,
}

/// Camera state for pan/zoom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraState {
    /// Pan offset
    pub pan: Position,
    
    /// Zoom level (1.0 = normal)
    pub zoom: f64,
    
    /// Zoom center point
    pub zoom_center: Position,
    
    /// Whether camera is currently being manipulated
    pub is_moving: bool,
}

/// Interaction configuration
#[derive(Debug, Clone)]
pub struct InteractionConfig {
    /// Whether dragging is enabled
    pub drag_enabled: bool,
    
    /// Whether to fix nodes when dragged
    pub fix_on_drag: bool,
    
    /// Whether zoom is enabled
    pub zoom_enabled: bool,
    
    /// Zoom limits
    pub zoom_min: f64,
    pub zoom_max: f64,
    
    /// Whether selection is enabled
    pub selection_enabled: bool,
    
    /// Whether multiple selection is allowed
    pub multi_select: bool,
    
    /// Drag sensitivity
    pub drag_sensitivity: f64,
    
    /// Zoom sensitivity
    pub zoom_sensitivity: f64,
}

/// Interaction events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Event type
    pub event_type: InteractionEventType,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Mouse/touch position
    pub position: Position,
    
    /// Target node (if any)
    pub target_node: Option<String>,
    
    /// Additional event data
    pub data: HashMap<String, serde_json::Value>,
}

/// Types of interaction events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionEventType {
    /// Mouse/pointer events
    MouseDown,
    MouseUp,
    MouseMove,
    MouseEnter,
    MouseLeave,
    Click,
    DoubleClick,
    
    /// Touch events
    TouchStart,
    TouchEnd,
    TouchMove,
    
    /// Drag events
    DragStart,
    DragMove,
    DragEnd,
    
    /// Zoom events
    Zoom,
    ZoomStart,
    ZoomEnd,
    
    /// Pan events
    Pan,
    PanStart,
    PanEnd,
    
    /// Selection events
    Select,
    Deselect,
    SelectAll,
    DeselectAll,
    
    /// Node-specific events
    NodeHover,
    NodeUnhover,
    NodePin,
    NodeUnpin,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            selected_nodes: HashSet::new(),
            hovered_node: None,
            dragged_nodes: HashMap::new(),
            pinned_nodes: HashSet::new(),
            camera: CameraState::default(),
            config: InteractionConfig::default(),
            event_history: Vec::new(),
        }
    }
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            pan: Position { x: 0.0, y: 0.0 },
            zoom: 1.0,
            zoom_center: Position { x: 0.0, y: 0.0 },
            is_moving: false,
        }
    }
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            drag_enabled: true,
            fix_on_drag: false,
            zoom_enabled: true,
            zoom_min: 0.1,
            zoom_max: 10.0,
            selection_enabled: true,
            multi_select: true,
            drag_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
        }
    }
}

impl InteractionState {
    /// Create new interaction state
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Start dragging a node
    pub fn start_drag(&mut self, node_id: String, position: Position) {
        let drag_state = DragState {
            node_id: node_id.clone(),
            start_position: position.clone(),
            offset: Position { x: 0.0, y: 0.0 },
            was_pinned: self.pinned_nodes.contains(&node_id),
            start_time: Self::current_timestamp(),
        };
        
        self.dragged_nodes.insert(node_id.clone(), drag_state);
        
        // Fix node if configured to do so
        if self.config.fix_on_drag {
            self.pinned_nodes.insert(node_id.clone());
        }
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::DragStart,
            timestamp: Self::current_timestamp(),
            position,
            target_node: Some(node_id),
            data: HashMap::new(),
        });
    }
    
    /// Update drag position
    pub fn update_drag(&mut self, node_id: &str, new_position: Position) {
        if let Some(drag_state) = self.dragged_nodes.get_mut(node_id) {
            drag_state.offset = Position {
                x: new_position.x - drag_state.start_position.x,
                y: new_position.y - drag_state.start_position.y,
            };
            
            self.add_event(InteractionEvent {
                event_type: InteractionEventType::DragMove,
                timestamp: Self::current_timestamp(),
                position: new_position,
                target_node: Some(node_id.to_string()),
                data: HashMap::new(),
            });
        }
    }
    
    /// End dragging a node
    pub fn end_drag(&mut self, node_id: &str, final_position: Position) {
        if let Some(drag_state) = self.dragged_nodes.remove(node_id) {
            // Restore pin state if it was changed
            if !drag_state.was_pinned && self.config.fix_on_drag {
                self.pinned_nodes.remove(node_id);
            }
            
            self.add_event(InteractionEvent {
                event_type: InteractionEventType::DragEnd,
                timestamp: Self::current_timestamp(),
                position: final_position,
                target_node: Some(node_id.to_string()),
                data: HashMap::new(),
            });
        }
    }
    
    /// Select a node
    pub fn select_node(&mut self, node_id: String) {
        if !self.config.multi_select {
            self.selected_nodes.clear();
        }
        
        self.selected_nodes.insert(node_id.clone());
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::Select,
            timestamp: Self::current_timestamp(),
            position: Position { x: 0.0, y: 0.0 }, // Position not relevant for selection
            target_node: Some(node_id),
            data: HashMap::new(),
        });
    }
    
    /// Deselect a node
    pub fn deselect_node(&mut self, node_id: &str) {
        self.selected_nodes.remove(node_id);
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::Deselect,
            timestamp: Self::current_timestamp(),
            position: Position { x: 0.0, y: 0.0 },
            target_node: Some(node_id.to_string()),
            data: HashMap::new(),
        });
    }
    
    /// Clear all selections
    pub fn clear_selection(&mut self) {
        self.selected_nodes.clear();
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::DeselectAll,
            timestamp: Self::current_timestamp(),
            position: Position { x: 0.0, y: 0.0 },
            target_node: None,
            data: HashMap::new(),
        });
    }
    
    /// Set hovered node
    pub fn set_hover(&mut self, node_id: Option<String>) {
        if let Some(old_hover) = &self.hovered_node {
            if Some(old_hover.clone()) != node_id {
                // Node unhover event
                self.add_event(InteractionEvent {
                    event_type: InteractionEventType::NodeUnhover,
                    timestamp: Self::current_timestamp(),
                    position: Position { x: 0.0, y: 0.0 },
                    target_node: Some(old_hover.clone()),
                    data: HashMap::new(),
                });
            }
        }
        
        if let Some(new_hover) = &node_id {
            if self.hovered_node.as_ref() != Some(new_hover) {
                // Node hover event
                self.add_event(InteractionEvent {
                    event_type: InteractionEventType::NodeHover,
                    timestamp: Self::current_timestamp(),
                    position: Position { x: 0.0, y: 0.0 },
                    target_node: Some(new_hover.clone()),
                    data: HashMap::new(),
                });
            }
        }
        
        self.hovered_node = node_id;
    }
    
    /// Pin a node
    pub fn pin_node(&mut self, node_id: String) {
        self.pinned_nodes.insert(node_id.clone());
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::NodePin,
            timestamp: Self::current_timestamp(),
            position: Position { x: 0.0, y: 0.0 },
            target_node: Some(node_id),
            data: HashMap::new(),
        });
    }
    
    /// Unpin a node
    pub fn unpin_node(&mut self, node_id: &str) {
        self.pinned_nodes.remove(node_id);
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::NodeUnpin,
            timestamp: Self::current_timestamp(),
            position: Position { x: 0.0, y: 0.0 },
            target_node: Some(node_id.to_string()),
            data: HashMap::new(),
        });
    }
    
    /// Update camera zoom
    pub fn set_zoom(&mut self, zoom: f64, center: Position) {
        let clamped_zoom = zoom.clamp(self.config.zoom_min, self.config.zoom_max);
        
        self.camera.zoom = clamped_zoom;
        self.camera.zoom_center = center;
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::Zoom,
            timestamp: Self::current_timestamp(),
            position: center,
            target_node: None,
            data: {
                let mut data = HashMap::new();
                data.insert("zoom".to_string(), serde_json::json!(clamped_zoom));
                data
            },
        });
    }
    
    /// Update camera pan
    pub fn set_pan(&mut self, pan: Position) {
        self.camera.pan = pan;
        
        self.add_event(InteractionEvent {
            event_type: InteractionEventType::Pan,
            timestamp: Self::current_timestamp(),
            position: pan,
            target_node: None,
            data: HashMap::new(),
        });
    }
    
    /// Check if a node is being dragged
    pub fn is_dragging(&self, node_id: &str) -> bool {
        self.dragged_nodes.contains_key(node_id)
    }
    
    /// Check if any node is being dragged
    pub fn is_dragging_any(&self) -> bool {
        !self.dragged_nodes.is_empty()
    }
    
    /// Check if a node is selected
    pub fn is_selected(&self, node_id: &str) -> bool {
        self.selected_nodes.contains(node_id)
    }
    
    /// Check if a node is pinned
    pub fn is_pinned(&self, node_id: &str) -> bool {
        self.pinned_nodes.contains(node_id)
    }
    
    /// Check if a node is hovered
    pub fn is_hovered(&self, node_id: &str) -> bool {
        self.hovered_node.as_ref() == Some(&node_id.to_string())
    }
    
    /// Get current drag offset for a node
    pub fn get_drag_offset(&self, node_id: &str) -> Option<Position> {
        self.dragged_nodes.get(node_id).map(|drag| drag.offset.clone())
    }
    
    /// Add event to history
    fn add_event(&mut self, event: InteractionEvent) {
        self.event_history.push(event);
        
        // Keep only recent events (last 100)
        if self.event_history.len() > 100 {
            self.event_history.remove(0);
        }
    }
    
    /// Get current timestamp
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    /// Get recent events of a specific type
    pub fn get_recent_events(&self, event_type: InteractionEventType, within_ms: u64) -> Vec<&InteractionEvent> {
        let cutoff = Self::current_timestamp() - within_ms;
        
        self.event_history
            .iter()
            .filter(|event| {
                std::mem::discriminant(&event.event_type) == std::mem::discriminant(&event_type) &&
                event.timestamp >= cutoff
            })
            .collect()
    }
    
    /// Clear event history
    pub fn clear_history(&mut self) {
        self.event_history.clear();
    }
    
    /// Apply interaction state to determine node visual states
    pub fn get_node_interaction_state(&self, node_id: &str) -> crate::viz::core::frame::NodeInteractionState {
        crate::viz::core::frame::NodeInteractionState {
            is_hovered: self.is_hovered(node_id),
            is_selected: self.is_selected(node_id),
            is_dragged: self.is_dragging(node_id),
            is_pinned: self.is_pinned(node_id),
            highlight: if self.is_selected(node_id) { 1.0 } else if self.is_hovered(node_id) { 0.5 } else { 0.0 },
        }
    }
}

/// Helper for gesture recognition
pub struct GestureRecognizer {
    /// Minimum distance for drag recognition
    pub drag_threshold: f64,
    
    /// Maximum time between clicks for double-click
    pub double_click_timeout: u64,
    
    /// Minimum velocity for momentum scrolling
    pub momentum_threshold: f64,
}

impl Default for GestureRecognizer {
    fn default() -> Self {
        Self {
            drag_threshold: 5.0,
            double_click_timeout: 300,
            momentum_threshold: 100.0,
        }
    }
}

impl GestureRecognizer {
    /// Detect if recent events constitute a drag gesture
    pub fn is_drag_gesture(&self, events: &[InteractionEvent]) -> bool {
        if events.len() < 2 {
            return false;
        }
        
        let start = &events[0];
        let end = &events[events.len() - 1];
        
        let distance = ((end.position.x - start.position.x).powi(2) + 
                       (end.position.y - start.position.y).powi(2)).sqrt();
        
        distance > self.drag_threshold
    }
    
    /// Detect double-click gesture
    pub fn is_double_click(&self, events: &[InteractionEvent]) -> bool {
        let clicks: Vec<&InteractionEvent> = events
            .iter()
            .filter(|e| matches!(e.event_type, InteractionEventType::Click))
            .collect();
        
        if clicks.len() >= 2 {
            let last_two = &clicks[clicks.len()-2..];
            let time_diff = last_two[1].timestamp - last_two[0].timestamp;
            
            return time_diff <= self.double_click_timeout;
        }
        
        false
    }
    
    /// Calculate momentum from recent movement
    pub fn calculate_momentum(&self, events: &[InteractionEvent]) -> Option<Position> {
        if events.len() < 3 {
            return None;
        }
        
        let recent: Vec<&InteractionEvent> = events
            .iter()
            .filter(|e| matches!(e.event_type, InteractionEventType::MouseMove | InteractionEventType::TouchMove))
            .rev()
            .take(5)
            .collect();
        
        if recent.len() < 2 {
            return None;
        }
        
        let start = recent[recent.len() - 1];
        let end = recent[0];
        let time_diff = end.timestamp - start.timestamp;
        
        if time_diff == 0 {
            return None;
        }
        
        let velocity_x = (end.position.x - start.position.x) / time_diff as f64 * 1000.0; // pixels per second
        let velocity_y = (end.position.y - start.position.y) / time_diff as f64 * 1000.0;
        
        let speed = (velocity_x.powi(2) + velocity_y.powi(2)).sqrt();
        
        if speed > self.momentum_threshold {
            Some(Position { x: velocity_x, y: velocity_y })
        } else {
            None
        }
    }
}