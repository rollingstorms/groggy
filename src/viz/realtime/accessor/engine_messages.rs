//! Engine message schema for Realtime Viz
//!
//! Defines EngineSnapshot and EngineUpdate types for communication
//! between DataSource, Engine, and Transport layers.

use crate::types::{AttrValue, EdgeId, NodeId};
use crate::viz::realtime::interaction::{NodeDragEvent, PointerEvent, WheelEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete snapshot of graph state for engine initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSnapshot {
    /// All nodes with their attributes
    pub nodes: Vec<Node>,
    /// All edges with their attributes
    pub edges: Vec<Edge>,
    /// Node positions in N-dimensional space
    pub positions: Vec<NodePosition>,
    /// Graph metadata
    pub meta: GraphMeta,
}

/// Node data for engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub attributes: HashMap<String, AttrValue>,
}

/// Edge data for engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub attributes: HashMap<String, AttrValue>,
}

/// N-dimensional position for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePosition {
    pub node_id: NodeId,
    /// Position coordinates in N-dimensional space
    pub coords: Vec<f64>,
}

/// Graph metadata for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMeta {
    pub node_count: usize,
    pub edge_count: usize,
    pub dimensions: usize,
    pub layout_method: String,
    pub embedding_method: String,
    pub has_positions: bool,
}

/// Updates that can be applied to the engine state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineUpdate {
    /// Node was added
    NodeAdded(Node),
    /// Node was removed
    NodeRemoved(NodeId),
    /// Node attributes changed
    NodeChanged {
        id: NodeId,
        attributes: HashMap<String, AttrValue>,
    },
    /// Edge was added
    EdgeAdded(Edge),
    /// Edge was removed
    EdgeRemoved(EdgeId),
    /// Edge attributes changed
    EdgeChanged {
        id: EdgeId,
        attributes: HashMap<String, AttrValue>,
    },
    /// Node position changed
    PositionDelta { node_id: NodeId, delta: Vec<f64> },
    /// Multiple position updates (batched)
    PositionsBatch(Vec<NodePosition>),
    /// Selection changed
    SelectionChanged {
        selected: Vec<NodeId>,
        deselected: Vec<NodeId>,
    },
    /// Embedding method changed
    EmbeddingChanged { method: String, dimensions: usize },
    /// Layout algorithm changed
    LayoutChanged {
        algorithm: String,
        params: HashMap<String, String>,
    },
    /// Snapshot was loaded (for synchronization tracking)
    SnapshotLoaded {
        node_count: usize,
        edge_count: usize,
    },
    /// Unified envelope carrying parameters, graph patches, and positions
    UpdateEnvelope(UpdateEnvelope),
}

/// Graph delta payload shared across engine, server, and browser layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPatch {
    pub nodes_added: Vec<Node>,
    pub nodes_removed: Vec<NodeId>,
    pub nodes_changed: Vec<(NodeId, serde_json::Value)>,
    pub edges_added: Vec<Edge>,
    pub edges_removed: Vec<EdgeId>,
    pub edges_changed: Vec<(EdgeId, serde_json::Value)>,
}

/// Position payload bundled with layout metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionsPayload {
    pub positions: Vec<NodePosition>,
    pub layout: Option<String>,
    pub params: Option<HashMap<String, String>>,
}

/// UI surface hints that may accompany engine updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiPayload {
    pub dialogs: Vec<serde_json::Value>,
    pub notifications: Vec<String>,
}

/// Envelope that consolidates realtime engine changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateEnvelope {
    pub timestamp_ms: u64,
    pub frame_id: u64,
    pub params_changed: Option<HashMap<String, serde_json::Value>>,
    pub graph_patch: Option<GraphPatch>,
    pub positions: Option<PositionsPayload>,
    pub ui: Option<UiPayload>,
    pub view_changed: Option<serde_json::Value>,
}

/// Control messages from client to server/engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMsg {
    /// Change embedding method and dimensionality
    ChangeEmbedding {
        method: String,
        k: usize,
        params: HashMap<String, String>,
    },
    /// Change layout algorithm
    ChangeLayout {
        algorithm: String,
        params: HashMap<String, String>,
    },
    /// Filter nodes/edges
    ApplyFilter {
        attribute: String,
        operator: String,
        value: String,
    },
    /// Clear all filters
    ClearFilters,
    /// Select nodes
    SelectNodes(Vec<NodeId>),
    /// Clear selection
    ClearSelection,

    /// Apply a direct position delta to a node (typically from drag)
    /// This is forwarded to the engine as an EngineUpdate::PositionDelta
    PositionDelta { node_id: NodeId, delta: Vec<f64> },

    /// Switch interaction controller
    SetInteractionController { mode: String },

    /// Pointer gesture event
    Pointer { event: PointerEvent },

    /// Wheel gesture event
    Wheel { event: WheelEvent },

    /// Node drag gesture event
    NodeDrag { event: NodeDragEvent },

    /// Rotate embedding axes
    RotateEmbedding {
        axis_i: usize,
        axis_j: usize,
        radians: f64,
    },

    /// Explicit view rotation (2D)
    SetViewRotation { radians: f64 },
}

impl EngineSnapshot {
    /// Create an empty snapshot
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            positions: Vec::new(),
            meta: GraphMeta {
                node_count: 0,
                edge_count: 0,
                dimensions: 2,
                layout_method: "force_directed".to_string(),
                embedding_method: "none".to_string(),
                has_positions: false,
            },
        }
    }

    /// Check if snapshot is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl EngineUpdate {
    /// Check if this update affects positions
    pub fn affects_positions(&self) -> bool {
        matches!(
            self,
            EngineUpdate::PositionDelta { .. }
                | EngineUpdate::PositionsBatch(_)
                | EngineUpdate::EmbeddingChanged { .. }
                | EngineUpdate::LayoutChanged { .. }
                | EngineUpdate::UpdateEnvelope(_)
        )
    }

    /// Check if this update affects the graph structure
    pub fn affects_structure(&self) -> bool {
        matches!(
            self,
            EngineUpdate::NodeAdded(_)
                | EngineUpdate::NodeRemoved(_)
                | EngineUpdate::EdgeAdded(_)
                | EngineUpdate::EdgeRemoved(_)
                | EngineUpdate::UpdateEnvelope(_)
        )
    }
}
