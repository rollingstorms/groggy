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

    // Visual styling fields (from VizConfig)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub border_width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_color: Option<String>,
}

/// Edge data for engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub attributes: HashMap<String, AttrValue>,

    // Visual styling fields (from VizConfig)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curvature: Option<f64>, // 0 = straight, positive = curve right, negative = curve left
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_size: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label_color: Option<String>,
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
    /// Switch interaction controller
    SetInteractionController { mode: String },

    /// Pointer gesture event
    Pointer { event: PointerEvent },

    /// Wheel gesture event
    Wheel { event: WheelEvent },

    /// Node drag gesture event
    NodeDrag { event: NodeDragEvent },

    /// Request table data window
    RequestTableData {
        offset: usize,
        window_size: usize,
        data_type: TableDataType,
        sort_columns: Vec<SortColumn>,
    },
}

/// Type of table data to request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TableDataType {
    /// Node table data
    Nodes,
    /// Edge table data
    Edges,
}

/// Column sorting specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortColumn {
    /// Column name to sort by
    pub column: String,
    /// Sort direction: "asc" or "desc"
    pub direction: String,
}

impl Node {
    /// Create a new node with just id and attributes (no styling)
    pub fn new(id: NodeId, attributes: HashMap<String, AttrValue>) -> Self {
        Self {
            id,
            attributes,
            color: None,
            size: None,
            shape: None,
            opacity: None,
            border_color: None,
            border_width: None,
            label: None,
            label_size: None,
            label_color: None,
        }
    }
}

impl Edge {
    /// Create a new edge with just id, source, target, and attributes (no styling)
    pub fn new(
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        attributes: HashMap<String, AttrValue>,
    ) -> Self {
        Self {
            id,
            source,
            target,
            attributes,
            color: None,
            width: None,
            opacity: None,
            style: None,
            curvature: None,
            label: None,
            label_size: None,
            label_color: None,
        }
    }
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
