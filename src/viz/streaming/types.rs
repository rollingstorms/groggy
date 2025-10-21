use super::data_source::{
    DataSchema, DataWindow, DataWindowMetadata, GraphEdge, GraphMetadata, GraphNode, NodePosition,
    Position,
};
use super::virtual_scroller::VirtualScrollConfig;
use serde::{Deserialize, Serialize};
use std::thread::JoinHandle as StdJoinHandle;
use tokio_util::sync::CancellationToken;

// =============================================================================
// Graph Visualization Data Structures for WebSocket Communication
// =============================================================================

/// Graph node data for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNodeData {
    pub id: String,
    pub label: Option<String>,
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
    pub position: Option<PositionData>,
}

/// Graph edge data for WebSocket transmission  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdgeData {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub weight: Option<f64>,
    pub attributes: std::collections::HashMap<String, serde_json::Value>,
}

/// 2D position for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionData {
    pub x: f64,
    pub y: f64,
}

/// Graph metadata for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadataData {
    pub node_count: usize,
    pub edge_count: usize,
    pub is_directed: bool,
    pub has_weights: bool,
    pub attribute_types: std::collections::HashMap<String, String>,
}

/// Node position with ID for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePositionData {
    pub node_id: String,
    pub position: PositionData,
}

// Conversion methods from internal types to WebSocket types
impl From<&GraphNode> for GraphNodeData {
    fn from(node: &GraphNode) -> Self {
        let attributes: std::collections::HashMap<String, serde_json::Value> = node
            .attributes
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    serde_json::Value::String(super::util::attr_value_to_display_text(v)),
                )
            })
            .collect();

        Self {
            id: node.id.clone(),
            label: node.label.clone(),
            attributes,
            position: node
                .position
                .as_ref()
                .map(|p| PositionData { x: p.x, y: p.y }),
        }
    }
}

impl From<&GraphEdge> for GraphEdgeData {
    fn from(edge: &GraphEdge) -> Self {
        let attributes: std::collections::HashMap<String, serde_json::Value> = edge
            .attributes
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    serde_json::Value::String(super::util::attr_value_to_display_text(v)),
                )
            })
            .collect();

        Self {
            id: edge.id.clone(),
            source: edge.source.clone(),
            target: edge.target.clone(),
            label: edge.label.clone(),
            weight: edge.weight,
            attributes,
        }
    }
}

impl From<&GraphMetadata> for GraphMetadataData {
    fn from(metadata: &GraphMetadata) -> Self {
        Self {
            node_count: metadata.node_count,
            edge_count: metadata.edge_count,
            is_directed: metadata.is_directed,
            has_weights: metadata.has_weights,
            attribute_types: metadata.attribute_types.clone(),
        }
    }
}

impl From<&NodePosition> for NodePositionData {
    fn from(pos: &NodePosition) -> Self {
        Self {
            node_id: pos.node_id.clone(),
            position: PositionData {
                x: pos.position.x,
                y: pos.position.y,
            },
        }
    }
}

/// Convert AttrValue to display text for HTML rendering
pub fn attr_value_to_display_text(attr: &crate::types::AttrValue) -> String {
    use crate::types::AttrValue;

    match attr {
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => f.to_string(),
        AttrValue::Text(s) => s.clone(),
        AttrValue::CompactText(s) => s.as_str().to_string(),
        AttrValue::SmallInt(i) => i.to_string(),
        AttrValue::Bool(b) => b.to_string(),
        AttrValue::FloatVec(v) => format!("[{} floats]", v.len()),
        AttrValue::Bytes(b) => format!("[{} bytes]", b.len()),
        AttrValue::CompressedText(_) => "[Compressed Text]".to_string(),
        AttrValue::CompressedFloatVec(_) => "[Compressed FloatVec]".to_string(),
        AttrValue::SubgraphRef(id) => format!("[Subgraph:{}]", id),
        AttrValue::NodeArray(nodes) => format!("[{} nodes]", nodes.len()),
        AttrValue::EdgeArray(edges) => format!("[{} edges]", edges.len()),
        AttrValue::Null => "null".to_string(),
        AttrValue::Json(json_str) => json_str.clone(),
        AttrValue::IntVec(v) => format!("[{} ints]", v.len()),
        AttrValue::TextVec(v) => format!("[{} strings]", v.len()),
        AttrValue::BoolVec(v) => format!("[{} bools]", v.len()),
    }
}

/// Convert AttrValue to JSON for WebSocket transmission
pub fn attr_value_to_json(attr: &crate::types::AttrValue) -> serde_json::Value {
    use crate::types::AttrValue;

    match attr {
        AttrValue::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
        AttrValue::Float(f) => serde_json::Number::from_f64(*f as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        AttrValue::Text(s) => serde_json::Value::String(s.clone()),
        AttrValue::Bool(b) => serde_json::Value::Bool(*b),
        AttrValue::CompactText(s) => serde_json::Value::String(s.as_str().to_string()),
        AttrValue::SmallInt(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
        AttrValue::FloatVec(v) => {
            let vec: Vec<serde_json::Value> = v
                .iter()
                .map(|&f| {
                    serde_json::Number::from_f64(f as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect();
            serde_json::Value::Array(vec)
        }
        AttrValue::Bytes(b) => serde_json::Value::String(format!("[{} bytes]", b.len())),
        AttrValue::CompressedText(_) => serde_json::Value::String("[Compressed Text]".to_string()),
        AttrValue::CompressedFloatVec(_) => {
            serde_json::Value::String("[Compressed FloatVec]".to_string())
        }
        AttrValue::SubgraphRef(id) => serde_json::Value::String(format!("[Subgraph:{}]", id)),
        AttrValue::NodeArray(nodes) => {
            serde_json::Value::String(format!("[{} nodes]", nodes.len()))
        }
        AttrValue::EdgeArray(edges) => {
            serde_json::Value::String(format!("[{} edges]", edges.len()))
        }
        AttrValue::Null => serde_json::Value::Null,
        AttrValue::Json(json_str) => {
            // Try to parse as JSON, fallback to string if invalid
            serde_json::from_str(json_str)
                .unwrap_or_else(|_| serde_json::Value::String(json_str.clone()))
        }
        AttrValue::IntVec(v) => serde_json::Value::Array(
            v.iter()
                .map(|&i| serde_json::Value::Number(i.into()))
                .collect(),
        ),
        AttrValue::TextVec(v) => serde_json::Value::Array(
            v.iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
        AttrValue::BoolVec(v) => {
            serde_json::Value::Array(v.iter().map(|&b| serde_json::Value::Bool(b)).collect())
        }
    }
}

/// Convert DataWindow to clean JSON for WebSocket transmission  
pub fn data_window_to_json(window: &DataWindow) -> JsonDataWindow {
    let clean_rows: Vec<Vec<WireCell>> = window
        .rows
        .iter()
        .map(|row| row.iter().map(attr_to_wire).collect())
        .collect();

    JsonDataWindow {
        headers: window.headers.clone(),
        rows: clean_rows,
        schema: window.schema.clone(),
        total_rows: window.total_rows,
        start_offset: window.start_offset,
        metadata: window.metadata.clone(),
    }
}

/// Wire cell type for leak-proof WebSocket transmission
/// Using #[serde(untagged)] ensures primitives serialize directly without enum tags
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireCell {
    N(i64),           // Integers
    F(f64),           // Floats
    S(String),        // Strings
    B(bool),          // Booleans
    A(Vec<WireCell>), // Arrays
    Null,             // Null values
}

/// Convert AttrValue to leak-proof WireCell
fn attr_to_wire(attr: &crate::types::AttrValue) -> WireCell {
    use crate::types::AttrValue;

    match attr {
        AttrValue::Int(i) => WireCell::N(*i),
        AttrValue::SmallInt(i) => WireCell::N(*i as i64),
        AttrValue::Float(f) => WireCell::F(*f as f64),
        AttrValue::Text(s) => WireCell::S(s.clone()),
        AttrValue::CompactText(s) => WireCell::S(s.as_str().to_string()),
        AttrValue::Bool(b) => WireCell::B(*b),
        AttrValue::FloatVec(v) => WireCell::A(v.iter().map(|&f| WireCell::F(f as f64)).collect()),
        AttrValue::Bytes(b) => WireCell::S(format!("[{} bytes]", b.len())),
        AttrValue::CompressedText(_) => WireCell::S("[Compressed Text]".to_string()),
        AttrValue::CompressedFloatVec(_) => WireCell::S("[Compressed FloatVec]".to_string()),
        AttrValue::SubgraphRef(id) => WireCell::S(format!("[Subgraph:{}]", id)),
        AttrValue::NodeArray(nodes) => WireCell::S(format!("[{} nodes]", nodes.len())),
        AttrValue::EdgeArray(edges) => WireCell::S(format!("[{} edges]", edges.len())),
        AttrValue::Null => WireCell::Null,
        AttrValue::Json(json_str) => WireCell::S(json_str.clone()),
        AttrValue::IntVec(v) => WireCell::S(format!("[{} ints]", v.len())),
        AttrValue::TextVec(v) => WireCell::S(format!("[{} strings]", v.len())),
        AttrValue::BoolVec(v) => WireCell::S(format!("[{} bools]", v.len())),
    }
}

/// Clean JSON-compatible DataWindow for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonDataWindow {
    /// Column headers
    pub headers: Vec<String>,

    /// Rows of data (each row is a vec of WireCell values - leak-proof)
    pub rows: Vec<Vec<WireCell>>,

    /// Schema information
    pub schema: DataSchema,

    /// Total number of rows in complete dataset
    pub total_rows: usize,

    /// Starting offset of this window
    pub start_offset: usize,

    /// Metadata for this window
    pub metadata: DataWindowMetadata,
}

/// WebSocket message protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSMessage {
    /// Initial data sent when client connects
    InitialData {
        window: JsonDataWindow,
        total_rows: usize,
        meta: ProtocolMeta,
    },

    /// Data update in response to scroll
    DataUpdate {
        new_window: JsonDataWindow,
        offset: usize,
        meta: ProtocolMeta,
    },

    /// Client requests scroll to offset
    ScrollRequest { offset: usize, window_size: usize },

    /// Client requests theme change
    ThemeChange { theme: String },

    /// Broadcast update to all clients
    BroadcastUpdate { update: DataUpdate },

    /// Error message
    Error { message: String, error_code: String },

    /// Server status/ping
    Status { stats: ServerStats },

    // NEW: Graph visualization message types
    /// Client requests graph data
    GraphDataRequest {
        layout_algorithm: Option<String>,
        theme: Option<String>,
    },

    /// Server responds with graph data
    GraphDataResponse {
        nodes: Vec<GraphNodeData>,
        edges: Vec<GraphEdgeData>,
        metadata: GraphMetadataData,
        layout_positions: Option<Vec<NodePositionData>>,
    },

    /// Client requests layout computation
    LayoutRequest {
        algorithm: String,
        parameters: std::collections::HashMap<String, serde_json::Value>,
    },

    /// Server responds with layout positions
    LayoutResponse {
        positions: Vec<NodePositionData>,
        algorithm: String,
    },

    /// Client requests graph metadata only
    MetadataRequest,

    /// Server responds with graph metadata
    MetadataResponse { metadata: GraphMetadataData },

    // Phase 7: Interactive Features - Node Interactions
    /// Client clicked on a node - request details
    NodeClickRequest {
        node_id: String,
        position: Option<Position>,
        modifier_keys: Vec<String>, // ctrl, shift, alt
    },

    /// Server responds with node details for display panel
    NodeClickResponse {
        node_id: String,
        node_data: GraphNodeData,
        neighbors: Vec<GraphNodeData>,
        connected_edges: Vec<GraphEdgeData>,
        analytics: NodeAnalytics,
    },

    /// Client hovered over a node - request tooltip data
    NodeHoverRequest { node_id: String, position: Position },

    /// Server responds with rich tooltip data
    NodeHoverResponse {
        node_id: String,
        tooltip_data: NodeTooltipData,
    },

    /// Client stopped hovering over node
    NodeHoverEnd { node_id: String },

    // Phase 7: Interactive Features - Edge Interactions
    /// Client clicked on an edge
    EdgeClickRequest {
        edge_id: String,
        position: Option<Position>,
    },

    /// Server responds with edge details
    EdgeClickResponse {
        edge_id: String,
        edge_data: GraphEdgeData,
        source_node: GraphNodeData,
        target_node: GraphNodeData,
        path_info: Option<PathInfo>,
    },

    /// Client hovered over an edge
    EdgeHoverRequest { edge_id: String, position: Position },

    /// Server responds with edge tooltip
    EdgeHoverResponse {
        edge_id: String,
        tooltip_data: EdgeTooltipData,
    },

    // Phase 7: Interactive Features - Multi-Node Selection
    /// Client selected multiple nodes (drag-to-select, shift+click, etc.)
    NodesSelectionRequest {
        node_ids: Vec<String>,
        selection_type: SelectionType,
        bounding_box: Option<BoundingBox>,
    },

    /// Server responds with bulk selection data
    NodesSelectionResponse {
        selected_nodes: Vec<GraphNodeData>,
        selection_analytics: SelectionAnalytics,
        bulk_operations: Vec<String>, // Available operations for selected nodes
    },

    /// Clear current selection
    ClearSelectionRequest,

    // Phase 7: Interactive Features - Keyboard Navigation & Search
    /// Client pressed keyboard shortcut
    KeyboardActionRequest {
        action: KeyboardAction,
        node_id: Option<String>, // Current focus node
    },

    /// Server responds to keyboard navigation
    KeyboardActionResponse {
        action: KeyboardAction,
        new_focus_node: Option<String>,
        highlight_changes: Vec<HighlightChange>,
    },

    /// Client search query
    SearchRequest {
        query: String,
        search_type: SearchType,
        filters: Vec<SearchFilter>,
    },

    /// Server responds with search results
    SearchResponse {
        results: Vec<SearchResult>,
        total_matches: usize,
        query_time_ms: u64,
    },
}

/// Data update broadcast to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUpdate {
    pub update_type: UpdateType,
    pub affected_rows: Vec<usize>,
    pub new_data: Option<JsonDataWindow>,
    pub timestamp: u64,
}

/// Type of data update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateType {
    Insert,
    Update,
    Delete,
    Refresh,
}

/// Client connection state
#[derive(Debug, Clone)]
pub struct ClientState {
    pub connection_id: ConnectionId,
    pub current_offset: usize,
    pub last_update: std::time::SystemTime,
    pub subscribed_updates: bool,
}

/// Metadata for protocol tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMeta {
    pub run_id: String,
    pub protocol_version: u8,
}

/// Server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    pub active_connections: usize,
    pub total_rows: usize,
    pub total_cols: usize,
    pub cache_stats: super::virtual_scroller::CacheStats,
    pub uptime: u64,
}

/// Connection identifier
pub type ConnectionId = String;

/// Server handle with shutdown support
#[derive(Debug)]
pub struct ServerHandle {
    pub port: u16,
    pub cancel: CancellationToken,
    pub thread: Option<StdJoinHandle<()>>,
}

impl ServerHandle {
    pub fn stop(mut self) {
        self.cancel.cancel();
        // Best-effort join; ignore panic to avoid poisoning tests
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.cancel.cancel();
        // Note: we can't join here because joining might block
        // The stop() method provides explicit cleanup when needed
    }
}

/// Streaming server configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Virtual scrolling configuration
    pub scroll_config: VirtualScrollConfig,

    /// WebSocket port
    pub port: u16,

    /// Maximum concurrent connections
    pub max_connections: usize,

    /// Auto-broadcast updates
    pub auto_broadcast: bool,

    /// Update throttle in milliseconds
    pub update_throttle_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            scroll_config: VirtualScrollConfig::default(),
            port: 0, // Use port 0 for automatic port assignment to avoid conflicts
            max_connections: 100,
            auto_broadcast: true,
            update_throttle_ms: 100,
        }
    }
}

/// Error types for streaming
#[derive(Debug)]
pub enum StreamingError {
    VirtualScroll(super::virtual_scroller::VirtualScrollError),
    WebSocket(String),
    Client(String),
    Server(String),
}

impl std::fmt::Display for StreamingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamingError::VirtualScroll(err) => write!(f, "Virtual scroll error: {}", err),
            StreamingError::WebSocket(msg) => write!(f, "WebSocket error: {}", msg),
            StreamingError::Client(msg) => write!(f, "Client error: {}", msg),
            StreamingError::Server(msg) => write!(f, "Server error: {}", msg),
        }
    }
}

impl std::error::Error for StreamingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamingError::VirtualScroll(err) => Some(err),
            _ => None,
        }
    }
}

impl From<super::virtual_scroller::VirtualScrollError> for StreamingError {
    fn from(err: super::virtual_scroller::VirtualScrollError) -> Self {
        StreamingError::VirtualScroll(err)
    }
}

pub type StreamingResult<T> = Result<T, StreamingError>;

// HTML escape utility function
// (moved) html_escape is defined in html.rs
// ============================================================================
// Phase 7: Interactive Features - Supporting Data Structures
// ============================================================================

/// Node analytics data for detail panel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnalytics {
    pub degree: usize,
    pub in_degree: Option<usize>,  // For directed graphs
    pub out_degree: Option<usize>, // For directed graphs
    pub centrality_measures: CentralityMeasures,
    pub clustering_coefficient: Option<f64>,
    pub community_id: Option<String>,
}

/// Centrality measures for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMeasures {
    pub betweenness: Option<f64>,
    pub closeness: Option<f64>,
    pub eigenvector: Option<f64>,
    pub page_rank: Option<f64>,
}

/// Rich tooltip data for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTooltipData {
    pub title: String,
    pub subtitle: Option<String>,
    pub primary_attributes: Vec<TooltipAttribute>,
    pub secondary_attributes: Vec<TooltipAttribute>,
    pub metrics: Vec<TooltipMetric>,
}

/// Rich tooltip data for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeTooltipData {
    pub title: String,
    pub source_label: String,
    pub target_label: String,
    pub weight_display: Option<String>,
    pub attributes: Vec<TooltipAttribute>,
    pub path_info: Option<String>,
}

/// Tooltip attribute display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipAttribute {
    pub name: String,
    pub value: String,
    pub display_type: AttributeDisplayType,
}

/// Tooltip metric display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipMetric {
    pub name: String,
    pub value: f64,
    pub format: MetricFormat,
    pub context: Option<String>, // e.g., "vs average: 2.3x higher"
}

/// How to display attribute in tooltip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeDisplayType {
    Text,
    Number,
    Badge,
    Link,
    Color,
}

/// How to format metric values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricFormat {
    Integer,
    Decimal { places: usize },
    Percentage { places: usize },
    Scientific,
}

/// Path information for edges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInfo {
    pub is_shortest_path: bool,
    pub path_length: usize,
    pub path_weight: Option<f64>,
    pub alternative_paths: usize,
}

/// Selection type for multi-node operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionType {
    Click,       // Single click
    ShiftClick,  // Shift+click to extend selection
    CtrlClick,   // Ctrl+click to toggle
    DragSelect,  // Drag to select multiple
    LassoSelect, // Lasso selection
    BoxSelect,   // Box selection
}

/// Bounding box for area selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

/// Analytics for multiple selected nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionAnalytics {
    pub node_count: usize,
    pub edge_count: usize, // Edges between selected nodes
    pub connected_components: usize,
    pub avg_degree: f64,
    pub total_weight: Option<f64>,
    pub communities_represented: Vec<String>,
}

/// Keyboard actions for navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyboardAction {
    // Navigation
    FocusNext,
    FocusPrevious,
    FocusNeighbor { direction: String },

    // Selection
    SelectFocused,
    SelectAll,
    ClearSelection,
    InvertSelection,

    // Layout
    ChangeLayout { algorithm: String },
    ZoomIn,
    ZoomOut,
    ZoomToFit,
    ResetView,

    // Search
    StartSearch,
    NextSearchResult,
    PrevSearchResult,

    // Display
    ToggleLabels,
    ToggleEdges,
    ToggleDetails,
}

/// Highlight change for visual feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightChange {
    pub element_id: String,
    pub element_type: HighlightElementType,
    pub highlight_type: HighlightType,
    pub duration_ms: Option<u64>,
}

/// Type of element to highlight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HighlightElementType {
    Node,
    Edge,
    Group,
}

/// Type of highlighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HighlightType {
    Focus,
    Selection,
    Hover,
    Search,
    Neighborhood,
    Path,
    Remove,
}

/// Search type for different search modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    Node,
    Edge,
    Attribute,
    Global,
}

/// Search filter for refined results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilter {
    pub field: String,
    pub operator: SearchOperator,
    pub value: serde_json::Value,
}

/// Search operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchOperator {
    Equals,
    Contains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    Between,
    In,
    Regex,
}

/// Search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub result_type: SearchResultType,
    pub id: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub relevance_score: f64,
    pub matched_fields: Vec<MatchedField>,
    pub highlight_data: HighlightData,
}

/// Type of search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchResultType {
    Node,
    Edge,
    Attribute,
}

/// Field that matched the search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedField {
    pub field_name: String,
    pub matched_text: String,
    pub context: String, // Surrounding text for context
}

/// Data for highlighting search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightData {
    pub element_id: String,
    pub highlight_regions: Vec<HighlightRegion>,
}

/// Region to highlight in search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightRegion {
    pub start: usize,
    pub length: usize,
    pub match_type: String,
}
