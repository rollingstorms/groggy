//! WebSocket bridge for realtime visualization
//!
//! Handles WebSocket connections and bridges engine messages to clients.

use crate::errors::GraphResult;
use crate::viz::realtime::accessor::{ControlMsg, EngineSnapshot, EngineUpdate};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_tungstenite::WebSocketStream;

/// Client ID type
pub type ClientId = usize;

/// WebSocket message frames for JSON communication
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Initial snapshot sent on connect
    #[serde(rename = "snapshot")]
    Snapshot {
        version: u32,
        payload: EngineSnapshot,
    },
    /// Update message for incremental changes
    #[serde(rename = "update")]
    Update {
        version: u32,
        payload: Box<EngineUpdate>,
    },
    /// Control message from client to server
    #[serde(rename = "control")]
    Control { version: u32, payload: ControlMsg },
    /// Acknowledgment of control message
    #[serde(rename = "control_ack")]
    ControlAck {
        version: u32,
        success: bool,
        message: String,
    },
    /// Table data response
    #[serde(rename = "table_data")]
    TableData {
        version: u32,
        payload: TableDataWindow,
    },
}

/// Table data window for streaming table view
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableDataWindow {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub total_rows: usize,
    pub start_offset: usize,
    pub data_type: String, // "nodes" or "edges"
}

/// WebSocket bridge for managing client connections and message broadcasting
pub struct WsBridge {
    /// Connected clients
    clients: Arc<Mutex<HashMap<ClientId, mpsc::UnboundedSender<WsMessage>>>>,
    /// Broadcast channel for updates
    update_tx: broadcast::Sender<EngineUpdate>,
    /// Next client ID
    next_client_id: Arc<Mutex<ClientId>>,
    /// Latest snapshot for new clients
    latest_snapshot: Arc<Mutex<Option<EngineSnapshot>>>,
    /// Control message sender to engine
    control_tx: Option<mpsc::UnboundedSender<(ClientId, ControlMsg)>>,
}

impl WsBridge {
    /// Parse control message with support for both UI and Python formats
    fn parse_control_message(text: &str) -> Result<ControlMsg, String> {
        // Debug message

        // First, try to parse as the expected WsMessage format (Python format)
        if let Ok(WsMessage::Control { .. }) = serde_json::from_str::<WsMessage>(text) {
            // Debug message
        }

        // If that fails, try to parse as direct JSON formats that UI might send

        // Try parsing as direct ControlMsg JSON
        if let Ok(_control_msg) = serde_json::from_str::<ControlMsg>(text) {
            // Debug message
        }

        // Try parsing as generic JSON and manually construct ControlMsg
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(text) {
            // Debug message

            // Check for common UI formats
            if let Some(obj) = json_val.as_object() {
                // Format 1: {"type": "ChangeLayout", "algorithm": "...", "params": {...}}
                if let Some(msg_type) = obj.get("type").and_then(|v| v.as_str()) {
                    match msg_type {
                        "ChangeLayout" => {
                            let _algorithm = obj
                                .get("algorithm")
                                .and_then(|v| v.as_str())
                                .unwrap_or("force_directed")
                                .to_string();
                            let _params: HashMap<String, String> = obj
                                .get("params")
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or_default();
                            // Debug message
                        }
                        "ChangeEmbedding" => {
                            let _method = obj
                                .get("method")
                                .and_then(|v| v.as_str())
                                .unwrap_or("spectral")
                                .to_string();
                            let _k = obj.get("k").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                            let _params: HashMap<String, String> = obj
                                .get("params")
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or_default();
                            // Debug message
                        }
                        "RequestTableData" => {
                            use crate::viz::realtime::accessor::{SortColumn, TableDataType};
                            let offset =
                                obj.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                            let window_size =
                                obj.get("window_size")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(100) as usize;
                            let data_type_str = obj
                                .get("data_type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("nodes");
                            let data_type = match data_type_str {
                                "edges" => TableDataType::Edges,
                                _ => TableDataType::Nodes,
                            };

                            // Parse sort_columns array
                            let sort_columns = obj
                                .get("sort_columns")
                                .and_then(|v| v.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|item| {
                                            let obj = item.as_object()?;
                                            let column = obj.get("column")?.as_str()?.to_string();
                                            let direction =
                                                obj.get("direction")?.as_str()?.to_string();
                                            Some(SortColumn { column, direction })
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();

                            return Ok(ControlMsg::RequestTableData {
                                offset,
                                window_size,
                                data_type,
                                sort_columns,
                            });
                        }
                        _ => {
                            // Debug message
                        }
                    }
                }

                // Format 2: {"layout": "honeycomb", "embedding": "spectral", ...}
                if let Some(layout) = obj.get("layout").and_then(|v| v.as_str()) {
                    let params = obj
                        .iter()
                        .filter(|(k, _)| *k != "layout")
                        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                        .collect();
                    // Debug message
                    return Ok(ControlMsg::ChangeLayout {
                        algorithm: layout.to_string(),
                        params,
                    });
                }
            }
        }

        Err(format!(
            "Unable to parse control message in any known format: {}",
            text
        ))
    }

    /// Create new WebSocket bridge
    pub fn new() -> Self {
        let (update_tx, _) = broadcast::channel(1000);

        Self {
            clients: Arc::new(Mutex::new(HashMap::new())),
            update_tx,
            next_client_id: Arc::new(Mutex::new(0)),
            latest_snapshot: Arc::new(Mutex::new(None)),
            control_tx: None,
        }
    }

    /// Set control message sender to engine
    pub fn set_control_sender(
        &mut self,
        control_tx: mpsc::UnboundedSender<(ClientId, ControlMsg)>,
    ) {
        self.control_tx = Some(control_tx);
    }

    /// Set the latest snapshot (for new clients)
    pub async fn set_snapshot(&self, snapshot: EngineSnapshot) {
        // Debug message
        *self.latest_snapshot.lock().await = Some(snapshot);
    }

    /// Get the latest snapshot (for debug endpoints)
    pub async fn get_snapshot(&self) -> Option<EngineSnapshot> {
        self.latest_snapshot.lock().await.clone()
    }

    /// Create browser-friendly JSON for EngineSnapshot that avoids AttrValue serialization issues
    /// This prevents "[object Object]" display by using direct JSON with simple string attributes
    fn create_browser_friendly_snapshot_json(snapshot: &EngineSnapshot) -> Result<String, String> {
        // Helper function to convert AttrValue to simple string
        let attr_to_string = |attr_value: &crate::types::AttrValue| -> String {
            match attr_value {
                crate::types::AttrValue::Float(f) => f.to_string(),
                crate::types::AttrValue::Int(i) => i.to_string(),
                crate::types::AttrValue::Text(s) => s.clone(),
                crate::types::AttrValue::Bool(b) => b.to_string(),
                crate::types::AttrValue::SmallInt(i) => i.to_string(),
                crate::types::AttrValue::CompactText(s) => s.as_str().to_string(),
                crate::types::AttrValue::Null => "null".to_string(),
                crate::types::AttrValue::FloatVec(vec) => {
                    if vec.len() <= 10 {
                        format!(
                            "[{}]",
                            vec.iter()
                                .map(|f| format!("{:.2}", f))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    } else {
                        format!(
                            "[{} values: {:.2}..{:.2}]",
                            vec.len(),
                            vec.first().unwrap_or(&0.0),
                            vec.last().unwrap_or(&0.0)
                        )
                    }
                }
                crate::types::AttrValue::IntVec(vec) => {
                    if vec.len() <= 10 {
                        format!(
                            "[{}]",
                            vec.iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    } else {
                        format!(
                            "[{} values: {}..{}]",
                            vec.len(),
                            vec.first().unwrap_or(&0),
                            vec.last().unwrap_or(&0)
                        )
                    }
                }
                crate::types::AttrValue::TextVec(vec) => {
                    if vec.len() <= 5 {
                        format!("[{}]", vec.join(", "))
                    } else {
                        format!(
                            "[{} items: {}, ...]",
                            vec.len(),
                            vec.first().map(|s| s.as_str()).unwrap_or("")
                        )
                    }
                }
                crate::types::AttrValue::BoolVec(vec) => {
                    let true_count = vec.iter().filter(|&&b| b).count();
                    format!(
                        "[{} bools: {} true, {} false]",
                        vec.len(),
                        true_count,
                        vec.len() - true_count
                    )
                }
                crate::types::AttrValue::SubgraphRef(id) => format!("Subgraph({})", id),
                crate::types::AttrValue::NodeArray(arr) => {
                    format!("NodeArray({} nodes)", arr.len())
                }
                crate::types::AttrValue::EdgeArray(arr) => {
                    format!("EdgeArray({} edges)", arr.len())
                }
                crate::types::AttrValue::Bytes(bytes) => format!("Bytes({} bytes)", bytes.len()),
                crate::types::AttrValue::CompressedText(data) => {
                    let ratio = data.compression_ratio();
                    format!(
                        "CompressedText({} bytes, {:.1}x compression)",
                        data.data.len(),
                        1.0 / ratio
                    )
                }
                crate::types::AttrValue::CompressedFloatVec(data) => {
                    let ratio = data.compression_ratio();
                    format!(
                        "CompressedFloatVec({} bytes, {:.1}x compression)",
                        data.data.len(),
                        1.0 / ratio
                    )
                }
                crate::types::AttrValue::Json(json) => match serde_json::to_string(json) {
                    Ok(json_str) => {
                        if json_str.len() <= 100 {
                            json_str
                        } else {
                            format!("JSON({} chars)", json_str.len())
                        }
                    }
                    Err(_) => "JSON(parse error)".to_string(),
                },
            }
        };

        // Create the browser-friendly JSON manually using serde_json::json! macro
        let browser_json = serde_json::json!({
            "type": "snapshot",
            "version": 1,
            "payload": {
                "nodes": snapshot.nodes.iter().map(|node| {
                    // Convert attributes to simple string key-value pairs
                    let simple_attrs: std::collections::HashMap<String, String> = node.attributes.iter()
                        .map(|(key, value)| (key.clone(), attr_to_string(value)))
                        .collect();

                    let mut node_json = serde_json::json!({
                        "id": node.id,
                        "attributes": simple_attrs
                    });

                    // Add styling fields if present
                    let obj = node_json.as_object_mut().unwrap();
                    if let Some(color) = &node.color {
                        obj.insert("color".to_string(), serde_json::json!(color));
                    }
                    if let Some(size) = node.size {
                        obj.insert("size".to_string(), serde_json::json!(size));
                    }
                    if let Some(shape) = &node.shape {
                        obj.insert("shape".to_string(), serde_json::json!(shape));
                    }
                    if let Some(opacity) = node.opacity {
                        obj.insert("opacity".to_string(), serde_json::json!(opacity));
                    }
                    if let Some(border_color) = &node.border_color {
                        obj.insert("border_color".to_string(), serde_json::json!(border_color));
                    }
                    if let Some(border_width) = node.border_width {
                        obj.insert("border_width".to_string(), serde_json::json!(border_width));
                    }
                    if let Some(label) = &node.label {
                        obj.insert("label".to_string(), serde_json::json!(label));
                    }
                    if let Some(label_color) = &node.label_color {
                        obj.insert("label_color".to_string(), serde_json::json!(label_color));
                    }
                    if let Some(label_size) = node.label_size {
                        obj.insert("label_size".to_string(), serde_json::json!(label_size));
                    }

                    node_json
                }).collect::<Vec<_>>(),
                "edges": snapshot.edges.iter().map(|edge| {
                    // Convert attributes to simple string key-value pairs
                    let simple_attrs: std::collections::HashMap<String, String> = edge.attributes.iter()
                        .map(|(key, value)| (key.clone(), attr_to_string(value)))
                        .collect();

                    let mut edge_json = serde_json::json!({
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "attributes": simple_attrs
                    });

                    // Add styling fields if present
                    let obj = edge_json.as_object_mut().unwrap();
                    if let Some(color) = &edge.color {
                        obj.insert("color".to_string(), serde_json::json!(color));
                    }
                    if let Some(width) = edge.width {
                        obj.insert("width".to_string(), serde_json::json!(width));
                    }
                    if let Some(opacity) = edge.opacity {
                        obj.insert("opacity".to_string(), serde_json::json!(opacity));
                    }
                    if let Some(style) = &edge.style {
                        obj.insert("style".to_string(), serde_json::json!(style));
                    }
                    if let Some(curvature) = edge.curvature {
                        obj.insert("curvature".to_string(), serde_json::json!(curvature));
                    }
                    if let Some(label) = &edge.label {
                        obj.insert("label".to_string(), serde_json::json!(label));
                    }
                    if let Some(label_size) = edge.label_size {
                        obj.insert("label_size".to_string(), serde_json::json!(label_size));
                    }
                    if let Some(label_color) = &edge.label_color {
                        obj.insert("label_color".to_string(), serde_json::json!(label_color));
                    }

                    edge_json
                }).collect::<Vec<_>>(),
                "positions": snapshot.positions,
                "meta": snapshot.meta
            }
        });

        serde_json::to_string(&browser_json).map_err(|e| format!("Failed to serialize: {}", e))
    }

    /// Broadcast update to all connected clients
    pub async fn broadcast_update(&self, update: EngineUpdate) -> GraphResult<()> {
        let _client_count = self.clients.lock().await.len();
        // Debug message

        // Send to broadcast channel; websocket tasks fan-out to clients
        let _ = self.update_tx.send(update);

        Ok(())
    }

    /// Send message to a specific client
    pub async fn send_to_client(&self, client_id: ClientId, message: WsMessage) -> GraphResult<()> {
        let clients = self.clients.lock().await;
        if let Some(client_tx) = clients.get(&client_id) {
            client_tx.send(message).map_err(|e| {
                crate::errors::GraphError::InvalidInput(format!(
                    "Failed to send to client {}: {}",
                    client_id, e
                ))
            })?;
        }
        Ok(())
    }

    /// Handle new WebSocket connection (already handshaken)
    pub async fn handle_websocket_stream(
        &self,
        stream: tokio::net::TcpStream,
        _addr: std::net::SocketAddr,
    ) -> GraphResult<()> {
        // Debug message

        // Convert to WebSocket stream (skip handshake since we already did it)
        use futures_util::StreamExt;

        let ws_stream = WebSocketStream::from_raw_socket(
            stream,
            tokio_tungstenite::tungstenite::protocol::Role::Server,
            None,
        )
        .await;
        let (ws_sender, mut ws_receiver) = ws_stream.split();
        let ws_sender: Arc<
            Mutex<
                futures_util::stream::SplitSink<
                    WebSocketStream<tokio::net::TcpStream>,
                    tokio_tungstenite::tungstenite::Message,
                >,
            >,
        > = Arc::new(Mutex::new(ws_sender));

        // Generate client ID
        let client_id = {
            let mut next_id = self.next_client_id.lock().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Debug message

        // Create message channel for this client
        let (client_tx, mut client_rx) = mpsc::unbounded_channel();

        // Add client to bridge
        {
            let mut clients = self.clients.lock().await;
            clients.insert(client_id, client_tx);
        }

        // Send initial snapshot if available
        if let Some(snapshot) = self.latest_snapshot.lock().await.clone() {
            // Debug message

            // Create a browser-friendly version using direct JSON to avoid AttrValue serialization issues
            let browser_friendly_json = Self::create_browser_friendly_snapshot_json(&snapshot);

            if let Ok(json_str) = browser_friendly_json {
                let ws_sender_clone = ws_sender.clone();
                tokio::spawn(async move {
                    let mut sender = ws_sender_clone.lock().await;
                    use futures_util::SinkExt;
                    let _ = sender
                        .send(tokio_tungstenite::tungstenite::Message::Text(json_str))
                        .await;
                });
            }
        }

        // Subscribe to broadcast updates
        let mut update_rx = self.update_tx.subscribe();

        // Handle messages from this client
        let _clients_clone = self.clients.clone();
        let ws_sender_clone = ws_sender.clone();

        // Spawn task to handle outgoing messages
        tokio::spawn(async move {
            use futures_util::SinkExt;

            loop {
                tokio::select! {
                    // Messages from client channel
                    msg = client_rx.recv() => {
                        if let Some(message) = msg {
                            if let Ok(json) = serde_json::to_string(&message) {
                                let mut sender = ws_sender_clone.lock().await;
                                if sender.send(tokio_tungstenite::tungstenite::Message::Text(json)).await.is_err() {
                                    break;
                                }
                            }
                        } else {
                            break;
                        }
                    }
                    // Broadcast updates
                    update = update_rx.recv() => {
                        if let Ok(update) = update {
                            let message = WsMessage::Update {
                                version: 1,
                                payload: Box::new(update),
                            };
                            if let Ok(json) = serde_json::to_string(&message) {
                                let mut sender = ws_sender_clone.lock().await;
                                if sender.send(tokio_tungstenite::tungstenite::Message::Text(json)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Remove client on disconnect
            // Debug message
        });

        // Handle incoming messages from WebSocket
        let clients_clone = self.clients.clone();
        let control_tx_clone = self.control_tx.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                        // Debug message

                        // Try to parse as control message
                        match Self::parse_control_message(&text) {
                            Ok(control_msg) => {
                                // Debug message
                                Self::handle_control_message_static(
                                    client_id,
                                    control_msg,
                                    control_tx_clone.clone(),
                                    clients_clone.clone(),
                                )
                                .await;
                            }
                            Err(_e) => {
                                // Debug message
                            }
                        }
                    }
                    Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => {
                        // Debug message
                        break;
                    }
                    Err(_e) => {
                        // Debug message
                        break;
                    }
                    _ => {}
                }
            }

            // Remove client on disconnect
            clients_clone.lock().await.remove(&client_id);
        });

        Ok(())
    }

    /// Handle control message from client (static version for tokio spawn)
    async fn handle_control_message_static(
        client_id: ClientId,
        control_msg: ControlMsg,
        control_tx: Option<mpsc::UnboundedSender<(ClientId, ControlMsg)>>,
        clients: Arc<Mutex<HashMap<ClientId, mpsc::UnboundedSender<WsMessage>>>>,
    ) {
        // Debug message

        // Forward to engine if we have a control sender
        if let Some(control_tx) = control_tx {
            if let Err(_e) = control_tx.send((client_id, control_msg.clone())) {
                // Debug message
                Self::send_control_ack_static(client_id, false, "Engine unavailable", clients)
                    .await;
                return;
            }
        } else {
            // Debug message
            Self::send_control_ack_static(client_id, false, "Engine not connected", clients).await;
            return;
        }

        // Send immediate ack (real response will come from engine)
        Self::send_control_ack_static(client_id, true, "Control message processed", clients).await;
    }

    /// Send control acknowledgment to client (static version)
    async fn send_control_ack_static(
        client_id: ClientId,
        success: bool,
        message: &str,
        clients: Arc<Mutex<HashMap<ClientId, mpsc::UnboundedSender<WsMessage>>>>,
    ) {
        let ack_msg = WsMessage::ControlAck {
            version: 1,
            success,
            message: message.to_string(),
        };

        let clients = clients.lock().await;
        if let Some(client_tx) = clients.get(&client_id) {
            if let Err(_e) = client_tx.send(ack_msg) {
                // Debug message
            }
        }
    }
}
