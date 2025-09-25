//! WebSocket bridge for realtime visualization
//!
//! Handles WebSocket connections and bridges engine messages to clients.

use crate::errors::{io_error_to_graph_error, GraphResult};
use crate::viz::realtime::accessor::{ControlMsg, EngineSnapshot, EngineUpdate};
use futures_util::{SinkExt, StreamExt};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_tungstenite::{accept_async, tungstenite::Message, WebSocketStream};

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
    Update { version: u32, payload: EngineUpdate },
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
        eprintln!("🔍 DEBUG: Attempting to parse control message: {}", text);

        // First, try to parse as the expected WsMessage format (Python format)
        if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(text) {
            if let WsMessage::Control { payload, .. } = ws_msg {
                eprintln!("✅ DEBUG: Parsed as WsMessage::Control format");
                return Ok(payload);
            }
        }

        // If that fails, try to parse as direct JSON formats that UI might send

        // Try parsing as direct ControlMsg JSON
        if let Ok(control_msg) = serde_json::from_str::<ControlMsg>(text) {
            eprintln!("✅ DEBUG: Parsed as direct ControlMsg format");
            return Ok(control_msg);
        }

        // Try parsing as generic JSON and manually construct ControlMsg
        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(text) {
            eprintln!("🔍 DEBUG: Trying to parse as generic JSON: {}", json_val);

            // Check for common UI formats
            if let Some(obj) = json_val.as_object() {
                // Format 1: {"type": "ChangeLayout", "algorithm": "...", "params": {...}}
                if let Some(msg_type) = obj.get("type").and_then(|v| v.as_str()) {
                    match msg_type {
                        "ChangeLayout" => {
                            let algorithm = obj
                                .get("algorithm")
                                .and_then(|v| v.as_str())
                                .unwrap_or("force_directed")
                                .to_string();
                            let params = obj
                                .get("params")
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or_default();
                            eprintln!("✅ DEBUG: Parsed as UI ChangeLayout format");
                            return Ok(ControlMsg::ChangeLayout { algorithm, params });
                        }
                        "ChangeEmbedding" => {
                            let method = obj
                                .get("method")
                                .and_then(|v| v.as_str())
                                .unwrap_or("spectral")
                                .to_string();
                            let k = obj.get("k").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                            let params = obj
                                .get("params")
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or_default();
                            eprintln!("✅ DEBUG: Parsed as UI ChangeEmbedding format");
                            return Ok(ControlMsg::ChangeEmbedding { method, k, params });
                        }
                        _ => {
                            eprintln!("⚠️  DEBUG: Unknown message type: {}", msg_type);
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
                    eprintln!("✅ DEBUG: Parsed as simple layout format");
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
        eprintln!(
            "📊 DEBUG: WsBridge storing snapshot with {} nodes, {} edges",
            snapshot.node_count(),
            snapshot.edge_count()
        );
        *self.latest_snapshot.lock().await = Some(snapshot);
    }

    /// Broadcast update to all connected clients
    pub async fn broadcast_update(&self, update: EngineUpdate) -> GraphResult<()> {
        let client_count = self.clients.lock().await.len();
        eprintln!("📡 DEBUG: Broadcasting update to {} clients", client_count);

        // Send to broadcast channel
        let _ = self.update_tx.send(update.clone());

        // Send directly to clients (fallback)
        let clients = self.clients.lock().await;
        let message = WsMessage::Update {
            version: 1,
            payload: update,
        };

        for (client_id, sender) in clients.iter() {
            if let Err(e) = sender.send(message.clone()) {
                eprintln!("⚠️  DEBUG: Failed to send to client {}: {}", client_id, e);
            }
        }

        Ok(())
    }

    /// Handle new WebSocket connection (already handshaken)
    pub async fn handle_websocket_stream(
        &self,
        stream: tokio::net::TcpStream,
        addr: std::net::SocketAddr,
    ) -> GraphResult<()> {
        eprintln!("🔗 DEBUG: New WebSocket stream from {}", addr);

        // Convert to WebSocket stream (skip handshake since we already did it)
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::WebSocketStream;

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

        eprintln!("👤 DEBUG: Assigned client ID: {}", client_id);

        // Create message channel for this client
        let (client_tx, mut client_rx) = mpsc::unbounded_channel();

        // Add client to bridge
        {
            let mut clients = self.clients.lock().await;
            clients.insert(client_id, client_tx);
        }

        // Send initial snapshot if available
        if let Some(snapshot) = self.latest_snapshot.lock().await.clone() {
            eprintln!("📊 DEBUG: Sending initial snapshot to client {}", client_id);
            let snapshot_msg = WsMessage::Snapshot {
                version: 1,
                payload: snapshot,
            };

            if let Ok(json) = serde_json::to_string(&snapshot_msg) {
                let ws_sender_clone = ws_sender.clone();
                tokio::spawn(async move {
                    let mut sender = ws_sender_clone.lock().await;
                    use futures_util::SinkExt;
                    let _ = sender
                        .send(tokio_tungstenite::tungstenite::Message::Text(json))
                        .await;
                });
            }
        }

        // Subscribe to broadcast updates
        let mut update_rx = self.update_tx.subscribe();

        // Handle messages from this client
        let clients_clone = self.clients.clone();
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
                                payload: update,
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
            eprintln!("🔌 DEBUG: Client {} disconnected", client_id);
            clients_clone.lock().await.remove(&client_id);
        });

        // Handle incoming messages from WebSocket
        let clients_clone = self.clients.clone();
        let control_tx_clone = self.control_tx.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(tokio_tungstenite::tungstenite::Message::Text(text)) => {
                        eprintln!("📨 DEBUG: Received from client {}: {}", client_id, text);

                        // Try to parse as control message
                        match Self::parse_control_message(&text) {
                            Ok(control_msg) => {
                                eprintln!("🎮 DEBUG: Parsed control message: {:?}", control_msg);
                                Self::handle_control_message_static(
                                    client_id,
                                    control_msg,
                                    control_tx_clone.clone(),
                                    clients_clone.clone(),
                                )
                                .await;
                            }
                            Err(e) => {
                                eprintln!("❌ DEBUG: Failed to parse control message: {}", e);
                                eprintln!("📝 DEBUG: Raw message: {}", text);
                            }
                        }
                    }
                    Ok(tokio_tungstenite::tungstenite::Message::Close(_)) => {
                        eprintln!("🔌 DEBUG: Client {} sent close", client_id);
                        break;
                    }
                    Err(e) => {
                        eprintln!("❌ DEBUG: WebSocket error for client {}: {}", client_id, e);
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

    /// Handle new WebSocket connection
    pub async fn handle_connection(
        &self,
        stream: tokio::net::TcpStream,
        addr: std::net::SocketAddr,
    ) -> GraphResult<()> {
        eprintln!("🔗 DEBUG: New WebSocket connection from {}", addr);

        let ws_stream = accept_async(stream).await.map_err(|e| {
            io_error_to_graph_error(
                std::io::Error::new(std::io::ErrorKind::InvalidData, e),
                "websocket_handshake",
                "tcp_stream",
            )
        })?;

        let (ws_sender, mut ws_receiver) = ws_stream.split();
        let ws_sender: Arc<
            Mutex<futures_util::stream::SplitSink<WebSocketStream<tokio::net::TcpStream>, Message>>,
        > = Arc::new(Mutex::new(ws_sender));

        // Generate client ID
        let client_id = {
            let mut next_id = self.next_client_id.lock().await;
            let id = *next_id;
            *next_id += 1;
            id
        };

        eprintln!("👤 DEBUG: Assigned client ID: {}", client_id);

        // Create message channel for this client
        let (client_tx, mut client_rx) = mpsc::unbounded_channel();

        // Add client to bridge
        {
            let mut clients = self.clients.lock().await;
            clients.insert(client_id, client_tx);
        }

        // Send initial snapshot if available
        if let Some(snapshot) = self.latest_snapshot.lock().await.clone() {
            eprintln!("📊 DEBUG: Sending initial snapshot to client {}", client_id);
            let snapshot_msg = WsMessage::Snapshot {
                version: 1,
                payload: snapshot,
            };

            if let Ok(json) = serde_json::to_string(&snapshot_msg) {
                let ws_sender_clone = ws_sender.clone();
                tokio::spawn(async move {
                    let mut sender = ws_sender_clone.lock().await;
                    use futures_util::SinkExt;
                    let _ = sender.send(Message::Text(json)).await;
                });
            }
        }

        // Subscribe to broadcast updates
        let mut update_rx = self.update_tx.subscribe();

        // Handle messages from this client
        let clients_clone = self.clients.clone();
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
                                if sender.send(Message::Text(json)).await.is_err() {
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
                                payload: update,
                            };
                            if let Ok(json) = serde_json::to_string(&message) {
                                let mut sender = ws_sender_clone.lock().await;
                                if sender.send(Message::Text(json)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Remove client on disconnect
            eprintln!("🔌 DEBUG: Client {} disconnected", client_id);
            clients_clone.lock().await.remove(&client_id);
        });

        // Handle incoming messages from WebSocket
        let clients_clone = self.clients.clone();
        let control_tx_clone = self.control_tx.clone();
        tokio::spawn(async move {
            use futures_util::StreamExt;

            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        eprintln!("📨 DEBUG: Received from client {}: {}", client_id, text);

                        // Try to parse as control message
                        match Self::parse_control_message(&text) {
                            Ok(control_msg) => {
                                eprintln!("🎮 DEBUG: Parsed control message: {:?}", control_msg);
                                Self::handle_control_message_static(
                                    client_id,
                                    control_msg,
                                    control_tx_clone.clone(),
                                    clients_clone.clone(),
                                )
                                .await;
                            }
                            Err(e) => {
                                eprintln!("❌ DEBUG: Failed to parse control message: {}", e);
                                eprintln!("📝 DEBUG: Raw message: {}", text);
                            }
                        }
                    }
                    Ok(Message::Close(_)) => {
                        eprintln!("🔌 DEBUG: Client {} sent close", client_id);
                        break;
                    }
                    Err(e) => {
                        eprintln!("❌ DEBUG: WebSocket error for client {}: {}", client_id, e);
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

    /// Get number of connected clients
    pub async fn client_count(&self) -> usize {
        self.clients.lock().await.len()
    }

    /// Get update channel sender for engine integration
    pub fn get_update_sender(&self) -> mpsc::UnboundedSender<EngineUpdate> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let bridge_tx = self.update_tx.clone();

        // Bridge mpsc to broadcast
        tokio::spawn(async move {
            while let Some(update) = rx.recv().await {
                let _ = bridge_tx.send(update);
            }
        });

        tx
    }

    /// Handle control message from client (static version for tokio spawn)
    async fn handle_control_message_static(
        client_id: ClientId,
        control_msg: ControlMsg,
        control_tx: Option<mpsc::UnboundedSender<(ClientId, ControlMsg)>>,
        clients: Arc<Mutex<HashMap<ClientId, mpsc::UnboundedSender<WsMessage>>>>,
    ) {
        eprintln!(
            "🎮 DEBUG: Processing control message from client {}: {:?}",
            client_id, control_msg
        );

        // Forward to engine if we have a control sender
        if let Some(control_tx) = control_tx {
            if let Err(e) = control_tx.send((client_id, control_msg.clone())) {
                eprintln!(
                    "❌ DEBUG: Failed to forward control message to engine: {}",
                    e
                );
                Self::send_control_ack_static(client_id, false, "Engine unavailable", clients)
                    .await;
                return;
            }
        } else {
            eprintln!("⚠️  DEBUG: No control sender available - engine not connected");
            Self::send_control_ack_static(client_id, false, "Engine not connected", clients).await;
            return;
        }

        // Send immediate ack (real response will come from engine)
        Self::send_control_ack_static(client_id, true, "Control message processed", clients).await;
    }

    /// Handle control message from client (instance method)
    async fn handle_control_message(&self, client_id: ClientId, control_msg: ControlMsg) {
        Self::handle_control_message_static(
            client_id,
            control_msg,
            self.control_tx.clone(),
            self.clients.clone(),
        )
        .await;
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
            if let Err(e) = client_tx.send(ack_msg) {
                eprintln!(
                    "❌ DEBUG: Failed to send control ack to client {}: {}",
                    client_id, e
                );
            }
        }
    }

    /// Send control acknowledgment to client (instance method)
    async fn send_control_ack(&self, client_id: ClientId, success: bool, message: &str) {
        Self::send_control_ack_static(client_id, success, message, self.clients.clone()).await;
    }
}
