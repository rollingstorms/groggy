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
                        if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                            match ws_msg {
                                WsMessage::Control { payload, .. } => {
                                    eprintln!("🎮 DEBUG: Control message: {:?}", payload);
                                    Self::handle_control_message_static(
                                        client_id,
                                        payload,
                                        control_tx_clone.clone(),
                                        clients_clone.clone(),
                                    )
                                    .await;
                                }
                                _ => {
                                    eprintln!("⚠️  DEBUG: Unexpected message type from client");
                                }
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
                        if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                            match ws_msg {
                                WsMessage::Control { payload, .. } => {
                                    eprintln!("🎮 DEBUG: Control message: {:?}", payload);
                                    Self::handle_control_message_static(
                                        client_id,
                                        payload,
                                        control_tx_clone.clone(),
                                        clients_clone.clone(),
                                    )
                                    .await;
                                }
                                _ => {
                                    eprintln!("⚠️  DEBUG: Unexpected message type from client");
                                }
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
