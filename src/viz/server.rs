//! WebSocket server for real-time graph visualization communication

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, RwLock};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use uuid::Uuid;

use crate::errors::{GraphResult, GraphError};
use super::{InteractiveOptions};
use crate::viz::streaming::data_source::{DataSource, GraphNode as VizNode, GraphEdge as VizEdge, GraphMetadata, Position};

/// WebSocket server for graph visualization
pub struct VizServer {
    /// Server address
    addr: SocketAddr,
    /// Graph data source
    graph_data: Arc<dyn DataSource>,
    /// Active WebSocket sessions
    sessions: Arc<RwLock<HashMap<SessionId, SessionState>>>,
    /// Broadcast channel for sending updates to all clients
    update_sender: broadcast::Sender<ServerMessage>,
    /// Server shutdown signal
    shutdown_sender: tokio::sync::oneshot::Sender<()>,
    /// Server handle for cleanup
    server_handle: tokio::task::JoinHandle<GraphResult<()>>,
}

impl VizServer {
    /// Create and start a new visualization server
    pub async fn new(
        graph_data: Arc<dyn DataSource + Send + Sync>,
        options: InteractiveOptions,
    ) -> GraphResult<Self> {
        // Bind to available port
        let addr: SocketAddr = if options.port == 0 {
            "127.0.0.1:0".parse().unwrap()
        } else {
            format!("127.0.0.1:{}", options.port).parse().unwrap()
        };
        
        let listener = TcpListener::bind(&addr).await
            .map_err(|e| GraphError::IoError { 
                operation: "bind server".to_string(),
                path: format!("{}", addr),
                underlying_error: e.to_string()
            })?;
        
        let actual_addr = listener.local_addr()
            .map_err(|e| GraphError::IoError { 
                operation: "get server address".to_string(),
                path: format!("{}", addr),
                underlying_error: e.to_string()
            })?;

        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (update_sender, _) = broadcast::channel(100);
        let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();

        // Clone data for the server task
        let server_graph_data = graph_data.clone();
        let server_sessions = sessions.clone();
        let server_update_sender = update_sender.clone();

        // Start the server task
        let server_handle = tokio::spawn(Self::run_server(
            listener,
            server_graph_data,
            server_sessions,
            server_update_sender,
            shutdown_receiver,
        ));

        Ok(Self {
            addr: actual_addr,
            graph_data,
            sessions,
            update_sender,
            shutdown_sender,
            server_handle,
        })
    }

    /// Get the port the server is running on
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Stop the server
    pub async fn stop(self) -> GraphResult<()> {
        // Send shutdown signal
        let _ = self.shutdown_sender.send(());
        
        // Wait for server to stop
        self.server_handle.await
            .map_err(|e| GraphError::InternalError { 
                message: format!("Server task panicked: {}", e),
                location: "VizServer::stop".to_string(),
                context: HashMap::new() 
            })?
    }

    /// Main server loop
    async fn run_server(
        listener: TcpListener,
        graph_data: Arc<dyn DataSource + Send + Sync>,
        sessions: Arc<RwLock<HashMap<SessionId, SessionState>>>,
        update_sender: broadcast::Sender<ServerMessage>,
        mut shutdown_receiver: tokio::sync::oneshot::Receiver<()>,
    ) -> GraphResult<()> {
        println!("🎨 Visualization server starting on {}", listener.local_addr().unwrap());

        loop {
            tokio::select! {
                // Handle new connections
                Ok((stream, addr)) = listener.accept() => {
                    let graph_data = graph_data.clone();
                    let sessions = sessions.clone();
                    let update_sender = update_sender.clone();
                    
                    tokio::spawn(Self::handle_connection(
                        stream,
                        addr,
                        graph_data,
                        sessions,
                        update_sender,
                    ));
                }
                
                // Handle shutdown signal
                _ = &mut shutdown_receiver => {
                    println!("🛑 Visualization server shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Handle a new WebSocket connection
    async fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        graph_data: Arc<dyn DataSource + Send + Sync>,
        sessions: Arc<RwLock<HashMap<SessionId, SessionState>>>,
        update_sender: broadcast::Sender<ServerMessage>,
    ) {
        println!("🔌 New connection from {}", addr);

        // Upgrade to WebSocket
        let ws_stream = match accept_async(stream).await {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("❌ WebSocket upgrade failed: {}", e);
                return;
            }
        };

        let session_id = SessionId::new();
        let (ws_sender, mut ws_receiver) = ws_stream.split();
        let mut update_receiver = update_sender.subscribe();

        // Create session state
        let session_state = SessionState {
            id: session_id.clone(),
            addr,
        };

        // Register session
        sessions.write().await.insert(session_id.clone(), session_state);

        // Send initial graph data
        let initial_data = ServerMessage::InitialData {
            nodes: graph_data.get_nodes(),
            edges: graph_data.get_edges(),
            metadata: graph_data.get_metadata(),
        };

        let ws_sender = Arc::new(tokio::sync::Mutex::new(ws_sender));
        let sender_for_initial = ws_sender.clone();

        // Send initial data
        tokio::spawn(async move {
            let mut sender = sender_for_initial.lock().await;
            if let Ok(msg) = serde_json::to_string(&initial_data) {
                let _ = sender.send(Message::Text(msg)).await;
            }
        });

        // Handle messages
        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                msg = ws_receiver.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Err(e) = Self::handle_client_message(
                                &session_id,
                                &text,
                                &graph_data,
                                &update_sender,
                            ).await {
                                eprintln!("❌ Error handling message: {}", e);
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            println!("👋 Client {} disconnected", session_id.0);
                            break;
                        }
                        Some(Ok(Message::Binary(_))) => {
                            // Handle binary messages if needed
                        }
                        Some(Ok(Message::Ping(_))) => {
                            // Handle ping messages
                        }
                        Some(Ok(Message::Pong(_))) => {
                            // Handle pong messages  
                        }
                        Some(Ok(Message::Frame(_))) => {
                            // Handle raw frames
                        }
                        Some(Err(e)) => {
                            eprintln!("❌ WebSocket error: {}", e);
                            break;
                        }
                        None => break,
                    }
                }
                
                // Handle server updates
                Ok(server_msg) = update_receiver.recv() => {
                    let sender = ws_sender.clone();
                    tokio::spawn(async move {
                        let mut sender = sender.lock().await;
                        if let Ok(msg) = serde_json::to_string(&server_msg) {
                            let _ = sender.send(Message::Text(msg)).await;
                        }
                    });
                }
            }
        }

        // Cleanup session
        sessions.write().await.remove(&session_id);
        println!("🧹 Session {} cleaned up", session_id.0);
    }

    /// Handle a message from a client
    async fn handle_client_message(
        session_id: &SessionId,
        message: &str,
        graph_data: &Arc<dyn DataSource + Send + Sync>,
        update_sender: &broadcast::Sender<ServerMessage>,
    ) -> GraphResult<()> {
        let client_message: ClientMessage = serde_json::from_str(message)
            .map_err(|e| GraphError::SerializationError { 
                data_type: "ClientMessage".to_string(),
                operation: "deserialize".to_string(),
                underlying_error: e.to_string()
            })?;

        match client_message {
            ClientMessage::GetNodeDetails { node_id } => {
                // Find node and send detailed information
                let nodes = graph_data.get_nodes();
                if let Some(node) = nodes.iter().find(|n| n.id == node_id) {
                    let response = ServerMessage::NodeDetails {
                        node: node.clone(),
                    };
                    let _ = update_sender.send(response);
                }
            }
            
            ClientMessage::FilterNodes { criteria } => {
                // Apply filter and send filtered node list
                let nodes = graph_data.get_nodes();
                let filtered_nodes: Vec<VizNode> = nodes.into_iter()
                    .filter(|node| Self::matches_criteria(node, &criteria))
                    .collect();
                
                let response = ServerMessage::FilteredNodes {
                    nodes: filtered_nodes,
                };
                let _ = update_sender.send(response);
            }
            
            ClientMessage::RequestLayout { algorithm } => {
                // Compute new layout (placeholder for now)
                println!("🎯 Layout request: {:?}", algorithm);
                // TODO: Implement layout computation
            }
        }

        Ok(())
    }

    /// Check if a node matches filter criteria
    fn matches_criteria(node: &VizNode, criteria: &FilterCriteria) -> bool {
        // Simple attribute-based filtering
        for (key, value) in &criteria.attributes {
            if let Some(node_value) = node.attributes.get(key) {
                if node_value != value {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

/// Unique session identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct SessionId(pub String);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// State for each WebSocket session
#[derive(Debug)]
pub struct SessionState {
    pub id: SessionId,
    pub addr: SocketAddr,
}

/// Messages sent from server to client
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    InitialData {
        nodes: Vec<VizNode>,
        edges: Vec<VizEdge>,
        metadata: GraphMetadata,
    },
    NodeDetails {
        node: VizNode,
    },
    FilteredNodes {
        nodes: Vec<VizNode>,
    },
    LayoutUpdate {
        positions: HashMap<String, Position>,
    },
    Error {
        message: String,
    },
}

/// Messages sent from client to server
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    GetNodeDetails {
        node_id: String,
    },
    FilterNodes {
        criteria: FilterCriteria,
    },
    RequestLayout {
        algorithm: String,
    },
}

/// Filter criteria for nodes/edges
#[derive(Debug, serde::Deserialize)]
pub struct FilterCriteria {
    pub attributes: HashMap<String, serde_json::Value>,
}