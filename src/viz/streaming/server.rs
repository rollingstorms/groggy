//! WebSocket server for real-time streaming table updates
//!
//! Provides WebSocket communication for streaming data to browser clients
//! with real-time updates and virtual scrolling support.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::net::{SocketAddr, IpAddr};
use std::thread::{self, JoinHandle as StdJoinHandle};
use serde::{Serialize, Deserialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use tokio::time::{sleep, Duration};
use tokio::{select, task::JoinHandle};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tokio_util::sync::CancellationToken;
use futures_util::{SinkExt, StreamExt};
use super::html::*;
use super::handlers::*;
use super::types::{
    WSMessage, StreamingConfig, StreamingResult, StreamingError,
    ServerHandle, ClientState, ConnectionId, ProtocolMeta,
    KeyboardAction, SearchType, DataUpdate, UpdateType,
    GraphNodeData, GraphEdgeData, GraphMetadataData, NodePositionData,
    HighlightChange, HighlightElementType, HighlightType,
    SearchFilter, MatchedField, SearchResult, SearchResultType,
    HighlightData, ServerStats,
    data_window_to_json
};

use super::data_source::{
    DataSource, DataWindow, DataSchema, DataWindowMetadata, GraphNode, GraphEdge, GraphMetadata, 
    LayoutAlgorithm, NodePosition, Position
};
use super::virtual_scroller::{VirtualScrollManager, VirtualScrollConfig};
use crate::viz::layouts::{LayoutEngine, ForceDirectedLayout, CircularLayout};

/// WebSocket server for streaming table data
#[derive(Debug, Clone)]
pub struct StreamingServer {
    /// Virtual scroll manager
    pub virtual_scroller: VirtualScrollManager,
    
    /// Data source being served
    pub data_source: Arc<dyn DataSource>,
    
    /// Active client connections
    active_connections: Arc<RwLock<HashMap<ConnectionId, ClientState>>>,
    
    /// Server configuration
    pub config: StreamingConfig,
    
    /// Actual port the server is running on (filled after start)
    actual_port: Option<u16>,
    
    /// Unique run identifier for this server instance
    pub run_id: String,
}

impl StreamingServer {
    /// Create new streaming server
    pub fn new(
        data_source: Arc<dyn DataSource>, 
        config: StreamingConfig
    ) -> Self {
        let virtual_scroller = VirtualScrollManager::new(config.scroll_config.clone());
        
        // Generate unique run identifier
        let run_id = format!("RUN{}", 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );
        
        Self {
            virtual_scroller,
            data_source,
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            config,
            actual_port: None,
            run_id,
        }
    }
    
    /// Get protocol metadata for this server instance
    fn get_meta(&self) -> ProtocolMeta {
        ProtocolMeta {
            run_id: self.run_id.clone(),
            protocol_version: 1,
        }
    }
    
    /// Centralized WebSocket message sending with poison-pill guards
    /// All WebSocket messages must go through this function to detect leaks
    async fn send_ws<T: serde::Serialize>(
        ws_sender: &mut futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
            tokio_tungstenite::tungstenite::Message
        >,
        msg: &T
    ) -> StreamingResult<()> {
        let json = serde_json::to_string(msg)
            .map_err(|e| StreamingError::WebSocket(format!("JSON serialization error: {}", e)))?;
        
        // Poison-pill guard: detect any tagged AttrValue JSON
        if json.contains(r#""Text":"#) || json.contains(r#""SmallInt":"#) || json.contains(r#""Int":"#) || 
           json.contains(r#""Float":"#) || json.contains(r#""Bool":"#) || json.contains(r#""CompactText":"#) {
            return Err(StreamingError::WebSocket("Poison guard: tagged AttrValue detected in WebSocket message".to_string()));
        }
        
        ws_sender.send(tokio_tungstenite::tungstenite::Message::Text(json)).await
            .map_err(|e| StreamingError::WebSocket(format!("Failed to send WebSocket message: {}", e)))?;
        
        Ok(())
    }
    
    /// Start WebSocket server on a dedicated runtime thread
    /// This avoids the Handle::block_on() deadlock issue by giving the server its own runtime
    pub fn start_background(&self, addr: IpAddr, port_hint: u16) -> StreamingResult<ServerHandle> {
        let server = self.clone();
        let cancel = CancellationToken::new();
        let cancel_child = cancel.clone();

        // Tell caller the bound port once we actually bind
        let (ready_tx, ready_rx) = oneshot::channel::<u16>();

        let thread: StdJoinHandle<()> = thread::Builder::new()
            .name("groggy-streaming".into())
            .spawn(move || {
                let rt = Runtime::new().expect("tokio runtime");
                // Keep this thread parked on the runtime until canceled
                rt.block_on(async move {
                    // 1) Bind first (real socket)
                    let listener = match TcpListener::bind((addr, port_hint)).await {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = ready_tx.send(0); // signal failure
                            eprintln!("❌ bind failed: {e}");
                            return;
                        }
                    };
                    let port = listener.local_addr().ok().map(|a| a.port()).unwrap_or(0);
                    
                    println!("✅ Bound and listening on {}:{}", addr, port);
                    println!("📊 Serving {} rows × {} columns", 
                            server.data_source.total_rows(), 
                            server.data_source.total_cols());

                    // 2) Notify the caller we're ready *after* binding
                    let _ = ready_tx.send(port);

                    // 3) Accept loop stays alive until cancel
                    if let Err(e) = server.accept_loop_direct(listener, cancel_child, port).await {
                        eprintln!("❌ accept loop error: {e}");
                    }
                });
            })
            .map_err(|e| StreamingError::Server(format!("failed to spawn server thread: {e}")))?;

        // Wait until the listener is actually bound (or failed)
        let port = ready_rx
            .blocking_recv()
            .map_err(|_| StreamingError::Server("server thread died before binding".into()))?;

        if port == 0 {
            return Err(StreamingError::Server("failed to bind listener".into()));
        }

        Ok(ServerHandle { port, cancel, thread: Some(thread) })
    }

    /// Resilient accept loop that runs directly on the runtime thread
    async fn accept_loop_direct(self, listener: TcpListener, cancel: CancellationToken, server_port: u16) -> StreamingResult<()> {
        loop {
            select! {
                _ = cancel.cancelled() => {
                    println!("🛑 shutdown requested; stopping accept loop");
                    break;
                }
                res = listener.accept() => {
                    match res {
                        Ok((stream, addr)) => {
                            println!("🔗 NEW CONNECTION: {}", addr);
                            // Spawn connection handler but keep the accept loop running
                            let server_clone = self.clone();
                            tokio::spawn(async move {
                                if let Err(e) = server_clone.handle_connection_with_port(stream, addr, server_port).await {
                                    eprintln!("🔻 connection {} ended with error: {}", addr, e);
                                } else {
                                    println!("✅ connection {} completed successfully", addr);
                                }
                            });
                        }
                        Err(e) => {
                            eprintln!("⚠️ accept error: {}; retrying", e);
                            sleep(Duration::from_millis(50)).await;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Handle individual client connections (HTTP + WebSocket) with known server port
    async fn handle_connection_with_port(&self, mut stream: TcpStream, addr: SocketAddr, server_port: u16) -> StreamingResult<()> {
        println!("🌐 HANDLING CONNECTION: {} on port {}", addr, server_port);
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        
        // Use peek() to sniff the request without consuming the stream
        let mut probe = [0u8; 1024];
        match stream.peek(&mut probe).await {
            Ok(n) if n > 0 => {
                let request_head = std::str::from_utf8(&probe[..n]).unwrap_or_default();
                
                if self.is_websocket_upgrade(&request_head) {
                    // This is a WebSocket upgrade request - pass untouched stream to tungstenite
                    println!("🔌 WebSocket upgrade request from {}", addr);
                    
                    match tokio_tungstenite::accept_async(stream).await {
                        Ok(ws_stream) => {
                            let conn_id = format!("{}:{}", addr.ip(), addr.port());
                            println!("📱 WebSocket client connected: {}", conn_id);
                            
                            if let Err(e) = self.handle_client_websocket(ws_stream, conn_id.clone()).await {
                                eprintln!("❌ WebSocket error for {}: {}", conn_id, e);
                            }
                            
                            // Clean up client connection
                            if let Ok(mut connections) = self.active_connections.write() {
                                connections.remove(&conn_id);
                            }
                            println!("📱 WebSocket client disconnected: {}", conn_id);
                        }
                        Err(e) => {
                            eprintln!("❌ WebSocket handshake failed for {}: {}", addr, e);
                            return Err(StreamingError::WebSocket(format!("WebSocket handshake failed: {}", e)));
                        }
                    }
                } else if request_head.starts_with("GET ") {
                    // This is a regular HTTP request - now we can safely read and parse
                    println!("🌐 HTTP request from {}", addr);
                    
                    self.handle_http_request_headers_only(stream, &request_head, addr, server_port).await?;
                } else {
                    eprintln!("❓ Unknown request type from {}: {}", addr, 
                              &request_head.chars().take(50).collect::<String>());
                }
            }
            Ok(_) => {
                eprintln!("📪 Empty request from {}", addr);
            }
            Err(e) => {
                eprintln!("❌ Failed to peek from {}: {}", addr, e);
                return Err(StreamingError::Server(format!("Failed to peek from stream: {}", e)));
            }
        }
        
        Ok(())
    }

    /// Handle individual client connections (HTTP + WebSocket) - legacy method
    async fn handle_connection(&self, stream: TcpStream, addr: SocketAddr) -> StreamingResult<()> {
        // Default to using the configured port
        let config_port = self.config.port;
        self.handle_connection_with_port(stream, addr, config_port).await
    }
    
    /// Check if the request is a WebSocket upgrade
    fn is_websocket_upgrade(&self, head: &str) -> bool {
        let h = head.to_ascii_lowercase();
        h.starts_with("get ")
            && h.contains("upgrade: websocket")
            && h.contains("connection: upgrade")
            && h.contains("sec-websocket-key:")
            && (h.contains("http/1.1") || h.contains("http/1.0"))
    }

    /// Handle WebSocket communication with a client
    async fn handle_client_websocket(
        &self,
        ws_stream: tokio_tungstenite::WebSocketStream<TcpStream>,
        conn_id: ConnectionId,
    ) -> StreamingResult<()> {
        use futures_util::{SinkExt, StreamExt};

        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        // Send initial data
        let initial_window = self.virtual_scroller.get_visible_window(self.data_source.as_ref())?;
        let converted_window = data_window_to_json(&initial_window);
        
        let initial_msg = WSMessage::InitialData {
            window: converted_window,
            total_rows: self.data_source.total_rows(),
            meta: self.get_meta(),
        };

        // Use centralized send function with poison-pill guard
        Self::send_ws(&mut ws_sender, &initial_msg).await?;

        // Add client to active connections
        let client_state = ClientState {
            connection_id: conn_id.clone(),
            current_offset: 0,
            last_update: std::time::SystemTime::now(),
            subscribed_updates: true,
        };

        if let Ok(mut connections) = self.active_connections.write() {
            connections.insert(conn_id.clone(), client_state);
        }

        // Handle incoming messages
        while let Some(msg_result) = ws_receiver.next().await {
            match msg_result {
                Ok(Message::Text(text)) => {
                    if let Err(e) = self.handle_client_message(&conn_id, &text, &mut ws_sender).await {
                        eprintln!("❌ Error handling message from {}: {}", conn_id, e);
                    }
                }
                Ok(Message::Close(_)) => {
                    println!("📱 Client {} requested close", conn_id);
                    break;
                }
                Err(e) => {
                    eprintln!("❌ WebSocket error for {}: {}", conn_id, e);
                    break;
                }
                _ => {} // Ignore other message types
            }
        }

        Ok(())
    }

    /// Handle incoming client message
    async fn handle_client_message(
        &self,
        conn_id: &ConnectionId,
        message: &str,
        ws_sender: &mut futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<TcpStream>,
            Message,
        >,
    ) -> StreamingResult<()> {
        println!("📩 HANDLE CLIENT MESSAGE: {} chars from {}", message.len(), conn_id);
        use futures_util::SinkExt;

        let parsed_msg: WSMessage = serde_json::from_str(message)
            .map_err(|e| StreamingError::Client(format!("Invalid JSON from {}: {}", conn_id, e)))?;

        match parsed_msg {
            WSMessage::ScrollRequest { offset, window_size: _ } => {
                // Handle scroll request
                let window = self.virtual_scroller.get_window_at_offset(
                    self.data_source.as_ref(),
                    offset,
                )?;

                // Update client state
                if let Ok(mut connections) = self.active_connections.write() {
                    if let Some(client) = connections.get_mut(conn_id) {
                        client.current_offset = offset;
                        client.last_update = std::time::SystemTime::now();
                    }
                }

                let response = WSMessage::DataUpdate {
                    new_window: data_window_to_json(&window),
                    offset,
                    meta: self.get_meta(),
                };

                // Use centralized send function with poison-pill guard
                Self::send_ws(ws_sender, &response).await?;
            }
            WSMessage::ThemeChange { theme } => {
                println!("🎨 Client {} changed theme to: {}", conn_id, theme);
                // Theme changes can be acknowledged or ignored for now
            }
            // NEW: Graph visualization message handlers
            WSMessage::GraphDataRequest { layout_algorithm, theme: _theme } => {
                println!("📊 Client {} requested graph data", conn_id);
                
                // Check if data source supports graph view
                if !self.data_source.supports_graph_view() {
                    let error_response = WSMessage::Error {
                        message: "Data source does not support graph visualization".to_string(),
                        error_code: "NO_GRAPH_SUPPORT".to_string(),
                    };
                    let error_json = serde_json::to_string(&error_response)
                        .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                    ws_sender.send(Message::Text(error_json)).await
                        .map_err(|e| StreamingError::WebSocket(format!("Failed to send error: {}", e)))?;
                    return Ok(());
                }
                
                // Get graph data
                let nodes = self.data_source.get_graph_nodes();
                let edges = self.data_source.get_graph_edges(); 
                let metadata = self.data_source.get_graph_metadata();
                
                // Convert to WebSocket data types
                let node_data: Vec<GraphNodeData> = nodes.iter().map(|n| n.into()).collect();
                let edge_data: Vec<GraphEdgeData> = edges.iter().map(|e| e.into()).collect();
                let metadata_data: GraphMetadataData = (&metadata).into();
                
                // Compute layout if algorithm specified
                let layout_positions = if let Some(algorithm_name) = layout_algorithm {
                    let algorithm = self.parse_layout_algorithm(&algorithm_name);
                    let positions = self.compute_layout_with_engine(algorithm);
                    Some(positions.iter().map(|p| p.into()).collect())
                } else {
                    None
                };
                
                let response = WSMessage::GraphDataResponse {
                    nodes: node_data,
                    edges: edge_data,
                    metadata: metadata_data,
                    layout_positions,
                };
                
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send graph data: {}", e)))?;
            }
            WSMessage::LayoutRequest { algorithm, parameters: _parameters } => {
                println!("🎯 Client {} requested layout: {}", conn_id, algorithm);
                
                if !self.data_source.supports_graph_view() {
                    let error_response = WSMessage::Error {
                        message: "Data source does not support graph visualization".to_string(),
                        error_code: "NO_GRAPH_SUPPORT".to_string(),
                    };
                    let error_json = serde_json::to_string(&error_response)
                        .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                    ws_sender.send(Message::Text(error_json)).await
                        .map_err(|e| StreamingError::WebSocket(format!("Failed to send error: {}", e)))?;
                    return Ok(());
                }
                
                // Parse and compute layout
                let layout_algo = self.parse_layout_algorithm(&algorithm);
                let positions = self.compute_layout_with_engine(layout_algo);
                let position_data: Vec<NodePositionData> = positions.iter().map(|p| p.into()).collect();
                
                let response = WSMessage::LayoutResponse {
                    positions: position_data,
                    algorithm,
                };
                
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send layout: {}", e)))?;
            }
            WSMessage::MetadataRequest => {
                println!("📋 Client {} requested metadata", conn_id);
                
                if !self.data_source.supports_graph_view() {
                    let error_response = WSMessage::Error {
                        message: "Data source does not support graph visualization".to_string(),
                        error_code: "NO_GRAPH_SUPPORT".to_string(),
                    };
                    let error_json = serde_json::to_string(&error_response)
                        .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                    ws_sender.send(Message::Text(error_json)).await
                        .map_err(|e| StreamingError::WebSocket(format!("Failed to send error: {}", e)))?;
                    return Ok(());
                }
                
                let metadata = self.data_source.get_graph_metadata();
                let metadata_data: GraphMetadataData = (&metadata).into();
                
                let response = WSMessage::MetadataResponse {
                    metadata: metadata_data,
                };
                
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send metadata: {}", e)))?;
            }
            
            // Phase 7: Interactive Features - Node Click Handler
            WSMessage::NodeClickRequest { node_id, position: _, modifier_keys } => {
                println!("🖱️ Client {} clicked node: {} (modifiers: {:?})", conn_id, node_id, modifier_keys);
                
                let response = self.handle_node_click_request(&node_id).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send node click response: {}", e)))?;
            }
            
            // Phase 7: Interactive Features - Node Hover Handler
            WSMessage::NodeHoverRequest { node_id, position: _ } => {
                println!("🎯 Client {} hovered node: {}", conn_id, node_id);
                
                let response = self.handle_node_hover_request(&node_id).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send node hover response: {}", e)))?;
            }
            
            WSMessage::NodeHoverEnd { node_id } => {
                println!("🎯 Client {} stopped hovering node: {}", conn_id, node_id);
                // No response needed - just for logging/cleanup
            }
            
            // Phase 7: Interactive Features - Edge Click Handler
            WSMessage::EdgeClickRequest { edge_id, position: _ } => {
                println!("🖱️ Client {} clicked edge: {}", conn_id, edge_id);
                
                let response = self.handle_edge_click_request(&edge_id).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send edge click response: {}", e)))?;
            }
            
            // Phase 7: Interactive Features - Edge Hover Handler  
            WSMessage::EdgeHoverRequest { edge_id, position: _ } => {
                println!("🎯 Client {} hovered edge: {}", conn_id, edge_id);
                
                let response = self.handle_edge_hover_request(&edge_id).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send edge hover response: {}", e)))?;
            }
            
            // Phase 7: Interactive Features - Multi-Node Selection Handler
            WSMessage::NodesSelectionRequest { node_ids, selection_type, bounding_box: _ } => {
                println!("🖱️ Client {} selected {} nodes (type: {:?})", conn_id, node_ids.len(), selection_type);
                
                let response = self.handle_nodes_selection_request(&node_ids).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send selection response: {}", e)))?;
            }
            
            WSMessage::ClearSelectionRequest => {
                println!("🖱️ Client {} cleared selection", conn_id);
                // No response needed - frontend handles clearing
            }
            
            // Phase 7: Interactive Features - Keyboard Navigation Handler
            WSMessage::KeyboardActionRequest { action, node_id } => {
                println!("⌨️ Client {} keyboard action: {:?} (focus: {:?})", conn_id, action, node_id);
                
                let response = self.handle_keyboard_action_request(action, node_id.as_deref()).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send keyboard response: {}", e)))?;
            }
            
            // Phase 7: Interactive Features - Search Handler
            WSMessage::SearchRequest { query, search_type, filters } => {
                println!("🔍 Client {} search: '{}' (type: {:?}, {} filters)", conn_id, query, search_type, filters.len());
                
                let response = self.handle_search_request(&query, search_type, &filters).await?;
                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON error: {}", e)))?;
                
                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send search response: {}", e)))?;
            }
            
            _ => {
                eprintln!("⚠️ Unhandled message type from client {}", conn_id);
            }
        }

        Ok(())
    }

    // ============================================================================
    // Phase 7: Interactive Features - Handler Methods  
    // ============================================================================
    
    /// Handle node click request - compute detailed node information
// moved to modules

    
    /// Handle node hover request - create rich tooltip data
// moved to modules

    
    /// Handle edge click request - compute detailed edge information
// moved to modules

    
    /// Handle edge hover request - create edge tooltip data
// moved to modules

    
    /// Handle multi-node selection request - compute bulk analytics
// moved to modules

    
    /// Handle keyboard action request - navigate and control graph
    async fn handle_keyboard_action_request(&self, action: KeyboardAction, current_focus: Option<&str>) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let mut highlight_changes = Vec::new();
        let mut new_focus_node = None;
        
        match &action {
            KeyboardAction::FocusNext => {
                if let Some(current) = current_focus {
                    // Find next node in list
                    if let Some(current_index) = nodes.iter().position(|n| n.id == current) {
                        let next_index = (current_index + 1) % nodes.len();
                        new_focus_node = Some(nodes[next_index].id.clone());
                    }
                } else if !nodes.is_empty() {
                    new_focus_node = Some(nodes[0].id.clone());
                }
                
                if let Some(ref focus_id) = new_focus_node {
                    highlight_changes.push(HighlightChange {
                        element_id: focus_id.clone(),
                        element_type: HighlightElementType::Node,
                        highlight_type: HighlightType::Focus,
                        duration_ms: None,
                    });
                }
            }
            KeyboardAction::FocusPrevious => {
                if let Some(current) = current_focus {
                    if let Some(current_index) = nodes.iter().position(|n| n.id == current) {
                        let prev_index = if current_index == 0 { nodes.len() - 1 } else { current_index - 1 };
                        new_focus_node = Some(nodes[prev_index].id.clone());
                    }
                } else if !nodes.is_empty() {
                    new_focus_node = Some(nodes[nodes.len() - 1].id.clone());
                }
                
                if let Some(ref focus_id) = new_focus_node {
                    highlight_changes.push(HighlightChange {
                        element_id: focus_id.clone(),
                        element_type: HighlightElementType::Node,
                        highlight_type: HighlightType::Focus,
                        duration_ms: None,
                    });
                }
            }
            _ => {
                // TODO: Implement other keyboard actions
                println!("⌨️ Keyboard action {:?} not yet implemented", action);
            }
        }
        
        Ok(WSMessage::KeyboardActionResponse {
            action,
            new_focus_node,
            highlight_changes,
        })
    }
    
    /// Handle search request - find nodes/edges matching query
    async fn handle_search_request(&self, query: &str, search_type: SearchType, _filters: &[SearchFilter]) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let start_time = std::time::Instant::now();
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let mut results = Vec::new();
        
        let query_lower = query.to_lowercase();
        
        match search_type {
            SearchType::Node | SearchType::Global => {
                for node in &nodes {
                    let mut relevance_score = 0.0;
                    let mut matched_fields = Vec::new();
                    
                    // Search in node ID
                    if node.id.to_lowercase().contains(&query_lower) {
                        relevance_score += 1.0;
                        matched_fields.push(MatchedField {
                            field_name: "id".to_string(),
                            matched_text: node.id.clone(),
                            context: format!("Node ID: {}", node.id),
                        });
                    }
                    
                    // Search in label
                    if let Some(label) = &node.label {
                        if label.to_lowercase().contains(&query_lower) {
                            relevance_score += 0.8;
                            matched_fields.push(MatchedField {
                                field_name: "label".to_string(),
                                matched_text: label.clone(),
                                context: format!("Label: {}", label),
                            });
                        }
                    }
                    
                    // Search in attributes
                    for (key, value) in &node.attributes {
                        let value_str = format!("{:?}", value).to_lowercase();
                        if key.to_lowercase().contains(&query_lower) || value_str.contains(&query_lower) {
                            relevance_score += 0.5;
                            matched_fields.push(MatchedField {
                                field_name: key.clone(),
                                matched_text: format!("{:?}", value),
                                context: format!("{}: {:?}", key, value),
                            });
                        }
                    }
                    
                    if relevance_score > 0.0 {
                        results.push(SearchResult {
                            result_type: SearchResultType::Node,
                            id: node.id.clone(),
                            title: node.label.clone().unwrap_or_else(|| node.id.clone()),
                            subtitle: Some(format!("Node • {} attributes", node.attributes.len())),
                            relevance_score,
                            matched_fields,
                            highlight_data: HighlightData {
                                element_id: node.id.clone(),
                                highlight_regions: vec![], // TODO: Implement text highlighting
                            },
                        });
                    }
                }
            }
            SearchType::Edge => {
                for edge in &edges {
                    let mut relevance_score = 0.0;
                    let mut matched_fields = Vec::new();
                    
                    if edge.id.to_lowercase().contains(&query_lower) {
                        relevance_score += 1.0;
                        matched_fields.push(MatchedField {
                            field_name: "id".to_string(),
                            matched_text: edge.id.clone(),
                            context: format!("Edge ID: {}", edge.id),
                        });
                    }
                    
                    if relevance_score > 0.0 {
                        results.push(SearchResult {
                            result_type: SearchResultType::Edge,
                            id: edge.id.clone(),
                            title: edge.label.clone().unwrap_or_else(|| format!("{} → {}", edge.source, edge.target)),
                            subtitle: Some(format!("Edge • {} → {}", edge.source, edge.target)),
                            relevance_score,
                            matched_fields,
                            highlight_data: HighlightData {
                                element_id: edge.id.clone(),
                                highlight_regions: vec![],
                            },
                        });
                    }
                }
            }
            SearchType::Attribute => {
                // TODO: Implement attribute-specific search
            }
        }
        
        // Sort by relevance score
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        
        let query_time_ms = start_time.elapsed().as_millis() as u64;
        let total_matches = results.len();
        
        Ok(WSMessage::SearchResponse {
            results,
            total_matches,
            query_time_ms,
        })
    }

    /// Parse layout algorithm string into LayoutAlgorithm enum
    fn parse_layout_algorithm(&self, algorithm_name: &str) -> LayoutAlgorithm {
        match algorithm_name.to_lowercase().as_str() {
            "force-directed" | "force_directed" | "spring" => LayoutAlgorithm::ForceDirected {
                charge: -300.0,
                distance: 50.0,
                iterations: 100,
            },
            "circular" | "circle" => LayoutAlgorithm::Circular {
                radius: Some(200.0),
                start_angle: 0.0,
            },
            "grid" | "matrix" => LayoutAlgorithm::Grid {
                columns: (self.data_source.get_graph_nodes().len() as f64).sqrt().ceil() as usize,
                cell_size: 100.0,
            },
            "hierarchical" | "tree" => LayoutAlgorithm::Hierarchical {
                direction: crate::viz::streaming::data_source::HierarchicalDirection::TopDown,
                layer_spacing: 100.0,
                node_spacing: 50.0,
            },
            _ => {
                // Default to force-directed for unknown algorithms
                eprintln!("⚠️ Unknown layout algorithm '{}', defaulting to force-directed", algorithm_name);
                LayoutAlgorithm::ForceDirected {
                    charge: -300.0,
                    distance: 50.0,
                    iterations: 100,
                }
            }
        }
    }

    /// Handle HTTP requests for the interactive table HTML page (headers only)
    async fn handle_http_request_headers_only(
        &self,
        mut stream: TcpStream,
        request_head: &str,
        _addr: SocketAddr,
        server_port: u16,
    ) -> StreamingResult<()> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let mut reader = BufReader::new(&mut stream);
        let mut headers = Vec::with_capacity(2048);

        // Read only HTTP headers (until \r\n\r\n)
        loop {
            let n = reader.read_until(b'\n', &mut headers).await
                .map_err(|e| StreamingError::Server(format!("Failed to read HTTP headers: {}", e)))?;

            if n == 0 || headers.len() > 64 * 1024 { // Guard against huge headers
                break;
            }

            if headers.ends_with(b"\r\n\r\n") {
                break;
            }
        }

        // Parse the request path
        let request_path = if let Some(first_line) = request_head.lines().next() {
            if let Some(path_start) = first_line.find(' ') {
                if let Some(path_end) = first_line.rfind(' ') {
                    first_line[path_start + 1..path_end].trim()
                } else {
                    "/"
                }
            } else {
                "/"
            }
        } else {
            "/"
        };

        // Handle different request paths
        let (content_type, content) = if request_path.starts_with("/css/") {
            // Serve CSS files
            self.serve_css_file(request_path).await?
        } else {
            // Serve the main HTML page
            let html_content = self.generate_interactive_html_with_port(server_port).await?;
            ("text/html; charset=utf-8".to_string(), html_content)
        };

        // Create HTTP response
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: {}\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            content_type,
            content.len(),
            content
        );

        // Send response to the original stream (not the BufReader)
        stream.write_all(response.as_bytes()).await
            .map_err(|e| StreamingError::Server(format!("Failed to write HTTP response: {}", e)))?;

        stream.flush().await
            .map_err(|e| StreamingError::Server(format!("Failed to flush HTTP response: {}", e)))?;

        // Close connection
        let _ = stream.shutdown().await;

        Ok(())
    }

    /// Serve CSS files from the embedded resources
    async fn serve_css_file(&self, path: &str) -> StreamingResult<(String, String)> {
        let css_content = match path {
            "/css/sleek.css" => include_str!("css/sleek.css"),
            "/css/graph_visualization.css" => include_str!("css/graph_visualization.css"),
            _ => {
                return Err(StreamingError::Server(format!("CSS file not found: {}", path)));
            }
        };

        Ok(("text/css; charset=utf-8".to_string(), css_content.to_string()))
    }

    /// Generate the HTML content for the interactive streaming table
// moved to modules

    
    /// Generate interactive HTML page with specified port
// moved to modules

    
    /// Broadcast update to all clients (for future real-time updates)
    pub async fn broadcast_update(&self, update: DataUpdate) -> StreamingResult<()> {
        // TODO: Implement broadcasting to active WebSocket connections
        // This would iterate through active_connections and send updates
        println!("📡 Broadcasting update to {} clients", 
                self.active_connections.read().map(|c| c.len()).unwrap_or(0));
        Ok(())
    }
    
    /// Get server statistics
    pub fn get_stats(&self) -> ServerStats {
        let connections = self.active_connections.read()
            .map(|c| c.len())
            .unwrap_or(0);
        
        ServerStats {
            active_connections: connections,
            total_rows: self.data_source.total_rows(),
            total_cols: self.data_source.total_cols(),
            cache_stats: self.virtual_scroller.get_cache_stats(),
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    /// Real-time layout updates are implemented through the existing streaming protocol
    /// Clients can request layout changes using LayoutRequest, and the server responds with LayoutResponse
    /// This provides real-time layout updates through the bidirectional WebSocket connection
    
    /// Compute layout using actual LayoutEngine implementations
    fn compute_layout_with_engine(&self, algorithm: LayoutAlgorithm) -> Vec<NodePosition> {
        // Get graph data from the data source
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        // Convert nodes and edges to the format expected by LayoutEngine
        let viz_nodes: Vec<crate::viz::streaming::data_source::GraphNode> = nodes;
        let viz_edges: Vec<crate::viz::streaming::data_source::GraphEdge> = edges;
        
        // Select the appropriate layout engine
        let positions = match algorithm {
            LayoutAlgorithm::ForceDirected { charge, distance, iterations } => {
                let engine = ForceDirectedLayout { 
                    charge, 
                    distance, 
                    iterations,
                    gravity: 0.1,
                    friction: 0.9,
                    theta: 0.8,
                    alpha: 1.0,
                    alpha_min: 0.001,
                    alpha_decay: 0.99,
                    link_strength: 1.0,
                    charge_strength: 1.0,
                    center_strength: 0.1,
                    collision_radius: 5.0,
                    bounds: None,
                    enable_barnes_hut: true,
                    enable_collision: false,
                    adaptive_cooling: true,
                    max_velocity: Some(100.0),
                    position_constraints: Vec::new(),
                };
                engine.compute_layout(&viz_nodes, &viz_edges)
            }
            LayoutAlgorithm::Circular { radius, start_angle: _ } => {
                let engine = CircularLayout { radius };
                engine.compute_layout(&viz_nodes, &viz_edges)
            }
            _ => {
                // For other algorithms, fall back to data source implementation
                return self.data_source.compute_layout(algorithm);
            }
        };
        
        // Convert positions to NodePosition format
        match positions {
            Ok(pos_vec) => {
                pos_vec.into_iter().map(|(id, pos)| NodePosition {
                    node_id: id,
                    position: pos,
                }).collect()
            }
            Err(e) => {
                eprintln!("⚠️ Layout computation failed: {}, falling back to data source", e);
                self.data_source.compute_layout(algorithm)
            }
        }
    }
}