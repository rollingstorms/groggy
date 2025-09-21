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
    virtual_scroller: VirtualScrollManager,
    
    /// Data source being served
    pub data_source: Arc<dyn DataSource>,
    
    /// Active client connections
    active_connections: Arc<RwLock<HashMap<ConnectionId, ClientState>>>,
    
    /// Server configuration
    pub config: StreamingConfig,
    
    /// Actual port the server is running on (filled after start)
    actual_port: Option<u16>,
    
    /// Unique run identifier for this server instance
    run_id: String,
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
    async fn handle_node_click_request(&self, node_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        // Find the clicked node
        let clicked_node = match nodes.iter().find(|n| n.id == node_id) {
            Some(node) => node,
            None => return Ok(WSMessage::Error {
                message: format!("Node '{}' not found", node_id),
                error_code: "NODE_NOT_FOUND".to_string(),
            }),
        };
        
        // Find connected neighbors and edges
        let mut neighbors = Vec::new();
        let mut connected_edges = Vec::new();
        
        for edge in &edges {
            if edge.source == node_id || edge.target == node_id {
                connected_edges.push(edge.into());
                
                // Add neighbor node
                let neighbor_id = if edge.source == node_id { &edge.target } else { &edge.source };
                if let Some(neighbor) = nodes.iter().find(|n| n.id == *neighbor_id) {
                    neighbors.push(neighbor.into());
                }
            }
        }
        
        // Compute basic analytics for the node
        let analytics = NodeAnalytics {
            degree: connected_edges.len(),
            in_degree: Some(edges.iter().filter(|e| e.target == node_id).count()),
            out_degree: Some(edges.iter().filter(|e| e.source == node_id).count()),
            centrality_measures: CentralityMeasures {
                betweenness: None, // TODO: Implement centrality calculations
                closeness: None,
                eigenvector: None,
                page_rank: None,
            },
            clustering_coefficient: None, // TODO: Implement clustering coefficient
            community_id: None, // TODO: Implement community detection
        };
        
        Ok(WSMessage::NodeClickResponse {
            node_id: node_id.to_string(),
            node_data: clicked_node.into(),
            neighbors,
            connected_edges,
            analytics,
        })
    }
    
    /// Handle node hover request - create rich tooltip data
    async fn handle_node_hover_request(&self, node_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        let hovered_node = match nodes.iter().find(|n| n.id == node_id) {
            Some(node) => node,
            None => return Ok(WSMessage::Error {
                message: format!("Node '{}' not found", node_id),
                error_code: "NODE_NOT_FOUND".to_string(),
            }),
        };
        
        // Create rich tooltip data
        let degree = edges.iter().filter(|e| e.source == node_id || e.target == node_id).count();
        
        let mut primary_attributes = Vec::new();
        let mut secondary_attributes = Vec::new();
        
        // Add key attributes as primary
        if let Some(label) = &hovered_node.label {
            primary_attributes.push(TooltipAttribute {
                name: "Label".to_string(),
                value: label.clone(),
                display_type: AttributeDisplayType::Text,
            });
        }
        
        // Add other attributes as secondary
        for (key, value) in &hovered_node.attributes {
            secondary_attributes.push(TooltipAttribute {
                name: key.clone(),
                value: format!("{:?}", value), // Basic formatting
                display_type: AttributeDisplayType::Text,
            });
        }
        
        let tooltip_data = NodeTooltipData {
            title: hovered_node.label.clone().unwrap_or_else(|| node_id.to_string()),
            subtitle: Some(format!("Node ID: {}", node_id)),
            primary_attributes,
            secondary_attributes,
            metrics: vec![
                TooltipMetric {
                    name: "Degree".to_string(),
                    value: degree as f64,
                    format: MetricFormat::Integer,
                    context: None,
                }
            ],
        };
        
        Ok(WSMessage::NodeHoverResponse {
            node_id: node_id.to_string(),
            tooltip_data,
        })
    }
    
    /// Handle edge click request - compute detailed edge information
    async fn handle_edge_click_request(&self, edge_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        let clicked_edge = match edges.iter().find(|e| e.id == edge_id) {
            Some(edge) => edge,
            None => return Ok(WSMessage::Error {
                message: format!("Edge '{}' not found", edge_id),
                error_code: "EDGE_NOT_FOUND".to_string(),
            }),
        };
        
        // Find source and target nodes
        let source_node = nodes.iter().find(|n| n.id == clicked_edge.source)
            .ok_or_else(|| StreamingError::Client(format!("Source node '{}' not found", clicked_edge.source)))?;
        let target_node = nodes.iter().find(|n| n.id == clicked_edge.target)
            .ok_or_else(|| StreamingError::Client(format!("Target node '{}' not found", clicked_edge.target)))?;
        
        // TODO: Compute path information (shortest path analysis)
        let path_info = None;
        
        Ok(WSMessage::EdgeClickResponse {
            edge_id: edge_id.to_string(),
            edge_data: clicked_edge.into(),
            source_node: source_node.into(),
            target_node: target_node.into(),
            path_info,
        })
    }
    
    /// Handle edge hover request - create edge tooltip data
    async fn handle_edge_hover_request(&self, edge_id: &str) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        let hovered_edge = match edges.iter().find(|e| e.id == edge_id) {
            Some(edge) => edge,
            None => return Ok(WSMessage::Error {
                message: format!("Edge '{}' not found", edge_id),
                error_code: "EDGE_NOT_FOUND".to_string(),
            }),
        };
        
        // Get node labels for display
        let source_label = nodes.iter().find(|n| n.id == hovered_edge.source)
            .and_then(|n| n.label.as_ref())
            .unwrap_or(&hovered_edge.source);
        let target_label = nodes.iter().find(|n| n.id == hovered_edge.target)
            .and_then(|n| n.label.as_ref())
            .unwrap_or(&hovered_edge.target);
        
        let mut attributes = Vec::new();
        for (key, value) in &hovered_edge.attributes {
            attributes.push(TooltipAttribute {
                name: key.clone(),
                value: format!("{:?}", value),
                display_type: AttributeDisplayType::Text,
            });
        }
        
        let tooltip_data = EdgeTooltipData {
            title: hovered_edge.label.clone().unwrap_or_else(|| format!("{} → {}", source_label, target_label)),
            source_label: source_label.to_string(),
            target_label: target_label.to_string(),
            weight_display: hovered_edge.weight.map(|w| format!("{:.3}", w)),
            attributes,
            path_info: None, // TODO: Add path analysis
        };
        
        Ok(WSMessage::EdgeHoverResponse {
            edge_id: edge_id.to_string(),
            tooltip_data,
        })
    }
    
    /// Handle multi-node selection request - compute bulk analytics
    async fn handle_nodes_selection_request(&self, node_ids: &[String]) -> StreamingResult<WSMessage> {
        if !self.data_source.supports_graph_view() {
            return Ok(WSMessage::Error {
                message: "Graph view not supported".to_string(),
                error_code: "NO_GRAPH_SUPPORT".to_string(),
            });
        }
        
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        
        // Find selected nodes
        let mut selected_nodes = Vec::new();
        for node_id in node_ids {
            if let Some(node) = nodes.iter().find(|n| n.id == *node_id) {
                selected_nodes.push(node.into());
            }
        }
        
        if selected_nodes.is_empty() {
            return Ok(WSMessage::Error {
                message: "No valid nodes found in selection".to_string(),
                error_code: "NO_VALID_NODES".to_string(),
            });
        }
        
        // Count edges between selected nodes
        let edge_count = edges.iter()
            .filter(|e| node_ids.contains(&e.source) && node_ids.contains(&e.target))
            .count();
        
        // Basic analytics
        let total_degree: usize = node_ids.iter()
            .map(|id| edges.iter().filter(|e| e.source == *id || e.target == *id).count())
            .sum();
        let avg_degree = if node_ids.is_empty() { 0.0 } else { total_degree as f64 / node_ids.len() as f64 };
        
        let selection_analytics = SelectionAnalytics {
            node_count: selected_nodes.len(),
            edge_count,
            connected_components: 1, // TODO: Implement component analysis
            avg_degree,
            total_weight: None, // TODO: Sum edge weights
            communities_represented: vec![], // TODO: Community analysis
        };
        
        let bulk_operations = vec![
            "Export Selection".to_string(),
            "Analyze Subgraph".to_string(),
            "Find Shortest Paths".to_string(),
            "Community Detection".to_string(),
        ];
        
        Ok(WSMessage::NodesSelectionResponse {
            selected_nodes,
            selection_analytics,
            bulk_operations,
        })
    }
    
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
        _request_head: &str,
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

        // Generate the interactive HTML page with the actual port
        let html_content = self.generate_interactive_html_with_port(server_port).await?;
        
        // Create HTTP response
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/html; charset=utf-8\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            html_content.len(),
            html_content
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

    /// Generate the HTML content for the interactive streaming table
    async fn generate_interactive_html(&self) -> StreamingResult<String> {
        // Use default port (fallback for compatibility)
        self.generate_interactive_html_with_port(self.config.port).await
    }
    
    /// Generate interactive HTML page with specified port
    async fn generate_interactive_html_with_port(&self, port: u16) -> StreamingResult<String> {
        // Get initial data window for the table
        let initial_window = self.virtual_scroller.get_visible_window(self.data_source.as_ref())?;
        let total_rows = self.data_source.total_rows();
        let total_cols = self.data_source.total_cols();

        // Generate column headers
        let column_names = self.data_source.get_column_names();
        let mut headers = Vec::new();
        for (col_idx, name) in column_names.iter().enumerate() {
            headers.push(format!(
                r#"<th class="col-header" data-col="{}">{}</th>"#,
                col_idx, html_escape(name)
            ));
        }
        let headers_html = headers.join("\n                        ");

        // Generate initial rows - Use AttrValue directly for HTML display
        let mut rows = Vec::new();
        for (row_idx, row) in initial_window.rows.iter().enumerate() {
            let mut cells = Vec::new();
            for (col_idx, cell_data) in row.iter().enumerate() {
                let display_value = attr_value_to_display_text(cell_data);
                cells.push(format!(
                    r#"<td class="cell" data-row="{}" data-col="{}">{}</td>"#,
                    initial_window.start_offset + row_idx,
                    col_idx,
                    html_escape(&display_value)
                ));
            }
            let row_html = format!(
                r#"<tr class="data-row" data-row="{}">{}</tr>"#,
                initial_window.start_offset + row_idx,
                cells.join("")
            );
            rows.push(row_html);
        }
        let rows_html = rows.join("\n                        ");

        // Use the actual port the server is running on for WebSocket connection
        let ws_port = port;
        
        // Get the sleek theme CSS
        let sleek_css = include_str!("../display/themes/sleek.css");

        let html = format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="groggy-run-id" content="{run_id}">
    <title>Groggy Interactive Visualization</title>
    <style>
        /* Sleek theme CSS */
        {sleek_css}
        
        /* Additional streaming-specific styles */
        body {{
            font-family: var(--font);
            margin: 0;
            padding: 20px;
            background: var(--bg);
        }}
        
        .table-container {{
            background: var(--bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            overflow: hidden;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        .table-header {{
            padding: 16px 20px;
            background: var(--hover);
            border-bottom: var(--border) solid var(--line);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .table-title {{
            font-size: 18px;
            font-weight: 600;
            color: var(--fg);
        }}
        
        .table-stats {{
            font-size: 14px;
            color: var(--muted);
        }}
        
        .connection-status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .status-connected {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-disconnected {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .loading {{
            text-align: center;
            padding: 20px;
            color: var(--muted);
        }}
        
        .error {{
            text-align: center;
            padding: 20px;
            color: #dc3545;
            background: #f8d7da;
            margin: 10px;
            border-radius: 4px;
        }}
        
        /* Apply sleek table styling to data table */
        #data-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 14px;
            line-height: 1.4;
            background-color: var(--bg);
            min-width: 100%;
        }}
        
        #data-table th,
        #data-table td {{
            padding: var(--cell-py) var(--cell-px);
            border-right: var(--border) solid var(--line);
            border-bottom: var(--border) solid var(--line);
        }}
        
        #data-table th:last-child,
        #data-table td:last-child {{
            border-right: none;
        }}
        
        #data-table thead th {{
            position: sticky;
            top: 0;
            z-index: 2;
            background: #fafafa;
            font-weight: 600;
            text-align: left;
            color: var(--fg);
        }}
        
        #data-table tbody tr:nth-child(even) td {{
            background: #f9f9f9;
        }}
        
        #data-table tbody tr:hover td {{
            background: var(--row-hover);
        }}
        
        .cell {{
            max-width: 240px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        /* Graph visualization styles */
        .view-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        
        .view-toggle {{
            display: flex;
            background: var(--hover);
            border-radius: 6px;
            padding: 2px;
        }}
        
        .view-toggle-btn {{
            padding: 6px 12px;
            border: none;
            background: transparent;
            color: var(--muted);
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .view-toggle-btn.active {{
            background: var(--bg);
            color: var(--fg);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .view-toggle-btn:hover {{
            background: var(--bg);
            color: var(--fg);
        }}
        
        .viz-container {{
            position: relative;
            height: 70vh;
            min-height: 400px;
        }}
        
        .table-view, .graph-view {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            transition: opacity 0.3s ease;
        }}
        
        .graph-canvas {{
            width: 100%;
            height: 100%;
            border: var(--border) solid var(--line);
            border-radius: var(--radius);
            background: var(--bg);
        }}
        
        .graph-controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 4px;
            background: rgba(255,255,255,0.9);
            border-radius: 6px;
            padding: 4px;
        }}
        
        .graph-btn {{
            padding: 6px;
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 4px;
            font-size: 14px;
            transition: background 0.2s ease;
        }}
        
        .graph-btn:hover {{
            background: rgba(0,0,0,0.1);
        }}
        
        .layout-controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            display: flex;
            gap: 8px;
            align-items: center;
            background: rgba(255,255,255,0.9);
            border-radius: 6px;
            padding: 8px;
        }}
        
        .layout-select {{
            padding: 4px 8px;
            border: 1px solid var(--line);
            border-radius: 4px;
            background: var(--bg);
            color: var(--fg);
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="groggy-display-container" data-theme="sleek">
        <div class="table-container">
            <div class="table-header">
                <div class="table-title">🎨 Interactive Visualization</div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div class="view-controls">
                        <div class="view-toggle">
                            <button id="table-view-btn" class="view-toggle-btn active">📊 Table</button>
                            <button id="graph-view-btn" class="view-toggle-btn">🕸️ Graph</button>
                            <button id="both-view-btn" class="view-toggle-btn">⚡ Both</button>
                        </div>
                    </div>
                    <div class="table-stats">{total_rows} rows × {total_cols} cols</div>
                    <div id="connection-status" class="connection-status status-disconnected">Connecting...</div>
                </div>
            </div>
            
            <div id="error-container"></div>
            
            <div class="viz-container">
                <!-- Table View -->
                <div id="table-view" class="table-view">
                    <table id="data-table" class="groggy-table theme-sleek">
                        <thead>
                            <tr>
                                {headers_html}
                            </tr>
                        </thead>
                        <tbody id="table-body">
                            {rows_html}
                        </tbody>
                    </table>
                    
                    <div id="loading" class="loading" style="display: none;">
                        Loading more data...
                    </div>
                </div>
                
                <!-- Graph View -->
                <div id="graph-view" class="graph-view" style="opacity: 0;">
                    <canvas id="graph-canvas" class="graph-canvas"></canvas>
                    
                    <!-- Graph Controls -->
                    <div class="graph-controls">
                        <button id="zoom-in-btn" class="graph-btn" title="Zoom In">🔍+</button>
                        <button id="zoom-out-btn" class="graph-btn" title="Zoom Out">🔍-</button>
                        <button id="center-btn" class="graph-btn" title="Center Graph">🎯</button>
                        <button id="screenshot-btn" class="graph-btn" title="Screenshot">📷</button>
                    </div>
                    
                    <!-- Layout Controls -->
                    <div class="layout-controls">
                        <label for="layout-select">Layout:</label>
                        <select id="layout-select" class="layout-select">
                            <option value="force-directed">Force</option>
                            <option value="circular">Circular</option>
                            <option value="grid">Grid</option>
                            <option value="hierarchical">Tree</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentOffset = 0;
        let totalRows = {total_rows};
        let isConnected = false;
        
        // Graph visualization variables
        let graphData = {{ nodes: [], edges: [], metadata: {{}} }};
        let currentView = 'table';
        let canvas = null;
        let ctx = null;
        let scale = 1.0;
        let translateX = 0;
        let translateY = 0;
        let selectedNode = null;
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;
        
        function updateConnectionStatus(connected) {{
            const statusEl = document.getElementById('connection-status');
            isConnected = connected;
            
            if (connected) {{
                statusEl.textContent = 'Connected';
                statusEl.className = 'connection-status status-connected';
            }} else {{
                statusEl.textContent = 'Disconnected';
                statusEl.className = 'connection-status status-disconnected';
            }}
        }}
        
        function showError(message) {{
            const errorContainer = document.getElementById('error-container');
            errorContainer.innerHTML = `<div class="error">Error: ${{message}}</div>`;
        }}
        
        // Graph rendering functions
        function initGraphCanvas() {{
            canvas = document.getElementById('graph-canvas');
            if (!canvas) return;
            
            ctx = canvas.getContext('2d');
            
            // Set canvas size
            const container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;
            
            // Add event listeners
            canvas.addEventListener('mousedown', onCanvasMouseDown);
            canvas.addEventListener('mousemove', onCanvasMouseMove);
            canvas.addEventListener('mouseup', onCanvasMouseUp);
            canvas.addEventListener('wheel', onCanvasWheel);
            
            // Initial render
            renderGraph();
        }}
        
        function renderGraph() {{
            if (!ctx || !canvas) return;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Save transform state
            ctx.save();
            
            // Apply zoom and pan
            ctx.translate(canvas.width / 2 + translateX, canvas.height / 2 + translateY);
            ctx.scale(scale, scale);
            
            // Render edges first (behind nodes)
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            graphData.edges.forEach(edge => {{
                const sourceNode = graphData.nodes.find(n => n.id === edge.source);
                const targetNode = graphData.nodes.find(n => n.id === edge.target);
                
                if (sourceNode && targetNode) {{
                    ctx.beginPath();
                    ctx.moveTo(sourceNode.x || 0, sourceNode.y || 0);
                    ctx.lineTo(targetNode.x || 0, targetNode.y || 0);
                    ctx.stroke();
                }}
            }});
            
            // Render nodes
            graphData.nodes.forEach(node => {{
                const x = node.x || 0;
                const y = node.y || 0;
                const radius = 8;
                
                // Node circle
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fillStyle = selectedNode === node.id ? '#ff6b6b' : '#4dabf7';
                ctx.fill();
                ctx.strokeStyle = '#333';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Node label
                if (node.label || node.id) {{
                    ctx.fillStyle = '#333';
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(node.label || node.id, x, y + radius + 12);
                }}
            }});
            
            // Restore transform state
            ctx.restore();
        }}
        
        function updateGraphData(data) {{
            graphData = data;
            
            // If no positions, generate simple circular layout
            if (graphData.nodes.length > 0 && !graphData.nodes[0].x) {{
                const centerX = 0;
                const centerY = 0;
                const radius = 100;
                const angleStep = (2 * Math.PI) / graphData.nodes.length;
                
                graphData.nodes.forEach((node, i) => {{
                    const angle = i * angleStep;
                    node.x = centerX + radius * Math.cos(angle);
                    node.y = centerY + radius * Math.sin(angle);
                }});
            }}
            
            renderGraph();
        }}
        
        function onCanvasMouseDown(e) {{
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            lastMouseX = mouseX;
            lastMouseY = mouseY;
            isDragging = true;
            
            // Check if clicking on a node
            const worldX = (mouseX - canvas.width / 2 - translateX) / scale;
            const worldY = (mouseY - canvas.height / 2 - translateY) / scale;
            
            selectedNode = null;
            graphData.nodes.forEach(node => {{
                const dx = worldX - (node.x || 0);
                const dy = worldY - (node.y || 0);
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < 10) {{
                    selectedNode = node.id;
                }}
            }});
            
            renderGraph();
        }}
        
        function onCanvasMouseMove(e) {{
            if (!isDragging) return;

            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const deltaX = mouseX - lastMouseX;
            const deltaY = mouseY - lastMouseY;

            if (selectedNode !== null) {{
                // Drag the selected node
                const node = graphData.nodes.find(n => n.id === selectedNode);
                if (node) {{
                    // Convert screen space delta to world space
                    node.x = (node.x || 0) + deltaX / scale;
                    node.y = (node.y || 0) + deltaY / scale;
                }}
            }} else {{
                // Drag the viewport
                translateX += deltaX;
                translateY += deltaY;
            }}

            lastMouseX = mouseX;
            lastMouseY = mouseY;

            renderGraph();
        }}
        
        function onCanvasMouseUp(e) {{
            isDragging = false;
            selectedNode = null; // Clear selection after dragging
        }}
        
        function onCanvasWheel(e) {{
            e.preventDefault();
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(0.1, Math.min(5.0, scale * zoomFactor));
            
            renderGraph();
        }}
        
        function requestGraphData() {{
            if (!ws || !isConnected) return;
            
            const message = {{
                type: 'GraphDataRequest',
                layout_algorithm: document.getElementById('layout-select').value,
                theme: 'sleek'
            }};
            
            ws.send(JSON.stringify(message));
        }}
        
        function connectWebSocket() {{
            const wsUrl = `ws://127.0.0.1:{ws_port}`;
            console.log('Connecting to WebSocket:', wsUrl);
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function(event) {{
                console.log('WebSocket connected');
                updateConnectionStatus(true);
                document.getElementById('error-container').innerHTML = '';
            }};
            
            ws.onmessage = function(event) {{
                try {{
                    const message = JSON.parse(event.data);
                    console.log('Received message:', message);
                    
                    if (message.type === 'InitialData') {{
                        updateTable(message.window, message.total_rows);
                        totalRows = message.total_rows;
                    }} else if (message.type === 'DataUpdate') {{
                        updateTable(message.new_window, totalRows);
                        currentOffset = message.offset;
                    }} else if (message.type === 'GraphDataResponse') {{
                        // Handle graph data from WebSocket
                        const graphDataFormatted = {{
                            nodes: message.nodes || [],
                            edges: message.edges || [],
                            metadata: message.metadata || {{}}
                        }};
                        
                        // Apply layout positions if provided
                        if (message.layout_positions) {{
                            message.layout_positions.forEach(pos => {{
                                const node = graphDataFormatted.nodes.find(n => n.id === pos.node_id);
                                if (node) {{
                                    node.x = pos.position.x;
                                    node.y = pos.position.y;
                                }}
                            }});
                        }}
                        
                        updateGraphData(graphDataFormatted);
                        console.log('Updated graph with', graphDataFormatted.nodes.length, 'nodes and', graphDataFormatted.edges.length, 'edges');
                    }} else if (message.type === 'LayoutResponse') {{
                        // Handle layout update from WebSocket
                        if (message.positions) {{
                            message.positions.forEach(pos => {{
                                const node = graphData.nodes.find(n => n.id === pos.node_id);
                                if (node) {{
                                    node.x = pos.position.x;
                                    node.y = pos.position.y;
                                }}
                            }});
                            renderGraph();
                        }}
                    }}
                }} catch (e) {{
                    console.error('Failed to parse WebSocket message:', e);
                    showError('Failed to parse server message');
                }}
            }};
            
            ws.onclose = function(event) {{
                console.log('WebSocket disconnected');
                updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {{
                    if (!isConnected) {{
                        connectWebSocket();
                    }}
                }}, 3000);
            }};
            
            ws.onerror = function(error) {{
                console.error('WebSocket error:', error);
                showError('WebSocket connection failed');
                updateConnectionStatus(false);
            }};
        }}
        
        function updateTable(dataWindow, total) {{
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            
            dataWindow.rows.forEach((row, rowIdx) => {{
                const tr = document.createElement('tr');
                tr.className = 'data-row';
                tr.dataset.row = dataWindow.offset + rowIdx;
                
                row.forEach((cell, colIdx) => {{
                    const td = document.createElement('td');
                    td.className = 'cell';
                    td.dataset.row = dataWindow.offset + rowIdx;
                    td.dataset.col = colIdx;
                    td.textContent = cell;
                    tr.appendChild(td);
                }});
                
                tbody.appendChild(tr);
            }});
            
            // Update stats
            const statsEl = document.querySelector('.table-stats');
            statsEl.textContent = `${{total}} rows × {total_cols} cols`;
        }}
        
        function requestScroll(offset, windowSize = 50) {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                const message = {{
                    type: 'ScrollRequest',
                    offset: offset,
                    window_size: windowSize
                }};
                ws.send(JSON.stringify(message));
            }}
        }}
        
        // Virtual scrolling support
        const tableContainer = document.querySelector('.table-container');
        let scrollTimeout = null;
        
        tableContainer.addEventListener('scroll', function() {{
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {{
                const scrollTop = tableContainer.scrollTop;
                const rowHeight = 45; // Approximate row height
                const newOffset = Math.floor(scrollTop / rowHeight);
                
                if (Math.abs(newOffset - currentOffset) > 10) {{
                    requestScroll(newOffset);
                }}
            }}, 150);
        }});
        
        // View toggle functionality
        function switchView(view) {{
            currentView = view;
            const tableView = document.getElementById('table-view');
            const graphView = document.getElementById('graph-view');
            
            // Update button states
            document.querySelectorAll('.view-toggle-btn').forEach(btn => btn.classList.remove('active'));
            
            if (view === 'table') {{
                tableView.style.opacity = '1';
                graphView.style.opacity = '0';
                document.getElementById('table-view-btn').classList.add('active');
            }} else if (view === 'graph') {{
                tableView.style.opacity = '0';
                graphView.style.opacity = '1';
                document.getElementById('graph-view-btn').classList.add('active');
                // Initialize canvas when switching to graph view
                setTimeout(initGraphCanvas, 100);
                // Request graph data if we don't have it
                if (graphData.nodes.length === 0) {{
                    requestGraphData();
                }}
            }} else if (view === 'both') {{
                tableView.style.opacity = '0.7';
                graphView.style.opacity = '0.7';
                document.getElementById('both-view-btn').classList.add('active');
                setTimeout(initGraphCanvas, 100);
                if (graphData.nodes.length === 0) {{
                    requestGraphData();
                }}
            }}
        }}
        
        // Graph controls functionality
        function setupGraphControls() {{
            // Zoom controls
            document.getElementById('zoom-in-btn')?.addEventListener('click', () => {{
                scale = Math.min(5.0, scale * 1.2);
                renderGraph();
            }});
            
            document.getElementById('zoom-out-btn')?.addEventListener('click', () => {{
                scale = Math.max(0.1, scale * 0.8);
                renderGraph();
            }});
            
            document.getElementById('center-btn')?.addEventListener('click', () => {{
                translateX = 0;
                translateY = 0;
                scale = 1.0;
                renderGraph();
            }});
            
            // Layout change
            document.getElementById('layout-select')?.addEventListener('change', (e) => {{
                requestGraphData();
            }});
            
            // View toggle buttons
            document.getElementById('table-view-btn')?.addEventListener('click', () => switchView('table'));
            document.getElementById('graph-view-btn')?.addEventListener('click', () => switchView('graph'));
            document.getElementById('both-view-btn')?.addEventListener('click', () => switchView('both'));
        }}
        
        // Initialize everything when page loads
        window.addEventListener('load', () => {{
            connectWebSocket();
            setupGraphControls();
            // Start with table view
            switchView('table');
        }});
    </script>
</body>
</html>"#,
            total_rows = total_rows,
            total_cols = total_cols,
            headers_html = headers_html,
            rows_html = rows_html,
            ws_port = ws_port,
            run_id = self.run_id
        );

        Ok(html)
    }
    
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
        let attributes: std::collections::HashMap<String, serde_json::Value> = node.attributes
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(attr_value_to_display_text(v))))
            .collect();

        Self {
            id: node.id.clone(),
            label: node.label.clone(),
            attributes,
            position: node.position.as_ref().map(|p| PositionData { x: p.x, y: p.y }),
        }
    }
}

impl From<&GraphEdge> for GraphEdgeData {
    fn from(edge: &GraphEdge) -> Self {
        let attributes: std::collections::HashMap<String, serde_json::Value> = edge.attributes
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(attr_value_to_display_text(v))))
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
            position: PositionData { x: pos.position.x, y: pos.position.y },
        }
    }
}

/// Convert AttrValue to display text for HTML rendering
fn attr_value_to_display_text(attr: &crate::types::AttrValue) -> String {
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
    }
}

/// Convert AttrValue to JSON for WebSocket transmission
fn attr_value_to_json(attr: &crate::types::AttrValue) -> serde_json::Value {
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
            let vec: Vec<serde_json::Value> = v.iter()
                .map(|&f| serde_json::Number::from_f64(f as f64)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null))
                .collect();
            serde_json::Value::Array(vec)
        },
        AttrValue::Bytes(b) => serde_json::Value::String(format!("[{} bytes]", b.len())),
        AttrValue::CompressedText(_) => serde_json::Value::String("[Compressed Text]".to_string()),
        AttrValue::CompressedFloatVec(_) => serde_json::Value::String("[Compressed FloatVec]".to_string()),
        AttrValue::SubgraphRef(id) => serde_json::Value::String(format!("[Subgraph:{}]", id)),
        AttrValue::NodeArray(nodes) => serde_json::Value::String(format!("[{} nodes]", nodes.len())),
        AttrValue::EdgeArray(edges) => serde_json::Value::String(format!("[{} edges]", edges.len())),
        AttrValue::Null => serde_json::Value::Null,
    }
}

/// Convert DataWindow to clean JSON for WebSocket transmission  
fn data_window_to_json(window: &DataWindow) -> JsonDataWindow {
    
    let clean_rows: Vec<Vec<WireCell>> = window.rows.iter()
        .map(|row| {
            row.iter().map(|attr| {
                attr_to_wire(attr)
            }).collect()
        })
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
    N(i64),        // Integers 
    F(f64),        // Floats
    S(String),     // Strings
    B(bool),       // Booleans
    A(Vec<WireCell>), // Arrays
    Null,          // Null values
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
    ScrollRequest {
        offset: usize,
        window_size: usize,
    },
    
    /// Client requests theme change
    ThemeChange {
        theme: String,
    },
    
    /// Broadcast update to all clients
    BroadcastUpdate {
        update: DataUpdate,
    },
    
    /// Error message
    Error {
        message: String,
        error_code: String,
    },
    
    /// Server status/ping
    Status {
        stats: ServerStats,
    },
    
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
    MetadataResponse {
        metadata: GraphMetadataData,
    },
    
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
    NodeHoverRequest {
        node_id: String,
        position: Position,
    },
    
    /// Server responds with rich tooltip data
    NodeHoverResponse {
        node_id: String,
        tooltip_data: NodeTooltipData,
    },
    
    /// Client stopped hovering over node
    NodeHoverEnd {
        node_id: String,
    },
    
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
    EdgeHoverRequest {
        edge_id: String,
        position: Position,
    },
    
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
    cancel: CancellationToken,
    thread: Option<StdJoinHandle<()>>,
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
            port: 0,  // Use port 0 for automatic port assignment to avoid conflicts
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

/// HTML escape utility function
fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

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
    Click,           // Single click
    ShiftClick,      // Shift+click to extend selection
    CtrlClick,       // Ctrl+click to toggle
    DragSelect,      // Drag to select multiple
    LassoSelect,     // Lasso selection
    BoxSelect,       // Box selection
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