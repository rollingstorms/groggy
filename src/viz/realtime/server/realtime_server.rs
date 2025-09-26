//! Realtime Server implementation
//!
//! HTTP + WebSocket server for realtime visualization.
//! Serves the UI at `/` with a `/ws` WebSocket endpoint.

use super::WsBridge;
use crate::api::graph::Graph;
use crate::errors::{io_error_to_graph_error, GraphError, GraphResult};
use crate::viz::realtime::accessor::{
    ControlMsg, EngineSnapshot, EngineUpdate, RealtimeVizAccessor,
};
use crate::viz::realtime::engine::ControlCommand;
use crate::viz::realtime::{RealTimeVizConfig, RealTimeVizEngine};
use base64::{engine::general_purpose, Engine as _};
use serde_json;
use sha1::{Digest, Sha1};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Handle for controlling a background realtime server
pub struct RealtimeServerHandle {
    /// Server port
    pub port: u16,
    /// Cancellation token to stop the server
    pub cancel: CancellationToken,
    /// Background thread handle
    pub thread: Option<std::thread::JoinHandle<()>>,
}

impl RealtimeServerHandle {
    /// Stop the server gracefully
    pub fn stop(mut self) {
        // Note: RealtimeServerHandle doesn't have verbosity - this is just cleanup
        self.cancel.cancel();
        if let Some(thread) = self.thread.take() {
            if let Err(e) = thread.join() {
                // Keep minimal error logging for server handle
                eprintln!("❌ Failed to join server thread: {:?}", e);
            } else {
                // Server stopped cleanly - no debug output needed
            }
        }
    }
}

/// Realtime server for Phase 2 transport layer
pub struct RealtimeServer {
    /// Server port
    port: u16,
    /// WebSocket bridge for client communication
    ws_bridge: Arc<WsBridge>,
    /// Realtime accessor for data
    accessor: Option<Arc<dyn RealtimeVizAccessor>>,
    /// Visualization engine for processing updates (shared with tasks)
    engine: Option<RealTimeVizEngine>,
    /// Verbosity level for debug output
    verbose: u8,
}

impl RealtimeServer {
    /// Create new realtime server
    pub fn new(port: u16) -> Self {
        Self {
            port,
            ws_bridge: Arc::new(WsBridge::new()),
            accessor: None,
            engine: None,
            verbose: 0,
        }
    }

    /// Set the data accessor
    pub fn with_accessor(mut self, accessor: Arc<dyn RealtimeVizAccessor>) -> Self {
        self.accessor = Some(accessor);
        self
    }

    /// Set verbosity level
    pub fn with_verbosity(mut self, verbose: u8) -> Self {
        self.verbose = verbose;
        self
    }

    /// Start the server
    pub async fn start(mut self) -> GraphResult<()> {
        if self.verbose >= 3 {
            eprintln!("🚀 DEBUG: Starting RealtimeServer on port {}", self.port);
        }

        // Create cancellation token for graceful shutdown
        let cancel_token = CancellationToken::new();
        // Optional control receiver (present when accessor exists)
        let mut control_rx_opt: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, ControlMsg)>> =
            None;

        // Load initial snapshot and create engine if accessor is available
        if let Some(ref accessor) = self.accessor {
            match accessor.initial_snapshot() {
                Ok(snapshot) => {
                    eprintln!(
                        "📊 DEBUG: Loaded initial snapshot with {} nodes, {} edges",
                        snapshot.node_count(),
                        snapshot.edge_count()
                    );

                    // Create visualization engine with the snapshot data
                    // For now, create a minimal graph - in practice we'd convert the snapshot properly
                    let temp_graph = crate::api::graph::Graph::new(); // TODO: Convert snapshot to proper graph
                    let engine_config = RealTimeVizConfig::default();
                    let mut engine = RealTimeVizEngine::new(temp_graph, engine_config);
                    if let Err(e) = engine.load_snapshot(snapshot.clone()).await {
                        if self.verbose >= 1 {
                            eprintln!("⚠️  INFO: Failed to load snapshot into engine: {}", e);
                        }
                    } else {
                        if self.verbose >= 2 {
                            eprintln!("✅ VERBOSE: Engine initialized with snapshot");
                        }
                    }

                    // Create control message channel between WebSocket bridge and engine
                    let (control_tx, control_rx) = tokio::sync::mpsc::unbounded_channel();
                    control_rx_opt = Some(control_rx);

                    // Set control sender in WebSocket bridge
                    Arc::get_mut(&mut self.ws_bridge)
                        .unwrap()
                        .set_control_sender(control_tx);

                    // Subscribe engine updates to WebSocket bridge
                    let engine_rx = engine.subscribe();
                    let ws_bridge_clone = self.ws_bridge.clone();

                    // Forward engine updates to WebSocket bridge
                    tokio::spawn(async move {
                        let mut engine_rx = engine_rx;
                        while let Ok(update) = engine_rx.recv().await {
                            if let Err(e) = ws_bridge_clone.broadcast_update(update).await {
                                eprintln!("⚠️  DEBUG: WebSocket bridge failed to broadcast engine update: {}", e);
                            }
                        }
                    });

                    self.engine = Some(engine);
                    self.ws_bridge.set_snapshot(snapshot).await;
                }
                Err(e) => {
                    eprintln!("⚠️  DEBUG: Failed to load initial snapshot: {}", e);
                }
            }
        }

        let addr = format!("127.0.0.1:{}", self.port);
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| io_error_to_graph_error(e, "bind_tcp_listener", &addr))?;

        if self.verbose >= 1 {
            eprintln!("✅ INFO: RealtimeServer listening on {}", addr);
        }
        if self.verbose >= 2 {
            eprintln!("🌐 VERBOSE: HTTP endpoint: http://127.0.0.1:{}/", self.port);
            eprintln!(
                "🔌 VERBOSE: WebSocket endpoint: ws://127.0.0.1:{}/ws",
                self.port
            );
        }

        let ws_bridge = Arc::clone(&self.ws_bridge);

        // Main server loop with cancellation support
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    eprintln!("🛑 DEBUG: Server cancellation requested, stopping accept loop");
                    break;
                },
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, addr)) => {
                            if self.verbose >= 2 {
                                eprintln!("🔗 VERBOSE: New connection from {}", addr);
                            }
                            let ws_bridge_clone = Arc::clone(&ws_bridge);
                            let cancel_clone = cancel_token.clone();
                            let port = self.port;
                            tokio::spawn(async move {
                                tokio::select! {
                                    _ = cancel_clone.cancelled() => {
                                        if port > 0 {
                                            eprintln!("🔌 DEBUG: Connection handler cancelled for {}", addr);
                                        }
                                    },
                                    result = Self::handle_connection(stream, addr, ws_bridge_clone, port) => {
                                        if let Err(e) = result {
                                            eprintln!("❌ DEBUG: Connection error for {}: {}", addr, e);
                                        }
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            eprintln!("❌ DEBUG: Accept error: {}", e);
                            // Continue accepting connections
                        }
                    }
                },
                control = async {
                    match control_rx_opt.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => None,
                    }
                }, if control_rx_opt.is_some() => {
                    if let Some((client_id, control_msg)) = control {
                        eprintln!("🎮 DEBUG: Processing control message from client {}: {:?}", client_id, control_msg);
                        if let Some(ref accessor) = self.accessor {
                            if let Err(e) = Self::process_control_message(accessor, control_msg.clone()).await {
                                eprintln!("❌ DEBUG: Accessor failed to process control message: {}", e);
                            }
                        }

                        // Engine-directed controls - apply ALL control messages to engine
                        if let Some(ref mut engine) = self.engine {
                            match control_msg {
                                ControlMsg::PositionDelta { node_id, delta } => {
                                    eprintln!("📍 DEBUG: Forwarding PositionDelta to engine: node_id={}, delta={:?}", node_id, delta);
                                    if let Err(e) = engine.apply(EngineUpdate::PositionDelta { node_id, delta }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply PositionDelta: {}", e);
                                    }
                                }
                                ControlMsg::ChangeEmbedding { method, k, params } => {
                                    eprintln!("🧠 DEBUG: Forwarding ChangeEmbedding to engine: method={}, k={}", method, k);
                                    if let Err(e) = engine.apply(EngineUpdate::EmbeddingChanged { method: method.clone(), dimensions: k }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply EmbeddingChanged: {}", e);
                                    }
                                }
                                ControlMsg::ChangeLayout { algorithm, params } => {
                                    eprintln!("📐 DEBUG: Forwarding ChangeLayout to engine: algorithm={}", algorithm);
                                    if let Err(e) = engine.apply(EngineUpdate::LayoutChanged { algorithm: algorithm.clone(), params: params.clone() }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply LayoutChanged: {}", e);
                                    }
                                }
                                ControlMsg::SetInteractionController { mode } => {
                                    eprintln!("🎮 DEBUG: Switching interaction controller to {}", mode);
                                    if let Err(e) = engine
                                        .handle_control_command(ControlCommand::SetInteractionController { mode: mode.clone() })
                                        .await
                                    {
                                        eprintln!("❌ DEBUG: Engine failed to set controller: {}", e);
                                    }
                                }
                                ControlMsg::Pointer { event } => {
                                    if let Err(e) = engine
                                        .handle_control_command(ControlCommand::Pointer { event: event.clone() })
                                        .await
                                    {
                                        eprintln!("❌ DEBUG: Engine failed to apply pointer event: {}", e);
                                    }
                                }
                                ControlMsg::Wheel { event } => {
                                    if let Err(e) = engine
                                        .handle_control_command(ControlCommand::Wheel { event: event.clone() })
                                        .await
                                    {
                                        eprintln!("❌ DEBUG: Engine failed to apply wheel event: {}", e);
                                    }
                                }
                                ControlMsg::NodeDrag { event } => {
                                    if let Err(e) = engine
                                        .handle_control_command(ControlCommand::NodeDrag { event: event.clone() })
                                        .await
                                    {
                                        eprintln!("❌ DEBUG: Engine failed to apply node drag event: {}", e);
                                    }
                                }
                                ControlMsg::RotateEmbedding { axis_i, axis_j, radians } => {
                                    if let Err(e) = engine
                                        .handle_control_command(ControlCommand::RotateEmbedding { axis_i, axis_j, radians })
                                        .await
                                    {
                                        eprintln!("❌ DEBUG: Engine failed to rotate embedding: {}", e);
                                    }
                                }
                                ControlMsg::SetViewRotation { radians } => {
                                    if let Err(e) = engine
                                        .handle_control_command(ControlCommand::SetViewRotation { radians })
                                        .await
                                    {
                                        eprintln!("❌ DEBUG: Engine failed to set view rotation: {}", e);
                                    }
                                }
                                ControlMsg::SelectNodes(node_ids) => {
                                    eprintln!("🎯 DEBUG: Forwarding SelectNodes to engine: {:?}", node_ids);
                                    if let Err(e) = engine.apply(EngineUpdate::SelectionChanged { selected: node_ids.clone(), deselected: vec![] }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply SelectionChanged: {}", e);
                                    }
                                }
                                ControlMsg::ClearSelection => {
                                    eprintln!("🎯 DEBUG: Forwarding ClearSelection to engine");
                                    if let Err(e) = engine.apply(EngineUpdate::SelectionChanged { selected: vec![], deselected: vec![] }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply ClearSelection: {}", e);
                                    }
                                }
                                _ => {
                                    eprintln!("⚠️  DEBUG: Control message type not implemented for engine: {:?}", control_msg);
                                }
                            }
                        }
                    } else {
                        eprintln!("🎮 DEBUG: Control message channel closed");
                        control_rx_opt = None; // stop selecting on it
                    }
                }
            }
        }

        eprintln!("🛑 DEBUG: RealtimeServer stopped");
        Ok(())
    }

    /// Handle incoming connection
    async fn handle_connection(
        stream: TcpStream,
        addr: SocketAddr,
        ws_bridge: Arc<WsBridge>,
        port: u16,
    ) -> GraphResult<()> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        // First, we need to read the HTTP request to check if it's WebSocket upgrade
        let mut buf_stream = BufReader::new(stream);
        let mut request_line = String::new();

        // Read the first line of the HTTP request
        buf_stream
            .read_line(&mut request_line)
            .await
            .map_err(|e| io_error_to_graph_error(e, "read_http_request_line", "tcp_stream"))?;

        eprintln!("📋 DEBUG: Request: {}", request_line.trim());

        // Parse the request
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(GraphError::IoError {
                operation: "parse_http_request".to_string(),
                path: "tcp_stream".to_string(),
                underlying_error: "Invalid HTTP request - missing method or path".to_string(),
            });
        }

        let method = parts[0];
        let path = parts[1];

        // Read headers until empty line
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            buf_stream
                .read_line(&mut line)
                .await
                .map_err(|e| io_error_to_graph_error(e, "read_http_header", "tcp_stream"))?;

            if line.trim().is_empty() {
                break;
            }
            headers.push(line.trim().to_string());
        }

        match (method, path) {
            ("GET", "/") | ("GET", "/index.html") => {
                eprintln!("🌐 DEBUG: Serving static HTML page to {}", addr);

                // Serve the static HTML file from web/index.html
                match Self::serve_static_file("web/index.html", "text/html").await {
                    Ok(response) => {
                        let mut stream = buf_stream.into_inner();
                        stream.write_all(response.as_bytes()).await.map_err(|e| {
                            io_error_to_graph_error(e, "write_http_response", "tcp_stream")
                        })?;

                        eprintln!("✅ DEBUG: Static HTML page served to {}", addr);
                    }
                    Err(e) => {
                        eprintln!("❌ DEBUG: Failed to serve static HTML: {}", e);

                        let response = format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                            e.len(),
                            e
                        );

                        let mut stream = buf_stream.into_inner();
                        stream.write_all(response.as_bytes()).await.map_err(|err| {
                            io_error_to_graph_error(err, "write_http_response", "tcp_stream")
                        })?;
                    }
                }
            }
            ("GET", "/config") => {
                eprintln!("🔧 DEBUG: Serving config endpoint to {}", addr);

                // Create runtime configuration JSON
                let config = serde_json::json!({
                    "ws_path": "/ws",
                    "port": port,
                    "version": "1.0.0"
                });

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    config.to_string().len(),
                    config
                );

                let mut stream = buf_stream.into_inner();
                stream
                    .write_all(response.as_bytes())
                    .await
                    .map_err(|e| io_error_to_graph_error(e, "write_http_response", "tcp_stream"))?;

                eprintln!("✅ DEBUG: Config served to {}", addr);
            }
            ("GET", path) if path.starts_with("/static/") => {
                eprintln!("📁 DEBUG: Serving static file {} to {}", path, addr);

                // Extract filename from path (remove /static/ prefix)
                let filename = path.strip_prefix("/static/").unwrap_or("");
                let file_path = format!("web/{}", filename);

                // Determine content type
                let content_type = match filename.split('.').last() {
                    Some("css") => "text/css",
                    Some("js") => "application/javascript",
                    Some("html") => "text/html",
                    Some("json") => "application/json",
                    Some("png") => "image/png",
                    Some("jpg") | Some("jpeg") => "image/jpeg",
                    Some("svg") => "image/svg+xml",
                    _ => "text/plain",
                };

                match Self::serve_static_file(&file_path, content_type).await {
                    Ok(response) => {
                        let mut stream = buf_stream.into_inner();
                        stream.write_all(response.as_bytes()).await.map_err(|e| {
                            io_error_to_graph_error(e, "write_http_response", "tcp_stream")
                        })?;

                        eprintln!("✅ DEBUG: Static file {} served to {}", filename, addr);
                    }
                    Err(e) => {
                        eprintln!("❌ DEBUG: Failed to serve static file {}: {}", filename, e);
                        let response = "HTTP/1.1 404 Not Found\r\n\r\nFile not found";
                        let mut stream = buf_stream.into_inner();
                        stream.write_all(response.as_bytes()).await.map_err(|e| {
                            io_error_to_graph_error(e, "write_http_response", "tcp_stream")
                        })?;
                    }
                }
            }
            ("GET", "/ws") => {
                // Check for WebSocket upgrade
                let mut is_websocket = false;
                let mut sec_websocket_key = None;

                for header in &headers {
                    let header_lower = header.to_lowercase();
                    if header_lower.contains("upgrade") && header_lower.contains("websocket") {
                        is_websocket = true;
                    }
                    if header_lower.starts_with("sec-websocket-key:") {
                        sec_websocket_key = header.split(':').nth(1).map(|s| s.trim().to_string());
                    }
                }

                if is_websocket && sec_websocket_key.is_some() {
                    eprintln!("🔌 DEBUG: WebSocket upgrade request from {}", addr);

                    // Perform WebSocket handshake manually
                    let websocket_key = sec_websocket_key.unwrap();
                    let accept_key = Self::generate_websocket_accept(&websocket_key);

                    let response = format!(
                        "HTTP/1.1 101 Switching Protocols\r\n\
                         Upgrade: websocket\r\n\
                         Connection: Upgrade\r\n\
                         Sec-WebSocket-Accept: {}\r\n\r\n",
                        accept_key
                    );

                    let mut stream = buf_stream.into_inner();
                    stream.write_all(response.as_bytes()).await.map_err(|e| {
                        io_error_to_graph_error(e, "write_websocket_handshake", "tcp_stream")
                    })?;

                    // Now handle as WebSocket connection
                    ws_bridge.handle_websocket_stream(stream, addr).await?;
                } else {
                    // Not a WebSocket request
                    let response = "HTTP/1.1 400 Bad Request\r\n\r\nWebSocket upgrade required";
                    let mut stream = buf_stream.into_inner();
                    stream.write_all(response.as_bytes()).await.map_err(|e| {
                        io_error_to_graph_error(e, "write_websocket_error", "tcp_stream")
                    })?;
                }
            }
            _ => {
                eprintln!("❌ DEBUG: 404 for {} {}", method, path);
                let response = "HTTP/1.1 404 Not Found\r\n\r\nNot Found";
                let mut stream = buf_stream.into_inner();
                stream
                    .write_all(response.as_bytes())
                    .await
                    .map_err(|e| io_error_to_graph_error(e, "write_http_response", "tcp_stream"))?;
            }
        }

        Ok(())
    }

    /// Start server with cancellation support
    pub async fn start_with_cancellation(
        mut self,
        cancel_token: CancellationToken,
        ready_tx: oneshot::Sender<u16>,
    ) -> GraphResult<()> {
        eprintln!(
            "🚀 DEBUG: Starting RealtimeServer on port {} with cancellation",
            self.port
        );

        // Optional control receiver (present when accessor exists)
        let mut control_rx_opt: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, ControlMsg)>> =
            None;

        // Load initial snapshot and create engine if accessor is available
        if let Some(ref accessor) = self.accessor {
            match accessor.initial_snapshot() {
                Ok(snapshot) => {
                    eprintln!(
                        "📊 DEBUG: Loaded initial snapshot with {} nodes, {} edges",
                        snapshot.node_count(),
                        snapshot.edge_count()
                    );

                    // Create visualization engine with the snapshot data
                    let temp_graph = crate::api::graph::Graph::new();
                    let engine_config = RealTimeVizConfig::default();
                    let mut engine = RealTimeVizEngine::new(temp_graph, engine_config);

                    // Load snapshot into engine
                    if let Err(e) = engine.load_snapshot(snapshot.clone()).await {
                        if self.verbose >= 1 {
                            eprintln!("⚠️  INFO: Failed to load snapshot into engine: {}", e);
                        }
                    } else {
                        if self.verbose >= 2 {
                            eprintln!("✅ VERBOSE: Engine initialized with snapshot");
                        }
                    }

                    // Create control message channel between WebSocket bridge and engine
                    let (control_tx, control_rx) = tokio::sync::mpsc::unbounded_channel();
                    control_rx_opt = Some(control_rx);

                    // Set control sender in WebSocket bridge
                    Arc::get_mut(&mut self.ws_bridge)
                        .unwrap()
                        .set_control_sender(control_tx);

                    // Subscribe engine updates to WebSocket bridge
                    let engine_rx = engine.subscribe();
                    let ws_bridge_clone = self.ws_bridge.clone();
                    let cancel_clone = cancel_token.clone();

                    // Forward engine updates to WebSocket bridge
                    tokio::spawn(async move {
                        let mut engine_rx = engine_rx;
                        loop {
                            tokio::select! {
                                _ = cancel_clone.cancelled() => {
                                    eprintln!("📡 DEBUG: Engine update task cancelled");
                                    break;
                                },
                                update_result = engine_rx.recv() => {
                                    match update_result {
                                        Ok(update) => {
                                            eprintln!("📡 DEBUG: Forwarding engine update to WebSocket clients: {:?}", update);
                                            if let Err(e) = ws_bridge_clone.broadcast_update(update).await {
                                                eprintln!("⚠️  DEBUG: WebSocket bridge failed to broadcast engine update: {}", e);
                                            }
                                        },
                                        Err(_) => {
                                            eprintln!("📡 DEBUG: Engine update channel closed");
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    });

                    self.engine = Some(engine);
                    self.ws_bridge.set_snapshot(snapshot).await;
                }
                Err(e) => {
                    eprintln!("⚠️  DEBUG: Failed to load initial snapshot: {}", e);
                }
            }
        }

        let addr = format!("127.0.0.1:{}", self.port);
        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| io_error_to_graph_error(e, "bind_tcp_listener", &addr))?;

        let actual_port = listener
            .local_addr()
            .map_err(|e| io_error_to_graph_error(e, "get_local_addr", "tcp_listener"))?
            .port();

        eprintln!("✅ DEBUG: RealtimeServer listening on {}", actual_port);
        eprintln!("🌐 DEBUG: HTTP endpoint: http://127.0.0.1:{}/", actual_port);
        eprintln!(
            "🔌 DEBUG: WebSocket endpoint: ws://127.0.0.1:{}/ws",
            actual_port
        );

        // Notify that the server is ready
        if let Err(_) = ready_tx.send(actual_port) {
            eprintln!("⚠️  DEBUG: Failed to send ready signal");
        }

        let ws_bridge = Arc::clone(&self.ws_bridge);

        // Main server loop with cancellation support
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    eprintln!("🛑 DEBUG: Server cancellation requested, stopping accept loop");
                    break;
                },
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, addr)) => {
                            eprintln!("🔗 DEBUG: New connection from {}", addr);

                            let ws_bridge_clone = Arc::clone(&ws_bridge);
                            let cancel_clone = cancel_token.clone();
                            let port = actual_port;

                            tokio::spawn(async move {
                                tokio::select! {
                                    _ = cancel_clone.cancelled() => {
                                        eprintln!("🔌 DEBUG: Connection handler cancelled for {}", addr);
                                    },
                                    result = Self::handle_connection(stream, addr, ws_bridge_clone, port) => {
                                        if let Err(e) = result {
                                            eprintln!("❌ DEBUG: Connection error for {}: {}", addr, e);
                                        }
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            eprintln!("❌ DEBUG: Accept error: {}", e);
                            // Continue accepting connections despite errors
                        }
                    }
                },
                control = async {
                    match control_rx_opt.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => None,
                    }
                }, if control_rx_opt.is_some() => {
                    if let Some((client_id, control_msg)) = control {
                        eprintln!("🎮 DEBUG: Processing control message from client {}: {:?}", client_id, control_msg);
                        // Accessor reactions
                        if let Some(ref accessor) = self.accessor {
                            if let Err(e) = Self::process_control_message(accessor, control_msg.clone()).await {
                                eprintln!("❌ DEBUG: Accessor failed to process control message: {}", e);
                            }
                        }
                        // Engine-directed controls - apply ALL control messages to engine
                        if let Some(ref mut engine) = self.engine {
                            match control_msg {
                                ControlMsg::PositionDelta { node_id, delta } => {
                                    eprintln!("📍 DEBUG: Forwarding PositionDelta to engine: node_id={}, delta={:?}", node_id, delta);
                                    if let Err(e) = engine.apply(EngineUpdate::PositionDelta { node_id, delta }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply PositionDelta: {}", e);
                                    }
                                }
                                ControlMsg::ChangeEmbedding { method, k, params } => {
                                    eprintln!("🧠 DEBUG: Forwarding ChangeEmbedding to engine: method={}, k={}", method, k);
                                    if let Err(e) = engine.apply(EngineUpdate::EmbeddingChanged { method: method.clone(), dimensions: k }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply EmbeddingChanged: {}", e);
                                    }
                                }
                                ControlMsg::ChangeLayout { algorithm, params } => {
                                    eprintln!("📐 DEBUG: Forwarding ChangeLayout to engine: algorithm={}", algorithm);
                                    if let Err(e) = engine.apply(EngineUpdate::LayoutChanged { algorithm: algorithm.clone(), params: params.clone() }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply LayoutChanged: {}", e);
                                    }
                                }
                                ControlMsg::SelectNodes(node_ids) => {
                                    eprintln!("🎯 DEBUG: Forwarding SelectNodes to engine: {:?}", node_ids);
                                    if let Err(e) = engine.apply(EngineUpdate::SelectionChanged { selected: node_ids.clone(), deselected: vec![] }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply SelectionChanged: {}", e);
                                    }
                                }
                                ControlMsg::ClearSelection => {
                                    eprintln!("🎯 DEBUG: Forwarding ClearSelection to engine");
                                    if let Err(e) = engine.apply(EngineUpdate::SelectionChanged { selected: vec![], deselected: vec![] }).await {
                                        eprintln!("❌ DEBUG: Engine failed to apply ClearSelection: {}", e);
                                    }
                                }
                                _ => {
                                    eprintln!("⚠️  DEBUG: Control message type not implemented for engine: {:?}", control_msg);
                                }
                            }
                        }
                    } else {
                        eprintln!("🎮 DEBUG: Control message channel closed");
                        control_rx_opt = None; // stop selecting on it
                    }
                },
            }
        }

        eprintln!("🛑 DEBUG: RealtimeServer stopped");
        Ok(())
    }

    /// Generate WebSocket accept key from client key
    fn generate_websocket_accept(client_key: &str) -> String {
        const WEBSOCKET_MAGIC: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        let mut hasher = Sha1::new();
        hasher.update(client_key.as_bytes());
        hasher.update(WEBSOCKET_MAGIC.as_bytes());
        let hash = hasher.finalize();
        general_purpose::STANDARD.encode(hash)
    }

    /// Serve static files from the filesystem
    async fn serve_static_file(file_path: &str, content_type: &str) -> Result<String, String> {
        use tokio::fs;

        match fs::read_to_string(file_path).await {
            Ok(content) => {
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
                    content_type,
                    content.len(),
                    content
                );
                Ok(response)
            }
            Err(e) => Err(format!("Failed to read file {}: {}", file_path, e)),
        }
    }

    /// Get WebSocket bridge for integration
    pub fn get_ws_bridge(&self) -> Arc<WsBridge> {
        Arc::clone(&self.ws_bridge)
    }

    /// Get server port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Process control message and trigger position updates
    async fn process_control_message(
        accessor: &Arc<dyn RealtimeVizAccessor>,
        control_msg: ControlMsg,
    ) -> GraphResult<()> {
        // The accessor acknowledges control intent; the engine now owns runtime updates.

        match &control_msg {
            ControlMsg::ChangeEmbedding { method, k, .. } => {
                eprintln!(
                    "🧠 DEBUG: Processing embedding change: {} with {} dimensions",
                    method, k
                );
            }
            ControlMsg::ChangeLayout { algorithm, .. } => {
                eprintln!("📐 DEBUG: Processing layout change: {}", algorithm);
            }
            ControlMsg::ApplyFilter {
                attribute,
                operator,
                value,
            } => {
                eprintln!(
                    "🔍 DEBUG: Processing filter: {} {} {}",
                    attribute, operator, value
                );
            }
            ControlMsg::ClearFilters => {
                eprintln!("🗑️  DEBUG: Processing clear filters");
            }
            ControlMsg::SelectNodes(node_ids) => {
                eprintln!("🎯 DEBUG: Processing node selection: {:?}", node_ids);
            }
            ControlMsg::ClearSelection => {
                eprintln!("🎯 DEBUG: Processing clear selection");
            }
            ControlMsg::PositionDelta { node_id, delta } => {
                eprintln!(
                    "📍 DEBUG: Received PositionDelta control: node_id={}, delta={:?}",
                    node_id, delta
                );
            }
            ControlMsg::SetInteractionController { mode } => {
                eprintln!("🎮 DEBUG: Requested controller mode {}", mode);
            }
            ControlMsg::Pointer { event } => {
                eprintln!("🖱️  DEBUG: Pointer event {:?}", event);
            }
            ControlMsg::Wheel { event } => {
                eprintln!("🖱️  DEBUG: Wheel event {:?}", event);
            }
            ControlMsg::NodeDrag { event } => {
                eprintln!("🖱️  DEBUG: Node drag event {:?}", event);
            }
            ControlMsg::RotateEmbedding {
                axis_i,
                axis_j,
                radians,
            } => {
                eprintln!(
                    "🔁 DEBUG: RotateEmbedding control axis=({}, {}) radians={}",
                    axis_i, axis_j, radians
                );
            }
            ControlMsg::SetViewRotation { radians } => {
                eprintln!("🔁 DEBUG: SetViewRotation control {}", radians);
            }
        }

        // Apply the control message through the accessor
        match accessor.apply_control(control_msg) {
            Ok(()) => {
                eprintln!(
                    "✅ DEBUG: Control message acknowledged by accessor (engine will emit updates)"
                );
            }
            Err(e) => {
                eprintln!("❌ DEBUG: Failed to apply control message: {}", e);
                return Err(e);
            }
        }

        Ok(())
    }
}

/// Create realtime server with accessor (caller must start it)
pub fn create_realtime_server(
    port: u16,
    accessor: Arc<dyn RealtimeVizAccessor>,
) -> GraphResult<RealtimeServer> {
    let server = RealtimeServer::new(port).with_accessor(accessor);

    eprintln!(
        "🚀 DEBUG: Created realtime server on port {} with accessor",
        port
    );

    Ok(server)
}

/// Start realtime server in background with proper cancellation support
pub fn start_realtime_background(
    port: u16,
    accessor: Arc<dyn RealtimeVizAccessor>,
    verbose: u8,
) -> GraphResult<RealtimeServerHandle> {
    let cancel = CancellationToken::new();
    let child_cancel = cancel.clone();
    let (ready_tx, ready_rx) = oneshot::channel();

    if verbose >= 3 {
        eprintln!(
            "🚀 DEBUG: Starting realtime server in background on port {}",
            port
        );
    }

    let thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            // Create and configure server
            let mut server = RealtimeServer::new(port)
                .with_accessor(accessor)
                .with_verbosity(verbose);

            // Start server with cancellation support
            match server.start_with_cancellation(child_cancel, ready_tx).await {
                Ok(()) => {
                    if verbose >= 2 {
                        eprintln!("✅ VERBOSE: Realtime server stopped cleanly");
                    }
                }
                Err(e) => {
                    if verbose >= 1 {
                        eprintln!("❌ INFO: Realtime server error: {}", e);
                    }
                }
            }
        });
    });

    // Wait for server to be ready with timeout to avoid indefinite blocking
    let actual_port = std::thread::spawn(move || match ready_rx.blocking_recv() {
        Ok(port) => port,
        Err(_) => {
            eprintln!(
                "⚠️  DEBUG: Could not get actual port from server, assuming {}",
                port
            );
            port
        }
    })
    .join()
    .map_err(|_| GraphError::IoError {
        operation: "start_realtime_background".to_string(),
        path: "background_thread".to_string(),
        underlying_error: "Failed to join thread waiting for ready signal".to_string(),
    })?;

    Ok(RealtimeServerHandle {
        port: actual_port,
        cancel,
        thread: Some(thread),
    })
}
