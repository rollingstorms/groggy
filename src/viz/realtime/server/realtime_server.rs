//! Realtime Server implementation
//!
//! HTTP + WebSocket server for realtime visualization.
//! Serves the UI at `/` with a `/ws` WebSocket endpoint.

use super::{WsBridge, WsMessage};
use crate::errors::{io_error_to_graph_error, GraphError, GraphResult};
use crate::viz::realtime::accessor::{ControlMsg, EngineUpdate, RealtimeVizAccessor};
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

// When building from python-groggy, CARGO_MANIFEST_DIR points to python-groggy/, so we need ../web/
// Embed static web assets at compile time
// CARGO_MANIFEST_DIR points to the groggy root when compiling this crate
const INDEX_HTML: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/web/index.html"));
const APP_JS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/web/app.js"));
const STYLES_CSS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/web/styles.css"));

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
            // Starting RealtimeServer
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
                    if let Err(_e) = engine.load_snapshot(snapshot.clone()).await {
                        if self.verbose >= 1 {
                            // Failed to load snapshot into engine
                        }
                    } else if self.verbose >= 2 {
                        // Engine initialized with snapshot
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
                            if let Err(_e) = ws_bridge_clone.broadcast_update(update).await {
                                // ⚠️  DEBUG: WebSocket bridge failed to broadcast engine update: {}
                            }
                        }
                    });

                    self.engine = Some(engine);
                    self.ws_bridge.set_snapshot(snapshot).await;
                }
                Err(_e) => {
                    // ⚠️  DEBUG: Failed to load initial snapshot: {}
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
                    // 🛑 DEBUG: Server cancellation requested, stopping accept loop
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
                                            // 🔌 DEBUG: Connection handler cancelled for {}
                                        }
                                    },
                                    result = Self::handle_connection(stream, addr, ws_bridge_clone, port) => {
                                        if let Err(_e) = result {
                                            // ❌ DEBUG: Connection error for {}: {}
                                        }
                                    }
                                }
                            });
                        }
                        Err(_e) => {
                            // ❌ DEBUG: Accept error: {}
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
                        // Handle RequestTableData before processing other controls
                        if let ControlMsg::RequestTableData { offset, window_size, data_type, sort_columns } = &control_msg {
                            if let Some(ref accessor) = self.accessor {
                                match accessor.get_table_data(data_type.clone(), *offset, *window_size, sort_columns.clone()) {
                                    Ok(table_window) => {
                                        let table_msg = WsMessage::TableData {
                                            version: 1,
                                            payload: table_window,
                                        };
                                        if let Err(e) = self.ws_bridge.send_to_client(client_id, table_msg).await {
                                            eprintln!("❌ Failed to send table data to client {}: {}", client_id, e);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Failed to fetch table data: {}", e);
                                    }
                                }
                            }
                            // Skip other control processing for table data requests
                            continue;
                        }

                        // 🎮 DEBUG: Processing control message from client {}: {:?}
                        if let Some(ref accessor) = self.accessor {
                            if let Err(_e) = Self::process_control_message(accessor, control_msg.clone()).await {
                                // ❌ DEBUG: Accessor failed to process control message: {}
                            }
                        }

                        // Engine-directed controls - apply ALL control messages to engine
                        if let Some(ref mut engine) = self.engine {
                            match &control_msg {
                                ControlMsg::ChangeEmbedding { method, k, .. } => {
                                    // 🧠 DEBUG: Forwarding ChangeEmbedding to engine: method={}, k={}
                                    if let Err(_e) = engine
                                        .apply(EngineUpdate::EmbeddingChanged {
                                            method: method.clone(),
                                            dimensions: *k,
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply EmbeddingChanged: {}
                                    }
                                }
                                ControlMsg::ChangeLayout { algorithm, params } => {
                                    // 📐 DEBUG: Forwarding ChangeLayout to engine: algorithm={}
                                    if let Err(_e) = engine
                                        .apply(EngineUpdate::LayoutChanged {
                                            algorithm: algorithm.clone(),
                                            params: params.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply LayoutChanged: {}
                                    }
                                }
                                ControlMsg::SetInteractionController { mode } => {
                                    // 🎮 DEBUG: Switching interaction controller to {}
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::SetInteractionController {
                                            mode: mode.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to set controller: {}
                                    }
                                }
                                ControlMsg::Pointer { event } => {
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::Pointer {
                                            event: event.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply pointer event: {}
                                    }
                                }
                                ControlMsg::Wheel { event } => {
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::Wheel {
                                            event: event.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply wheel event: {}
                                    }
                                }
                                ControlMsg::NodeDrag { event } => {
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::NodeDrag {
                                            event: event.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply node drag event: {}
                                    }
                                }
                                ControlMsg::RequestTableData { offset, window_size, data_type, sort_columns } => {
                                    eprintln!("🔍 RequestTableData received: offset={}, window_size={}, data_type={:?}", offset, window_size, data_type);
                                    // Fetch table data from accessor
                                    if let Some(ref accessor) = self.accessor {
                                        eprintln!("  ✓ Accessor available");
                                        match accessor.get_table_data(data_type.clone(), *offset, *window_size, sort_columns.clone()) {
                                            Ok(table_window) => {
                                                eprintln!("  ✓ Table data fetched: {} headers, {} rows", table_window.headers.len(), table_window.rows.len());
                                                // Send table data response to requesting client
                                                let table_msg = WsMessage::TableData {
                                                    version: 1,
                                                    payload: table_window,
                                                };
                                                eprintln!("  → Sending table data to client {}", client_id);
                                                if let Err(e) = self.ws_bridge.send_to_client(client_id, table_msg).await {
                                                    eprintln!("  ❌ Failed to send table data: {}", e);
                                                } else {
                                                    eprintln!("  ✓ Table data sent successfully");
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("  ❌ Failed to fetch table data: {}", e);
                                            }
                                        }
                                    } else {
                                        eprintln!("  ❌ No accessor available!");
                                    }
                                }
                            }
                        }
                    } else {
                        // 🎮 DEBUG: Control message channel closed
                        control_rx_opt = None; // stop selecting on it
                    }
                }
            }
        }

        // 🛑 DEBUG: RealtimeServer stopped
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

        // 📋 DEBUG: Request handling

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
                // 🌐 DEBUG: Serving static HTML page to {}

                let mut stream = buf_stream.into_inner();
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n",
                    INDEX_HTML.len()
                );
                stream
                    .write_all(header.as_bytes())
                    .await
                    .map_err(|e| io_error_to_graph_error(e, "write_http_response", "tcp_stream"))?;
                stream
                    .write_all(INDEX_HTML.as_bytes())
                    .await
                    .map_err(|e| io_error_to_graph_error(e, "write_http_response", "tcp_stream"))?;

                // ✅ DEBUG: Embedded HTML page served to {}
            }
            ("GET", "/config") => {
                // 🔧 DEBUG: Serving config endpoint to {}

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

                // ✅ DEBUG: Config served to {}
            }
            ("GET", path) if path.starts_with("/static/") => {
                // 📁 DEBUG: Serving embedded static file {} to {}

                let filename = path.strip_prefix("/static/").unwrap_or("");
                let (content_opt, content_type) = match filename {
                    "styles.css" => (Some(STYLES_CSS.as_bytes()), "text/css"),
                    "app.js" => (Some(APP_JS.as_bytes()), "application/javascript"),
                    _ => {
                        // ❌ DEBUG: Unknown static asset requested: {}
                        (None, "text/plain")
                    }
                };

                let mut stream = buf_stream.into_inner();
                match content_opt {
                    Some(content) => {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-cache, no-store, must-revalidate\r\n\r\n",
                            content_type,
                            content.len()
                        );
                        stream.write_all(header.as_bytes()).await.map_err(|e| {
                            io_error_to_graph_error(e, "write_http_response", "tcp_stream")
                        })?;
                        stream.write_all(content).await.map_err(|e| {
                            io_error_to_graph_error(e, "write_http_response", "tcp_stream")
                        })?;
                        // ✅ DEBUG: Embedded static file {} served to {}
                    }
                    None => {
                        let body = format!("File {} not found", filename);
                        let header = format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n",
                            body.len()
                        );
                        stream.write_all(header.as_bytes()).await.map_err(|e| {
                            io_error_to_graph_error(e, "write_http_response", "tcp_stream")
                        })?;
                        stream.write_all(body.as_bytes()).await.map_err(|e| {
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

                if let Some(websocket_key) = is_websocket.then_some(()).and(sec_websocket_key) {
                    // 🔌 DEBUG: WebSocket upgrade request from {}

                    // Perform WebSocket handshake manually
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
            ("GET", "/debug/nodes") => {
                // 🔍 DEBUG: Serving debug nodes endpoint

                let snapshot = ws_bridge.get_snapshot().await;
                let nodes_json = match snapshot {
                    Some(snapshot) => {
                        serde_json::to_string_pretty(&snapshot.nodes).unwrap_or_else(|e| {
                            format!("{{\"error\": \"Failed to serialize nodes: {}\"}}", e)
                        })
                    }
                    None => "{\"error\": \"No snapshot available\"}".to_string(),
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    nodes_json.len(),
                    nodes_json
                );

                let mut stream = buf_stream.into_inner();
                stream
                    .write_all(response.as_bytes())
                    .await
                    .map_err(|e| io_error_to_graph_error(e, "write_debug_nodes", "tcp_stream"))?;
            }
            ("GET", "/debug/edges") => {
                // 🔍 DEBUG: Serving debug edges endpoint

                let snapshot = ws_bridge.get_snapshot().await;
                let edges_json = match snapshot {
                    Some(snapshot) => {
                        serde_json::to_string_pretty(&snapshot.edges).unwrap_or_else(|e| {
                            format!("{{\"error\": \"Failed to serialize edges: {}\"}}", e)
                        })
                    }
                    None => "{\"error\": \"No snapshot available\"}".to_string(),
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    edges_json.len(),
                    edges_json
                );

                let mut stream = buf_stream.into_inner();
                stream
                    .write_all(response.as_bytes())
                    .await
                    .map_err(|e| io_error_to_graph_error(e, "write_debug_edges", "tcp_stream"))?;
            }
            ("GET", "/debug/snapshot") => {
                // 🔍 DEBUG: Serving complete debug snapshot

                let snapshot = ws_bridge.get_snapshot().await;
                let snapshot_json = match snapshot {
                    Some(snapshot) => {
                        // Create a simplified view for debugging with string attributes
                        let debug_data = serde_json::json!({
                            "nodes_count": snapshot.nodes.len(),
                            "edges_count": snapshot.edges.len(),
                            "nodes": snapshot.nodes.iter().map(|node| {
                                // Convert AttrValue to simple strings for browser display
                                let simple_attrs: std::collections::HashMap<String, String> = node.attributes.iter().map(|(key, attr_value)| {
                                    let simple_value = match attr_value {
                                        crate::types::AttrValue::Float(f) => f.to_string(),
                                        crate::types::AttrValue::Int(i) => i.to_string(),
                                        crate::types::AttrValue::Text(s) => s.clone(),
                                        crate::types::AttrValue::Bool(b) => b.to_string(),
                                        crate::types::AttrValue::SmallInt(i) => i.to_string(),
                                        crate::types::AttrValue::CompactText(s) => s.as_str().to_string(),
                                        crate::types::AttrValue::Null => "null".to_string(),
                                        crate::types::AttrValue::FloatVec(vec) => {
                                            if vec.len() <= 10 {
                                                format!("[{}]", vec.iter().map(|f| format!("{:.2}", f)).collect::<Vec<_>>().join(", "))
                                            } else {
                                                format!("[{} values: {:.2}..{:.2}]", vec.len(), vec.first().unwrap_or(&0.0), vec.last().unwrap_or(&0.0))
                                            }
                                        },
                                        crate::types::AttrValue::IntVec(vec) => {
                                            if vec.len() <= 10 {
                                                format!("[{}]", vec.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "))
                                            } else {
                                                format!("[{} values: {}..{}]", vec.len(), vec.first().unwrap_or(&0), vec.last().unwrap_or(&0))
                                            }
                                        },
                                        crate::types::AttrValue::TextVec(vec) => {
                                            if vec.len() <= 5 {
                                                format!("[{}]", vec.join(", "))
                                            } else {
                                                format!("[{} items: {}, ...]", vec.len(), vec.first().map(|s| s.as_str()).unwrap_or(""))
                                            }
                                        },
                                        crate::types::AttrValue::BoolVec(vec) => {
                                            let true_count = vec.iter().filter(|&&b| b).count();
                                            format!("[{} bools: {} true, {} false]", vec.len(), true_count, vec.len() - true_count)
                                        },
                                        crate::types::AttrValue::SubgraphRef(id) => format!("Subgraph({})", id),
                                        crate::types::AttrValue::NodeArray(ref arr) => format!("NodeArray({} nodes)", arr.len()),
                                        crate::types::AttrValue::EdgeArray(ref arr) => format!("EdgeArray({} edges)", arr.len()),
                                        crate::types::AttrValue::Bytes(ref bytes) => format!("Bytes({} bytes)", bytes.len()),
                                        crate::types::AttrValue::CompressedText(ref data) => {
                                            let ratio = data.compression_ratio();
                                            format!("CompressedText({} bytes, {:.1}x compression)", data.data.len(), 1.0 / ratio)
                                        },
                                        crate::types::AttrValue::CompressedFloatVec(ref data) => {
                                            let ratio = data.compression_ratio();
                                            format!("CompressedFloatVec({} bytes, {:.1}x compression)", data.data.len(), 1.0 / ratio)
                                        },
                                        crate::types::AttrValue::Json(ref json) => {
                                            match serde_json::to_string(json) {
                                                Ok(json_str) => {
                                                    if json_str.len() <= 100 {
                                                        json_str
                                                    } else {
                                                        format!("JSON({} chars)", json_str.len())
                                                    }
                                                }
                                                Err(_) => "JSON(parse error)".to_string(),
                                            }
                                        }
                                    };
                                    (key.clone(), simple_value)
                                }).collect();

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
                                // Convert AttrValue to simple strings for browser display
                                let simple_attrs: std::collections::HashMap<String, String> = edge.attributes.iter().map(|(key, attr_value)| {
                                    let simple_value = match attr_value {
                                        crate::types::AttrValue::Float(f) => f.to_string(),
                                        crate::types::AttrValue::Int(i) => i.to_string(),
                                        crate::types::AttrValue::Text(s) => s.clone(),
                                        crate::types::AttrValue::Bool(b) => b.to_string(),
                                        crate::types::AttrValue::SmallInt(i) => i.to_string(),
                                        _ => format!("{:?}", attr_value) // Fallback for other types
                                    };
                                    (key.clone(), simple_value)
                                }).collect();

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
                            }).collect::<Vec<_>>()
                        });
                        serde_json::to_string_pretty(&debug_data).unwrap_or_else(|e| {
                            format!("{{\"error\": \"Failed to serialize snapshot: {}\"}}", e)
                        })
                    }
                    None => "{\"error\": \"No snapshot available\"}".to_string(),
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    snapshot_json.len(),
                    snapshot_json
                );

                let mut stream = buf_stream.into_inner();
                stream.write_all(response.as_bytes()).await.map_err(|e| {
                    io_error_to_graph_error(e, "write_debug_snapshot", "tcp_stream")
                })?;
            }
            _ => {
                // ❌ DEBUG: 404 for {} {}
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
        // Optional control receiver (present when accessor exists)
        let mut control_rx_opt: Option<tokio::sync::mpsc::UnboundedReceiver<(usize, ControlMsg)>> =
            None;

        // Load initial snapshot and create engine if accessor is available
        if let Some(ref accessor) = self.accessor {
            match accessor.initial_snapshot() {
                Ok(snapshot) => {
                    // Create visualization engine with the snapshot data
                    let temp_graph = crate::api::graph::Graph::new();
                    let engine_config = RealTimeVizConfig::default();
                    let mut engine = RealTimeVizEngine::new(temp_graph, engine_config);

                    // Load snapshot into engine
                    if let Err(_e) = engine.load_snapshot(snapshot.clone()).await {
                        if self.verbose >= 1 {
                            // Failed to load snapshot into engine
                        }
                    } else if self.verbose >= 2 {
                        // Engine initialized with snapshot
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
                                    // 📡 DEBUG: Engine update task cancelled
                                    break;
                                },
                                update_result = engine_rx.recv() => {
                                    match update_result {
                                        Ok(update) => {
                                            // 📡 DEBUG: Forwarding engine update to WebSocket clients: {:?}
                                            if let Err(_e) = ws_bridge_clone.broadcast_update(update).await {
                                                // ⚠️  DEBUG: WebSocket bridge failed to broadcast engine update: {}
                                            }
                                        },
                                        Err(_) => {
                                            // 📡 DEBUG: Engine update channel closed
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
                Err(_e) => {
                    // ⚠️  DEBUG: Failed to load initial snapshot: {}
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

        // Print server URL for user
        eprintln!("Visualization server: http://127.0.0.1:{}/", actual_port);

        // Notify that the server is ready
        if ready_tx.send(actual_port).is_err() {
            // ⚠️  DEBUG: Failed to send ready signal
        }

        let ws_bridge = Arc::clone(&self.ws_bridge);

        // Main server loop with cancellation support
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    // 🛑 DEBUG: Server cancellation requested, stopping accept loop
                    break;
                },
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((stream, addr)) => {
                            // 🔗 DEBUG: New connection from {}

                            let ws_bridge_clone = Arc::clone(&ws_bridge);
                            let cancel_clone = cancel_token.clone();
                            let port = actual_port;

                            tokio::spawn(async move {
                                tokio::select! {
                                    _ = cancel_clone.cancelled() => {
                                        // 🔌 DEBUG: Connection handler cancelled for {}
                                    },
                                    result = Self::handle_connection(stream, addr, ws_bridge_clone, port) => {
                                        if let Err(_e) = result {
                                            // ❌ DEBUG: Connection error for {}: {}
                                        }
                                    }
                                }
                            });
                        }
                        Err(_e) => {
                            // ❌ DEBUG: Accept error: {}
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
                        // Handle RequestTableData before processing other controls
                        if let ControlMsg::RequestTableData { offset, window_size, data_type, sort_columns } = &control_msg {
                            if let Some(ref accessor) = self.accessor {
                                match accessor.get_table_data(data_type.clone(), *offset, *window_size, sort_columns.clone()) {
                                    Ok(table_window) => {
                                        let table_msg = WsMessage::TableData {
                                            version: 1,
                                            payload: table_window,
                                        };
                                        if let Err(e) = self.ws_bridge.send_to_client(client_id, table_msg).await {
                                            eprintln!("❌ Failed to send table data to client {}: {}", client_id, e);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Failed to fetch table data: {}", e);
                                    }
                                }
                            }
                            // Skip other control processing for table data requests
                            continue;
                        }

                        // 🎮 DEBUG: Processing control message from client {}: {:?}
                        // Accessor reactions
                        if let Some(ref accessor) = self.accessor {
                            if let Err(_e) = Self::process_control_message(accessor, control_msg.clone()).await {
                                // ❌ DEBUG: Accessor failed to process control message: {}
                            }
                        }
                        // Engine-directed controls - apply ALL control messages to engine
                        if let Some(ref mut engine) = self.engine {
                            match &control_msg {
                                ControlMsg::ChangeEmbedding { method, k, .. } => {
                                    // 🧠 DEBUG: Forwarding ChangeEmbedding to engine: method={}, k={}
                                    if let Err(_e) = engine
                                        .apply(EngineUpdate::EmbeddingChanged {
                                            method: method.clone(),
                                            dimensions: *k,
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply EmbeddingChanged: {}
                                    }
                                }
                                ControlMsg::ChangeLayout { algorithm, params } => {
                                    // 📐 DEBUG: Forwarding ChangeLayout to engine: algorithm={}
                                    if let Err(_e) = engine
                                        .apply(EngineUpdate::LayoutChanged {
                                            algorithm: algorithm.clone(),
                                            params: params.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply LayoutChanged: {}
                                    }
                                }
                                ControlMsg::SetInteractionController { mode } => {
                                    // 🎮 DEBUG: Switching interaction controller to {}
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::SetInteractionController {
                                            mode: mode.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to set controller: {}
                                    }
                                }
                                ControlMsg::Pointer { event } => {
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::Pointer {
                                            event: event.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply pointer event: {}
                                    }
                                }
                                ControlMsg::Wheel { event } => {
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::Wheel {
                                            event: event.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply wheel event: {}
                                    }
                                }
                                ControlMsg::NodeDrag { event } => {
                                    if let Err(_e) = engine
                                        .handle_control_command(ControlCommand::NodeDrag {
                                            event: event.clone(),
                                        })
                                        .await
                                    {
                                        // ❌ DEBUG: Engine failed to apply node drag event: {}
                                    }
                                }
                                ControlMsg::RequestTableData { offset, window_size, data_type, sort_columns } => {
                                    eprintln!("🔍 RequestTableData received: offset={}, window_size={}, data_type={:?}", offset, window_size, data_type);
                                    // Fetch table data from accessor
                                    if let Some(ref accessor) = self.accessor {
                                        eprintln!("  ✓ Accessor available");
                                        match accessor.get_table_data(data_type.clone(), *offset, *window_size, sort_columns.clone()) {
                                            Ok(table_window) => {
                                                eprintln!("  ✓ Table data fetched: {} headers, {} rows", table_window.headers.len(), table_window.rows.len());
                                                // Send table data response to requesting client
                                                let table_msg = WsMessage::TableData {
                                                    version: 1,
                                                    payload: table_window,
                                                };
                                                eprintln!("  → Sending table data to client {}", client_id);
                                                if let Err(e) = self.ws_bridge.send_to_client(client_id, table_msg).await {
                                                    eprintln!("  ❌ Failed to send table data: {}", e);
                                                } else {
                                                    eprintln!("  ✓ Table data sent successfully");
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("  ❌ Failed to fetch table data: {}", e);
                                            }
                                        }
                                    } else {
                                        eprintln!("  ❌ No accessor available!");
                                    }
                                }
                            }
                        }
                    } else {
                        // 🎮 DEBUG: Control message channel closed
                        control_rx_opt = None; // stop selecting on it
                    }
                },
            }
        }

        // 🛑 DEBUG: RealtimeServer stopped
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
            ControlMsg::ChangeEmbedding { .. } => {
                // 🧠 DEBUG: Processing embedding change
            }
            ControlMsg::ChangeLayout { .. } => {
                // 📐 DEBUG: Processing layout change: {}
            }
            ControlMsg::SetInteractionController { mode: _ } => {
                // 🎮 DEBUG: Requested controller mode {}
            }
            ControlMsg::Pointer { event: _ } => {
                // 🖱️  DEBUG: Pointer event {:?}
            }
            ControlMsg::Wheel { event: _ } => {
                // 🖱️  DEBUG: Wheel event {:?}
            }
            ControlMsg::NodeDrag { event: _ } => {
                // 🖱️  DEBUG: Node drag event {:?}
            }
            ControlMsg::RequestTableData {
                offset: _,
                window_size: _,
                data_type: _,
                sort_columns: _,
            } => {
                // 📊 DEBUG: Table data request: offset={}, window_size={}, type={:?}
            }
        }

        // Apply the control message through the accessor
        match accessor.apply_control(control_msg) {
            Ok(()) => {
                // ✅ DEBUG: Control message acknowledged by accessor
            }
            Err(e) => {
                // ❌ DEBUG: Failed to apply control message: {}
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
            let server = RealtimeServer::new(port)
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
