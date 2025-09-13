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
use futures_util;

use super::data_source::{DataSource, DataWindow};
use super::virtual_scroller::{VirtualScrollManager, VirtualScrollConfig};

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
}

impl StreamingServer {
    /// Create new streaming server
    pub fn new(
        data_source: Arc<dyn DataSource>, 
        config: StreamingConfig
    ) -> Self {
        let virtual_scroller = VirtualScrollManager::new(config.scroll_config.clone());
        
        Self {
            virtual_scroller,
            data_source,
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            config,
            actual_port: None,
        }
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
                            // Spawn connection handler but keep the accept loop running
                            let server_clone = self.clone();
                            tokio::spawn(async move {
                                if let Err(e) = server_clone.handle_connection_with_port(stream, addr, server_port).await {
                                    eprintln!("🔻 connection {} ended with error: {}", addr, e);
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
        let initial_msg = WSMessage::InitialData {
            window: initial_window,
            total_rows: self.data_source.total_rows(),
        };

        let initial_json = serde_json::to_string(&initial_msg)
            .map_err(|e| StreamingError::WebSocket(format!("JSON serialization error: {}", e)))?;
        
        ws_sender.send(Message::Text(initial_json)).await
            .map_err(|e| StreamingError::WebSocket(format!("Failed to send initial data: {}", e)))?;

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
                    new_window: window,
                    offset,
                };

                let response_json = serde_json::to_string(&response)
                    .map_err(|e| StreamingError::WebSocket(format!("JSON serialization error: {}", e)))?;

                ws_sender.send(Message::Text(response_json)).await
                    .map_err(|e| StreamingError::WebSocket(format!("Failed to send data update: {}", e)))?;
            }
            WSMessage::ThemeChange { theme } => {
                println!("🎨 Client {} changed theme to: {}", conn_id, theme);
                // Theme changes can be acknowledged or ignored for now
            }
            _ => {
                eprintln!("⚠️ Unhandled message type from client {}", conn_id);
            }
        }

        Ok(())
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

        // Generate initial rows
        let mut rows = Vec::new();
        for (row_idx, row) in initial_window.rows.iter().enumerate() {
            let mut cells = Vec::new();
            for (col_idx, cell_data) in row.iter().enumerate() {
                cells.push(format!(
                    r#"<td class="cell" data-row="{}" data-col="{}">{}</td>"#,
                    initial_window.start_offset + row_idx,
                    col_idx,
                    html_escape(&format!("{}", cell_data))
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
    <title>Groggy Interactive Table</title>
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
    </style>
</head>
<body>
    <div class="groggy-display-container" data-theme="sleek">
        <div class="table-container">
            <div class="table-header">
            <div class="table-title">🐸 Interactive Streaming Table</div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <div class="table-stats">{total_rows} rows × {total_cols} cols</div>
                <div id="connection-status" class="connection-status status-disconnected">Connecting...</div>
            </div>
        </div>
        
        <div id="error-container"></div>
        
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
    </div>

    <script>
        let ws = null;
        let currentOffset = 0;
        let totalRows = {total_rows};
        let isConnected = false;
        
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
        
        // Connect when page loads
        window.addEventListener('load', connectWebSocket);
    </script>
</body>
</html>"#,
            total_rows = total_rows,
            total_cols = total_cols,
            headers_html = headers_html,
            rows_html = rows_html,
            ws_port = ws_port
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
}

/// WebSocket message protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WSMessage {
    /// Initial data sent when client connects
    InitialData {
        window: DataWindow,
        total_rows: usize,
    },
    
    /// Data update in response to scroll
    DataUpdate {
        new_window: DataWindow,
        offset: usize,
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
}

/// Data update broadcast to clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataUpdate {
    pub update_type: UpdateType,
    pub affected_rows: Vec<usize>,
    pub new_data: Option<DataWindow>,
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