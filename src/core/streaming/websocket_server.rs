//! WebSocket server for real-time streaming table updates
//!
//! Provides WebSocket communication for streaming data to browser clients
//! with real-time updates and virtual scrolling support.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

use super::data_source::{DataSource, DataWindow};
use super::virtual_scroller::{VirtualScrollManager, VirtualScrollConfig};

/// WebSocket server for streaming table data
#[derive(Debug)]
pub struct StreamingServer {
    /// Virtual scroll manager
    virtual_scroller: VirtualScrollManager,
    
    /// Data source being served
    data_source: Arc<dyn DataSource>,
    
    /// Active client connections
    active_connections: Arc<RwLock<HashMap<ConnectionId, ClientState>>>,
    
    /// Server configuration
    config: StreamingConfig,
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
        }
    }
    
    /// Start WebSocket server (placeholder for now)
    pub fn start(&self, port: u16) -> StreamingResult<ServerHandle> {
        // TODO: Implement actual WebSocket server with tokio-tungstenite
        println!("Starting streaming server on port {}", port);
        println!("Data source: {} rows, {} columns", 
                self.data_source.total_rows(), 
                self.data_source.total_cols());
        
        Ok(ServerHandle {
            port,
            is_running: true,
        })
    }
    
    /// Handle new client connection
    pub fn handle_client_connection(&self, conn_id: ConnectionId) -> StreamingResult<()> {
        let initial_window = self.virtual_scroller.get_visible_window(self.data_source.as_ref())?;
        
        let client_state = ClientState {
            connection_id: conn_id.clone(),
            current_offset: 0,
            last_update: std::time::SystemTime::now(),
            subscribed_updates: true,
        };
        
        if let Ok(mut connections) = self.active_connections.write() {
            connections.insert(conn_id, client_state);
        }
        
        // Send initial data (placeholder)
        self.send_to_client(conn_id, WSMessage::InitialData {
            window: initial_window,
            total_rows: self.data_source.total_rows(),
        })
    }
    
    /// Handle client scroll request
    pub fn handle_client_scroll(
        &self, 
        conn_id: ConnectionId, 
        offset: usize
    ) -> StreamingResult<()> {
        let window = self.virtual_scroller.get_window_at_offset(
            self.data_source.as_ref(), 
            offset
        )?;
        
        // Update client state
        if let Ok(mut connections) = self.active_connections.write() {
            if let Some(client) = connections.get_mut(&conn_id) {
                client.current_offset = offset;
                client.last_update = std::time::SystemTime::now();
            }
        }
        
        self.send_to_client(conn_id, WSMessage::DataUpdate {
            new_window: window,
            offset,
        })
    }
    
    /// Broadcast update to all clients
    pub fn broadcast_update(&self, update: DataUpdate) -> StreamingResult<()> {
        if let Ok(connections) = self.active_connections.read() {
            for conn_id in connections.keys() {
                self.send_to_client(conn_id.clone(), WSMessage::BroadcastUpdate {
                    update: update.clone(),
                })?;
            }
        }
        
        Ok(())
    }
    
    /// Send message to specific client (placeholder)
    fn send_to_client(&self, _conn_id: ConnectionId, message: WSMessage) -> StreamingResult<()> {
        // TODO: Implement actual WebSocket message sending
        println!("Sending message to client: {:?}", message);
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

/// Server handle
#[derive(Debug)]
pub struct ServerHandle {
    pub port: u16,
    pub is_running: bool,
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
            port: 8080,
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