//! Streaming adapter that wraps the unified core engine
//!
//! Per UNIFIED_VIZ_MIGRATION_PLAN.md: "StreamingAdapter wraps the core 
//! VizEngine for WebSocket streaming while using the unified physics
//! and rendering pipeline."

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::net::IpAddr;
use tokio::sync::oneshot;
use tokio::runtime::Runtime;
use serde::{Serialize, Deserialize};

use crate::errors::GraphResult;
use crate::viz::core::{VizEngine, VizConfig, VizFrame};
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge};
use crate::viz::streaming::websocket_server::{StreamingServer, StreamingConfig};
use super::traits::{
    VizAdapter, StreamingAdapter as StreamingAdapterTrait, AdapterResult, AdapterConfig, 
    StreamingResult, AdapterError
};

/// Streaming adapter that wraps VizEngine for WebSocket server integration
pub struct StreamingAdapter {
    /// Unified core engine (single source of truth)
    core: VizEngine,
    
    /// WebSocket server for streaming
    server: Option<StreamingServer>,
    
    /// Active WebSocket clients
    clients: Arc<RwLock<HashMap<String, WebSocketClient>>>,
    
    /// Streaming configuration
    streaming_config: StreamingConfig,
    
    /// Whether streaming is active
    is_streaming: bool,
    
    /// Current frame for broadcasting
    current_frame: Option<VizFrame>,
    
    /// Runtime for async operations
    runtime: Option<Arc<Runtime>>,
}

/// WebSocket client state
#[derive(Debug, Clone)]
pub struct WebSocketClient {
    /// Client ID
    pub id: String,
    
    /// Client IP address
    pub ip: IpAddr,
    
    /// Connection timestamp
    pub connected_at: std::time::SystemTime,
    
    /// Last frame sent to this client
    pub last_frame_id: Option<String>,
    
    /// Client-specific view state
    pub view_state: ClientViewState,
}

/// Client-specific view state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientViewState {
    /// Camera zoom level
    pub zoom: f64,
    
    /// Camera pan offset
    pub pan: (f64, f64),
    
    /// Selected nodes
    pub selected_nodes: Vec<String>,
    
    /// Currently hovered node
    pub hovered_node: Option<String>,
}

impl Default for ClientViewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: (0.0, 0.0),
            selected_nodes: Vec::new(),
            hovered_node: None,
        }
    }
}

impl StreamingAdapter {
    /// Create new streaming adapter with unified core engine
    pub fn new(config: AdapterConfig, streaming_config: StreamingConfig) -> Self {
        // Convert adapter config to VizEngine config
        let viz_config = VizConfig {
            width: config.width,
            height: config.height,
            physics_enabled: config.physics_enabled,
            continuous_physics: true, // Enable for streaming
            target_fps: 60.0,
            interactions_enabled: config.interactions_enabled,
            auto_fit: config.auto_fit,
            fit_padding: 50.0,
        };
        
        // Create unified core engine
        let core = VizEngine::new(viz_config);
        
        Self {
            core,
            server: None,
            clients: Arc::new(RwLock::new(HashMap::new())),
            streaming_config,
            is_streaming: false,
            current_frame: None,
            runtime: None,
        }
    }
    
    /// Start the WebSocket server and begin streaming
    pub async fn start_server(&mut self) -> GraphResult<String> {
        // Create Tokio runtime if needed
        if self.runtime.is_none() {
            let runtime = Runtime::new()
                .map_err(|e| AdapterError::StreamingFailed(format!("Failed to create runtime: {}", e)))?;
            self.runtime = Some(Arc::new(runtime));
        }
        
        // Generate initial frame from core engine
        self.update_frame()?;
        
        // TODO: Integration with existing StreamingServer
        // For now, return a placeholder URL
        let url = format!("ws://{}:{}", 
                         self.streaming_config.host.unwrap_or("localhost".to_string()),
                         self.streaming_config.port);
        
        self.is_streaming = true;
        
        Ok(url)
    }
    
    /// Stop the WebSocket server
    pub fn stop_server(&mut self) -> GraphResult<()> {
        if let Some(_server) = self.server.take() {
            // TODO: Properly shutdown server
        }
        
        self.is_streaming = false;
        self.clients.write().unwrap().clear();
        
        Ok(())
    }
    
    /// Update current frame from core engine
    fn update_frame(&mut self) -> GraphResult<()> {
        // Use unified core engine to generate frame
        let frame = self.core.update()?;
        
        // Broadcast frame to all connected clients
        if self.is_streaming {
            self.broadcast_frame(&frame)?;
        }
        
        self.current_frame = Some(frame);
        Ok(())
    }
    
    /// Broadcast frame to all connected WebSocket clients
    fn broadcast_frame(&self, frame: &VizFrame) -> GraphResult<()> {
        let clients = self.clients.read().unwrap();
        
        if clients.is_empty() {
            return Ok(());
        }
        
        // Serialize frame to JSON for WebSocket transmission
        let frame_json = frame.to_json()
            .map_err(|e| AdapterError::StreamingFailed(format!("Frame serialization failed: {}", e)))?;
        
        // Create streaming message
        let message = StreamingMessage {
            message_type: MessageType::FrameUpdate,
            data: serde_json::json!({
                "frame": frame_json,
                "timestamp": frame.metadata.timestamp
            }),
        };
        
        // TODO: Send to actual WebSocket connections
        // For now, just log the broadcast
        println!("Broadcasting frame {} to {} clients", 
                frame.metadata.frame_id, 
                clients.len());
        
        Ok(())
    }
    
    /// Handle client interaction events
    pub fn handle_client_interaction(&mut self, client_id: &str, interaction: ClientInteraction) -> GraphResult<()> {
        match interaction {
            ClientInteraction::NodeSelect { node_id } => {
                self.core.select_node(node_id);
            }
            ClientInteraction::NodeHover { node_id } => {
                self.core.set_hover(Some(node_id));
            }
            ClientInteraction::NodeDrag { node_id, position } => {
                self.core.update_drag(&node_id, position);
            }
            ClientInteraction::CameraMove { zoom, pan } => {
                self.core.set_zoom(zoom, crate::viz::streaming::data_source::Position { x: 0.0, y: 0.0 });
                self.core.set_pan(crate::viz::streaming::data_source::Position { x: pan.0, y: pan.1 });
            }
        }
        
        // Update and broadcast new frame
        self.update_frame()?;
        
        Ok(())
    }
    
    /// Add a new WebSocket client
    pub fn add_client(&mut self, client: WebSocketClient) {
        let mut clients = self.clients.write().unwrap();
        clients.insert(client.id.clone(), client);
    }
    
    /// Remove a WebSocket client
    pub fn remove_client(&mut self, client_id: &str) {
        let mut clients = self.clients.write().unwrap();
        clients.remove(client_id);
    }
    
    /// Get current streaming statistics
    pub fn get_stats(&self) -> StreamingStats {
        let clients = self.clients.read().unwrap();
        let frame_info = self.core.get_frame_info();
        
        StreamingStats {
            active_connections: clients.len(),
            total_frames_sent: frame_info.frame_count,
            is_streaming: self.is_streaming,
            physics_running: self.core.is_simulation_running(),
            current_fps: if frame_info.target_fps > 0.0 { frame_info.target_fps } else { 0.0 },
        }
    }
}

impl VizAdapter for StreamingAdapter {
    fn set_data(&mut self, nodes: Vec<VizNode>, edges: Vec<VizEdge>) -> GraphResult<()> {
        // Delegate to unified core engine
        self.core.set_data(nodes, edges)?;
        
        // Update current frame with new data
        self.update_frame()?;
        
        Ok(())
    }
    
    fn render(&mut self) -> GraphResult<AdapterResult> {
        // Update frame from core engine
        self.update_frame()?;
        
        // Return streaming-specific result
        let stats = self.get_stats();
        let result = StreamingResult {
            url: format!("ws://{}:{}", 
                        self.streaming_config.host.as_deref().unwrap_or("localhost"),
                        self.streaming_config.port),
            port: self.streaming_config.port,
            connections: stats.active_connections,
            status: if self.is_streaming { "streaming".to_string() } else { "stopped".to_string() },
        };
        
        Ok(AdapterResult::Streaming(result))
    }
    
    fn get_frame(&mut self) -> GraphResult<VizFrame> {
        // Always get fresh frame from unified core
        self.core.update()
    }
    
    fn is_ready(&self) -> bool {
        // Ready if we have data loaded
        !self.core.get_positions().is_empty()
    }
    
    fn get_config(&self) -> AdapterConfig {
        // Convert VizEngine config back to AdapterConfig
        AdapterConfig {
            width: 800.0, // TODO: Get from actual engine config
            height: 600.0,
            physics_enabled: self.core.is_simulation_running(),
            interactions_enabled: true, // TODO: Get from engine
            auto_fit: true,
            theme: None,
            custom_styles: HashMap::new(),
        }
    }
    
    fn update_config(&mut self, config: AdapterConfig) -> GraphResult<()> {
        // Convert to VizEngine config and update
        let viz_config = VizConfig {
            width: config.width,
            height: config.height,
            physics_enabled: config.physics_enabled,
            continuous_physics: true,
            target_fps: 60.0,
            interactions_enabled: config.interactions_enabled,
            auto_fit: config.auto_fit,
            fit_padding: 50.0,
        };
        
        self.core.set_config(viz_config)?;
        
        Ok(())
    }
}

impl StreamingAdapterTrait for StreamingAdapter {
    fn start_streaming(&mut self) -> GraphResult<()> {
        // Start async server in background
        // TODO: Implement actual async server startup
        self.is_streaming = true;
        Ok(())
    }
    
    fn stop_streaming(&mut self) -> GraphResult<()> {
        self.stop_server()
    }
    
    fn is_streaming(&self) -> bool {
        self.is_streaming
    }
    
    fn connection_count(&self) -> usize {
        self.clients.read().unwrap().len()
    }
}

// === WebSocket Message Types ===

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMessage {
    pub message_type: MessageType,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Frame update with new visualization data
    FrameUpdate,
    /// Client interaction event
    Interaction,
    /// Configuration change
    ConfigUpdate,
    /// Status message
    Status,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientInteraction {
    /// Node selection
    NodeSelect { node_id: String },
    /// Node hover
    NodeHover { node_id: String },
    /// Node drag
    NodeDrag { 
        node_id: String, 
        position: crate::viz::streaming::data_source::Position 
    },
    /// Camera movement
    CameraMove { zoom: f64, pan: (f64, f64) },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    pub active_connections: usize,
    pub total_frames_sent: u64,
    pub is_streaming: bool,
    pub physics_running: bool,
    pub current_fps: f64,
}

/// Builder for creating streaming adapters with configuration
pub struct StreamingAdapterBuilder {
    adapter_config: AdapterConfig,
    streaming_config: StreamingConfig,
}

impl StreamingAdapterBuilder {
    pub fn new() -> Self {
        Self {
            adapter_config: AdapterConfig::default(),
            streaming_config: StreamingConfig::default(),
        }
    }
    
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.adapter_config.width = width;
        self.adapter_config.height = height;
        self
    }
    
    pub fn with_port(mut self, port: u16) -> Self {
        self.streaming_config.port = port;
        self
    }
    
    pub fn with_physics(mut self, enabled: bool) -> Self {
        self.adapter_config.physics_enabled = enabled;
        self
    }
    
    pub fn build(self) -> StreamingAdapter {
        StreamingAdapter::new(self.adapter_config, self.streaming_config)
    }
}

impl Default for StreamingAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}