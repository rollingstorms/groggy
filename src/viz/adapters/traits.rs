//! Common traits and interfaces for visualization adapters
//!
//! Defines the common interface that all adapters implement to ensure
//! consistent behavior across different output formats.

use crate::errors::GraphResult;
use crate::viz::core::frame::VizFrame;
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge};

/// Result types for different adapters
#[derive(Debug, Clone)]
pub enum AdapterResult {
    /// Streaming server result
    Streaming(StreamingResult),
    /// Jupyter widget result  
    Jupyter(JupyterResult),
    /// File export result
    File(FileResult),
}

/// Streaming adapter result
#[derive(Debug, Clone)]
pub struct StreamingResult {
    /// Server URL
    pub url: String,
    /// Port number
    pub port: u16,
    /// Number of active connections
    pub connections: usize,
    /// Server status
    pub status: String,
}

/// Jupyter widget result
#[derive(Debug, Clone)]
pub struct JupyterResult {
    /// Widget HTML content
    pub html: String,
    /// Widget JavaScript code
    pub javascript: String,
    /// Widget metadata
    pub metadata: JupyterMetadata,
}

#[derive(Debug, Clone)]
pub struct JupyterMetadata {
    /// Widget ID
    pub widget_id: String,
    /// Model data size
    pub data_size_bytes: usize,
    /// Render time
    pub render_time_ms: f64,
}

/// File export result
#[derive(Debug, Clone)]
pub struct FileResult {
    /// Output file path
    pub file_path: String,
    /// File size in bytes
    pub file_size_bytes: usize,
    /// Export format
    pub format: String,
    /// Export time
    pub export_time_ms: f64,
}

/// Common adapter trait that all visualization adapters implement
pub trait VizAdapter {
    /// Set the graph data for visualization
    fn set_data(&mut self, nodes: Vec<VizNode>, edges: Vec<VizEdge>) -> GraphResult<()>;
    
    /// Render the visualization using the adapter's specific format
    fn render(&mut self) -> GraphResult<AdapterResult>;
    
    /// Get the current frame from the underlying engine
    fn get_frame(&mut self) -> GraphResult<VizFrame>;
    
    /// Check if the adapter is ready for rendering
    fn is_ready(&self) -> bool;
    
    /// Get adapter-specific configuration
    fn get_config(&self) -> AdapterConfig;
    
    /// Update adapter configuration
    fn update_config(&mut self, config: AdapterConfig) -> GraphResult<()>;
}

/// Configuration that can be applied to any adapter
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Canvas dimensions
    pub width: f64,
    pub height: f64,
    
    /// Whether physics simulation is enabled
    pub physics_enabled: bool,
    
    /// Whether interactions are enabled
    pub interactions_enabled: bool,
    
    /// Auto-fit graph to canvas
    pub auto_fit: bool,
    
    /// Theme name
    pub theme: Option<String>,
    
    /// Custom styling
    pub custom_styles: std::collections::HashMap<String, String>,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            physics_enabled: true,
            interactions_enabled: true,
            auto_fit: true,
            theme: None,
            custom_styles: std::collections::HashMap::new(),
        }
    }
}

/// Helper trait for adapters that support real-time updates
pub trait StreamingAdapter: VizAdapter {
    /// Start streaming updates
    fn start_streaming(&mut self) -> GraphResult<()>;
    
    /// Stop streaming updates
    fn stop_streaming(&mut self) -> GraphResult<()>;
    
    /// Check if currently streaming
    fn is_streaming(&self) -> bool;
    
    /// Get number of active connections
    fn connection_count(&self) -> usize;
}

/// Helper trait for adapters that support static export
pub trait ExportAdapter: VizAdapter {
    /// Export to file with specified format
    fn export_to_file(&mut self, path: &str, format: ExportFormat) -> GraphResult<FileResult>;
    
    /// Get available export formats
    fn supported_formats(&self) -> Vec<ExportFormat>;
}

/// Export format options
#[derive(Debug, Clone, PartialEq)]
pub enum ExportFormat {
    /// SVG vector graphics
    SVG,
    /// HTML with embedded SVG
    HTML,
    /// PNG raster image (requires external renderer)
    PNG { dpi: u32 },
    /// PDF document (requires external renderer)
    PDF,
    /// JSON data export
    JSON,
}

impl ExportFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::SVG => "svg",
            ExportFormat::HTML => "html",
            ExportFormat::PNG { .. } => "png",
            ExportFormat::PDF => "pdf",
            ExportFormat::JSON => "json",
        }
    }
    
    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::SVG => "image/svg+xml",
            ExportFormat::HTML => "text/html",
            ExportFormat::PNG { .. } => "image/png",
            ExportFormat::PDF => "application/pdf",
            ExportFormat::JSON => "application/json",
        }
    }
}

/// Error types specific to adapters
#[derive(Debug, Clone)]
pub enum AdapterError {
    /// Engine not initialized
    EngineNotInitialized,
    /// Data not set
    DataNotSet,
    /// Rendering failed
    RenderingFailed(String),
    /// Export failed
    ExportFailed(String),
    /// Streaming failed
    StreamingFailed(String),
    /// Configuration error
    ConfigurationError(String),
}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterError::EngineNotInitialized => write!(f, "Visualization engine not initialized"),
            AdapterError::DataNotSet => write!(f, "Graph data not set"),
            AdapterError::RenderingFailed(msg) => write!(f, "Rendering failed: {}", msg),
            AdapterError::ExportFailed(msg) => write!(f, "Export failed: {}", msg),
            AdapterError::StreamingFailed(msg) => write!(f, "Streaming failed: {}", msg),
            AdapterError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Convert AdapterError to GraphError
impl From<AdapterError> for crate::errors::GraphError {
    fn from(err: AdapterError) -> Self {
        crate::errors::GraphError::InvalidInput(err.to_string())
    }
}