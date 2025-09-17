//! Graph Visualization Module
//! 
//! Unified visualization system combining streaming tables and interactive graphs.
//! Built on existing display and streaming infrastructure.

use std::sync::Arc;
use std::net::IpAddr;
use crate::errors::{GraphResult, GraphError};
use streaming::websocket_server::{StreamingServer, StreamingConfig};
use streaming::data_source::{DataSource, LayoutAlgorithm, HierarchicalDirection};
use streaming::virtual_scroller::VirtualScrollConfig;

// Migrated infrastructure modules
pub mod streaming;  // Migrated from core/streaming
pub mod display;    // Migrated from core/display

// New visualization modules
pub mod layouts;    // Graph layout algorithms
pub mod themes;     // Graph visualization themes

// Legacy - deprecated in favor of unified streaming infrastructure
// pub mod server;

/// Main visualization module providing interactive and static graph visualization
#[derive(Debug, Clone)]
pub struct VizModule {
    /// Unified data source for visualization (supports both tables and graphs)
    data_source: Arc<dyn DataSource>,
    /// Current visualization configuration
    config: VizConfig,
}

impl VizModule {
    /// Create a new visualization module with any DataSource
    pub fn new(data_source: Arc<dyn DataSource>) -> Self {
        Self {
            data_source,
            config: VizConfig::default(),
        }
    }

    /// Launch interactive browser-based visualization using streaming infrastructure
    pub fn interactive(&self, options: Option<InteractiveOptions>) -> GraphResult<InteractiveViz> {
        let opts = options.unwrap_or_default();
        
        // Create streaming configuration from visualization options
        let streaming_config = StreamingConfig {
            port: opts.port,
            scroll_config: VirtualScrollConfig {
                window_size: 50, // Default window size
                buffer_size: 100,
                cache_size: 50,
                auto_preload: true,
                cache_timeout_secs: 300,
            },
            max_connections: 100,
            auto_broadcast: true,
            update_throttle_ms: 100,
        };
        
        // Create streaming server with the data source
        let streaming_server = StreamingServer::new(
            self.data_source.clone(),
            streaming_config,
        );
        
        Ok(InteractiveViz {
            streaming_server,
            config: opts,
            viz_config: self.config.clone(),
        })
    }

    /// Generate static visualization export (PNG, SVG, PDF)
    pub fn static_viz(&self, options: StaticOptions) -> GraphResult<StaticViz> {
        // TODO: Implement static visualization export
        Err(GraphError::NotImplemented { 
            feature: "Static visualization".to_string(),
            tracking_issue: Some("https://github.com/anthropics/groggy/issues/viz-static".to_string())
        })
    }

    /// Update the configuration for this visualization module
    pub fn with_config(mut self, config: VizConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Convenience method to create VizModule from a NodesTable
    pub fn from_nodes_table(nodes_table: Arc<crate::storage::table::NodesTable>) -> Self {
        Self::new(nodes_table as Arc<dyn DataSource>)
    }
    
    /// Convenience method to create VizModule from an EdgesTable
    pub fn from_edges_table(edges_table: Arc<crate::storage::table::EdgesTable>) -> Self {
        Self::new(edges_table as Arc<dyn DataSource>)
    }
    
    /// Convenience method to create VizModule from a GraphTable
    pub fn from_graph_table(graph_table: Arc<crate::storage::table::GraphTable>) -> Self {
        Self::new(graph_table as Arc<dyn DataSource>)
    }
    
    /// Check if the data source supports graph visualization
    pub fn supports_graph_view(&self) -> bool {
        self.data_source.supports_graph_view()
    }
    
    /// Get basic statistics about the data source
    pub fn get_info(&self) -> DataSourceInfo {
        let supports_graph = self.data_source.supports_graph_view();
        let total_rows = self.data_source.total_rows();
        let total_cols = self.data_source.total_cols();
        
        let graph_info = if supports_graph {
            let metadata = self.data_source.get_graph_metadata();
            Some(GraphInfo {
                node_count: metadata.node_count,
                edge_count: metadata.edge_count,
                is_directed: metadata.is_directed,
                has_weights: metadata.has_weights,
            })
        } else {
            None
        };
        
        DataSourceInfo {
            total_rows,
            total_cols,
            supports_graph,
            graph_info,
            source_type: self.data_source.get_schema().source_type,
        }
    }
}

/// Configuration for the visualization module
#[derive(Debug, Clone)]
pub struct VizConfig {
    /// Default theme for visualizations
    pub default_theme: String,
    /// Default layout algorithm
    pub default_layout: LayoutAlgorithm,
    /// Performance optimization settings
    pub performance: PerformanceConfig,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            default_theme: "light".to_string(),
            default_layout: LayoutAlgorithm::ForceDirected {
                charge: -300.0,
                distance: 50.0,
                iterations: 100,
            },
            performance: PerformanceConfig::default(),
        }
    }
}

/// Performance configuration for handling large graphs
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Maximum nodes before enabling clustering
    pub clustering_threshold: usize,
    /// Enable GPU acceleration when available
    pub gpu_acceleration: bool,
    /// Memory limit for client-side caching (MB)
    pub memory_limit_mb: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            clustering_threshold: 1000,
            gpu_acceleration: true,
            memory_limit_mb: 100,
        }
    }
}

// Layout algorithms and directions are now imported from streaming::data_source
// This eliminates duplication and ensures consistency

/// Options for interactive visualization
#[derive(Debug, Clone)]
pub struct InteractiveOptions {
    /// Server port (0 for automatic)
    pub port: u16,
    /// Layout algorithm to use
    pub layout: LayoutAlgorithm,
    /// Visual theme
    pub theme: String,
    /// Canvas dimensions
    pub width: u32,
    pub height: u32,
    /// Enable specific interaction features
    pub interactions: InteractionConfig,
}

impl Default for InteractiveOptions {
    fn default() -> Self {
        Self {
            port: 0, // Auto-assign port
            layout: LayoutAlgorithm::ForceDirected {
                charge: -300.0,
                distance: 50.0,
                iterations: 100,
            },
            theme: "light".to_string(),
            width: 1200,
            height: 800,
            interactions: InteractionConfig::default(),
        }
    }
}

/// Configuration for interaction features
#[derive(Debug, Clone)]
pub struct InteractionConfig {
    /// Enable node clicking for details
    pub clickable_nodes: bool,
    /// Enable edge hovering for tooltips
    pub hoverable_edges: bool,
    /// Enable multi-node selection
    pub selectable_regions: bool,
    /// Enable zoom and pan controls
    pub zoom_controls: bool,
    /// Show filtering panel
    pub filter_panel: bool,
    /// Show search box
    pub search_box: bool,
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            clickable_nodes: true,
            hoverable_edges: true,
            selectable_regions: true,
            zoom_controls: true,
            filter_panel: true,
            search_box: true,
        }
    }
}

/// Options for static visualization export
#[derive(Debug, Clone)]
pub struct StaticOptions {
    /// Output filename
    pub filename: String,
    /// Export format
    pub format: ExportFormat,
    /// Layout algorithm
    pub layout: LayoutAlgorithm,
    /// Visual theme
    pub theme: String,
    /// Resolution for raster formats
    pub dpi: u32,
    /// Canvas dimensions
    pub width: u32,
    pub height: u32,
}

/// Supported export formats for static visualization
#[derive(Debug, Clone)]
pub enum ExportFormat {
    PNG,
    SVG,
    PDF,
}

/// Active interactive visualization session using streaming infrastructure
pub struct InteractiveViz {
    streaming_server: StreamingServer,
    config: InteractiveOptions,
    viz_config: VizConfig,
}

impl InteractiveViz {
    /// Start the visualization server and return the URL
    pub fn start(&self, bind_addr: Option<IpAddr>) -> GraphResult<InteractiveVizSession> {
        let addr = bind_addr.unwrap_or_else(|| "127.0.0.1".parse().unwrap());
        let port_hint = if self.config.port == 0 { 8080 } else { self.config.port };
        
        // Start the streaming server in background
        let server_handle = self.streaming_server.start_background(addr, port_hint)
            .map_err(|e| GraphError::internal(&format!("Failed to start visualization server: {}", e), "VizModule::start"))?;
        
        let actual_port = server_handle.port;
        let url = format!("http://{}:{}", addr, actual_port);
        
        println!("🚀 Interactive visualization server started at: {}", url);
        
        if self.streaming_server.data_source.supports_graph_view() {
            let metadata = self.streaming_server.data_source.get_graph_metadata();
            println!("📊 Graph visualization: {} nodes, {} edges", 
                    metadata.node_count, metadata.edge_count);
        } else {
            println!("📋 Table visualization: {} rows × {} columns", 
                    self.streaming_server.data_source.total_rows(),
                    self.streaming_server.data_source.total_cols());
        }
        
        Ok(InteractiveVizSession {
            server_handle,
            url,
            config: self.config.clone(),
        })
    }
}

/// Active visualization session with running server
pub struct InteractiveVizSession {
    server_handle: streaming::websocket_server::ServerHandle,
    url: String,
    config: InteractiveOptions,
}

impl InteractiveVizSession {
    /// Get the URL where the visualization is accessible
    pub fn url(&self) -> &str {
        &self.url
    }
    
    /// Get the port the server is running on
    pub fn port(&self) -> u16 {
        self.server_handle.port
    }
    
    /// Stop the visualization server
    pub fn stop(self) {
        self.server_handle.stop()
    }
}

/// Static visualization output
pub struct StaticViz {
    /// Output file path
    pub file_path: String,
    /// Generated content size in bytes
    pub size_bytes: usize,
}

// All graph data structures and traits are now provided by the streaming::data_source module
// This ensures consistency and eliminates duplication between visualization and streaming components

/// Information about a data source for visualization
#[derive(Debug, Clone)]
pub struct DataSourceInfo {
    /// Total number of rows in the data source
    pub total_rows: usize,
    /// Total number of columns in the data source
    pub total_cols: usize,
    /// Whether the data source supports graph visualization
    pub supports_graph: bool,
    /// Graph-specific information (if available)
    pub graph_info: Option<GraphInfo>,
    /// Type of the data source (e.g., "nodes_table", "edges_table", "graph_table")
    pub source_type: String,
}

/// Graph-specific information
#[derive(Debug, Clone)]
pub struct GraphInfo {
    /// Number of nodes in the graph
    pub node_count: usize,
    /// Number of edges in the graph
    pub edge_count: usize,
    /// Whether the graph is directed
    pub is_directed: bool,
    /// Whether the graph has weighted edges
    pub has_weights: bool,
}