//! 🎯 Unified Visualization API - Single Entry Point
//!
//! Phase 4: Unified API that provides a single entry point for all visualization
//! backends while using the unified core engine and adapters from previous phases.
//!
//! This module implements the final API design from UNIFIED_VIZ_MIGRATION_PLAN.md:
//! - Single VizModule entry point
//! - Backend routing with enum dispatch  
//! - Convenience methods for common use cases
//! - Consistent configuration across all backends

use crate::errors::GraphResult;
use crate::viz::core::{VizEngine, VizConfig};
use crate::viz::adapters::{
    VizAdapter, StreamingAdapter, JupyterAdapter, FileAdapter,
    AdapterResult, AdapterConfig, ExportFormat
};
use crate::viz::streaming::websocket_server::StreamingConfig;
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge};
use std::collections::HashMap;
use std::path::PathBuf;

/// Unified visualization backend types
#[derive(Debug, Clone, PartialEq)]
pub enum VizBackend {
    /// Jupyter notebook widget
    Jupyter,
    /// WebSocket streaming server
    Streaming { port: Option<u16> },
    /// Static file export
    File { path: String, format: Option<ExportFormat> },
    /// Local in-process visualization
    Local,
}

/// Result from visualization rendering
#[derive(Debug, Clone)]
pub enum VizResult {
    /// Jupyter widget result
    Widget(crate::viz::adapters::JupyterResult),
    /// Streaming server result
    Server(crate::viz::adapters::StreamingResult),
    /// File export result
    File(crate::viz::adapters::FileResult),
    /// Local visualization result
    Local(String), // URL or identifier for local display
}

/// Unified visualization module - single entry point for all backends
pub struct VizModule {
    /// Reference to the graph for data extraction
    graph_ref: std::sync::Arc<dyn GraphDataProvider>,
    
    /// Unified configuration applied to all backends
    config: VizConfig,
    
    /// Adapter-specific configurations
    adapter_configs: HashMap<String, AdapterConfig>,
    
    /// Cached adapters for reuse
    adapters: HashMap<String, Box<dyn VizAdapter>>,
}

/// Trait for extracting graph data (implemented by Graph)
pub trait GraphDataProvider: Send + Sync {
    fn get_viz_nodes(&self) -> GraphResult<Vec<VizNode>>;
    fn get_viz_edges(&self) -> GraphResult<Vec<VizEdge>>;
    fn get_node_count(&self) -> usize;
    fn get_edge_count(&self) -> usize;
}

impl VizModule {
    /// Create new visualization module from graph data provider
    pub fn new(graph: std::sync::Arc<dyn GraphDataProvider>) -> Self {
        Self {
            graph_ref: graph,
            config: VizConfig::default(),
            adapter_configs: HashMap::new(),
            adapters: HashMap::new(),
        }
    }
    
    /// 🎯 Unified render method - routes to appropriate backend
    pub fn render(&mut self, backend: VizBackend) -> GraphResult<VizResult> {
        let nodes = self.graph_ref.get_viz_nodes()?;
        let edges = self.graph_ref.get_viz_edges()?;
        
        match backend {
            VizBackend::Jupyter => {
                let adapter = self.get_or_create_jupyter_adapter()?;
                adapter.set_data(nodes, edges)?;
                let result = adapter.render()?;
                
                match result {
                    AdapterResult::Jupyter(jupyter_result) => Ok(VizResult::Widget(jupyter_result)),
                    _ => Err(crate::errors::GraphError::InvalidInput("Expected Jupyter result".to_string())),
                }
            }
            
            VizBackend::Streaming { port } => {
                let adapter = self.get_or_create_streaming_adapter(port)?;
                adapter.set_data(nodes, edges)?;
                let result = adapter.render()?;
                
                match result {
                    AdapterResult::Streaming(streaming_result) => Ok(VizResult::Server(streaming_result)),
                    _ => Err(crate::errors::GraphError::InvalidInput("Expected Streaming result".to_string())),
                }
            }
            
            VizBackend::File { path, format } => {
                let adapter = self.get_or_create_file_adapter()?;
                adapter.set_data(nodes, edges)?;
                
                // Determine format from path extension if not specified
                let export_format = format.unwrap_or_else(|| {
                    self.detect_format_from_path(&path)
                });
                
                let file_adapter = adapter.as_any().downcast_mut::<FileAdapter>()
                    .ok_or_else(|| crate::errors::GraphError::InvalidInput("Expected FileAdapter".to_string()))?;
                
                let result = file_adapter.export_to_file(&path, export_format)?;
                Ok(VizResult::File(result))
            }
            
            VizBackend::Local => {
                // For local backend, create a simple HTML file in temp directory
                let temp_path = std::env::temp_dir().join("groggy_local_viz.html");
                let file_backend = VizBackend::File { 
                    path: temp_path.to_string_lossy().to_string(), 
                    format: Some(ExportFormat::HTML) 
                };
                
                match self.render(file_backend)? {
                    VizResult::File(file_result) => {
                        Ok(VizResult::Local(format!("file://{}", file_result.file_path)))
                    }
                    _ => Err(crate::errors::GraphError::InvalidInput("Local render failed".to_string())),
                }
            }
        }
    }
    
    /// 🎯 Convenience method: Create Jupyter widget
    pub fn widget(&mut self) -> GraphResult<VizResult> {
        self.render(VizBackend::Jupyter)
    }
    
    /// 🎯 Convenience method: Start streaming server
    pub fn serve(&mut self, port: Option<u16>) -> GraphResult<VizResult> {
        self.render(VizBackend::Streaming { port })
    }
    
    /// 🎯 Convenience method: Save to file
    pub fn save(&mut self, path: &str) -> GraphResult<VizResult> {
        self.render(VizBackend::File { path: path.to_string(), format: None })
    }
    
    /// 🎯 Convenience method: Save with specific format
    pub fn save_as(&mut self, path: &str, format: ExportFormat) -> GraphResult<VizResult> {
        self.render(VizBackend::File { path: path.to_string(), format: Some(format) })
    }
    
    /// 🎯 Convenience method: Open local visualization
    pub fn show(&mut self) -> GraphResult<VizResult> {
        self.render(VizBackend::Local)
    }
    
    /// ⚙️ Configure visualization settings
    pub fn configure(&mut self, config: VizConfig) -> &mut Self {
        self.config = config;
        
        // Clear cached adapters so they pick up new config
        self.adapters.clear();
        
        self
    }
    
    /// ⚙️ Configure specific adapter
    pub fn configure_adapter(&mut self, backend_type: &str, config: AdapterConfig) -> &mut Self {
        self.adapter_configs.insert(backend_type.to_string(), config);
        
        // Remove cached adapter to force recreation with new config
        self.adapters.remove(backend_type);
        
        self
    }
    
    /// 🏗️ Get or create Jupyter adapter
    fn get_or_create_jupyter_adapter(&mut self) -> GraphResult<&mut Box<dyn VizAdapter>> {
        if !self.adapters.contains_key("jupyter") {
            let adapter_config = self.adapter_configs
                .get("jupyter")
                .cloned()
                .unwrap_or_else(|| self.create_default_adapter_config());
            
            let jupyter_adapter = JupyterAdapter::new(adapter_config);
            self.adapters.insert("jupyter".to_string(), Box::new(jupyter_adapter));
        }
        
        Ok(self.adapters.get_mut("jupyter").unwrap())
    }
    
    /// 🏗️ Get or create Streaming adapter
    fn get_or_create_streaming_adapter(&mut self, port: Option<u16>) -> GraphResult<&mut Box<dyn VizAdapter>> {
        let key = format!("streaming_{}", port.unwrap_or(8080));
        
        if !self.adapters.contains_key(&key) {
            let adapter_config = self.adapter_configs
                .get("streaming")
                .cloned()
                .unwrap_or_else(|| self.create_default_adapter_config());
            
            let mut streaming_config = StreamingConfig::default();
            if let Some(p) = port {
                streaming_config.port = p;
            }
            
            let streaming_adapter = StreamingAdapter::new(adapter_config, streaming_config);
            self.adapters.insert(key.clone(), Box::new(streaming_adapter));
        }
        
        Ok(self.adapters.get_mut(&key).unwrap())
    }
    
    /// 🏗️ Get or create File adapter
    fn get_or_create_file_adapter(&mut self) -> GraphResult<&mut Box<dyn VizAdapter>> {
        if !self.adapters.contains_key("file") {
            let adapter_config = self.adapter_configs
                .get("file")
                .cloned()
                .unwrap_or_else(|| self.create_default_adapter_config());
            
            let file_adapter = FileAdapter::new(adapter_config, None);
            self.adapters.insert("file".to_string(), Box::new(file_adapter));
        }
        
        Ok(self.adapters.get_mut("file").unwrap())
    }
    
    /// 🏗️ Create default adapter configuration
    fn create_default_adapter_config(&self) -> AdapterConfig {
        AdapterConfig {
            width: self.config.width,
            height: self.config.height,
            physics_enabled: self.config.physics_enabled,
            interactions_enabled: self.config.interactions_enabled,
            auto_fit: self.config.auto_fit,
            theme: None,
            custom_styles: HashMap::new(),
        }
    }
    
    /// 🔍 Detect export format from file path
    fn detect_format_from_path(&self, path: &str) -> ExportFormat {
        let path_lower = path.to_lowercase();
        
        if path_lower.ends_with(".html") || path_lower.ends_with(".htm") {
            ExportFormat::HTML
        } else if path_lower.ends_with(".svg") {
            ExportFormat::SVG
        } else if path_lower.ends_with(".json") {
            ExportFormat::JSON
        } else if path_lower.ends_with(".png") {
            ExportFormat::PNG { dpi: 300 }
        } else if path_lower.ends_with(".pdf") {
            ExportFormat::PDF
        } else {
            // Default to HTML for unknown extensions
            ExportFormat::HTML
        }
    }
    
    /// 📊 Get visualization statistics
    pub fn stats(&self) -> VizStats {
        VizStats {
            node_count: self.graph_ref.get_node_count(),
            edge_count: self.graph_ref.get_edge_count(),
            active_adapters: self.adapters.len(),
            config: self.config.clone(),
        }
    }
    
    /// 🧹 Clear cached adapters
    pub fn clear_cache(&mut self) {
        self.adapters.clear();
    }
}

/// Visualization statistics
#[derive(Debug, Clone)]
pub struct VizStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub active_adapters: usize,
    pub config: VizConfig,
}

/// Builder for VizModule with fluent configuration
pub struct VizModuleBuilder {
    graph: Option<std::sync::Arc<dyn GraphDataProvider>>,
    config: VizConfig,
    adapter_configs: HashMap<String, AdapterConfig>,
}

impl VizModuleBuilder {
    pub fn new() -> Self {
        Self {
            graph: None,
            config: VizConfig::default(),
            adapter_configs: HashMap::new(),
        }
    }
    
    pub fn with_graph(mut self, graph: std::sync::Arc<dyn GraphDataProvider>) -> Self {
        self.graph = Some(graph);
        self
    }
    
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }
    
    pub fn with_physics(mut self, enabled: bool) -> Self {
        self.config.physics_enabled = enabled;
        self
    }
    
    pub fn with_adapter_config(mut self, backend: &str, config: AdapterConfig) -> Self {
        self.adapter_configs.insert(backend.to_string(), config);
        self
    }
    
    pub fn build(self) -> GraphResult<VizModule> {
        let graph = self.graph.ok_or_else(|| {
            crate::errors::GraphError::InvalidInput("Graph data provider required".to_string())
        })?;
        
        let mut module = VizModule::new(graph);
        module.config = self.config;
        module.adapter_configs = self.adapter_configs;
        
        Ok(module)
    }
}

impl Default for VizModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Trait to add additional functionality to VizAdapter for downcasting
pub trait VizAdapterExt {
    fn as_any(&mut self) -> &mut dyn std::any::Any;
}

// Blanket implementation for all VizAdapter types
impl<T: VizAdapter + 'static> VizAdapterExt for T {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}