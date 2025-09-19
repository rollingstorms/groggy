//! File adapter that wraps the unified core engine for static export
//!
//! Per UNIFIED_VIZ_MIGRATION_PLAN.md: "FileAdapter wraps the core
//! VizEngine for static file export (HTML/SVG/PNG)"

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

use crate::errors::GraphResult;
use crate::viz::core::{VizEngine, VizConfig, VizFrame};
use crate::viz::core::rendering::{RenderingEngine, RenderingConfig, RenderFormat};
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge};
use super::traits::{
    VizAdapter, ExportAdapter, AdapterResult, AdapterConfig, FileResult, 
    ExportFormat, AdapterError
};

/// File adapter that wraps VizEngine for static file export
pub struct FileAdapter {
    /// Unified core engine (single source of truth)
    core: VizEngine,
    
    /// Rendering engine for different formats
    renderer: RenderingEngine,
    
    /// Export configuration
    export_config: FileExportConfig,
    
    /// Output directory
    output_dir: PathBuf,
    
    /// Last generated frame
    current_frame: Option<VizFrame>,
}

/// Configuration for file export
#[derive(Debug, Clone)]
pub struct FileExportConfig {
    /// Default export format
    pub default_format: ExportFormat,
    
    /// Include metadata in exports
    pub include_metadata: bool,
    
    /// Optimize file size
    pub optimize: bool,
    
    /// Custom export settings per format
    pub format_settings: HashMap<ExportFormat, FormatSettings>,
}

#[derive(Debug, Clone)]
pub struct FormatSettings {
    /// Quality/compression settings (0-100)
    pub quality: Option<u32>,
    
    /// Include interactive features
    pub interactive: bool,
    
    /// Custom styling
    pub custom_styles: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for FileExportConfig {
    fn default() -> Self {
        let mut format_settings = HashMap::new();
        
        // SVG settings
        format_settings.insert(ExportFormat::SVG, FormatSettings {
            quality: None,
            interactive: false,
            custom_styles: String::new(),
            metadata: HashMap::new(),
        });
        
        // HTML settings
        format_settings.insert(ExportFormat::HTML, FormatSettings {
            quality: None,
            interactive: true,
            custom_styles: String::new(),
            metadata: HashMap::new(),
        });
        
        // PNG settings
        format_settings.insert(ExportFormat::PNG { dpi: 300 }, FormatSettings {
            quality: Some(90),
            interactive: false,
            custom_styles: String::new(),
            metadata: HashMap::new(),
        });
        
        Self {
            default_format: ExportFormat::SVG,
            include_metadata: true,
            optimize: true,
            format_settings,
        }
    }
}

impl FileAdapter {
    /// Create new file adapter with unified core engine
    pub fn new(config: AdapterConfig, output_dir: Option<PathBuf>) -> Self {
        // Convert adapter config to VizEngine config
        let viz_config = VizConfig {
            width: config.width,
            height: config.height,
            physics_enabled: config.physics_enabled,
            continuous_physics: false, // Static export doesn't need continuous updates
            target_fps: 1.0, // Minimal FPS for static export
            interactions_enabled: false, // Interactions handled per format
            auto_fit: config.auto_fit,
            fit_padding: 50.0,
        };
        
        // Create unified core engine
        let core = VizEngine::new(viz_config);
        
        // Create rendering engine with file-optimized settings
        let render_config = RenderingConfig {
            format: RenderFormat::SVG, // Default, will be changed per export
            width: config.width,
            height: config.height,
            include_animations: false, // Static by default
            include_interactions: false, // Will be set per format
            background_color: "#ffffff".to_string(),
            show_grid: false,
            grid_config: Default::default(),
        };
        
        let renderer = RenderingEngine::new(render_config);
        
        let output_dir = output_dir.unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });
        
        Self {
            core,
            renderer,
            export_config: FileExportConfig::default(),
            output_dir,
            current_frame: None,
        }
    }
    
    /// Export visualization to file with specified format
    pub fn export_to_file_with_format(&mut self, filename: &str, format: ExportFormat) -> GraphResult<FileResult> {
        let start_time = std::time::Instant::now();
        
        // Generate frame from unified core engine
        let frame = self.core.update()?;
        self.current_frame = Some(frame.clone());
        
        // Configure renderer for the specific format
        self.configure_renderer_for_format(&format)?;
        
        // Render using unified rendering pipeline
        let render_output = self.renderer.render(&frame)?;
        
        // Process content based on format
        let processed_content = self.process_content_for_format(&render_output.content, &format, &frame)?;
        
        // Determine output path
        let output_path = self.get_output_path(filename, &format)?;
        
        // Write to file
        fs::write(&output_path, processed_content.as_bytes())
            .map_err(|e| AdapterError::ExportFailed(format!("Failed to write file: {}", e)))?;
        
        let export_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let file_size = fs::metadata(&output_path)
            .map(|m| m.len() as usize)
            .unwrap_or(processed_content.len());
        
        let result = FileResult {
            file_path: output_path.to_string_lossy().to_string(),
            file_size_bytes: file_size,
            format: self.format_to_string(&format),
            export_time_ms: export_time,
        };
        
        Ok(result)
    }
    
    /// Configure renderer for specific export format
    fn configure_renderer_for_format(&mut self, format: &ExportFormat) -> GraphResult<()> {
        let format_settings = self.export_config.format_settings
            .get(format)
            .cloned()
            .unwrap_or_default();
        
        // Update renderer configuration
        let render_format = match format {
            ExportFormat::SVG => RenderFormat::SVG,
            ExportFormat::HTML => RenderFormat::HTML,
            ExportFormat::JSON => RenderFormat::JSON,
            ExportFormat::PNG { .. } => RenderFormat::PNG { scale: 1.0 },
            ExportFormat::PDF => RenderFormat::PDF,
        };
        
        self.renderer.config.format = render_format;
        self.renderer.config.include_interactions = format_settings.interactive;
        self.renderer.config.include_animations = false; // Static export
        
        Ok(())
    }
    
    /// Process rendered content for specific format
    fn process_content_for_format(&self, content: &str, format: &ExportFormat, frame: &VizFrame) -> GraphResult<String> {
        let format_settings = self.export_config.format_settings
            .get(format)
            .cloned()
            .unwrap_or_default();
        
        let mut processed = content.to_string();
        
        match format {
            ExportFormat::HTML => {
                // Add additional HTML structure and metadata
                processed = self.wrap_html_content(&processed, frame, &format_settings)?;
            }
            
            ExportFormat::SVG => {
                // Add SVG metadata and optimization
                processed = self.optimize_svg_content(&processed, &format_settings)?;
            }
            
            ExportFormat::JSON => {
                // Add export metadata to JSON
                processed = self.add_json_metadata(&processed, frame)?;
            }
            
            ExportFormat::PNG { .. } | ExportFormat::PDF => {
                return Err(AdapterError::ExportFailed(
                    format!("Format {:?} requires external renderer", format)
                ).into());
            }
        }
        
        // Add custom styles if specified
        if !format_settings.custom_styles.is_empty() {
            processed = self.apply_custom_styles(&processed, &format_settings.custom_styles, format)?;
        }
        
        Ok(processed)
    }
    
    /// Wrap HTML content with complete document structure
    fn wrap_html_content(&self, content: &str, frame: &VizFrame, settings: &FormatSettings) -> GraphResult<String> {
        let metadata_html = if self.export_config.include_metadata {
            format!(
                r#"
    <div class="metadata" style="margin-top: 20px; padding: 10px; background: #f5f5f5; border-radius: 4px; font-size: 12px;">
        <strong>Generated by Groggy Graph Library</strong><br>
        Nodes: {}, Edges: {}<br>
        Generated: {}<br>
        Frame ID: {}
    </div>"#,
                frame.nodes.len(),
                frame.edges.len(),
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                frame.metadata.frame_id
            )
        } else {
            String::new()
        };
        
        let interactive_js = if settings.interactive {
            r#"
    <script>
    // Basic interaction support for exported HTML
    document.addEventListener('DOMContentLoaded', function() {
        const svg = document.querySelector('svg');
        if (!svg) return;
        
        // Add basic zoom/pan
        let scale = 1;
        let panX = 0, panY = 0;
        
        svg.addEventListener('wheel', function(e) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale *= delta;
            updateTransform();
        });
        
        let isDragging = false;
        let startX, startY;
        
        svg.addEventListener('mousedown', function(e) {
            isDragging = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            this.style.cursor = 'grabbing';
        });
        
        document.addEventListener('mousemove', function(e) {
            if (!isDragging) return;
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            updateTransform();
        });
        
        document.addEventListener('mouseup', function() {
            isDragging = false;
            svg.style.cursor = 'grab';
        });
        
        function updateTransform() {
            svg.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
        }
    });
    </script>"#
        } else {
            ""
        };
        
        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Groggy Graph Visualization</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .visualization {{
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
            background: white;
        }}
        svg {{
            display: block;
            width: 100%;
            height: auto;
            cursor: grab;
        }}
        {custom_styles}
    </style>
</head>
<body>
    <div class="container">
        <h1>Graph Visualization</h1>
        <div class="visualization">
            {content}
        </div>
        {metadata_html}
    </div>
    {interactive_js}
</body>
</html>"#,
            content = content,
            metadata_html = metadata_html,
            interactive_js = interactive_js,
            custom_styles = settings.custom_styles
        );
        
        Ok(html)
    }
    
    /// Optimize SVG content for file export
    fn optimize_svg_content(&self, content: &str, settings: &FormatSettings) -> GraphResult<String> {
        let mut optimized = content.to_string();
        
        if self.export_config.optimize {
            // Remove unnecessary whitespace
            optimized = optimized
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("");
            
            // Add metadata if requested
            if self.export_config.include_metadata {
                let metadata = format!(
                    r#"<!-- Generated by Groggy Graph Library at {} -->"#,
                    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
                );
                
                optimized = optimized.replace("<svg", &format!("{}\n<svg", metadata));
            }
        }
        
        Ok(optimized)
    }
    
    /// Add metadata to JSON export
    fn add_json_metadata(&self, content: &str, frame: &VizFrame) -> GraphResult<String> {
        if !self.export_config.include_metadata {
            return Ok(content.to_string());
        }
        
        // Parse JSON and add metadata
        let mut json: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| AdapterError::ExportFailed(format!("Invalid JSON: {}", e)))?;
        
        let metadata = serde_json::json!({
            "export_metadata": {
                "generated_by": "Groggy Graph Library",
                "generated_at": chrono::Utc::now().to_rfc3339(),
                "frame_id": frame.metadata.frame_id,
                "node_count": frame.nodes.len(),
                "edge_count": frame.edges.len(),
                "export_format": "JSON"
            }
        });
        
        if let serde_json::Value::Object(ref mut map) = json {
            map.insert("_metadata".to_string(), metadata);
        }
        
        serde_json::to_string_pretty(&json)
            .map_err(|e| AdapterError::ExportFailed(format!("JSON serialization failed: {}", e)).into())
    }
    
    /// Apply custom styles to content
    fn apply_custom_styles(&self, content: &str, styles: &str, format: &ExportFormat) -> GraphResult<String> {
        match format {
            ExportFormat::HTML => {
                // Insert styles into HTML head
                Ok(content.replace("</style>", &format!("{}\n</style>", styles)))
            }
            ExportFormat::SVG => {
                // Add style element to SVG
                let style_element = format!("<style><![CDATA[{}]]></style>", styles);
                Ok(content.replace("<svg", &format!("{}\n<svg", style_element)))
            }
            _ => Ok(content.to_string()), // Styles not applicable to other formats
        }
    }
    
    /// Get output path for file
    fn get_output_path(&self, filename: &str, format: &ExportFormat) -> GraphResult<PathBuf> {
        let path = Path::new(filename);
        
        let output_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.output_dir.join(path)
        };
        
        // Ensure correct extension
        let extension = format.extension();
        let final_path = if output_path.extension().map(|e| e.to_str()) == Some(Some(extension)) {
            output_path
        } else {
            output_path.with_extension(extension)
        };
        
        // Create parent directories if needed
        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| AdapterError::ExportFailed(format!("Failed to create directory: {}", e)))?;
        }
        
        Ok(final_path)
    }
    
    /// Convert ExportFormat to string
    fn format_to_string(&self, format: &ExportFormat) -> String {
        match format {
            ExportFormat::SVG => "SVG".to_string(),
            ExportFormat::HTML => "HTML".to_string(),
            ExportFormat::PNG { dpi } => format!("PNG ({}dpi)", dpi),
            ExportFormat::PDF => "PDF".to_string(),
            ExportFormat::JSON => "JSON".to_string(),
        }
    }
    
    /// Set output directory
    pub fn set_output_directory(&mut self, dir: PathBuf) -> GraphResult<()> {
        if !dir.exists() {
            fs::create_dir_all(&dir)
                .map_err(|e| AdapterError::ConfigurationError(format!("Failed to create directory: {}", e)))?;
        }
        
        self.output_dir = dir;
        Ok(())
    }
    
    /// Update export configuration
    pub fn update_export_config(&mut self, config: FileExportConfig) {
        self.export_config = config;
    }
    
    /// Generate preview without saving to file
    pub fn generate_preview(&mut self, format: ExportFormat) -> GraphResult<String> {
        // Generate frame from unified core engine
        let frame = self.core.update()?;
        
        // Configure renderer for the format
        self.configure_renderer_for_format(&format)?;
        
        // Render using unified rendering pipeline
        let render_output = self.renderer.render(&frame)?;
        
        // Process content for format
        self.process_content_for_format(&render_output.content, &format, &frame)
    }
}

impl VizAdapter for FileAdapter {
    fn set_data(&mut self, nodes: Vec<VizNode>, edges: Vec<VizEdge>) -> GraphResult<()> {
        // Delegate to unified core engine
        self.core.set_data(nodes, edges)?;
        
        // Clear cached frame
        self.current_frame = None;
        
        Ok(())
    }
    
    fn render(&mut self) -> GraphResult<AdapterResult> {
        // For file adapter, render means export to default format
        let result = self.export_to_file_with_format(
            "graph_export", 
            self.export_config.default_format.clone()
        )?;
        
        Ok(AdapterResult::File(result))
    }
    
    fn get_frame(&mut self) -> GraphResult<VizFrame> {
        // Get fresh frame from unified core
        self.core.update()
    }
    
    fn is_ready(&self) -> bool {
        // Ready if we have data loaded
        !self.core.get_positions().is_empty()
    }
    
    fn get_config(&self) -> AdapterConfig {
        AdapterConfig {
            width: 800.0, // TODO: Get from actual renderer config
            height: 600.0,
            physics_enabled: self.core.is_simulation_running(),
            interactions_enabled: false, // Static exports don't have interactions by default
            auto_fit: true,
            theme: None,
            custom_styles: HashMap::new(),
        }
    }
    
    fn update_config(&mut self, config: AdapterConfig) -> GraphResult<()> {
        // Update VizEngine config
        let viz_config = VizConfig {
            width: config.width,
            height: config.height,
            physics_enabled: config.physics_enabled,
            continuous_physics: false,
            target_fps: 1.0,
            interactions_enabled: false, // Static export
            auto_fit: config.auto_fit,
            fit_padding: 50.0,
        };
        
        self.core.set_config(viz_config)?;
        
        // Update renderer config
        self.renderer.config.width = config.width;
        self.renderer.config.height = config.height;
        
        Ok(())
    }
}

impl ExportAdapter for FileAdapter {
    fn export_to_file(&mut self, path: &str, format: ExportFormat) -> GraphResult<FileResult> {
        self.export_to_file_with_format(path, format)
    }
    
    fn supported_formats(&self) -> Vec<ExportFormat> {
        vec![
            ExportFormat::SVG,
            ExportFormat::HTML,
            ExportFormat::JSON,
            // PNG and PDF require external renderers
        ]
    }
}

/// Builder for creating file adapters with configuration
pub struct FileAdapterBuilder {
    config: AdapterConfig,
    export_config: FileExportConfig,
    output_dir: Option<PathBuf>,
}

impl FileAdapterBuilder {
    pub fn new() -> Self {
        Self {
            config: AdapterConfig::default(),
            export_config: FileExportConfig::default(),
            output_dir: None,
        }
    }
    
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }
    
    pub fn with_output_directory<P: Into<PathBuf>>(mut self, dir: P) -> Self {
        self.output_dir = Some(dir.into());
        self
    }
    
    pub fn with_default_format(mut self, format: ExportFormat) -> Self {
        self.export_config.default_format = format;
        self
    }
    
    pub fn with_optimization(mut self, optimize: bool) -> Self {
        self.export_config.optimize = optimize;
        self
    }
    
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.export_config.include_metadata = include;
        self
    }
    
    pub fn build(self) -> FileAdapter {
        let mut adapter = FileAdapter::new(self.config, self.output_dir);
        adapter.update_export_config(self.export_config);
        adapter
    }
}

impl Default for FileAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FormatSettings {
    fn default() -> Self {
        Self {
            quality: None,
            interactive: false,
            custom_styles: String::new(),
            metadata: HashMap::new(),
        }
    }
}