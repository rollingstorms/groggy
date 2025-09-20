//! Graph Visualization Module
//! 
//! Unified visualization system combining streaming tables and interactive graphs.
//! Built on existing display and streaming infrastructure.
//!
//! The new unified approach uses a single render() method with backend switching:
//! - render(backend=VizBackend::Jupyter) - Jupyter notebook embedding
//! - render(backend=VizBackend::Streaming) - Interactive WebSocket server
//! - render(backend=VizBackend::File) - Static file export (HTML/SVG/PNG)
//! - render(backend=VizBackend::Local) - Self-contained HTML

use std::sync::Arc;
use std::net::IpAddr;
use crate::errors::{GraphResult, GraphError};
use streaming::websocket_server::{StreamingServer, StreamingConfig};
use streaming::data_source::{DataSource, LayoutAlgorithm, HierarchicalDirection};
use streaming::virtual_scroller::VirtualScrollConfig;

/// Visualization backend options for unified rendering
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VizBackend {
    /// Jupyter notebook embedding with optimized display
    Jupyter,
    /// WebSocket interactive server with real-time updates
    Streaming,
    /// Static file export (HTML/SVG/PNG)
    File,
    /// Self-contained HTML with embedded data
    Local,
}

impl VizBackend {
    /// Convert string to VizBackend with validation
    pub fn from_string(backend: &str) -> GraphResult<Self> {
        match backend.to_lowercase().as_str() {
            "jupyter" => Ok(VizBackend::Jupyter),
            "streaming" => Ok(VizBackend::Streaming),
            "file" => Ok(VizBackend::File),
            "local" => Ok(VizBackend::Local),
            _ => Err(GraphError::InvalidInput(
                format!("Invalid backend '{}'. Expected: jupyter, streaming, file, or local", backend)
            )),
        }
    }
    
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            VizBackend::Jupyter => "jupyter",
            VizBackend::Streaming => "streaming", 
            VizBackend::File => "file",
            VizBackend::Local => "local",
        }
    }
}

/// Unified render options for all backends
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    // Universal parameters
    pub layout: Option<LayoutAlgorithm>,
    pub theme: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub title: Option<String>,
    
    // Backend-specific parameters
    pub port: Option<u16>,              // For streaming backend
    pub filename: Option<String>,       // For file backend
    pub format: Option<ExportFormat>,   // For file backend
    pub dpi: Option<u32>,              // For file backend
    pub auto_open: Option<bool>,       // For streaming backend
}

/// Result of rendering operation (different types for different backends)
pub enum RenderResult {
    /// HTML content (for jupyter and local backends)
    Html(String),
    /// Interactive visualization session (for streaming backend)
    Interactive(InteractiveViz),
    /// Static visualization result (for file backend)
    Static(StaticViz),
}

// Migrated infrastructure modules
pub mod streaming;  // Migrated from core/streaming
pub mod display;    // Migrated from core/display

// New visualization modules
pub mod core;       // ✅ Unified visualization core engine
pub mod layouts;    // Graph layout algorithms
pub mod themes;     // Graph visualization themes
pub mod unified;    // ✅ Unified API entry point

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
    
    /// 🎯 CONVENIENCE METHODS - Simplified unified API
    /// These methods provide the same interface as described in UNIFIED_VIZ_MIGRATION_PLAN.md
    
    /// Create Jupyter widget (unified core)
    pub fn widget(&mut self) -> GraphResult<RenderResult> {
        self.render(VizBackend::Jupyter, RenderOptions::default())
    }
    
    /// Start streaming server (unified core)
    pub fn serve(&mut self, port: Option<u16>) -> GraphResult<RenderResult> {
        let options = RenderOptions {
            port,
            ..Default::default()
        };
        self.render(VizBackend::Streaming, options)
    }
    
    /// Save to file (unified core)
    pub fn save(&mut self, path: &str) -> GraphResult<RenderResult> {
        let options = RenderOptions {
            filename: Some(path.to_string()),
            ..Default::default()
        };
        self.render(VizBackend::File, options)
    }
    
    /// Show local visualization (unified core)
    pub fn show(&mut self) -> GraphResult<RenderResult> {
        self.render(VizBackend::Local, RenderOptions::default())
    }

    /// 🎯 UNIFIED VISUALIZATION METHOD - The One Command To Rule Them All
    ///
    /// This is the new unified approach that replaces separate methods with
    /// a single command that adapts to different backends.
    ///
    /// # Arguments
    /// * `backend` - Target backend (Jupyter/Streaming/File/Local)
    /// * `options` - Backend-specific options and universal parameters
    ///
    /// # Returns
    /// * `RenderResult` - Different return types based on backend
    ///
    /// # Examples
    /// ```rust
    /// use groggy::viz::{VizModule, VizBackend, RenderOptions};
    /// 
    /// // Jupyter notebook embedding
    /// let result = viz_module.render(VizBackend::Jupyter, RenderOptions::default())?;
    /// 
    /// // Interactive server
    /// let result = viz_module.render(VizBackend::Streaming, RenderOptions {
    ///     port: Some(8080),
    ///     ..Default::default()
    /// })?;
    /// 
    /// // Static file export
    /// let result = viz_module.render(VizBackend::File, RenderOptions {
    ///     filename: Some("graph.html".to_string()),
    ///     ..Default::default()
    /// })?;
    /// ```
    pub fn render(&self, backend: VizBackend, options: RenderOptions) -> GraphResult<RenderResult> {
        match backend {
            VizBackend::Jupyter => self.render_jupyter(options),
            VizBackend::Streaming => self.render_streaming(options),
            VizBackend::File => self.render_file(options),
            VizBackend::Local => self.render_local(options),
        }
    }

    /// Render for Jupyter notebook with optimized embedding
    fn render_jupyter(&self, options: RenderOptions) -> GraphResult<RenderResult> {
        // Generate Jupyter-optimized HTML
        let html = self.generate_jupyter_html(&options)?;
        Ok(RenderResult::Html(html))
    }

    /// Render for streaming WebSocket server
    fn render_streaming(&self, options: RenderOptions) -> GraphResult<RenderResult> {
        // Delegate to existing interactive method
        let interactive_opts = InteractiveOptions {
            port: options.port.unwrap_or(8080),
            layout: options.layout.unwrap_or(LayoutAlgorithm::ForceDirected { 
                charge: -30.0, 
                distance: 50.0, 
                iterations: 100 
            }),
            theme: options.theme.unwrap_or_else(|| "light".to_string()),
            width: options.width.unwrap_or(800),
            height: options.height.unwrap_or(600),
            interactions: InteractionConfig::default(),
        };
        
        let interactive_viz = self.interactive(Some(interactive_opts))?;
        Ok(RenderResult::Interactive(interactive_viz))
    }

    /// Render for static file export
    fn render_file(&self, options: RenderOptions) -> GraphResult<RenderResult> {
        let filename = options.filename.ok_or_else(|| {
            GraphError::InvalidInput("filename is required for file backend".to_string())
        })?;

        let format = options.format.unwrap_or(ExportFormat::HTML);
        
        // Delegate to existing static_viz method
        let static_opts = StaticOptions {
            filename,
            format,
            layout: options.layout.unwrap_or(LayoutAlgorithm::ForceDirected { 
                charge: -30.0, 
                distance: 50.0, 
                iterations: 100 
            }),
            theme: options.theme.unwrap_or_else(|| "light".to_string()),
            dpi: options.dpi.unwrap_or(300),
            width: options.width.unwrap_or(800),
            height: options.height.unwrap_or(600),
        };
        
        let static_viz = self.static_viz(static_opts)?;
        Ok(RenderResult::Static(static_viz))
    }

    /// Render for self-contained HTML
    fn render_local(&self, options: RenderOptions) -> GraphResult<RenderResult> {
        // Generate self-contained HTML
        let html = self.generate_local_html(&options)?;
        Ok(RenderResult::Html(html))
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

    /// Generate static visualization export (PNG, SVG, PDF, HTML)
    pub fn static_viz(&self, options: StaticOptions) -> GraphResult<StaticViz> {
        match options.format {
            ExportFormat::HTML => {
                self.generate_static_html(&options)
            },
            ExportFormat::SVG => {
                self.generate_simple_svg(&options)
            },
            _ => Err(GraphError::NotImplemented { 
                feature: format!("{:?} export", options.format),
                tracking_issue: Some("https://github.com/anthropics/groggy/issues/viz-static".to_string())
            })
        }
    }
    
    /// Generate static HTML file with embedded graph data
    fn generate_static_html(&self, options: &StaticOptions) -> GraphResult<StaticViz> {
        use std::fs;
        
        // Get graph data from the data source
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let metadata = self.data_source.get_graph_metadata();
        
        // Convert to JSON
        let nodes_json = serde_json::to_string(&nodes)
            .map_err(|e| GraphError::internal(&format!("Failed to serialize nodes: {}", e), "generate_static_html"))?;
        let edges_json = serde_json::to_string(&edges)
            .map_err(|e| GraphError::internal(&format!("Failed to serialize edges: {}", e), "generate_static_html"))?;
        
        // Read the HTML template
        let html_template = self.get_html_template()?;
        
        // Replace template variables
        let html = html_template
            .replace("{{TITLE}}", "Graph Visualization")
            .replace("{{NODE_COUNT}}", &metadata.node_count.to_string())
            .replace("{{EDGE_COUNT}}", &metadata.edge_count.to_string())
            .replace("{{WIDTH}}", &options.width.to_string())
            .replace("{{HEIGHT}}", &options.height.to_string())
            .replace("{{LAYOUT}}", &format!("{:?}", options.layout).to_lowercase())
            .replace("{{THEME}}", &options.theme)
            .replace("{{NODES_JSON}}", &nodes_json)
            .replace("{{EDGES_JSON}}", &edges_json)
            .replace("{{USE_WEBSOCKET}}", "false");
        
        // Write to file
        fs::write(&options.filename, &html)
            .map_err(|e| GraphError::internal(&format!("Failed to write HTML file: {}", e), "generate_static_html"))?;
        
        Ok(StaticViz {
            file_path: options.filename.clone(),
            size_bytes: html.len(),
        })
    }
    
    /// Generate simple SVG export
    fn generate_simple_svg(&self, options: &StaticOptions) -> GraphResult<StaticViz> {
        use std::fs;
        
        // Get graph data and positions
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let positions = self.data_source.compute_layout(options.layout.clone());
        
        // Build SVG
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <style>
                    .node {{ fill: #007bff; stroke: #fff; stroke-width: 2px; }}
                    .edge {{ stroke: #999; stroke-width: 1px; }}
                    .label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
                </style>
            </defs>"#,
            options.width, options.height
        );
        
        // Draw edges
        for edge in &edges {
            if let (Some(src_pos), Some(dst_pos)) = (
                positions.iter().find(|p| p.node_id == edge.source),
                positions.iter().find(|p| p.node_id == edge.target)
            ) {
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="edge"/>"#,
                    src_pos.position.x, src_pos.position.y, 
                    dst_pos.position.x, dst_pos.position.y
                ));
            }
        }
        
        // Draw nodes
        for (node, pos) in nodes.iter().zip(positions.iter()) {
            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="8" class="node"/>"#,
                pos.position.x, pos.position.y
            ));
            
            if let Some(label) = &node.label {
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" class="label">{}</text>"#,
                    pos.position.x, pos.position.y - 15.0, label
                ));
            }
        }
        
        svg.push_str("</svg>");
        
        // Write to file
        fs::write(&options.filename, &svg)
            .map_err(|e| GraphError::internal(&format!("Failed to write SVG file: {}", e), "generate_simple_svg"))?;
        
        Ok(StaticViz {
            file_path: options.filename.clone(),
            size_bytes: svg.len(),
        })
    }
    
    /// Get HTML template with embedded CSS and JS
    fn get_html_template(&self) -> GraphResult<String> {
        // Simple template without complex JavaScript for now
        Ok(r###"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{TITLE}}</title>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 20px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls { text-align: center; margin: 20px 0; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        #graph-canvas { border: 1px solid #ddd; border-radius: 8px; display: block; margin: 0 auto; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .info { text-align: center; margin-top: 20px; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{TITLE}}</h1>
            <p><strong>{{NODE_COUNT}}</strong> nodes, <strong>{{EDGE_COUNT}}</strong> edges</p>
        </div>
        
        <div class="controls">
            <button onclick="resetView()">Reset View</button>
            <button onclick="toggleLayout()">Change Layout</button>
            <button onclick="exportSVG()">Export SVG</button>
        </div>
        
        <canvas id="graph-canvas" width="{{WIDTH}}" height="{{HEIGHT}}"></canvas>
        
        <div class="info">
            <p>Click and drag to pan • Scroll to zoom • Current layout: <span id="layout-name">{{LAYOUT}}</span></p>
            <p>Theme: {{THEME}} • Generated by Groggy</p>
        </div>
    </div>

    <script>
        // Static graph data (no WebSocket needed!)
        const USE_WEBSOCKET = {{USE_WEBSOCKET}};
        const graphData = {
            nodes: {{NODES_JSON}},
            edges: {{EDGES_JSON}},
            layout: "{{LAYOUT}}",
            theme: "{{THEME}}"
        };
        
        // Canvas setup
        const canvas = document.getElementById('graph-canvas');
        const ctx = canvas.getContext('2d');
        
        // State
        let positions = [];
        let camera = { x: 0, y: 0, zoom: 1 };
        let isDragging = false;
        let lastMouse = { x: 0, y: 0 };
        let currentLayout = graphData.layout;
        
        console.log('📊 Loaded graph:', graphData.nodes.length, 'nodes,', graphData.edges.length, 'edges');
        
        // Layout calculation
        function calculateLayout() {
            const width = canvas.width;
            const height = canvas.height;
            const padding = 50;
            positions = [];
            
            switch (currentLayout) {
                case 'circular':
                    calculateCircularLayout(width, height, padding);
                    break;
                case 'grid':
                    calculateGridLayout(width, height, padding);
                    break;
                case 'force-directed':
                default:
                    calculateForceLayout(width, height, padding);
                    break;
            }
        }
        
        function calculateCircularLayout(width, height, padding) {
            const centerX = width / 2;
            const centerY = height / 2;
            const radius = Math.min(width, height) / 2 - padding;
            
            graphData.nodes.forEach((node, i) => {
                const angle = (i * 2 * Math.PI) / graphData.nodes.length;
                positions.push({
                    id: node.id,
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle)
                });
            });
        }
        
        function calculateGridLayout(width, height, padding) {
            const cols = Math.ceil(Math.sqrt(graphData.nodes.length));
            const cellW = (width - 2 * padding) / cols;
            const cellH = (height - 2 * padding) / Math.ceil(graphData.nodes.length / cols);
            
            graphData.nodes.forEach((node, i) => {
                positions.push({
                    id: node.id,
                    x: padding + (i % cols) * cellW + cellW / 2,
                    y: padding + Math.floor(i / cols) * cellH + cellH / 2
                });
            });
        }
        
        function calculateForceLayout(width, height, padding) {
            // Simple force-directed layout
            const nodes = graphData.nodes.map((node, i) => ({
                id: node.id,
                x: padding + Math.random() * (width - 2 * padding),
                y: padding + Math.random() * (height - 2 * padding),
                vx: 0, vy: 0
            }));
            
            // Simulation
            for (let iter = 0; iter < 100; iter++) {
                // Repulsion
                for (let i = 0; i < nodes.length; i++) {
                    for (let j = i + 1; j < nodes.length; j++) {
                        const dx = nodes[j].x - nodes[i].x;
                        const dy = nodes[j].y - nodes[i].y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const force = 1000 / (dist * dist);
                        
                        nodes[i].vx -= force * dx / dist;
                        nodes[i].vy -= force * dy / dist;
                        nodes[j].vx += force * dx / dist;
                        nodes[j].vy += force * dy / dist;
                    }
                }
                
                // Attraction
                graphData.edges.forEach(edge => {
                    const src = nodes.find(n => n.id === edge.source);
                    const dst = nodes.find(n => n.id === edge.target);
                    if (src && dst) {
                        const dx = dst.x - src.x;
                        const dy = dst.y - src.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const force = dist * 0.01;
                        
                        src.vx += force * dx / dist;
                        src.vy += force * dy / dist;
                        dst.vx -= force * dx / dist;
                        dst.vy -= force * dy / dist;
                    }
                });
                
                // Update positions
                nodes.forEach(node => {
                    node.x += node.vx * 0.1;
                    node.y += node.vy * 0.1;
                    node.vx *= 0.9;
                    node.vy *= 0.9;
                    
                    // Bounds
                    node.x = Math.max(padding, Math.min(width - padding, node.x));
                    node.y = Math.max(padding, Math.min(height - padding, node.y));
                });
            }
            
            positions = nodes.map(n => ({ id: n.id, x: n.x, y: n.y }));
        }
        
        // Rendering
        function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(camera.x, camera.y);
            ctx.scale(camera.zoom, camera.zoom);
            
            // Edges
            ctx.strokeStyle = '#999';
            ctx.lineWidth = 1;
            graphData.edges.forEach(edge => {
                const src = positions.find(p => p.id === edge.source);
                const dst = positions.find(p => p.id === edge.target);
                if (src && dst) {
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.lineTo(dst.x, dst.y);
                    ctx.stroke();
                }
            });
            
            // Nodes
            positions.forEach(pos => {
                const node = graphData.nodes.find(n => n.id === pos.id);
                
                ctx.fillStyle = node.color || '#007bff';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 8, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                if (node.label) {
                    ctx.fillStyle = '#333';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(node.label, pos.x, pos.y - 15);
                }
            });
            
            ctx.restore();
        }
        
        // Event handling
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            lastMouse = { x: e.clientX, y: e.clientY };
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (isDragging) {
                camera.x += e.clientX - lastMouse.x;
                camera.y += e.clientY - lastMouse.y;
                lastMouse = { x: e.clientX, y: e.clientY };
                render();
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            camera.zoom *= zoomFactor;
            camera.zoom = Math.max(0.1, Math.min(5, camera.zoom));
            render();
        });
        
        // Controls
        function resetView() {
            camera = { x: 0, y: 0, zoom: 1 };
            render();
        }
        
        function toggleLayout() {
            const layouts = ['force-directed', 'circular', 'grid'];
            const idx = layouts.indexOf(currentLayout);
            currentLayout = layouts[(idx + 1) % layouts.length];
            document.getElementById('layout-name').textContent = currentLayout;
            calculateLayout();
            render();
        }
        
        function exportSVG() {
            // Simple SVG export
            const svgContent = generateSVG();
            const blob = new Blob([svgContent], { type: 'image/svg+xml' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'graph.svg';
            a.click();
            URL.revokeObjectURL(url);
        }
        
        function generateSVG() {
            let svg = `<svg width="${canvas.width}" height="${canvas.height}" xmlns="http://www.w3.org/2000/svg">`;
            
            // Edges
            graphData.edges.forEach(edge => {
                const src = positions.find(p => p.id === edge.source);
                const dst = positions.find(p => p.id === edge.target);
                if (src && dst) {
                    svg += `<line x1="${src.x}" y1="${src.y}" x2="${dst.x}" y2="${dst.y}" stroke="#999" stroke-width="1"/>`;
                }
            });
            
            // Nodes
            positions.forEach(pos => {
                const node = graphData.nodes.find(n => n.id === pos.id);
                svg += `<circle cx="${pos.x}" cy="${pos.y}" r="8" fill="${node.color || '#007bff'}" stroke="#fff" stroke-width="2"/>`;
                if (node.label) {
                    svg += `<text x="${pos.x}" y="${pos.y - 15}" text-anchor="middle" font-size="12" fill="#333">${node.label}</text>`;
                }
            });
            
            svg += '</svg>';
            return svg;
        }
        
        // Initialize
        calculateLayout();
        render();
        
        console.log('🚀 Graph visualization ready! Layout:', currentLayout, 'Theme:', graphData.theme);
    </script>
</body>
</html>"###.to_string())
    }

    /// Generate Jupyter-optimized HTML for embedding
    fn generate_jupyter_html(&self, options: &RenderOptions) -> GraphResult<String> {
        // Get graph data from the data source
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let metadata = self.data_source.get_graph_metadata();
        
        // Convert to JSON
        let nodes_json = serde_json::to_string(&nodes)
            .map_err(|e| GraphError::internal(&format!("Failed to serialize nodes: {}", e), "generate_jupyter_html"))?;
        let edges_json = serde_json::to_string(&edges)
            .map_err(|e| GraphError::internal(&format!("Failed to serialize edges: {}", e), "generate_jupyter_html"))?;
        
        let width = options.width.unwrap_or(800);
        let height = options.height.unwrap_or(600);
        let canvas_id = format!("viz-canvas-{}", std::ptr::addr_of!(*self) as usize);
        
        // Generate Jupyter-optimized HTML (simpler than full template for embedding)
        let html = format!(r###"
<div style="width: {}px; height: {}px; border: 1px solid #ddd; position: relative; background: #fafafa;">
    <canvas id="{}" width="{}" height="{}" 
            style="display: block; background: white;"></canvas>
    <div style="position: absolute; top: 5px; right: 5px; font-size: 12px; color: #666;">
        {} nodes, {} edges
    </div>
    <script>
    // Jupyter-optimized visualization with embedded data
    const nodes = {};
    const edges = {};
    console.log('Jupyter viz loaded:', nodes.length, 'nodes', edges.length, 'edges');
    </script>
</div>
"###, width, height, canvas_id, width, height, metadata.node_count, metadata.edge_count, 
             nodes_json, edges_json);
        
        Ok(html)
    }

    /// Generate self-contained HTML for local use
    fn generate_local_html(&self, options: &RenderOptions) -> GraphResult<String> {
        // For local HTML, we can reuse the existing static HTML generation
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let metadata = self.data_source.get_graph_metadata();
        
        // Convert to JSON
        let nodes_json = serde_json::to_string(&nodes)
            .map_err(|e| GraphError::internal(&format!("Failed to serialize nodes: {}", e), "generate_local_html"))?;
        let edges_json = serde_json::to_string(&edges)
            .map_err(|e| GraphError::internal(&format!("Failed to serialize edges: {}", e), "generate_local_html"))?;
        
        // Read the HTML template
        let html_template = self.get_html_template()?;
        
        let width = options.width.unwrap_or(800);
        let height = options.height.unwrap_or(600);
        let layout = options.layout.as_ref().map(|l| format!("{:?}", l).to_lowercase()).unwrap_or_else(|| "force-directed".to_string());
        let theme = options.theme.as_ref().map(|s| s.as_str()).unwrap_or("light");
        let title = options.title.as_ref().map(|s| s.as_str()).unwrap_or("Graph Visualization");
        
        // Replace template variables
        let html = html_template
            .replace("{{TITLE}}", title)
            .replace("{{NODE_COUNT}}", &metadata.node_count.to_string())
            .replace("{{EDGE_COUNT}}", &metadata.edge_count.to_string())
            .replace("{{WIDTH}}", &width.to_string())
            .replace("{{HEIGHT}}", &height.to_string())
            .replace("{{LAYOUT}}", &layout)
            .replace("{{THEME}}", theme)
            .replace("{{NODES_JSON}}", &nodes_json)
            .replace("{{EDGES_JSON}}", &edges_json)
            .replace("{{USE_WEBSOCKET}}", "false");
        
        Ok(html)
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
    HTML,
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