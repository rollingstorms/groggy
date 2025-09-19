//! Unified rendering pipeline for visualization
//!
//! Provides rendering abstractions that work across all backends (SVG, HTML, Canvas, etc.)

use std::collections::HashMap;
use crate::viz::core::frame::{VizFrame, FrameNode, FrameEdge, NodeShape, LineStyle};
use crate::errors::GraphResult;

/// Unified rendering engine that outputs to different formats
pub struct RenderingEngine {
    /// Rendering configuration
    pub config: RenderingConfig,
    
    /// Theme settings
    pub theme: Theme,
    
    /// Custom styles
    pub custom_styles: HashMap<String, String>,
}

/// Rendering configuration
#[derive(Debug, Clone)]
pub struct RenderingConfig {
    /// Output format
    pub format: RenderFormat,
    
    /// Canvas dimensions
    pub width: f64,
    pub height: f64,
    
    /// Whether to include animations
    pub include_animations: bool,
    
    /// Whether to include interactions
    pub include_interactions: bool,
    
    /// Background color
    pub background_color: String,
    
    /// Whether to show grid
    pub show_grid: bool,
    
    /// Grid settings
    pub grid_config: GridConfig,
}

/// Supported rendering formats
#[derive(Debug, Clone)]
pub enum RenderFormat {
    /// SVG for vector graphics
    SVG,
    
    /// HTML with embedded SVG
    HTML,
    
    /// Canvas-based rendering commands
    Canvas,
    
    /// JSON data for frontend consumption
    JSON,
    
    /// PNG raster image (requires external renderer)
    PNG { scale: f64 },
    
    /// PDF vector document (requires external renderer)
    PDF,
}

/// Theme configuration
#[derive(Debug, Clone)]
pub struct Theme {
    /// Theme name
    pub name: String,
    
    /// Node color palette
    pub node_colors: Vec<String>,
    
    /// Edge color palette
    pub edge_colors: Vec<String>,
    
    /// Background color
    pub background_color: String,
    
    /// Text color
    pub text_color: String,
    
    /// Grid color
    pub grid_color: String,
    
    /// Selection color
    pub selection_color: String,
    
    /// Hover color
    pub hover_color: String,
}

/// Grid configuration
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Grid spacing
    pub spacing: f64,
    
    /// Grid line width
    pub line_width: f64,
    
    /// Grid opacity
    pub opacity: f64,
    
    /// Grid color
    pub color: String,
}

/// Rendering output
pub struct RenderOutput {
    /// Rendered content
    pub content: String,
    
    /// Output format
    pub format: RenderFormat,
    
    /// Additional metadata
    pub metadata: RenderMetadata,
}

/// Rendering metadata
#[derive(Debug, Clone)]
pub struct RenderMetadata {
    /// Rendering time in milliseconds
    pub render_time_ms: f64,
    
    /// Output size in bytes
    pub size_bytes: usize,
    
    /// Number of nodes rendered
    pub node_count: usize,
    
    /// Number of edges rendered
    pub edge_count: usize,
    
    /// Canvas dimensions
    pub dimensions: (f64, f64),
}

impl Default for RenderingConfig {
    fn default() -> Self {
        Self {
            format: RenderFormat::SVG,
            width: 800.0,
            height: 600.0,
            include_animations: true,
            include_interactions: true,
            background_color: "#ffffff".to_string(),
            show_grid: false,
            grid_config: GridConfig::default(),
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            spacing: 50.0,
            line_width: 1.0,
            opacity: 0.1,
            color: "#cccccc".to_string(),
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            node_colors: vec![
                "#4CAF50".to_string(), // Green
                "#2196F3".to_string(), // Blue
                "#FF9800".to_string(), // Orange
                "#9C27B0".to_string(), // Purple
                "#F44336".to_string(), // Red
                "#00BCD4".to_string(), // Cyan
                "#FFEB3B".to_string(), // Yellow
                "#795548".to_string(), // Brown
            ],
            edge_colors: vec![
                "#999999".to_string(), // Gray
                "#666666".to_string(), // Dark gray
                "#cccccc".to_string(), // Light gray
            ],
            background_color: "#ffffff".to_string(),
            text_color: "#333333".to_string(),
            grid_color: "#eeeeee".to_string(),
            selection_color: "#ff4444".to_string(),
            hover_color: "#4444ff".to_string(),
        }
    }
}

impl RenderingEngine {
    /// Create new rendering engine
    pub fn new(config: RenderingConfig) -> Self {
        Self {
            config,
            theme: Theme::default(),
            custom_styles: HashMap::new(),
        }
    }
    
    /// Render a frame to the specified format
    pub fn render(&self, frame: &VizFrame) -> GraphResult<RenderOutput> {
        let start_time = std::time::Instant::now();
        
        let content = match self.config.format {
            RenderFormat::SVG => self.render_svg(frame)?,
            RenderFormat::HTML => self.render_html(frame)?,
            RenderFormat::Canvas => self.render_canvas_commands(frame)?,
            RenderFormat::JSON => self.render_json(frame)?,
            RenderFormat::PNG { .. } => {
                return Err(crate::errors::GraphError::InvalidInput(
                    "PNG rendering requires external renderer".to_string()
                ));
            }
            RenderFormat::PDF => {
                return Err(crate::errors::GraphError::InvalidInput(
                    "PDF rendering requires external renderer".to_string()
                ));
            }
        };
        
        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let content_size = content.len();
        
        Ok(RenderOutput {
            content,
            format: self.config.format.clone(),
            metadata: RenderMetadata {
                render_time_ms: render_time,
                size_bytes: content_size,
                node_count: frame.nodes.len(),
                edge_count: frame.edges.len(),
                dimensions: (self.config.width, self.config.height),
            },
        })
    }
    
    /// Render frame as SVG
    fn render_svg(&self, frame: &VizFrame) -> GraphResult<String> {
        let mut svg = String::new();
        
        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));
        
        // Background
        svg.push_str(&format!(
            r#"<rect width="100%" height="100%" fill="{}"/>"#,
            self.config.background_color
        ));
        
        // Grid (if enabled)
        if self.config.show_grid {
            svg.push_str(&self.render_svg_grid());
        }
        
        // Render edges first (so they appear behind nodes)
        for edge in &frame.edges {
            svg.push_str(&self.render_svg_edge(edge, frame)?);
        }
        
        // Render nodes
        for node in &frame.nodes {
            svg.push_str(&self.render_svg_node(node)?);
        }
        
        // Interactions (if enabled)
        if self.config.include_interactions {
            svg.push_str(&self.render_svg_interactions());
        }
        
        // Animations (if enabled)
        if self.config.include_animations {
            svg.push_str(&self.render_svg_animations(frame));
        }
        
        svg.push_str("</svg>");
        
        Ok(svg)
    }
    
    /// Render SVG grid
    fn render_svg_grid(&self) -> String {
        let mut grid = String::new();
        let spacing = self.config.grid_config.spacing;
        
        // Vertical lines
        let mut x = 0.0;
        while x <= self.config.width {
            grid.push_str(&format!(
                r#"<line x1="{}" y1="0" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="{}"/>"#,
                x, x, self.config.height,
                self.config.grid_config.color,
                self.config.grid_config.line_width,
                self.config.grid_config.opacity
            ));
            x += spacing;
        }
        
        // Horizontal lines
        let mut y = 0.0;
        while y <= self.config.height {
            grid.push_str(&format!(
                r#"<line x1="0" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="{}"/>"#,
                y, self.config.width, y,
                self.config.grid_config.color,
                self.config.grid_config.line_width,
                self.config.grid_config.opacity
            ));
            y += spacing;
        }
        
        grid
    }
    
    /// Render SVG node
    fn render_svg_node(&self, node: &FrameNode) -> GraphResult<String> {
        let mut node_svg = String::new();
        
        // Adjust position to canvas center
        let x = node.position.x + self.config.width / 2.0;
        let y = node.position.y + self.config.height / 2.0;
        
        // Node shape
        match node.visual.shape {
            NodeShape::Circle => {
                node_svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}"/>"#,
                    x, y, node.visual.radius,
                    node.visual.fill_color,
                    node.visual.stroke_color,
                    node.visual.stroke_width,
                    node.visual.opacity
                ));
            }
            NodeShape::Square => {
                let size = node.visual.radius * 2.0;
                node_svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}"/>"#,
                    x - node.visual.radius, y - node.visual.radius, size, size,
                    node.visual.fill_color,
                    node.visual.stroke_color,
                    node.visual.stroke_width,
                    node.visual.opacity
                ));
            }
            NodeShape::Triangle => {
                let r = node.visual.radius;
                let points = format!("{},{} {},{} {},{}",
                    x, y - r,
                    x - r * 0.866, y + r * 0.5,
                    x + r * 0.866, y + r * 0.5
                );
                node_svg.push_str(&format!(
                    r#"<polygon points="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}"/>"#,
                    points,
                    node.visual.fill_color,
                    node.visual.stroke_color,
                    node.visual.stroke_width,
                    node.visual.opacity
                ));
            }
            _ => {
                // Default to circle for other shapes
                node_svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}" opacity="{}"/>"#,
                    x, y, node.visual.radius,
                    node.visual.fill_color,
                    node.visual.stroke_color,
                    node.visual.stroke_width,
                    node.visual.opacity
                ));
            }
        }
        
        // Node label
        if let Some(label) = &node.visual.label {
            if node.visual.label_style.visible {
                node_svg.push_str(&format!(
                    r#"<text x="{}" y="{}" text-anchor="middle" font-family="{}" font-size="{}" fill="{}">{}</text>"#,
                    x, y + 4.0, // Slight offset for better centering
                    node.visual.label_style.font_family,
                    node.visual.label_style.font_size,
                    node.visual.label_style.color,
                    Self::escape_xml(label)
                ));
            }
        }
        
        // Interaction highlights
        if node.interaction_state.is_selected {
            node_svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="{}" fill="none" stroke="{}" stroke-width="3" opacity="0.8"/>"#,
                x, y, node.visual.radius + 5.0,
                self.theme.selection_color
            ));
        }
        
        if node.interaction_state.is_hovered {
            node_svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="{}" fill="none" stroke="{}" stroke-width="2" opacity="0.6"/>"#,
                x, y, node.visual.radius + 3.0,
                self.theme.hover_color
            ));
        }
        
        Ok(node_svg)
    }
    
    /// Render SVG edge
    fn render_svg_edge(&self, edge: &FrameEdge, frame: &VizFrame) -> GraphResult<String> {
        // Find source and target nodes
        let source_node = frame.nodes.iter().find(|n| n.id == edge.source);
        let target_node = frame.nodes.iter().find(|n| n.id == edge.target);
        
        if let (Some(source), Some(target)) = (source_node, target_node) {
            let mut edge_svg = String::new();
            
            // Adjust positions to canvas center
            let x1 = source.position.x + self.config.width / 2.0;
            let y1 = source.position.y + self.config.height / 2.0;
            let x2 = target.position.x + self.config.width / 2.0;
            let y2 = target.position.y + self.config.height / 2.0;
            
            // Line style
            let stroke_dasharray = match edge.visual.line_style {
                LineStyle::Solid => "",
                LineStyle::Dashed => "stroke-dasharray=\"5,5\"",
                LineStyle::Dotted => "stroke-dasharray=\"2,2\"",
                LineStyle::DashDot => "stroke-dasharray=\"5,2,2,2\"",
            };
            
            edge_svg.push_str(&format!(
                r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" opacity="{}" {}/>"#,
                x1, y1, x2, y2,
                edge.visual.stroke_color,
                edge.visual.stroke_width,
                edge.visual.opacity,
                stroke_dasharray
            ));
            
            // Arrow markers (if any)
            if edge.visual.markers.end.is_some() {
                let angle = (y2 - y1).atan2(x2 - x1);
                let marker_x = x2 - (target.visual.radius + 5.0) * angle.cos();
                let marker_y = y2 - (target.visual.radius + 5.0) * angle.sin();
                
                edge_svg.push_str(&format!(
                    r#"<polygon points="{},{} {},{} {},{}" fill="{}"/>"#,
                    marker_x, marker_y,
                    marker_x - 8.0 * angle.cos() + 4.0 * angle.sin(),
                    marker_y - 8.0 * angle.sin() - 4.0 * angle.cos(),
                    marker_x - 8.0 * angle.cos() - 4.0 * angle.sin(),
                    marker_y - 8.0 * angle.sin() + 4.0 * angle.cos(),
                    edge.visual.stroke_color
                ));
            }
            
            // Edge label
            if let Some(label) = &edge.visual.label {
                if edge.visual.label_style.visible {
                    let mid_x = (x1 + x2) / 2.0;
                    let mid_y = (y1 + y2) / 2.0;
                    
                    edge_svg.push_str(&format!(
                        r#"<text x="{}" y="{}" text-anchor="middle" font-family="{}" font-size="{}" fill="{}">{}</text>"#,
                        mid_x, mid_y,
                        edge.visual.label_style.font_family,
                        edge.visual.label_style.font_size,
                        edge.visual.label_style.color,
                        Self::escape_xml(label)
                    ));
                }
            }
            
            Ok(edge_svg)
        } else {
            Ok(String::new()) // Skip edges with missing nodes
        }
    }
    
    /// Render frame as HTML
    fn render_html(&self, frame: &VizFrame) -> GraphResult<String> {
        let svg_content = self.render_svg(frame)?;
        
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Groggy Graph Visualization</title>
    <style>
        body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
        .viz-container {{ border: 1px solid #ddd; display: inline-block; }}
        .stats {{ margin-top: 10px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>Graph Visualization</h1>
    <div class="viz-container">
        {}
    </div>
    <div class="stats">
        {} nodes, {} edges | Generated at {}
    </div>
</body>
</html>"#,
            svg_content,
            frame.nodes.len(),
            frame.edges.len(),
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );
        
        Ok(html)
    }
    
    /// Render frame as JSON
    fn render_json(&self, frame: &VizFrame) -> GraphResult<String> {
        serde_json::to_string_pretty(frame)
            .map_err(|e| crate::errors::GraphError::InvalidInput(format!("JSON serialization failed: {}", e)))
    }
    
    /// Render canvas drawing commands
    fn render_canvas_commands(&self, frame: &VizFrame) -> GraphResult<String> {
        let mut commands = Vec::new();
        
        // Canvas setup
        commands.push(format!("clearRect(0, 0, {}, {})", self.config.width, self.config.height));
        commands.push(format!("fillStyle = '{}'", self.config.background_color));
        commands.push(format!("fillRect(0, 0, {}, {})", self.config.width, self.config.height));
        
        // Grid (if enabled)
        if self.config.show_grid {
            commands.extend(self.render_canvas_grid());
        }
        
        // Render edges
        for edge in &frame.edges {
            commands.extend(self.render_canvas_edge(edge, frame)?);
        }
        
        // Render nodes
        for node in &frame.nodes {
            commands.extend(self.render_canvas_node(node)?);
        }
        
        Ok(commands.join("\n"))
    }
    
    /// Render canvas grid commands
    fn render_canvas_grid(&self) -> Vec<String> {
        let mut commands = Vec::new();
        let spacing = self.config.grid_config.spacing;
        
        commands.push(format!("strokeStyle = '{}'", self.config.grid_config.color));
        commands.push(format!("lineWidth = {}", self.config.grid_config.line_width));
        commands.push(format!("globalAlpha = {}", self.config.grid_config.opacity));
        commands.push("beginPath()".to_string());
        
        // Vertical lines
        let mut x = 0.0;
        while x <= self.config.width {
            commands.push(format!("moveTo({}, 0)", x));
            commands.push(format!("lineTo({}, {})", x, self.config.height));
            x += spacing;
        }
        
        // Horizontal lines
        let mut y = 0.0;
        while y <= self.config.height {
            commands.push(format!("moveTo(0, {})", y));
            commands.push(format!("lineTo({}, {})", self.config.width, y));
            y += spacing;
        }
        
        commands.push("stroke()".to_string());
        commands.push("globalAlpha = 1.0".to_string());
        
        commands
    }
    
    /// Render canvas node commands
    fn render_canvas_node(&self, node: &FrameNode) -> GraphResult<Vec<String>> {
        let mut commands = Vec::new();
        
        let x = node.position.x + self.config.width / 2.0;
        let y = node.position.y + self.config.height / 2.0;
        
        commands.push(format!("globalAlpha = {}", node.visual.opacity));
        commands.push(format!("fillStyle = '{}'", node.visual.fill_color));
        commands.push(format!("strokeStyle = '{}'", node.visual.stroke_color));
        commands.push(format!("lineWidth = {}", node.visual.stroke_width));
        
        // Draw shape
        match node.visual.shape {
            NodeShape::Circle => {
                commands.push("beginPath()".to_string());
                commands.push(format!("arc({}, {}, {}, 0, 2 * Math.PI)", x, y, node.visual.radius));
                commands.push("fill()".to_string());
                commands.push("stroke()".to_string());
            }
            NodeShape::Square => {
                let size = node.visual.radius * 2.0;
                commands.push(format!("fillRect({}, {}, {}, {})", 
                    x - node.visual.radius, y - node.visual.radius, size, size));
                commands.push(format!("strokeRect({}, {}, {}, {})", 
                    x - node.visual.radius, y - node.visual.radius, size, size));
            }
            _ => {
                // Default to circle
                commands.push("beginPath()".to_string());
                commands.push(format!("arc({}, {}, {}, 0, 2 * Math.PI)", x, y, node.visual.radius));
                commands.push("fill()".to_string());
                commands.push("stroke()".to_string());
            }
        }
        
        // Label
        if let Some(label) = &node.visual.label {
            if node.visual.label_style.visible {
                commands.push(format!("fillStyle = '{}'", node.visual.label_style.color));
                commands.push(format!("font = '{}px {}'", 
                    node.visual.label_style.font_size, node.visual.label_style.font_family));
                commands.push("textAlign = 'center'".to_string());
                commands.push("textBaseline = 'middle'".to_string());
                commands.push(format!("fillText('{}', {}, {})", label, x, y));
            }
        }
        
        commands.push("globalAlpha = 1.0".to_string());
        
        Ok(commands)
    }
    
    /// Render canvas edge commands
    fn render_canvas_edge(&self, edge: &FrameEdge, frame: &VizFrame) -> GraphResult<Vec<String>> {
        let mut commands = Vec::new();
        
        // Find source and target nodes
        let source_node = frame.nodes.iter().find(|n| n.id == edge.source);
        let target_node = frame.nodes.iter().find(|n| n.id == edge.target);
        
        if let (Some(source), Some(target)) = (source_node, target_node) {
            let x1 = source.position.x + self.config.width / 2.0;
            let y1 = source.position.y + self.config.height / 2.0;
            let x2 = target.position.x + self.config.width / 2.0;
            let y2 = target.position.y + self.config.height / 2.0;
            
            commands.push(format!("globalAlpha = {}", edge.visual.opacity));
            commands.push(format!("strokeStyle = '{}'", edge.visual.stroke_color));
            commands.push(format!("lineWidth = {}", edge.visual.stroke_width));
            
            // Set line dash pattern
            match edge.visual.line_style {
                LineStyle::Solid => commands.push("setLineDash([])".to_string()),
                LineStyle::Dashed => commands.push("setLineDash([5, 5])".to_string()),
                LineStyle::Dotted => commands.push("setLineDash([2, 2])".to_string()),
                LineStyle::DashDot => commands.push("setLineDash([5, 2, 2, 2])".to_string()),
            }
            
            commands.push("beginPath()".to_string());
            commands.push(format!("moveTo({}, {})", x1, y1));
            commands.push(format!("lineTo({}, {})", x2, y2));
            commands.push("stroke()".to_string());
            commands.push("setLineDash([])".to_string()); // Reset line dash
            commands.push("globalAlpha = 1.0".to_string());
        }
        
        Ok(commands)
    }
    
    /// Render SVG interactions (placeholder)
    fn render_svg_interactions(&self) -> String {
        // Placeholder for interaction handlers
        r#"<script><![CDATA[
            // Interaction handlers would go here
            console.log('Groggy interactive visualization loaded');
        ]]></script>"#.to_string()
    }
    
    /// Render SVG animations (placeholder)
    fn render_svg_animations(&self, _frame: &VizFrame) -> String {
        // Placeholder for animations
        if self.config.include_animations {
            r#"<style>
                .groggy-node { transition: all 0.3s ease; }
                .groggy-edge { transition: all 0.3s ease; }
            </style>"#.to_string()
        } else {
            String::new()
        }
    }
    
    /// Escape XML special characters
    fn escape_xml(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }
    
    /// Set theme
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }
    
    /// Add custom style
    pub fn add_custom_style(&mut self, key: String, value: String) {
        self.custom_styles.insert(key, value);
    }
}

/// Predefined themes
impl Theme {
    /// Dark theme
    pub fn dark() -> Self {
        Self {
            name: "dark".to_string(),
            node_colors: vec![
                "#81C784".to_string(), // Light green
                "#64B5F6".to_string(), // Light blue
                "#FFB74D".to_string(), // Light orange
                "#BA68C8".to_string(), // Light purple
                "#E57373".to_string(), // Light red
                "#4DD0E1".to_string(), // Light cyan
                "#FFF176".to_string(), // Light yellow
                "#A1887F".to_string(), // Light brown
            ],
            edge_colors: vec![
                "#666666".to_string(),
                "#888888".to_string(),
                "#555555".to_string(),
            ],
            background_color: "#1e1e1e".to_string(),
            text_color: "#ffffff".to_string(),
            grid_color: "#333333".to_string(),
            selection_color: "#ff6666".to_string(),
            hover_color: "#6666ff".to_string(),
        }
    }
    
    /// High contrast theme
    pub fn high_contrast() -> Self {
        Self {
            name: "high_contrast".to_string(),
            node_colors: vec![
                "#000000".to_string(), // Black
                "#ffffff".to_string(), // White
                "#ff0000".to_string(), // Red
                "#0000ff".to_string(), // Blue
                "#00ff00".to_string(), // Green
                "#ffff00".to_string(), // Yellow
                "#ff00ff".to_string(), // Magenta
                "#00ffff".to_string(), // Cyan
            ],
            edge_colors: vec![
                "#000000".to_string(),
                "#666666".to_string(),
            ],
            background_color: "#ffffff".to_string(),
            text_color: "#000000".to_string(),
            grid_color: "#cccccc".to_string(),
            selection_color: "#ff0000".to_string(),
            hover_color: "#0000ff".to_string(),
        }
    }
}