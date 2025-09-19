//! Jupyter adapter that wraps the unified core engine
//!
//! Per UNIFIED_VIZ_MIGRATION_PLAN.md: "JupyterAdapter wraps the core
//! VizEngine for widget integration with Python bridge"

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::errors::GraphResult;
use crate::viz::core::{VizEngine, VizConfig, VizFrame};
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge};
use super::traits::{
    VizAdapter, AdapterResult, AdapterConfig, JupyterResult, JupyterMetadata, AdapterError
};

/// Jupyter adapter that wraps VizEngine for notebook widget integration
pub struct JupyterAdapter {
    /// Unified core engine (single source of truth)
    core: VizEngine,
    
    /// Python bridge for widget communication
    python_bridge: Option<PyO3Bridge>,
    
    /// Widget state
    widget_state: JupyterWidgetState,
    
    /// Current widget HTML
    cached_html: Option<String>,
    
    /// Widget configuration
    widget_config: JupyterWidgetConfig,
}

/// Python bridge for widget communication
#[derive(Debug)]
pub struct PyO3Bridge {
    /// Widget model data
    pub model_data: HashMap<String, serde_json::Value>,
    
    /// Widget ID
    pub widget_id: String,
    
    /// Communication channel state
    pub comm_open: bool,
}

/// Jupyter widget state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterWidgetState {
    /// Widget ID
    pub widget_id: String,
    
    /// Current model version
    pub model_version: u64,
    
    /// Widget title
    pub title: String,
    
    /// Widget dimensions
    pub width: f64,
    pub height: f64,
    
    /// Interactive features enabled
    pub interactive: bool,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Configuration for Jupyter widget behavior
#[derive(Debug, Clone)]
pub struct JupyterWidgetConfig {
    /// Auto-update when data changes
    pub auto_update: bool,
    
    /// Include interactive features
    pub interactive: bool,
    
    /// Widget theme
    pub theme: WidgetTheme,
    
    /// Custom CSS styles
    pub custom_css: String,
    
    /// Animation settings
    pub animations: AnimationConfig,
}

#[derive(Debug, Clone)]
pub enum WidgetTheme {
    Default,
    Dark,
    Light,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct AnimationConfig {
    pub enabled: bool,
    pub duration_ms: u32,
    pub easing: String,
}

impl Default for JupyterWidgetConfig {
    fn default() -> Self {
        Self {
            auto_update: true,
            interactive: true,
            theme: WidgetTheme::Default,
            custom_css: String::new(),
            animations: AnimationConfig {
                enabled: true,
                duration_ms: 300,
                easing: "ease-in-out".to_string(),
            },
        }
    }
}

impl JupyterAdapter {
    /// Create new Jupyter adapter with unified core engine
    pub fn new(config: AdapterConfig) -> Self {
        // Convert adapter config to VizEngine config
        let viz_config = VizConfig {
            width: config.width,
            height: config.height,
            physics_enabled: config.physics_enabled,
            continuous_physics: false, // Static for widgets unless explicitly animated
            target_fps: 30.0, // Lower FPS for widget efficiency
            interactions_enabled: config.interactions_enabled,
            auto_fit: config.auto_fit,
            fit_padding: 20.0, // Smaller padding for widgets
        };
        
        // Create unified core engine
        let core = VizEngine::new(viz_config);
        
        // Generate unique widget ID
        let widget_id = format!("groggy_widget_{}", fastrand::u64(..));
        
        let widget_state = JupyterWidgetState {
            widget_id: widget_id.clone(),
            model_version: 0,
            title: "Groggy Graph Visualization".to_string(),
            width: config.width,
            height: config.height,
            interactive: config.interactions_enabled,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        
        Self {
            core,
            python_bridge: None,
            widget_state,
            cached_html: None,
            widget_config: JupyterWidgetConfig::default(),
        }
    }
    
    /// Initialize Python bridge for widget communication
    pub fn initialize_python_bridge(&mut self) -> GraphResult<()> {
        let bridge = PyO3Bridge {
            model_data: HashMap::new(),
            widget_id: self.widget_state.widget_id.clone(),
            comm_open: false,
        };
        
        self.python_bridge = Some(bridge);
        Ok(())
    }
    
    /// Generate widget HTML with embedded visualization
    pub fn generate_widget_html(&mut self) -> GraphResult<String> {
        // Get current frame from unified core engine
        let frame = self.core.update()?;
        
        // Render frame to SVG using unified rendering pipeline
        let render_output = self.core.render(&frame)?;
        
        // Generate complete widget HTML
        let html = self.create_widget_html(&render_output.content, &frame)?;
        
        // Cache the HTML
        self.cached_html = Some(html.clone());
        
        // Update widget state
        self.widget_state.model_version += 1;
        self.widget_state.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        Ok(html)
    }
    
    /// Create complete widget HTML with styling and interactions
    fn create_widget_html(&self, svg_content: &str, frame: &VizFrame) -> GraphResult<String> {
        let widget_id = &self.widget_state.widget_id;
        let theme_css = self.get_theme_css();
        let interaction_js = self.get_interaction_javascript();
        
        let html = format!(
            r#"
<div id="{widget_id}" class="groggy-widget" style="
    width: {width}px;
    height: {height}px;
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    background: #f9f9f9;
    font-family: Arial, sans-serif;
    position: relative;
    overflow: hidden;
">
    <div class="groggy-header" style="
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <span>{title}</span>
        <div class="groggy-stats" style="font-size: 12px; color: #666;">
            {node_count} nodes, {edge_count} edges
        </div>
    </div>
    
    <div class="groggy-content" style="
        width: 100%;
        height: calc(100% - 40px);
        border: 1px solid #ddd;
        background: white;
        border-radius: 4px;
        overflow: hidden;
        position: relative;
    ">
        {svg_content}
    </div>
    
    {theme_css}
    {interaction_js}
</div>
"#,
            widget_id = widget_id,
            width = self.widget_state.width,
            height = self.widget_state.height,
            title = self.widget_state.title,
            node_count = frame.nodes.len(),
            edge_count = frame.edges.len(),
            svg_content = svg_content,
            theme_css = theme_css,
            interaction_js = interaction_js,
        );
        
        Ok(html)
    }
    
    /// Get CSS for the current theme
    fn get_theme_css(&self) -> String {
        let base_css = r#"
<style>
.groggy-widget .groggy-content svg {
    width: 100%;
    height: 100%;
}

.groggy-widget .groggy-node {
    cursor: pointer;
    transition: all 0.2s ease;
}

.groggy-widget .groggy-node:hover {
    filter: brightness(1.1);
}

.groggy-widget .groggy-edge {
    transition: all 0.2s ease;
}
"#;
        
        let theme_css = match &self.widget_config.theme {
            WidgetTheme::Dark => r#"
.groggy-widget {
    background: #2d2d2d !important;
    border-color: #666 !important;
    color: white !important;
}

.groggy-widget .groggy-content {
    background: #1e1e1e !important;
    border-color: #444 !important;
}
"#,
            WidgetTheme::Light => r#"
.groggy-widget {
    background: #ffffff !important;
    border-color: #e0e0e0 !important;
}

.groggy-widget .groggy-content {
    background: #fafafa !important;
}
"#,
            WidgetTheme::Custom(css) => css.as_str(),
            WidgetTheme::Default => "",
        };
        
        format!("{}\n{}\n{}\n</style>", 
                base_css, 
                theme_css, 
                self.widget_config.custom_css)
    }
    
    /// Get JavaScript for widget interactions
    fn get_interaction_javascript(&self) -> String {
        if !self.widget_config.interactive {
            return String::new();
        }
        
        format!(
            r#"
<script>
(function() {{
    const widget = document.getElementById('{widget_id}');
    if (!widget) return;
    
    // Add interaction handlers
    const svg = widget.querySelector('svg');
    if (!svg) return;
    
    // Node interaction handlers
    const nodes = svg.querySelectorAll('circle, rect, polygon');
    nodes.forEach(node => {{
        node.addEventListener('click', function(e) {{
            // Handle node selection
            const nodeId = this.getAttribute('data-node-id') || 'unknown';
            console.log('Node clicked:', nodeId);
            
            // Add selection styling
            this.style.stroke = '#ff4444';
            this.style.strokeWidth = '3';
        }});
        
        node.addEventListener('mouseenter', function(e) {{
            // Handle node hover
            this.style.opacity = '0.8';
        }});
        
        node.addEventListener('mouseleave', function(e) {{
            // Remove hover styling
            this.style.opacity = '1';
        }});
    }});
    
    // Add zoom/pan if enabled
    if ({interactive}) {{
        let isDragging = false;
        let startX, startY;
        
        svg.addEventListener('mousedown', function(e) {{
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            this.style.cursor = 'grabbing';
        }});
        
        document.addEventListener('mousemove', function(e) {{
            if (!isDragging) return;
            
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            // Apply pan transformation
            const currentTransform = svg.style.transform || '';
            svg.style.transform = currentTransform + ` translate(${{deltaX}}px, ${{deltaY}}px)`;
            
            startX = e.clientX;
            startY = e.clientY;
        }});
        
        document.addEventListener('mouseup', function() {{
            isDragging = false;
            svg.style.cursor = 'grab';
        }});
    }}
}})();
</script>
"#,
            widget_id = self.widget_state.widget_id,
            interactive = self.widget_config.interactive
        )
    }
    
    /// Generate JavaScript code for widget model
    pub fn generate_widget_javascript(&self) -> String {
        format!(
            r#"
// Groggy Widget JavaScript Model
const groggyWidgetModel = {{
    widgetId: "{widget_id}",
    modelVersion: {model_version},
    nodeCount: {node_count},
    edgeCount: {edge_count},
    interactive: {interactive},
    
    // Widget communication methods
    updateData: function(nodes, edges) {{
        // Send data update to Python backend
        console.log('Updating widget data:', nodes.length, 'nodes,', edges.length, 'edges');
    }},
    
    handleInteraction: function(interaction) {{
        // Send interaction to Python backend
        console.log('Widget interaction:', interaction);
    }}
}};
"#,
            widget_id = self.widget_state.widget_id,
            model_version = self.widget_state.model_version,
            node_count = if let Some(ref bridge) = self.python_bridge {
                bridge.model_data.get("node_count").and_then(|v| v.as_u64()).unwrap_or(0)
            } else { 0 },
            edge_count = if let Some(ref bridge) = self.python_bridge {
                bridge.model_data.get("edge_count").and_then(|v| v.as_u64()).unwrap_or(0)
            } else { 0 },
            interactive = self.widget_config.interactive
        )
    }
    
    /// Update widget configuration
    pub fn update_widget_config(&mut self, config: JupyterWidgetConfig) -> GraphResult<()> {
        self.widget_config = config;
        
        // Invalidate cached HTML to force regeneration
        self.cached_html = None;
        
        Ok(())
    }
    
    /// Get widget model data for Python bridge
    pub fn get_model_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        
        data.insert("widget_id".to_string(), serde_json::json!(self.widget_state.widget_id));
        data.insert("model_version".to_string(), serde_json::json!(self.widget_state.model_version));
        data.insert("title".to_string(), serde_json::json!(self.widget_state.title));
        data.insert("width".to_string(), serde_json::json!(self.widget_state.width));
        data.insert("height".to_string(), serde_json::json!(self.widget_state.height));
        data.insert("interactive".to_string(), serde_json::json!(self.widget_state.interactive));
        
        // Add current positions from core engine
        let positions = self.core.get_positions();
        data.insert("positions".to_string(), serde_json::json!(positions));
        
        data
    }
}

impl VizAdapter for JupyterAdapter {
    fn set_data(&mut self, nodes: Vec<VizNode>, edges: Vec<VizEdge>) -> GraphResult<()> {
        // Delegate to unified core engine
        self.core.set_data(nodes, edges)?;
        
        // Update Python bridge model data
        if let Some(ref mut bridge) = self.python_bridge {
            bridge.model_data.insert("node_count".to_string(), serde_json::json!(nodes.len()));
            bridge.model_data.insert("edge_count".to_string(), serde_json::json!(edges.len()));
        }
        
        // Invalidate cached HTML
        self.cached_html = None;
        
        Ok(())
    }
    
    fn render(&mut self) -> GraphResult<AdapterResult> {
        let start_time = std::time::Instant::now();
        
        // Generate widget HTML using unified core
        let html = self.generate_widget_html()?;
        let javascript = self.generate_widget_javascript();
        
        let render_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        let metadata = JupyterMetadata {
            widget_id: self.widget_state.widget_id.clone(),
            data_size_bytes: html.len() + javascript.len(),
            render_time_ms: render_time,
        };
        
        let result = JupyterResult {
            html,
            javascript,
            metadata,
        };
        
        Ok(AdapterResult::Jupyter(result))
    }
    
    fn get_frame(&mut self) -> GraphResult<VizFrame> {
        // Get fresh frame from unified core
        self.core.update()
    }
    
    fn is_ready(&self) -> bool {
        // Ready if we have data and optionally a Python bridge
        !self.core.get_positions().is_empty()
    }
    
    fn get_config(&self) -> AdapterConfig {
        AdapterConfig {
            width: self.widget_state.width,
            height: self.widget_state.height,
            physics_enabled: self.core.is_simulation_running(),
            interactions_enabled: self.widget_state.interactive,
            auto_fit: true,
            theme: match &self.widget_config.theme {
                WidgetTheme::Dark => Some("dark".to_string()),
                WidgetTheme::Light => Some("light".to_string()),
                WidgetTheme::Custom(_) => Some("custom".to_string()),
                WidgetTheme::Default => None,
            },
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
            target_fps: 30.0,
            interactions_enabled: config.interactions_enabled,
            auto_fit: config.auto_fit,
            fit_padding: 20.0,
        };
        
        self.core.set_config(viz_config)?;
        
        // Update widget state
        self.widget_state.width = config.width;
        self.widget_state.height = config.height;
        self.widget_state.interactive = config.interactions_enabled;
        
        // Update widget config theme
        if let Some(theme_name) = config.theme {
            self.widget_config.theme = match theme_name.as_str() {
                "dark" => WidgetTheme::Dark,
                "light" => WidgetTheme::Light,
                _ => WidgetTheme::Default,
            };
        }
        
        // Invalidate cached HTML
        self.cached_html = None;
        
        Ok(())
    }
}

/// Builder for creating Jupyter adapters with configuration
pub struct JupyterAdapterBuilder {
    config: AdapterConfig,
    widget_config: JupyterWidgetConfig,
}

impl JupyterAdapterBuilder {
    pub fn new() -> Self {
        Self {
            config: AdapterConfig::default(),
            widget_config: JupyterWidgetConfig::default(),
        }
    }
    
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }
    
    pub fn with_theme(mut self, theme: WidgetTheme) -> Self {
        self.widget_config.theme = theme;
        self
    }
    
    pub fn with_interactions(mut self, enabled: bool) -> Self {
        self.config.interactions_enabled = enabled;
        self.widget_config.interactive = enabled;
        self
    }
    
    pub fn with_title(mut self, title: String) -> Self {
        // Note: Title will be set when creating the adapter
        self
    }
    
    pub fn build(self) -> JupyterAdapter {
        let mut adapter = JupyterAdapter::new(self.config);
        adapter.update_widget_config(self.widget_config).expect("Failed to update widget config");
        adapter
    }
}

impl Default for JupyterAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}