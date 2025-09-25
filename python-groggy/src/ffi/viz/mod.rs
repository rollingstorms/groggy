//! Python FFI for Visualization System
//!
//! Provides Python bindings for the Groggy visualization module,
//! enabling interactive graph and table visualizations from Python.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use std::collections::HashMap;

use groggy::viz::{
    VizModule, VizConfig, InteractiveOptions, InteractiveViz, InteractiveVizSession,
    StaticOptions, StaticViz, ExportFormat, InteractionConfig, PerformanceConfig
};
use groggy::viz::streaming::data_source::{DataSource, LayoutAlgorithm};
use groggy::errors::GraphResult;
use crate::ffi::errors::PyGraphError;

// =============================================================================
// PyVizConfig - Configuration class for visualization customization  
// =============================================================================

/// Python wrapper for visualization configuration
/// 
/// Provides a convenient way to configure visualization options
/// with sensible defaults and validation.
#[pyclass(name = "VizConfig", module = "groggy")]
pub struct PyVizConfig {
    pub port: u16,
    pub layout: String,
    pub theme: String,
    pub width: u32,
    pub height: u32,
    pub auto_open: bool,
}

#[pymethods]
impl PyVizConfig {
    /// Create a new VizConfig with default settings
    /// 
    /// # Arguments
    /// * `port` - Port number (default: 0 for auto-assign)
    /// * `layout` - Layout algorithm (default: "force-directed")
    /// * `theme` - Visual theme (default: "light")
    /// * `width` - Canvas width (default: 1200)
    /// * `height` - Canvas height (default: 800)
    /// * `auto_open` - Automatically open browser (default: true)
    #[new]
    #[pyo3(signature = (port=0, layout="force-directed".to_string(), theme="light".to_string(), width=1200, height=800, auto_open=true))]
    pub fn new(
        port: u16,
        layout: String, 
        theme: String,
        width: u32,
        height: u32,
        auto_open: bool
    ) -> PyResult<Self> {
        // Validate layout algorithm
        parse_layout_algorithm(&layout)?;
        
        // Validate theme
        if !matches!(theme.as_str(), "light" | "dark" | "publication" | "minimal") {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid theme '{}'. Use 'light', 'dark', 'publication', or 'minimal'", theme)
            ));
        }
        
        Ok(PyVizConfig {
            port,
            layout,
            theme,
            width,
            height,
            auto_open,
        })
    }
    
    /// Get the port number
    #[getter]
    pub fn port(&self) -> u16 {
        self.port
    }
    
    /// Set the port number
    #[setter]
    pub fn set_port(&mut self, port: u16) {
        self.port = port;
    }
    
    /// Get the layout algorithm
    #[getter] 
    pub fn layout(&self) -> &str {
        &self.layout
    }
    
    /// Set the layout algorithm with validation
    #[setter]
    pub fn set_layout(&mut self, layout: String) -> PyResult<()> {
        parse_layout_algorithm(&layout)?;
        self.layout = layout;
        Ok(())
    }
    
    /// Get the theme
    #[getter]
    pub fn theme(&self) -> &str {
        &self.theme
    }
    
    /// Set the theme with validation
    #[setter]
    pub fn set_theme(&mut self, theme: String) -> PyResult<()> {
        if !matches!(theme.as_str(), "light" | "dark" | "publication" | "minimal") {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid theme '{}'. Use 'light', 'dark', 'publication', or 'minimal'", theme)
            ));
        }
        self.theme = theme;
        Ok(())
    }
    
    /// Get the width
    #[getter]
    pub fn width(&self) -> u32 {
        self.width
    }
    
    /// Set the width
    #[setter]
    pub fn set_width(&mut self, width: u32) {
        self.width = width;
    }
    
    /// Get the height
    #[getter]
    pub fn height(&self) -> u32 {
        self.height
    }
    
    /// Set the height
    #[setter]
    pub fn set_height(&mut self, height: u32) {
        self.height = height;
    }
    
    /// Get auto_open setting
    #[getter]
    pub fn auto_open(&self) -> bool {
        self.auto_open
    }
    
    /// Set auto_open setting
    #[setter]
    pub fn set_auto_open(&mut self, auto_open: bool) {
        self.auto_open = auto_open;
    }
    
    /// Create a preset configuration for publication-ready outputs
    #[staticmethod]
    pub fn publication(width: Option<u32>, height: Option<u32>) -> PyResult<PyVizConfig> {
        PyVizConfig::new(
            0,
            "hierarchical".to_string(),
            "publication".to_string(),
            width.unwrap_or(1600),
            height.unwrap_or(1200),
            false
        )
    }
    
    /// Create a preset configuration for interactive exploration
    #[staticmethod]
    pub fn interactive(port: Option<u16>) -> PyResult<PyVizConfig> {
        PyVizConfig::new(
            port.unwrap_or(0),
            "force-directed".to_string(),
            "dark".to_string(),
            1200,
            800,
            true
        )
    }
    
    /// String representation
    fn __str__(&self) -> String {
        format!("VizConfig(port={}, layout='{}', theme='{}', {}×{}, auto_open={})",
                self.port, self.layout, self.theme, self.width, self.height, self.auto_open)
    }
}

// =============================================================================
// PyVizModule - Main visualization wrapper
// =============================================================================

/// Python wrapper for VizModule
/// 
/// Provides unified visualization capabilities for graph data,
/// supporting both interactive browser-based visualization and
/// static export functionality.
#[pyclass(name = "VizModule", module = "groggy")]
pub struct PyVizModule {
    pub(crate) inner: VizModule,
}

#[pymethods]
impl PyVizModule {
    /// Launch interactive visualization in browser
    /// 
    /// # Arguments
    /// * `port` - Optional port number (0 for auto-assign)
    /// * `layout` - Layout algorithm: "force-directed", "circular", "grid", "hierarchical"
    /// * `theme` - Visual theme: "light", "dark", "publication", "minimal"
    /// * `width` - Canvas width in pixels
    /// * `height` - Canvas height in pixels
    /// 
    /// # Returns
    /// PyInteractiveViz session object
    /// 
    /// # Examples
    /// ```python
    /// viz = table.interactive()
    /// session = viz.interactive(port=8080, layout="force-directed")
    /// print(f"Visualization available at: {session.url()}")
    /// ```
    pub fn interactive(
        &self,
        port: Option<u16>,
        layout: Option<String>,
        theme: Option<String>,
        width: Option<u32>,
        height: Option<u32>
    ) -> PyResult<PyInteractiveViz> {
        let layout_algo = parse_layout_algorithm(layout.as_deref().unwrap_or("force-directed"))?;
        
        let options = InteractiveOptions {
            port: port.unwrap_or(0),
            layout: layout_algo,
            theme: theme.unwrap_or_else(|| "light".to_string()),
            width: width.unwrap_or(1200),
            height: height.unwrap_or(800),
            interactions: InteractionConfig::default(),
            show_labels: true,
            auto_open: false,
        };
        
        let interactive_viz = self.inner.interactive(Some(options))
            .map_err(PyGraphError::from)?;
        
        Ok(PyInteractiveViz {
            inner: Some(interactive_viz),
        })
    }
    
    /// Generate static visualization export
    /// 
    /// # Arguments
    /// * `filename` - Output filename
    /// * `format` - Export format: "svg", "html" (png and pdf not yet implemented)
    /// * `layout` - Layout algorithm
    /// * `theme` - Visual theme
    /// * `dpi` - Resolution for raster formats (unused for svg/html)
    /// * `width` - Canvas width
    /// * `height` - Canvas height
    /// 
    /// # Returns
    /// PyStaticViz with export information
    pub fn static_viz(
        &self,
        filename: String,
        format: Option<String>,
        layout: Option<String>,
        theme: Option<String>,
        dpi: Option<u32>,
        width: Option<u32>,
        height: Option<u32>
    ) -> PyResult<PyStaticViz> {
        let export_format = match format.as_deref().unwrap_or("png") {
            "png" => ExportFormat::PNG,
            "svg" => ExportFormat::SVG,
            "pdf" => ExportFormat::PDF,
            "html" => ExportFormat::HTML,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Format must be 'png', 'svg', 'pdf', or 'html'"
            )),
        };
        
        let layout_algo = parse_layout_algorithm(layout.as_deref().unwrap_or("force-directed"))?;
        
        let options = StaticOptions {
            filename,
            format: export_format,
            layout: layout_algo,
            theme: theme.unwrap_or_else(|| "light".to_string()),
            dpi: dpi.unwrap_or(300),
            width: width.unwrap_or(1200),
            height: height.unwrap_or(800),
        };
        
        let static_viz = self.inner.static_viz(options)
            .map_err(PyGraphError::from)?;
        
        Ok(PyStaticViz {
            inner: static_viz,
        })
    }
    
    /// Check if the data source supports graph visualization
    pub fn supports_graph_view(&self) -> bool {
        self.inner.supports_graph_view()
    }
    
    /// Get information about the data source
    pub fn info(&self, py: Python) -> PyResult<PyObject> {
        let info = self.inner.get_info();
        let dict = PyDict::new(py);

        dict.set_item("total_rows", info.total_rows)?;
        dict.set_item("total_cols", info.total_cols)?;
        dict.set_item("supports_graph", info.supports_graph)?;
        dict.set_item("source_type", info.source_type)?;

        if let Some(graph_info) = info.graph_info {
            let graph_dict = PyDict::new(py);
            graph_dict.set_item("node_count", graph_info.node_count)?;
            graph_dict.set_item("edge_count", graph_info.edge_count)?;
            graph_dict.set_item("is_directed", graph_info.is_directed)?;
            graph_dict.set_item("has_weights", graph_info.has_weights)?;
            dict.set_item("graph_info", graph_dict)?;
        }

        Ok(dict.to_object(py))
    }

    /// Generate embedded iframe HTML for visualization in Jupyter notebooks
    ///
    /// This method creates a streaming server for the data source and returns iframe HTML
    /// that can be embedded directly in Jupyter notebooks. It supports both table
    /// and graph visualization views depending on the data source type.
    ///
    /// # Returns
    /// * HTML iframe code for embedding
    ///
    /// # Examples
    /// ```python
    /// # Create graph and get iframe for embedding
    /// g = groggy.generators.karate_club()
    /// viz = g.viz()
    /// iframe_html = viz.interactive_embed()
    ///
    /// # Use in Jupyter
    /// from IPython.display import HTML, display
    /// display(HTML(iframe_html))
    /// ```
    pub fn interactive_embed(&self) -> PyResult<String> {
        println!("🎯 VizModule::interactive_embed called");

        // Create an InteractiveViz using the existing infrastructure
        let interactive_viz = self.inner.interactive(None)
            .map_err(PyGraphError::from)?;

        // Start the server to get the URL and port
        let addr = "127.0.0.1".parse().unwrap();
        let session = interactive_viz.start(Some(addr))
            .map_err(PyGraphError::from)?;

        let port = session.port();
        let iframe_html = format!(
            r#"<iframe src="http://127.0.0.1:{port}/" width="100%" height="420" style="border:0;border-radius:12px;"></iframe>"#,
            port = port
        );

        println!("🖼️  VizModule iframe generated for port {}", port);

        // Note: The session goes out of scope here, but the server should keep running
        // because of the streaming server's background thread architecture
        std::mem::forget(session); // Prevent session from being dropped

        Ok(iframe_html)
    }

    /// Show interactive visualization in browser (shortcut for the most common workflow)
    /// 
    /// This method provides a convenient shortcut for launching a real-time visualization
    /// with default settings. It's equivalent to calling the underlying VizModule's show() method.
    /// 
    /// # Returns
    /// None (opens browser automatically)
    /// 
    /// # Examples
    /// ```python
    /// g = groggy.Graph()
    /// g.add_node(name="Alice")
    /// g.add_node(name="Bob") 
    /// g.add_edge(0, 1)
    /// g.viz().show()  # Opens browser with real-time visualization
    /// ```
    pub fn show(&mut self, py: Python) -> PyResult<()> {
        py.allow_threads(|| {
            match self.inner.show() {
                Ok(_) => Ok(()),
                Err(e) => Err(PyGraphError::from(e)),
            }
        })?;
        Ok(())
    }
}

// =============================================================================
// PyInteractiveViz - Active visualization session
// =============================================================================

/// Python wrapper for InteractiveViz
#[pyclass(name = "InteractiveViz", module = "groggy", unsendable)]
pub struct PyInteractiveViz {
    inner: Option<InteractiveViz>,
}

#[pymethods]
impl PyInteractiveViz {
    /// Start the visualization server
    /// 
    /// # Arguments
    /// * `bind_addr` - IP address to bind to (default: "127.0.0.1")
    /// 
    /// # Returns
    /// PyInteractiveVizSession with server details
    pub fn start(&mut self, bind_addr: Option<String>) -> PyResult<PyInteractiveVizSession> {
        let addr = if let Some(addr_str) = bind_addr {
            addr_str.parse().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid IP address: {}", e)
                )
            })?
        } else {
            "127.0.0.1".parse().unwrap()
        };
        
        let interactive_viz = self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "InteractiveViz has already been started"
            )
        })?;
        
        let session = interactive_viz.start(Some(addr))
            .map_err(PyGraphError::from)?;
        
        Ok(PyInteractiveVizSession {
            inner: session,
        })
    }
}

// =============================================================================
// PyInteractiveVizSession - Running visualization session
// =============================================================================

/// Python wrapper for InteractiveVizSession
#[pyclass(name = "InteractiveVizSession", module = "groggy")]
pub struct PyInteractiveVizSession {
    inner: InteractiveVizSession,
}

#[pymethods]
impl PyInteractiveVizSession {
    /// Get the URL where visualization is accessible
    pub fn url(&self) -> String {
        self.inner.url().to_string()
    }
    
    /// Get the port the server is running on
    pub fn port(&self) -> u16 {
        self.inner.port()
    }
    
    /// Stop the visualization server
    pub fn stop(&mut self) {
        // Note: This is a placeholder - we'll need to implement server stopping
        // when the actual server implementation is available
    }
}

// =============================================================================
// PyStaticViz - Static export result
// =============================================================================

/// Python wrapper for StaticViz
#[pyclass(name = "StaticViz", module = "groggy")]
pub struct PyStaticViz {
    inner: StaticViz,
}

#[pymethods]
impl PyStaticViz {
    /// Get the output file path
    pub fn file_path(&self) -> &str {
        &self.inner.file_path
    }
    
    /// Get the file size in bytes
    pub fn size_bytes(&self) -> usize {
        self.inner.size_bytes
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Parse layout algorithm string into LayoutAlgorithm enum
fn parse_layout_algorithm(layout: &str) -> PyResult<LayoutAlgorithm> {
    match layout {
        "force-directed" | "force" => Ok(LayoutAlgorithm::ForceDirected {
            charge: -300.0,
            distance: 50.0,
            iterations: 100,
        }),
        "circular" | "circle" => Ok(LayoutAlgorithm::Circular {
            radius: Some(100.0),
            start_angle: 0.0,
        }),
        "grid" => Ok(LayoutAlgorithm::Grid {
            columns: 10, // Auto-calculate default
            cell_size: 50.0,
        }),
        "hierarchical" | "tree" => Ok(LayoutAlgorithm::Hierarchical {
            direction: groggy::viz::streaming::data_source::HierarchicalDirection::TopDown,
            layer_spacing: 100.0,
            node_spacing: 50.0,
        }),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown layout algorithm: {}. Use 'force-directed', 'circular', 'grid', or 'hierarchical'", layout)
        )),
    }
}
