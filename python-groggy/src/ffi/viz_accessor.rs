//! FFI VizAccessor implementation for interactive visualization.
//!
//! Provides a pandas-like .viz accessor for visualization operations on subgraphs,
//! arrays, and tables with support for different backends (jupyter, server).

use futures_util::SinkExt;
use groggy::viz::streaming::GraphDataSource;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::Mutex;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

/// Macro for conditional debug printing based on verbosity level
macro_rules! debug_print {
    ($verbose:expr, $level:expr, $($arg:tt)*) => {
        if $verbose >= $level {
            eprintln!($($arg)*);
        }
    };
}

/// Server info for tracking active visualization servers
#[derive(Debug, Clone)]
struct ServerInfo {
    port: u16,
    data_source_id: String,
}

/// Global registry of active visualization servers
static SERVER_REGISTRY: std::sync::LazyLock<Mutex<HashMap<u16, ServerInfo>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Find an available port in the given range
fn find_available_port(start_port: u16) -> Result<u16, String> {
    for port in start_port..=start_port + 200 {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => {
                // Port is available - no debug needed unless verbose
                return Ok(port);
            }
            Err(_e) => {
                // Port unavailable - continue trying next port
                continue;
            }
        }
    }
    Err(format!(
        "No available ports found in range {}-{}",
        start_port,
        start_port + 200
    ))
}

/// VizAccessor provides visualization methods for groggy objects
#[pyclass]
pub struct VizAccessor {
    /// The underlying data source for visualization
    data_source: Option<GraphDataSource>,
    /// Object type for fallback handling
    object_type: String,
    /// Unique identifier for this data source
    data_source_id: String,
}

#[pymethods]
impl VizAccessor {
    /// Show visualization with Realtime backend using proper DataSource integration
    /// Now supports server reuse - updates existing servers instead of creating new ones
    #[pyo3(signature = (layout = "honeycomb".to_string(), verbose = None, **kwargs))]
    fn show(
        &self,
        py: Python,
        layout: String,
        verbose: Option<u8>,
        kwargs: Option<&pyo3::types::PyDict>,
    ) -> PyResult<PyObject> {
        let verbose = verbose.unwrap_or(0);
        debug_print!(verbose, 2, "📺 VizAccessor.show()");
        let (alg, viz_config) = self.parse_layout_kwargs_typed(kwargs, &layout, verbose)?;
        self.ensure_server_and_display_iframe(
            py,
            verbose,
            &alg,
            &viz_config,
            /*open_browser=*/ false,
        )
    }

    /// Show visualization in standalone server mode using realtime backend
    #[pyo3(signature = (verbose=None))]
    fn server(&self, py: Python, verbose: Option<u8>) -> PyResult<PyObject> {
        let verbose = verbose.unwrap_or(0);
        debug_print!(verbose, 2, "🖥️ VizAccessor.server()");
        // No kwargs here, so use default honeycomb and empty viz_config
        let (alg, viz_config) = self.parse_layout_kwargs_typed(None, "honeycomb", verbose)?;
        self.ensure_server_and_display_iframe(
            py,
            verbose,
            &alg,
            &viz_config,
            /*open_browser=*/ true,
        )
    }

    /// Update visualization parameters - sends control message to existing server
    #[pyo3(signature = (verbose=None, **kwargs))]
    fn update(&self, py: Python, verbose: Option<u8>, kwargs: Option<&PyDict>) -> PyResult<()> {
        let verbose = verbose.unwrap_or(0);
        debug_print!(verbose, 2, "🔄 VizAccessor.update()");

        let (layout, viz_config) = self.parse_layout_kwargs_typed(kwargs, "honeycomb", verbose)?;
        if let Some(info) = self.get_server_info() {
            self.send_control_message_to_server(
                py,
                verbose,
                info.port,
                layout,
                viz_config.layout_params,
            )?;
            debug_print!(
                verbose,
                1,
                "✅ Visualization parameters updated successfully"
            );
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No active visualization server found. Use .show() first to create a visualization."
            ))
        }
    }

    /// Create a basic fallback visualization for objects that don't have direct viz support
    fn create_fallback_visualization(&self) -> String {
        format!(
            r#"<div style="padding: 20px; border: 1px solid #ccc; border-radius: 5px;">
                <h3>Visualization not yet implemented</h3>
                <p>Object type: {}</p>
                <p>Visualization support for this object type is coming soon.</p>
            </div>"#,
            self.object_type
        )
    }

    fn __repr__(&self) -> String {
        format!("VizAccessor({})", self.object_type)
    }
}

impl VizAccessor {
    /// Create a new VizAccessor with a data source (for graph-based objects)
    pub fn with_data_source(data_source: GraphDataSource, object_type: String) -> Self {
        let data_source_id = format!("{:p}", &data_source as *const _);
        Self {
            data_source: Some(data_source),
            object_type,
            data_source_id,
        }
    }

    /// Create a new VizAccessor without a data source (for fallback)
    pub fn without_data_source(object_type: String) -> Self {
        let data_source_id = format!("fallback_{}", std::process::id());
        Self {
            data_source: None,
            object_type,
            data_source_id,
        }
    }

    /// Get server info for this viz instance from the global registry
    fn get_server_info(&self) -> Option<ServerInfo> {
        let registry = SERVER_REGISTRY.lock().unwrap();
        // Find server by data source ID
        for (port, server_info) in registry.iter() {
            if server_info.data_source_id == self.data_source_id {
                return Some(ServerInfo {
                    port: *port,
                    data_source_id: server_info.data_source_id.clone(),
                });
            }
        }
        None
    }

    /// Normalize aliases so front/back-end agree
    fn normalize_layout_name(&self, s: &str) -> String {
        match s.to_lowercase().as_str() {
            "force" | "force_directed" | "force-directed" | "spring" => {
                "force_directed".to_string()
            }
            "circle" | "circular" => "circular".to_string(),
            "grid" | "matrix" => "grid".to_string(),
            "hex" | "hexagonal" | "honeycomb" => "honeycomb".to_string(),
            other => other.to_string(),
        }
    }

    /// Typed kwargs parser with a consistent fallback layout
    fn parse_layout_kwargs_typed(
        &self,
        kwargs: Option<&PyDict>,
        fallback_layout: &str, // e.g. "honeycomb"
        verbose: u8,
    ) -> PyResult<(String, groggy::viz::realtime::VizConfig)> {
        use groggy::viz::realtime::VizConfig;
        use pyo3::types::{PyList, PyTuple};

        let mut layout = self.normalize_layout_name(fallback_layout);
        let mut viz_config = VizConfig::new();

        if let Some(kw) = kwargs {
            debug_print!(
                verbose,
                3,
                "📝 Processing visualization parameters from kwargs..."
            );

            for (k, v) in kw.iter() {
                let key = k.extract::<String>().unwrap_or_else(|_| k.to_string());

                match key.as_str() {
                    // Layout algorithm
                    "layout" | "layout_algorithm" => {
                        if let Ok(s) = v.extract::<String>() {
                            layout = self.normalize_layout_name(&s);
                            viz_config.layout_algorithm = Some(s);
                        } else {
                            let s = v.to_string();
                            layout = self.normalize_layout_name(&s);
                            viz_config.layout_algorithm = Some(s);
                        }
                        debug_print!(verbose, 3, "  🎯 layout={}", layout);
                    }

                    // Node styling parameters
                    "node_color" => {
                        viz_config.node_color = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🎨 node_color parameter set");
                    }
                    "node_size" => {
                        viz_config.node_size = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📏 node_size parameter set");
                    }
                    "node_shape" => {
                        viz_config.node_shape = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🔶 node_shape parameter set");
                    }
                    "node_opacity" => {
                        viz_config.node_opacity = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  👻 node_opacity parameter set");
                    }
                    "node_border_color" => {
                        viz_config.node_border_color = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🖍️ node_border_color parameter set");
                    }
                    "node_border_width" => {
                        viz_config.node_border_width = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📐 node_border_width parameter set");
                    }

                    // Edge styling parameters
                    "edge_color" => {
                        viz_config.edge_color = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🎨 edge_color parameter set");
                    }
                    "edge_width" => {
                        viz_config.edge_width = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📏 edge_width parameter set");
                    }
                    "edge_opacity" => {
                        viz_config.edge_opacity = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  👻 edge_opacity parameter set");
                    }
                    "edge_style" => {
                        viz_config.edge_style = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  ➖ edge_style parameter set");
                    }

                    // Label parameters
                    "label" | "node_label" => {
                        viz_config.node_label = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🏷️ node_label parameter set");
                    }
                    "edge_label" => {
                        viz_config.edge_label = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🏷️ edge_label parameter set");
                    }
                    "label_size" => {
                        viz_config.label_size = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📏 label_size parameter set");
                    }
                    "label_color" => {
                        viz_config.label_color = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🎨 label_color parameter set");
                    }
                    "edge_label_size" => {
                        viz_config.edge_label_size = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📏 edge_label_size parameter set");
                    }
                    "edge_label_color" => {
                        viz_config.edge_label_color = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  🎨 edge_label_color parameter set");
                    }

                    // Position parameters
                    "x" => {
                        viz_config.x = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📍 x parameter set");
                    }
                    "y" => {
                        viz_config.y = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📍 y parameter set");
                    }
                    "z" => {
                        viz_config.z = self.parse_viz_parameter(v, verbose)?;
                        debug_print!(verbose, 3, "  📍 z parameter set");
                    }

                    // Filtering parameters
                    "show_nodes_where" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.show_nodes_where = Some(s);
                            debug_print!(verbose, 3, "  🔍 show_nodes_where filter set");
                        }
                    }
                    "show_edges_where" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.show_edges_where = Some(s);
                            debug_print!(verbose, 3, "  🔍 show_edges_where filter set");
                        }
                    }
                    "highlight_nodes_where" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.highlight_nodes_where = Some(s);
                            debug_print!(verbose, 3, "  ✨ highlight_nodes_where filter set");
                        }
                    }
                    "highlight_edges_where" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.highlight_edges_where = Some(s);
                            debug_print!(verbose, 3, "  ✨ highlight_edges_where filter set");
                        }
                    }

                    // Scaling parameters
                    "node_size_range" => {
                        if let Ok(tuple) = v.downcast::<PyTuple>() {
                            if tuple.len() == 2 {
                                if let (Ok(min), Ok(max)) = (
                                    tuple.get_item(0)?.extract::<f64>(),
                                    tuple.get_item(1)?.extract::<f64>(),
                                ) {
                                    viz_config.node_size_range = Some((min, max));
                                    debug_print!(
                                        verbose,
                                        3,
                                        "  📏 node_size_range=[{}, {}]",
                                        min,
                                        max
                                    );
                                }
                            }
                        }
                    }
                    "edge_width_range" => {
                        if let Ok(tuple) = v.downcast::<PyTuple>() {
                            if tuple.len() == 2 {
                                if let (Ok(min), Ok(max)) = (
                                    tuple.get_item(0)?.extract::<f64>(),
                                    tuple.get_item(1)?.extract::<f64>(),
                                ) {
                                    viz_config.edge_width_range = Some((min, max));
                                    debug_print!(
                                        verbose,
                                        3,
                                        "  📏 edge_width_range=[{}, {}]",
                                        min,
                                        max
                                    );
                                }
                            }
                        }
                    }

                    // Color parameters
                    "color_palette" => {
                        if let Ok(list) = v.downcast::<PyList>() {
                            let palette: Result<Vec<String>, _> =
                                list.iter().map(|item| item.extract::<String>()).collect();
                            if let Ok(palette) = palette {
                                viz_config.color_palette = Some(palette);
                                debug_print!(
                                    verbose,
                                    3,
                                    "  🎨 color_palette with {} colors",
                                    viz_config.color_palette.as_ref().unwrap().len()
                                );
                            }
                        }
                    }
                    "color_scale_type" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.color_scale_type = Some(s);
                            debug_print!(verbose, 3, "  📊 color_scale_type set");
                        }
                    }

                    // Tooltip parameters
                    "tooltip_columns" => {
                        if let Ok(list) = v.downcast::<PyList>() {
                            let columns: Result<Vec<String>, _> =
                                list.iter().map(|item| item.extract::<String>()).collect();
                            if let Ok(columns) = columns {
                                viz_config.tooltip_columns = columns;
                                debug_print!(
                                    verbose,
                                    3,
                                    "  💬 tooltip_columns with {} columns",
                                    viz_config.tooltip_columns.len()
                                );
                            }
                        }
                    }

                    // Interaction parameters
                    "click_behavior" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.click_behavior = Some(s);
                            debug_print!(verbose, 3, "  🖱️ click_behavior set");
                        }
                    }
                    "hover_behavior" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.hover_behavior = Some(s);
                            debug_print!(verbose, 3, "  🖱️ hover_behavior set");
                        }
                    }
                    "selection_mode" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.selection_mode = Some(s);
                            debug_print!(verbose, 3, "  ✅ selection_mode set");
                        }
                    }
                    "zoom_behavior" => {
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.zoom_behavior = Some(s);
                            debug_print!(verbose, 3, "  🔍 zoom_behavior set");
                        }
                    }

                    // Layout parameters (legacy support)
                    _ => {
                        // For any other parameter, treat it as a layout parameter
                        if let Ok(s) = v.extract::<String>() {
                            viz_config.layout_params.insert(key.clone(), s.clone());
                            debug_print!(verbose, 3, "  🔧 layout param {}={}", key, s);
                        } else if let Ok(f) = v.extract::<f64>() {
                            viz_config.layout_params.insert(key.clone(), f.to_string());
                            debug_print!(verbose, 3, "  🔧 layout param {}={}", key, f);
                        } else if let Ok(i) = v.extract::<i64>() {
                            viz_config.layout_params.insert(key.clone(), i.to_string());
                            debug_print!(verbose, 3, "  🔧 layout param {}={}", key, i);
                        } else if let Ok(b) = v.extract::<bool>() {
                            viz_config.layout_params.insert(key.clone(), b.to_string());
                            debug_print!(verbose, 3, "  🔧 layout param {}={}", key, b);
                        } else {
                            let s = v.to_string();
                            viz_config.layout_params.insert(key.clone(), s.clone());
                            debug_print!(verbose, 3, "  🔧 layout param {}={}", key, s);
                        }
                    }
                }
            }
        }

        debug_print!(
            verbose,
            3,
            "📊 Final visualization config: algorithm='{}', styled parameters parsed",
            layout
        );
        Ok((layout, viz_config))
    }

    /// Parse a Python value into a VizParameter (array, column name, or single value)
    fn parse_viz_parameter<T>(
        &self,
        py_value: &PyAny,
        verbose: u8,
    ) -> PyResult<groggy::viz::realtime::VizParameter<T>>
    where
        for<'a> T: pyo3::FromPyObject<'a>,
    {
        use groggy::viz::realtime::VizParameter;
        use pyo3::types::{PyList, PyTuple};

        // Try to extract as a string first (column name)
        if let Ok(column_name) = py_value.extract::<String>() {
            debug_print!(verbose, 3, "    📝 Parsed as column: {}", column_name);
            return Ok(VizParameter::Column(column_name));
        }

        // Try to extract as a list (array of values)
        if let Ok(py_list) = py_value.downcast::<PyList>() {
            let values: Result<Vec<T>, _> =
                py_list.iter().map(|item| item.extract::<T>()).collect();
            if let Ok(values) = values {
                debug_print!(
                    verbose,
                    3,
                    "    📊 Parsed as array with {} values",
                    values.len()
                );
                return Ok(VizParameter::Array(values));
            }
        }

        // Try to extract as a tuple (array of values)
        if let Ok(py_tuple) = py_value.downcast::<PyTuple>() {
            let values: Result<Vec<T>, _> =
                py_tuple.iter().map(|item| item.extract::<T>()).collect();
            if let Ok(values) = values {
                debug_print!(
                    verbose,
                    3,
                    "    📊 Parsed as tuple array with {} values",
                    values.len()
                );
                return Ok(VizParameter::Array(values));
            }
        }

        // Try to extract as a single value
        if let Ok(value) = py_value.extract::<T>() {
            debug_print!(verbose, 3, "    🎯 Parsed as single value");
            return Ok(VizParameter::Value(value));
        }

        // If all parsing fails, return None
        debug_print!(verbose, 3, "    ❌ Could not parse parameter, using None");
        Ok(VizParameter::None)
    }

    /// Boot (or reuse) the realtime server in background and display iframe.
    /// This is the single shared engine used by `show`/`server`.
    fn ensure_server_and_display_iframe(
        &self,
        py: Python,
        verbose: u8,
        layout: &str,
        viz_config: &groggy::viz::realtime::VizConfig,
        open_browser: bool, // true for `.server()`, false for `.show()`
    ) -> PyResult<PyObject> {
        if let Some(info) = self.get_server_info() {
            // Reuse: just send a control message and re-display iframe
            self.send_control_message_to_server(
                py,
                verbose,
                info.port,
                layout.to_string(),
                viz_config.layout_params.clone(),
            )?;
            let html = format!(
                r#"<div style="position: relative;">
<iframe src="http://127.0.0.1:{}/" width="100%" height="640" frameborder="0" style="border: 1px solid #ddd;"></iframe>
</div>"#,
                info.port
            );
            py.run(
                &format!(
                    r#"
from IPython.display import HTML, display
display(HTML(r'''{html}'''))
    "#,
                    html = html.replace("'", "\\'")
                ),
                None,
                None,
            )?;
            return Ok(py.None());
        }

        // Start a fresh server (background, non-blocking)
        let ds = match &self.data_source {
            Some(ds) => ds.clone(),
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "No data source available for {}",
                    self.object_type
                )));
            }
        };
        let data_source_id = self.data_source_id.clone();

        let (port, iframe_html) = py.allow_threads(|| -> Result<(u16, String), String> {
            use groggy::viz::realtime::accessor::{DataSourceRealtimeAccessor, RealtimeVizAccessor};
            use groggy::viz::realtime::server::start_realtime_background;
            use groggy::viz::streaming::data_source::LayoutAlgorithm;
            use std::sync::Arc;

            // Build layout algorithm from normalized layout + params (keep same rules as before)
            let algo = match layout {
                "force_directed" => {
                    let iterations = viz_config.layout_params.get("iterations").and_then(|s| s.parse().ok()).unwrap_or(100);
                    let charge = viz_config.layout_params.get("charge").and_then(|s| s.parse().ok()).unwrap_or(-300.0);
                    let distance = viz_config.layout_params.get("distance").and_then(|s| s.parse().ok()).unwrap_or(50.0);
                    LayoutAlgorithm::ForceDirected { charge, distance, iterations }
                }
                "circular" => {
                    let radius = viz_config.layout_params.get("radius").and_then(|s| s.parse::<f64>().ok());
                    let start_angle = viz_config.layout_params.get("start_angle").and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    LayoutAlgorithm::Circular { radius, start_angle }
                }
                "grid" => {
                    let columns = viz_config.layout_params.get("columns").and_then(|s| s.parse().ok()).unwrap_or(5);
                    let cell_size = viz_config.layout_params.get("cell_size").and_then(|s| s.parse().ok()).unwrap_or(100.0);
                    LayoutAlgorithm::Grid { columns, cell_size }
                }
                _ => { // honeycomb default
                    let cell_size = viz_config.layout_params.get("cell_size").and_then(|s| s.parse().ok()).unwrap_or(40.0);
                    let energy_optimization = viz_config.layout_params.get("energy_optimization").and_then(|s| s.parse().ok()).unwrap_or(true);
                    let iterations = viz_config.layout_params.get("iterations").and_then(|s| s.parse().ok()).unwrap_or(100);
                    LayoutAlgorithm::Honeycomb { cell_size, energy_optimization, iterations }
                }
            };

            let ds_arc = Arc::new(ds);

            let accessor = DataSourceRealtimeAccessor::with_layout_and_config(
                ds_arc,
                algo,
                Some(viz_config.clone())
            );

            // Validate data access once (snapshot)
            accessor.initial_snapshot().map_err(|e| format!("Snapshot creation failed: {}", e))?;

            let port = find_available_port(8080)?;
            let accessor_arc = Arc::new(accessor);
            let handle = start_realtime_background(port, accessor_arc, verbose)
                .map_err(|e| format!("Failed to start realtime server: {}", e))?;
            let actual_port = handle.port;

            let iframe_html = format!(
r#"<div style="position: relative;">
<iframe src="http://127.0.0.1:{}/" width="100%" height="640" frameborder="0" style="border: 1px solid #ddd;"></iframe>
</div>"#, actual_port);

            Ok((actual_port, iframe_html))
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Register server
        {
            let mut registry = SERVER_REGISTRY.lock().unwrap();
            registry.insert(
                port,
                ServerInfo {
                    port,
                    data_source_id,
                },
            );
        }

        // Apply requested layout to freshly started server
        if let Err(e) = self.send_control_message_to_server(
            py,
            verbose,
            port,
            layout.to_string(),
            viz_config.layout_params.clone(),
        ) {
            debug_print!(
                verbose,
                1,
                "⚠️ Failed to apply initial layout '{}': {}",
                layout,
                e
            );
        }

        if open_browser {
            py.run(
                &format!(
                    "import webbrowser; webbrowser.open('http://127.0.0.1:{}/')",
                    port
                ),
                None,
                None,
            )?;
        }

        py.run(
            &format!(
                r#"
from IPython.display import HTML, display
display(HTML(r'''{html}'''))
    "#,
                html = iframe_html.replace("'", "\\'")
            ),
            None,
            None,
        )?;

        Ok(py.None())
    }

    /// Send control message to server - extracted from update_existing_server for reuse
    fn send_control_message_to_server(
        &self,
        py: Python,
        verbose: u8,
        port: u16,
        layout: String,
        layout_params: HashMap<String, String>,
    ) -> PyResult<()> {
        debug_print!(
            verbose,
            2,
            "📡 Sending ChangeLayout control message to server on port {}",
            port
        );

        let result = py.allow_threads(move || -> Result<String, String> {
            // Create tokio runtime for WebSocket client
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

            rt.block_on(async {
                // Connect to the existing server's WebSocket endpoint
                let ws_url = format!("ws://127.0.0.1:{}/ws", port);
                debug_print!(verbose, 3, "🔗 Connecting to WebSocket: {}", ws_url);

                match connect_async(&ws_url).await {
                    Ok((mut ws_stream, _)) => {
                        debug_print!(verbose, 3, "✅ WebSocket connected successfully");

                        // Serialize the control message to match server expectations
                        let control_json = serde_json::json!({
                            "type": "control",
                            "version": 1,
                            "payload": {
                                "ChangeLayout": {
                                    "algorithm": layout.clone(),
                                    "params": layout_params.clone()
                                }
                            }
                        });

                        let message_text = serde_json::to_string(&control_json)
                            .map_err(|e| format!("Failed to serialize control message: {}", e))?;

                        debug_print!(verbose, 3, "📡 Sending WebSocket message: {}", message_text);

                        // Send the control message
                        match ws_stream.send(Message::Text(message_text)).await {
                            Ok(()) => {
                                debug_print!(
                                    verbose,
                                    2,
                                    "✅ ChangeLayout control message sent successfully"
                                );

                                // Wait briefly for server to process and send position updates
                                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                                // Close the WebSocket connection cleanly
                                let _ = ws_stream.close(None).await;
                                Ok("Control message sent successfully".to_string())
                            }
                            Err(e) => {
                                debug_print!(
                                    verbose,
                                    1,
                                    "❌ Failed to send WebSocket message: {}",
                                    e
                                );
                                Err(format!("Failed to send WebSocket message: {}", e))
                            }
                        }
                    }
                    Err(e) => {
                        debug_print!(
                            verbose,
                            1,
                            "❌ Failed to connect to WebSocket {}: {}",
                            ws_url,
                            e
                        );
                        Err(format!("Failed to connect to WebSocket: {}", e))
                    }
                }
            })
        });

        match result {
            Ok(_) => {
                debug_print!(
                    verbose,
                    2,
                    "🔄 Server update successful - layout parameters sent"
                );
                Ok(())
            }
            Err(e) => {
                debug_print!(verbose, 1, "❌ Failed to send control message: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to send control message: {}",
                    e
                )))
            }
        }
    }

    /// Parse layout parameters from Python kwargs
    #[allow(dead_code)]
    fn parse_layout_parameters(
        &self,
        kwargs: &PyDict,
        verbose: u8,
    ) -> PyResult<(String, HashMap<String, String>)> {
        let mut layout = "force_directed".to_string();
        let mut layout_params = HashMap::new();

        debug_print!(verbose, 3, "📝 Processing layout parameters from kwargs...");

        for (key, value) in kwargs.iter() {
            let key_str = key.to_string();
            if key_str == "layout" {
                layout = value.to_string();
            } else {
                let value_str = value.to_string();
                layout_params.insert(key_str.clone(), value_str.clone());
                debug_print!(verbose, 3, "  🔧 Parameter: {}={}", key_str, value_str);
            }
        }

        debug_print!(
            verbose,
            3,
            "📊 Final layout parameters: algorithm='{}', params={:?}",
            layout,
            layout_params
        );
        Ok((layout, layout_params))
    }

    /// Update existing server with new layout parameters via WebSocket ControlMsg
    #[allow(dead_code)]
    fn update_existing_server(
        &self,
        py: Python,
        port: u16,
        layout: String,
        layout_params: HashMap<String, String>,
        verbose: u8,
    ) -> PyResult<PyObject> {
        debug_print!(
            verbose,
            2,
            "📡 Sending ChangeLayout control message to server on port {}",
            port
        );

        let result = py.allow_threads(move || -> Result<String, String> {
            use groggy::viz::realtime::accessor::ControlMsg;

            // Build the control message
            let _control_msg = ControlMsg::ChangeLayout {
                algorithm: layout.clone(),
                params: layout_params.clone(),
            };

            debug_print!(verbose, 3, "📤 ChangeLayout message created: algorithm={}, params={:?}", layout, layout_params);

            // Create tokio runtime for WebSocket client
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

            let send_result = rt.block_on(async {
                // Connect to the existing server's WebSocket endpoint
                let ws_url = format!("ws://127.0.0.1:{}/ws", port);
                debug_print!(verbose, 3, "🔗 Connecting to WebSocket: {}", ws_url);

                match connect_async(&ws_url).await {
                    Ok((mut ws_stream, _)) => {
                        debug_print!(verbose, 3, "✅ WebSocket connected successfully");

                        // Serialize the control message to match server expectations
                        let control_json = serde_json::json!({
                            "type": "control",
                            "version": 1,
                            "payload": {
                                "ChangeLayout": {
                                    "algorithm": layout.clone(),
                                    "params": layout_params.clone()
                                }
                            }
                        });

                        let message_text = serde_json::to_string(&control_json)
                            .map_err(|e| format!("Failed to serialize control message: {}", e))?;

                        debug_print!(verbose, 3, "📡 Sending WebSocket message: {}", message_text);

                        // Send the control message
                        match ws_stream.send(Message::Text(message_text)).await {
                            Ok(()) => {
                                debug_print!(verbose, 2, "✅ ChangeLayout control message sent successfully");

                                // Close the WebSocket connection cleanly
                                let _ = ws_stream.close(None).await;
                                Ok("Control message sent successfully".to_string())
                            }
                            Err(e) => {
                                debug_print!(verbose, 1, "❌ Failed to send WebSocket message: {}", e);
                                Err(format!("Failed to send WebSocket message: {}", e))
                            }
                        }
                    }
                    Err(e) => {
                        debug_print!(verbose, 1, "❌ Failed to connect to WebSocket {}: {}", ws_url, e);
                        Err(format!("Failed to connect to WebSocket: {}", e))
                    }
                }
            });

            match send_result {
                Ok(_) => {
                    // Return iframe HTML that reuses the existing server
                    let iframe_html = format!(
                        r#"<div style="position: relative;">
    <iframe src="http://127.0.0.1:{}/" width="100%" height="640" frameborder="0" style="border: 1px solid #ddd;"></iframe>
    <div style="font-size: 12px; color: #666; margin-top: 5px;">
        🔄 <strong>Updated Server</strong> on port {} | Layout changed to: {} | Parameters updated
        <div style="margin-top: 3px;">
            ✅ Server reuse - no new instance created | ✅ ChangeLayout control message sent via WebSocket
        </div>
    </div>
</div>"#,
                        port, port, layout
                    );
                    Ok(iframe_html)
                }
                Err(e) => {
                    debug_print!(verbose, 1, "❌ WebSocket control message failed: {}", e);
                    Err(format!("WebSocket control message failed: {}", e))
                }
            }
        });

        match result {
            Ok(html) => {
                // Auto-display in Jupyter using display(HTML())
                py.run(
                    &format!(
                        r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
except Exception as e:
    print(f"Display error: {{e}}")
"#,
                        html = html.replace("'", "\\'")
                    ),
                    None,
                    None,
                )?;

                Ok(py.None())
            }
            Err(e) => {
                debug_print!(verbose, 1, "❌ Failed to update existing server: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to update existing server: {}",
                    e
                )))
            }
        }
    }
}

/// Extract port number from iframe HTML
#[allow(dead_code)]
fn extract_port_from_iframe(iframe_html: &str) -> Option<u16> {
    // Simple regex-free port extraction
    if let Some(start) = iframe_html.find("http://127.0.0.1:") {
        let port_start = start + "http://127.0.0.1:".len();
        if let Some(end) = iframe_html[port_start..].find('"') {
            return iframe_html[port_start..port_start + end].parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_port_from_iframe() {
        let iframe = r#"<iframe src="http://127.0.0.1:8080/" width="100%" height="420"></iframe>"#;
        assert_eq!(extract_port_from_iframe(iframe), Some(8080));

        let no_port = r#"<iframe src="http://example.com" width="100%" height="420"></iframe>"#;
        assert_eq!(extract_port_from_iframe(no_port), None);
    }
}
