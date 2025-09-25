//! FFI VizAccessor implementation for interactive visualization.
//!
//! Provides a pandas-like .viz accessor for visualization operations on subgraphs,
//! arrays, and tables with support for different backends (jupyter, server).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use groggy::api::graph::GraphDataSource;
use groggy::errors::GraphResult;
use std::net::TcpListener;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures_util::{SinkExt, StreamExt};

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
    for port in start_port..=start_port+200 {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => {
                eprintln!("🔍 DEBUG: Port {} is available", port);
                return Ok(port);
            },
            Err(e) => {
                eprintln!("🔍 DEBUG: Port {} unavailable: {}", port, e);
                continue;
            }
        }
    }
    Err(format!("No available ports found in range {}-{}", start_port, start_port+200))
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
    /// Show honeycomb visualization with n-dimensional rotation controls
    fn show_honeycomb(&self, py: Python) -> PyResult<PyObject> {
        let iframe_html = format!(r#"
<div style="border: 2px dashed #ff6b6b; padding: 20px; margin: 10px; border-radius: 8px; background: #f8f9fa;">
    <h3 style="color: #ff6b6b; margin-top: 0;">🍯 Honeycomb N-Dimensional Controls</h3>
    <p><strong>Canvas Dragging for N-Dimensional Rotation:</strong></p>
    <ul>
        <li><strong>Left Mouse + Drag:</strong> Rotate in dimensions 0-1</li>
        <li><strong>Left + Shift + Drag:</strong> Rotate in dimensions 0-1 (explicit)</li>
        <li><strong>Left + Ctrl + Drag:</strong> Rotate in higher dimensions (2-3)</li>
        <li><strong>Right Mouse + Drag:</strong> Multi-dimensional rotation (0-2, 1-3)</li>
        <li><strong>Middle Mouse + Drag:</strong> Rotate across all dimension pairs</li>
    </ul>
    <p><strong>Node Dragging for Direct Manipulation:</strong></p>
    <ul>
        <li><strong>Drag Nodes:</strong> Move individual points in n-dimensional space</li>
        <li><strong>Screen X/Y:</strong> Maps to multi-dimensional coordinates</li>
        <li><strong>Higher Dimensions:</strong> Affected based on movement patterns</li>
    </ul>
    <p><strong>Features:</strong></p>
    <ul>
        <li>✨ <strong>Momentum Rotation:</strong> Smooth rotation continues after release</li>
        <li>🎯 <strong>Real-time Updates:</strong> Immediate visual feedback via WebSocket</li>
        <li>⚡ <strong>60 FPS:</strong> Smooth animations and interactions</li>
        <li>🔧 <strong>Adaptive Quality:</strong> Performance optimization based on complexity</li>
    </ul>
    <p style="color: #666; font-style: italic;">Note: This is the advanced honeycomb layout with 5D embeddings projected to 2D hexagonal grid.
    Use regular .show() for traditional force-directed layouts.</p>
    <iframe src="http://localhost:8080" width="100%" height="600" frameborder="0" style="border: 1px solid #ddd; border-radius: 4px; margin-top: 10px;">
        <p>Honeycomb visualization with n-dimensional rotation controls</p>
    </iframe>
</div>
"#);

        // Auto-display in Jupyter using display(HTML())
        py.run(&format!(
            r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
except ImportError:
    print("IPython not available for auto-display")
except Exception as e:
    print(f"Display error: {{e}}")
"#,
            html = iframe_html.replace("'", "\\'")
        ), None, None)?;

        Ok(py.None())
    }

    /// Show visualization with Realtime backend using proper DataSource integration
    /// Now supports server reuse - updates existing servers instead of creating new ones
    #[pyo3(signature = (layout = "honeycomb".to_string(), **kwargs))]
    fn show(&self, py: Python, layout: String, kwargs: Option<&pyo3::types::PyDict>) -> PyResult<PyObject> {
        eprintln!("🚀 DEBUG: VizAccessor.show() called with layout='{}' and kwargs: {:?}", layout, kwargs);
        eprintln!("📊 DEBUG: DataSource ID: {}", self.data_source_id);

        // Extract layout parameters from kwargs
        let mut layout_params = std::collections::HashMap::new();
        if let Some(kwargs_dict) = kwargs {
            eprintln!("📝 DEBUG: Processing layout parameters from kwargs...");
            for (key, value) in kwargs_dict.iter() {
                if let Ok(key_str) = key.extract::<String>() {
                    if let Ok(value_str) = value.extract::<String>() {
                        layout_params.insert(key_str.clone(), value_str.clone());
                        eprintln!("  🔧 Parameter: {}={}", key_str, value_str);
                    } else if let Ok(value_int) = value.extract::<i64>() {
                        layout_params.insert(key_str.clone(), value_int.to_string());
                        eprintln!("  🔧 Parameter: {}={}", key_str, value_int);
                    } else if let Ok(value_float) = value.extract::<f64>() {
                        layout_params.insert(key_str.clone(), value_float.to_string());
                        eprintln!("  🔧 Parameter: {}={}", key_str, value_float);
                    }
                }
            }
        }

        eprintln!("📊 DEBUG: Final layout parameters: algorithm='{}', params={:?}", layout, layout_params);

        if let Some(ref data_source) = self.data_source {
            // Check if we have an existing server for this data source
            let existing_server = {
                let registry = SERVER_REGISTRY.lock().unwrap();
                registry.values().find(|info| info.data_source_id == self.data_source_id).map(|info| info.port)
            };

            if let Some(existing_port) = existing_server {
                eprintln!("🔄 DEBUG: Found existing server on port {} - sending ChangeLayout control message", existing_port);
                return self.update_existing_server(py, existing_port, layout, layout_params);
            }

            eprintln!("🆕 DEBUG: No existing server found - creating new server");
            let iframe_html = py.allow_threads(move || -> Result<String, String> {
                eprintln!("✅ DEBUG: Found user's GraphDataSource - implementing Phase 1 integration");

                // Phase 1: Create DataSourceRealtimeAccessor with layout
                use groggy::viz::realtime::accessor::{DataSourceRealtimeAccessor, RealtimeVizAccessor};
                use groggy::viz::streaming::data_source::LayoutAlgorithm;
                use std::sync::Arc;

                // Convert layout string and params to LayoutAlgorithm
                let layout_algorithm = match layout.as_str() {
                    "force_directed" | "force-directed" | "spring" => {
                        let iterations = layout_params.get("iterations")
                            .and_then(|s| s.parse().ok()).unwrap_or(100);
                        let charge = layout_params.get("charge")
                            .and_then(|s| s.parse().ok()).unwrap_or(-300.0);
                        let distance = layout_params.get("distance")
                            .and_then(|s| s.parse().ok()).unwrap_or(50.0);

                        eprintln!("⚡ DEBUG: Creating ForceDirected layout: iterations={}, charge={}, distance={}",
                                 iterations, charge, distance);

                        LayoutAlgorithm::ForceDirected { charge, distance, iterations }
                    },
                    "circular" | "circle" => {
                        let radius = layout_params.get("radius")
                            .and_then(|s| s.parse::<f64>().ok());
                        let start_angle = layout_params.get("start_angle")
                            .and_then(|s| s.parse().ok()).unwrap_or(0.0);

                        eprintln!("⭕ DEBUG: Creating Circular layout: radius={:?}, start_angle={}",
                                 radius, start_angle);

                        LayoutAlgorithm::Circular { radius, start_angle }
                    },
                    "grid" | "matrix" => {
                        let columns = layout_params.get("columns")
                            .and_then(|s| s.parse().ok()).unwrap_or(5);
                        let cell_size = layout_params.get("cell_size")
                            .and_then(|s| s.parse().ok()).unwrap_or(100.0);

                        eprintln!("▦ DEBUG: Creating Grid layout: columns={}, cell_size={}",
                                 columns, cell_size);

                        LayoutAlgorithm::Grid { columns, cell_size }
                    },
                    "honeycomb" | "hexagonal" | "hex" => {
                        let cell_size = layout_params.get("cell_size")
                            .and_then(|s| s.parse().ok()).unwrap_or(40.0);
                        let energy_optimization = layout_params.get("energy_optimization")
                            .and_then(|s| s.parse().ok()).unwrap_or(true);
                        let iterations = layout_params.get("iterations")
                            .and_then(|s| s.parse().ok()).unwrap_or(100);

                        eprintln!("🔶 DEBUG: Creating Honeycomb layout: cell_size={}, energy_optimization={}, iterations={}",
                                 cell_size, energy_optimization, iterations);

                        LayoutAlgorithm::Honeycomb { cell_size, energy_optimization, iterations }
                    },
                    _ => {
                        eprintln!("⚠️  DEBUG: Unknown layout '{}', defaulting to honeycomb", layout);
                        LayoutAlgorithm::Honeycomb { cell_size: 40.0, energy_optimization: true, iterations: 100 }
                    }
                };

                // Convert GraphDataSource to Arc<dyn DataSource>
                let data_source_arc = Arc::new(data_source.clone());
                let accessor = DataSourceRealtimeAccessor::with_layout(data_source_arc, layout_algorithm);

                eprintln!("🔧 DEBUG: Created DataSourceRealtimeAccessor");

                // Get initial snapshot using proper accessor
                match accessor.initial_snapshot() {
                    Ok(snapshot) => {
                        eprintln!("📊 DEBUG: Got snapshot: {} nodes, {} edges",
                                 snapshot.node_count(), snapshot.edge_count());

                        // Phase 2: Start realtime server with accessor
                        eprintln!("🚀 DEBUG: Starting Phase 2 realtime server with accessor");
                        use groggy::viz::realtime::server::create_realtime_server;

                        let accessor_arc = Arc::new(accessor);

                        // Find an available port starting from specified or default port
                        let start_port = layout_params.get("port")
                            .and_then(|s| s.parse::<u16>().ok())
                            .unwrap_or(8080);
                        let port = find_available_port(start_port)?;
                        eprintln!("🔍 DEBUG: Found available port: {} (searched from {})", port, start_port);

                        // Register this server in our registry
                        {
                            let mut registry = SERVER_REGISTRY.lock().unwrap();
                            registry.insert(port, ServerInfo {
                                port,
                                data_source_id: self.data_source_id.clone(),
                            });
                            eprintln!("📝 DEBUG: Registered server on port {} for data source {}", port, self.data_source_id);
                        }

                        // Use the same threading pattern as streaming server
                        // Start server in background thread (like streaming server does)
                        eprintln!("🚀 DEBUG: Starting realtime server using streaming server pattern");
                        
                        let server_accessor = accessor_arc.clone();
                        let server_port = port;
                        
                        // Start server in background thread (same pattern as streaming server)
                        std::thread::Builder::new()
                            .name("groggy-realtime".into())
                            .spawn(move || {
                                let rt = tokio::runtime::Runtime::new().expect("tokio runtime for realtime server");
                                rt.block_on(async move {
                                    match create_realtime_server(server_port, server_accessor) {
                                        Ok(server) => {
                                            eprintln!("✅ DEBUG: Realtime server created on port {}", server_port);
                                            if let Err(e) = server.start().await {
                                                eprintln!("❌ DEBUG: Realtime server error: {}", e);
                                            }
                                        },
                                        Err(e) => {
                                            eprintln!("❌ DEBUG: Failed to create realtime server: {}", e);
                                        }
                                    }
                                });
                            })
                            .expect("Failed to spawn realtime server thread");
                        
                        // Give server a moment to start up
                        std::thread::sleep(std::time::Duration::from_millis(200));
                        
                        // Create iframe HTML pointing to our realtime server (now with static file serving)
                        let iframe_html = format!(
                            r#"<div style="position: relative;">
    <iframe src="http://127.0.0.1:{}/" width="100%" height="640" frameborder="0" style="border: 1px solid #ddd;"></iframe>
    <div style="font-size: 12px; color: #666; margin-top: 5px;">
        🍯 <strong>Realtime Server</strong> on port {} | 🎮 Static Files + WebSocket + Canvas + Interactive controls
        <div style="margin-top: 3px;">
            ✅ File-based UI (HTML/JS/CSS) with /config endpoint | ✅ Layout parameters: {}
        </div>
    </div>
</div>"#,
                            port, port, layout
                        );
                        
                        eprintln!("📊 DEBUG: Created realtime server iframe HTML");
                        Ok(iframe_html)
                    },
                    Err(e) => {
                        eprintln!("❌ DEBUG: Snapshot creation failed: {}", e);
                        Err(format!("Snapshot creation failed: {}", e))
                    }
                }
            });

            match iframe_html {
                Ok(html) => {
                    // Auto-display in Jupyter using display(HTML())
                    py.run(&format!(
                        r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
    print("🍯 DEBUG: Using Realtime integration (Phase 2 complete - WebSocket + Canvas)")
except ImportError:
    print("IPython not available for auto-display")
except Exception as e:
    print(f"Display error: {{e}}")
"#,
                        html = html.replace("'", "\\'")
                    ), None, None)?;

                    Ok(py.None())
                },
                Err(e) => {
                    eprintln!("❌ DEBUG: Realtime integration failed: {}", e);
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Realtime visualization failed: {}", e)
                    ))
                }
            }
        } else {
            let fallback = self.create_fallback_visualization();
            py.run(&format!(
                r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
    print("⚠️  DEBUG: No data source available - using fallback")
except ImportError:
    print("IPython not available for auto-display")
except Exception as e:
    print(f"Display error: {{e}}")
"#,
                html = fallback.replace("'", "\\'")
            ), None, None)?;
            Ok(py.None())
        }
    }

    /// Show visualization with honeycomb layout
    fn honeycomb(&self, py: Python, cell_size: Option<f64>, energy_optimization: Option<bool>, iterations: Option<usize>) -> PyResult<PyObject> {
        eprintln!("🔍 Python FFI: honeycomb method called with cell_size={:?}, energy_opt={:?}, iterations={:?}", 
                 cell_size, energy_optimization, iterations);
        
        if let Some(ref data_source) = self.data_source {
            eprintln!("🔍 Python FFI: data_source found, calling interactive_embed_with_layout");
            
            // Create honeycomb layout with specified parameters
            let honeycomb_html = py.allow_threads(|| {
                use groggy::viz::streaming::data_source::LayoutAlgorithm;

                let layout = LayoutAlgorithm::Honeycomb {
                    cell_size: cell_size.unwrap_or(40.0),
                    energy_optimization: energy_optimization.unwrap_or(true),
                    iterations: iterations.unwrap_or(100),
                };

                eprintln!("🔍 Python FFI: About to call interactive_embed_with_layout with {:?}", layout);
                let result = data_source.interactive_embed_with_layout(layout);
                eprintln!("🔍 Python FFI: interactive_embed_with_layout returned");
                
                result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Honeycomb visualization failed: {}", e)
                    ))
            })?;

            // Create and return IPython HTML object
            let html_obj = py.run(&format!(
                r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
    print("🍯 Honeycomb layout applied (cell_size: {cell_size}, optimization: {energy_opt}, iterations: {iterations})")
    _result = _html_obj
except ImportError:
    print("IPython not available for auto-display")
    _result = None
except Exception as e:
    print(f"Display error: {{e}}")
    _result = None
"#,
                html = honeycomb_html.replace("'", "\\'"),
                cell_size = cell_size.unwrap_or(40.0),
                energy_opt = energy_optimization.unwrap_or(true),
                iterations = iterations.unwrap_or(100)
            ), None, None)?;

            // Return the HTML object or None if IPython not available
            match py.eval("_result", None, None) {
                Ok(obj) => Ok(obj.to_object(py)),
                Err(_) => Ok(py.None()),
            }
        } else {
            let fallback = self.create_fallback_visualization();
            py.run(&format!(
                r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
    print("🍯 Honeycomb layout requested but not available for this object type")
except ImportError:
    print("IPython not available for auto-display")
except Exception as e:
    print(f"Display error: {{e}}")
"#,
                html = fallback.replace("'", "\\'")
            ), None, None)?;
            Ok(py.None())
        }
    }

    /// Show visualization in standalone server mode using realtime backend
    fn server(&self, py: Python) -> PyResult<PyObject> {
        eprintln!("🚀 DEBUG: VizAccessor.server() called - using REALTIME BACKEND!");

        if let Some(ref data_source) = self.data_source {
            let data_source_id = self.data_source_id.clone();
            let server_handle = py.allow_threads(|| -> Result<(u16, String), String> {
                eprintln!("✅ DEBUG: Found user's GraphDataSource - starting realtime server");

                // Create DataSourceRealtimeAccessor
                use groggy::viz::realtime::accessor::{DataSourceRealtimeAccessor, RealtimeVizAccessor};
                use std::sync::Arc;

                let data_source_arc = Arc::new(data_source.clone());
                let accessor = DataSourceRealtimeAccessor::new(data_source_arc);

                eprintln!("🔧 DEBUG: Created DataSourceRealtimeAccessor");

                // Get initial snapshot using proper accessor
                match accessor.initial_snapshot() {
                    Ok(snapshot) => {
                        eprintln!("📊 DEBUG: Got snapshot: {} nodes, {} edges",
                                 snapshot.node_count(), snapshot.edge_count());

                        // Start realtime server with accessor
                        eprintln!("🚀 DEBUG: Starting realtime server with accessor");
                        use groggy::viz::realtime::server::start_realtime_background;

                        let accessor_arc = Arc::new(accessor);

                        // Find an available port starting from 8080
                        let port = find_available_port(8080).unwrap_or(8080);
                        eprintln!("🔍 DEBUG: Found available port: {}", port);

                        // Start server in background
                        match start_realtime_background(port, accessor_arc) {
                            Ok(handle) => {
                                let actual_port = handle.port;
                                eprintln!("✅ DEBUG: Realtime server started on port {}", actual_port);

                                // Convert handle to a string representation for Python
                                let handle_info = format!("RealtimeServerHandle(port={})", actual_port);
                                Ok((actual_port, handle_info))
                            },
                            Err(e) => {
                                eprintln!("❌ DEBUG: Failed to start realtime server: {}", e);
                                Err(format!("Failed to start realtime server: {}", e))
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("❌ DEBUG: Snapshot creation failed: {}", e);
                        Err(format!("Snapshot creation failed: {}", e))
                    }
                }
            });

            match server_handle {
                Ok((port, _handle_info)) => {
                    // Register server in global registry
                    {
                        let mut registry = SERVER_REGISTRY.lock().unwrap();
                        registry.insert(port, ServerInfo {
                            port,
                            data_source_id: data_source_id.clone(),
                        });
                        eprintln!("📝 DEBUG: Registered server on port {} for data source {}", port, data_source_id);
                    }

                    println!("🚀 Realtime visualization server started at http://127.0.0.1:{}/", port);
                    println!("🎮 WebSocket endpoint: ws://127.0.0.1:{}/ws", port);

                    // Open in browser using Python's webbrowser module
                    py.run(&format!(
                        "import webbrowser; webbrowser.open('http://127.0.0.1:{}/')",
                        port
                    ), None, None)?;

                    // Return a simple server info object
                    let server_info = py.eval(&format!(
                        "type('ServerInfo', (), {{'port': {}, 'url': 'http://127.0.0.1:{}/', 'stop': lambda: print('Use Ctrl+C to stop server')}})()",
                        port, port
                    ), None, None)?;

                    Ok(server_info.to_object(py))
                },
                Err(e) => {
                    eprintln!("❌ DEBUG: Realtime server startup failed: {}", e);
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Realtime server startup failed: {}", e)
                    ))
                }
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                format!("Server visualization not yet implemented for {}", self.object_type)
            ))
        }
    }

    /// Update visualization parameters - sends control message to existing server
    #[pyo3(signature = (**kwargs))]
    fn update(&self, py: Python, kwargs: Option<&PyDict>) -> PyResult<()> {
        eprintln!("🔄 DEBUG: VizAccessor.update() called");

        // Check if we have an active server for this viz instance
        if let Some(server_info) = self.get_server_info() {
            let port = server_info.port;
            eprintln!("📡 DEBUG: Found active server on port {} for this viz instance", port);

            // Parse layout parameters from kwargs
            let (layout, layout_params) = if let Some(kwargs) = kwargs {
                self.parse_layout_parameters(kwargs)?
            } else {
                // Default to current layout with no changes
                ("force_directed".to_string(), std::collections::HashMap::new())
            };

            eprintln!("🎯 DEBUG: Updating server with layout: {}, params: {:?}", layout, layout_params);

            // Send control message to existing server
            self.send_control_message_to_server(py, port, layout, layout_params)?;

            eprintln!("✅ DEBUG: Visualization parameters updated successfully");
            Ok(())
        } else {
            eprintln!("❌ DEBUG: No active server found for this viz instance");
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

    /// Send control message to server - extracted from update_existing_server for reuse
    fn send_control_message_to_server(&self, py: Python, port: u16, layout: String, layout_params: HashMap<String, String>) -> PyResult<()> {
        eprintln!("📡 DEBUG: Sending ChangeLayout control message to server on port {}", port);

        let result = py.allow_threads(move || -> Result<String, String> {
            // Create tokio runtime for WebSocket client
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

            rt.block_on(async {
                // Connect to the existing server's WebSocket endpoint
                let ws_url = format!("ws://127.0.0.1:{}/ws", port);
                eprintln!("🔗 DEBUG: Connecting to WebSocket: {}", ws_url);

                match connect_async(&ws_url).await {
                    Ok((mut ws_stream, _)) => {
                        eprintln!("✅ DEBUG: WebSocket connected successfully");

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

                        eprintln!("📡 DEBUG: Sending WebSocket message: {}", message_text);

                        // Send the control message
                        match ws_stream.send(Message::Text(message_text)).await {
                            Ok(()) => {
                                eprintln!("✅ DEBUG: ChangeLayout control message sent successfully");

                                // Wait briefly for server to process and send position updates
                                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                                // Close the WebSocket connection cleanly
                                let _ = ws_stream.close(None).await;
                                Ok("Control message sent successfully".to_string())
                            }
                            Err(e) => {
                                eprintln!("❌ DEBUG: Failed to send WebSocket message: {}", e);
                                Err(format!("Failed to send WebSocket message: {}", e))
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ DEBUG: Failed to connect to WebSocket {}: {}", ws_url, e);
                        Err(format!("Failed to connect to WebSocket: {}", e))
                    }
                }
            })
        });

        match result {
            Ok(_) => {
                eprintln!("🔄 DEBUG: Server update successful - layout parameters sent");
                Ok(())
            },
            Err(e) => {
                eprintln!("❌ DEBUG: Failed to send control message: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to send control message: {}", e)
                ))
            }
        }
    }

    /// Parse layout parameters from Python kwargs
    fn parse_layout_parameters(&self, kwargs: &PyDict) -> PyResult<(String, HashMap<String, String>)> {
        let mut layout = "force_directed".to_string();
        let mut layout_params = HashMap::new();

        eprintln!("📝 DEBUG: Processing layout parameters from kwargs...");

        for (key, value) in kwargs.iter() {
            let key_str = key.to_string();
            if key_str == "layout" {
                layout = value.to_string();
            } else {
                let value_str = value.to_string();
                layout_params.insert(key_str.clone(), value_str.clone());
                eprintln!("  🔧 Parameter: {}={}", key_str, value_str);
            }
        }

        eprintln!("📊 DEBUG: Final layout parameters: algorithm='{}', params={:?}", layout, layout_params);
        Ok((layout, layout_params))
    }

    /// Update existing server with new layout parameters via WebSocket ControlMsg
    fn update_existing_server(&self, py: Python, port: u16, layout: String, layout_params: HashMap<String, String>) -> PyResult<PyObject> {
        eprintln!("📡 DEBUG: Sending ChangeLayout control message to server on port {}", port);

        let result = py.allow_threads(move || -> Result<String, String> {
            use groggy::viz::realtime::accessor::ControlMsg;

            // Build the control message
            let control_msg = ControlMsg::ChangeLayout {
                algorithm: layout.clone(),
                params: layout_params.clone(),
            };

            eprintln!("📤 DEBUG: ChangeLayout message created: algorithm={}, params={:?}", layout, layout_params);

            // Create tokio runtime for WebSocket client
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

            let send_result = rt.block_on(async {
                // Connect to the existing server's WebSocket endpoint
                let ws_url = format!("ws://127.0.0.1:{}/ws", port);
                eprintln!("🔗 DEBUG: Connecting to WebSocket: {}", ws_url);

                match connect_async(&ws_url).await {
                    Ok((mut ws_stream, _)) => {
                        eprintln!("✅ DEBUG: WebSocket connected successfully");

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

                        eprintln!("📡 DEBUG: Sending WebSocket message: {}", message_text);

                        // Send the control message
                        match ws_stream.send(Message::Text(message_text)).await {
                            Ok(()) => {
                                eprintln!("✅ DEBUG: ChangeLayout control message sent successfully");

                                // Close the WebSocket connection cleanly
                                let _ = ws_stream.close(None).await;
                                Ok("Control message sent successfully".to_string())
                            }
                            Err(e) => {
                                eprintln!("❌ DEBUG: Failed to send WebSocket message: {}", e);
                                Err(format!("Failed to send WebSocket message: {}", e))
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ DEBUG: Failed to connect to WebSocket {}: {}", ws_url, e);
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
                    eprintln!("❌ DEBUG: WebSocket control message failed: {}", e);
                    Err(format!("WebSocket control message failed: {}", e))
                }
            }
        });

        match result {
            Ok(html) => {
                // Auto-display in Jupyter using display(HTML())
                py.run(&format!(
                    r#"
try:
    from IPython.display import HTML, display
    _html_obj = HTML(r'''{html}''')
    display(_html_obj)
    print("🔄 DEBUG: Server reuse successful - layout updated without creating new server")
except ImportError:
    print("IPython not available for auto-display")
except Exception as e:
    print(f"Display error: {{e}}")
"#,
                    html = html.replace("'", "\\'")
                ), None, None)?;

                Ok(py.None())
            },
            Err(e) => {
                eprintln!("❌ DEBUG: Failed to update existing server: {}", e);
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to update existing server: {}", e)
                ))
            }
        }
    }
}

/// Extract port number from iframe HTML
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
