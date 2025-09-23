//! FFI VizAccessor implementation for interactive visualization.
//!
//! Provides a pandas-like .viz accessor for visualization operations on subgraphs,
//! arrays, and tables with support for different backends (jupyter, server).

use pyo3::prelude::*;
use groggy::api::graph::GraphDataSource;
use groggy::errors::GraphResult;

/// VizAccessor provides visualization methods for groggy objects
#[pyclass]
pub struct VizAccessor {
    /// The underlying data source for visualization
    data_source: Option<GraphDataSource>,
    /// Object type for fallback handling
    object_type: String,
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

    /// Show visualization with auto-display in Jupyter or return HTML for non-Jupyter
    fn show(&self, py: Python) -> PyResult<PyObject> {
        if let Some(ref data_source) = self.data_source {
            let iframe_html = py.allow_threads(|| {
                data_source.interactive_embed()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Jupyter visualization failed: {}", e)
                    ))
            })?;

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
        } else {
            let fallback = self.create_fallback_visualization();
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

    /// Show visualization in standalone server mode
    fn server(&self, py: Python) -> PyResult<()> {
        if let Some(ref data_source) = self.data_source {
            let iframe_html = py.allow_threads(|| {
                data_source.interactive_embed()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        format!("Server visualization failed: {}", e)
                    ))
            })?;

            // Extract port from iframe and open in browser
            if let Some(port) = extract_port_from_iframe(&iframe_html) {
                println!("🚀 Visualization server started at http://127.0.0.1:{}", port);

                // Open in browser using Python's webbrowser module
                py.run(&format!(
                    "import webbrowser; webbrowser.open('http://127.0.0.1:{}')",
                    port
                ), None, None)?;

                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not extract server port from visualization"
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                format!("Server visualization not yet implemented for {}", self.object_type)
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
        Self {
            data_source: Some(data_source),
            object_type,
        }
    }

    /// Create a new VizAccessor without a data source (for fallback)
    pub fn without_data_source(object_type: String) -> Self {
        Self {
            data_source: None,
            object_type,
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
        let iframe = r#"<iframe src="http://127.0.0.1:8080" width="100%" height="420"></iframe>"#;
        assert_eq!(extract_port_from_iframe(iframe), Some(8080));

        let no_port = r#"<iframe src="http://example.com" width="100%" height="420"></iframe>"#;
        assert_eq!(extract_port_from_iframe(no_port), None);
    }
}