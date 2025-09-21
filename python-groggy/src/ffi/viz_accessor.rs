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