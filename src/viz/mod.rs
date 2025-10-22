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
//!
//! ## Multi-dimensional Embedding System
//!
//! The visualization system includes a comprehensive multi-dimensional embedding
//! framework for advanced graph layouts, particularly the honeycomb layout with
//! programmable projections and energy-based optimization.

use crate::errors::{GraphError, GraphResult};
// use ; // TODO: add missing import
use std::net::IpAddr;
use std::sync::Arc;
use streaming::data_source::{DataSource, LayoutAlgorithm};
use streaming::server::StreamingServer;
use streaming::types::StreamingConfig;
use streaming::virtual_scroller::VirtualScrollConfig;

pub mod embeddings;
pub mod projection;
pub mod realtime;

/// Visualization backend options for unified rendering
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VizBackend {
    /// Jupyter notebook embedding with optimized display
    Jupyter,
    /// WebSocket interactive server with real-time updates
    Streaming,
    /// Advanced real-time visualization with Phase 3 features
    Realtime,
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
            "realtime" => Ok(VizBackend::Realtime),
            "file" => Ok(VizBackend::File),
            "local" => Ok(VizBackend::Local),
            _ => Err(GraphError::InvalidInput(format!(
                "Invalid backend '{}'. Expected: jupyter, streaming, realtime, file, or local",
                backend
            ))),
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            VizBackend::Jupyter => "jupyter",
            VizBackend::Streaming => "streaming",
            VizBackend::Realtime => "realtime",
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
    pub verbose: Option<u8>, // Verbosity level: 0=quiet, 1=info, 2=verbose, 3=debug

    // Backend-specific parameters
    pub port: Option<u16>,            // For streaming backend
    pub filename: Option<String>,     // For file backend
    pub format: Option<ExportFormat>, // For file backend
    pub dpi: Option<u32>,             // For file backend
    pub auto_open: Option<bool>,      // For streaming backend
}

/// Result of rendering operation (different types for different backends)
pub enum RenderResult {
    /// HTML content (for jupyter and local backends)
    Html(String),
    /// Interactive visualization session (for streaming backend)
    Interactive(InteractiveViz),
    /// Static visualization result (for file backend)
    Static(StaticViz),
    /// Real-time visualization with Phase 3 features (for realtime backend)
    RealTime(RealTimeVisualization),
}

// Working visualization modules
pub mod display; // Table/array/matrix formatting (essential for storage)
pub mod layouts;
pub mod streaming; // The working WebSocket + Canvas server // Graph layout algorithms (used by streaming)

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

    // Start realtime visualization server
    // pub fn server(&mut self, port: Option<u16>) -> GraphResult<RenderResult> {
    //     eprintln!("🚀 INFO: server() called - starting REALTIME backend server!");
    //     let options = RenderOptions {
    //         port,
    //         ..Default::default()
    //     };
    //     self.render(VizBackend::Realtime, options)
    // }

    /// Save to file (unified core)
    pub fn save(&mut self, path: &str) -> GraphResult<RenderResult> {
        let options = RenderOptions {
            filename: Some(path.to_string()),
            ..Default::default()
        };
        self.render(VizBackend::File, options)
    }

    /// 🚀 Show interactive real-time visualization
    ///
    /// This is the main visualization method that launches the real-time system with:
    /// - Interactive controls for all layout algorithms including honeycomb n-dimensional rotation
    /// - Real-time parameter adjustment and streaming updates
    /// - Performance monitoring with adaptive quality
    /// - WebSocket broadcasting for live updates
    /// - Support for traditional layouts (force-directed, etc.) and honeycomb layout
    /// - N-dimensional embeddings with UMAP, t-SNE, PCA projections
    /// - Advanced physics simulation and momentum-based interactions
    ///
    /// # Arguments
    /// * `verbose` - Verbosity level (0=quiet, 1=info, 2=verbose, 3=debug). Defaults to 0.
    ///
    /// # Returns
    /// * `RenderResult::RealTime` - Contains the real-time visualization engine with advanced features
    ///
    /// # Examples
    /// ```ignore
    /// use groggy::viz::VizModule;
    ///
    /// let mut viz = graph.viz();
    /// let result = viz.show(Some(2))?; // Verbose debug output
    /// let result = viz.show(Some(0))?; // Quiet mode
    /// let result = viz.show(None)?; // Quiet mode (default)
    ///
    /// // Real-time visualization with n-dimensional honeycomb controls is now running
    /// ```ignore
    pub fn show(&mut self, verbose: Option<u8>) -> GraphResult<RenderResult> {
        let verbose = verbose.unwrap_or(0);

        if verbose >= 1 {
            eprintln!("🚀 INFO: show() called - using REALTIME backend");
        }

        if verbose >= 2 {
            eprintln!(
                "📊 VERBOSE: Data source type: {}",
                self.data_source.get_schema().source_type
            );
        }

        if self.data_source.supports_graph_view() && verbose >= 2 {
            let metadata = self.data_source.get_graph_metadata();
            eprintln!(
                "🍯 VERBOSE: Graph data - {} nodes, {} edges",
                metadata.node_count, metadata.edge_count
            );
        }

        let options = RenderOptions {
            verbose: Some(verbose),
            ..Default::default()
        };
        self.render(VizBackend::Realtime, options)
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
    /// ```ignore
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
    /// ```ignore
    pub fn render(&self, backend: VizBackend, options: RenderOptions) -> GraphResult<RenderResult> {
        let verbose = options.verbose.unwrap_or(0);

        if verbose >= 3 {
            eprintln!("🎯 DEBUG: render() called with backend: {:?}", backend);
        }
        match backend {
            VizBackend::Jupyter => {
                if verbose >= 3 {
                    eprintln!("📓 DEBUG: Calling render_jupyter()");
                }
                self.render_jupyter(options)
            }
            VizBackend::Streaming => {
                if verbose >= 2 {
                    eprintln!("🌊 VERBOSE: Streaming backend DISABLED - redirecting to realtime");
                }
                self.render_realtime(options)
            }
            VizBackend::Realtime => {
                if verbose >= 3 {
                    eprintln!("⚡ DEBUG: Calling render_realtime()");
                }
                self.render_realtime(options)
            }
            VizBackend::File => {
                if verbose >= 3 {
                    eprintln!("📄 DEBUG: Calling render_file()");
                }
                self.render_file(options)
            }
            VizBackend::Local => {
                if verbose >= 3 {
                    eprintln!("🏠 DEBUG: Calling render_local()");
                }
                self.render_local(options)
            }
        }
    }

    /// Render for Jupyter notebook with optimized embedding
    fn render_jupyter(&self, options: RenderOptions) -> GraphResult<RenderResult> {
        // Generate Jupyter-optimized HTML
        let html = self.generate_jupyter_html(&options)?;
        Ok(RenderResult::Html(html))
    }

    /// Render for advanced real-time visualization with Phase 3 features
    fn render_realtime(&self, options: RenderOptions) -> GraphResult<RenderResult> {
        let verbose = options.verbose.unwrap_or(0);

        if verbose >= 3 {
            eprintln!("🔥 DEBUG: render_realtime() called!");
            eprintln!("⚙️  DEBUG: Importing realtime modules...");
        }

        // Use fully-qualified paths for realtime types to avoid name collisions with streaming::types::StreamingConfig
        use crate::viz::embeddings::{EmbeddingConfig, EmbeddingMethod};
        use crate::viz::projection::{
            HoneycombConfig, InterpolationConfig, ProjectionConfig, ProjectionMethod, QualityConfig,
        };
        use crate::viz::realtime::{InteractionConfig, RealTimeConfig, RealTimeVizConfig};

        if verbose >= 3 {
            eprintln!("✅ DEBUG: Realtime modules imported successfully!");
        }

        // Determine embedding and projection methods based on layout
        // Default to Honeycomb for real-time backend if no layout specified
        let layout = options
            .layout
            .as_ref()
            .unwrap_or(&LayoutAlgorithm::Honeycomb {
                cell_size: 40.0,
                energy_optimization: true,
                iterations: 500,
            });

        if verbose >= 2 {
            eprintln!("🍯 VERBOSE: Layout determined: {:?}", layout);
        }
        let (embedding_method, projection_method, is_honeycomb) = match layout {
            LayoutAlgorithm::Honeycomb { .. } => {
                if verbose >= 2 {
                    eprintln!("🍯 VERBOSE: HONEYCOMB LAYOUT DETECTED!");
                    eprintln!(
                        "🔥 VERBOSE: Setting up N-DIMENSIONAL EMBEDDINGS with EnergyND method"
                    );
                    eprintln!("🎯 VERBOSE: Setting up UMAP projection for multi-dimensional space");
                    eprintln!("🎮 VERBOSE: Honeycomb controls will be ENABLED (is_honeycomb=true)");
                }
                // For honeycomb layout, use multi-dimensional embedding + honeycomb projection
                (
                    EmbeddingMethod::EnergyND {
                        iterations: 500,
                        learning_rate: 0.01,
                        annealing: true,
                    },
                    ProjectionMethod::UMAP {
                        n_neighbors: 15,
                        min_dist: 0.1,
                        n_epochs: 200,
                        negative_sample_rate: 5.0,
                    },
                    true,
                )
            }
            LayoutAlgorithm::ForceDirected { .. } => {
                // For force-directed, use 2D embedding with PCA projection
                (
                    EmbeddingMethod::EnergyND {
                        iterations: 300,
                        learning_rate: 0.02,
                        annealing: false,
                    },
                    ProjectionMethod::PCA {
                        center: true,
                        standardize: false,
                    },
                    false,
                )
            }
            LayoutAlgorithm::Circular { .. } => {
                // For circular layout, use spectral embedding
                (
                    EmbeddingMethod::Spectral {
                        normalized: true,
                        eigenvalue_threshold: 1e-10,
                    },
                    ProjectionMethod::PCA {
                        center: true,
                        standardize: true,
                    },
                    false,
                )
            }
            _ => {
                // Default to energy-based with PCA projection
                (
                    EmbeddingMethod::EnergyND {
                        iterations: 400,
                        learning_rate: 0.015,
                        annealing: true,
                    },
                    ProjectionMethod::PCA {
                        center: true,
                        standardize: true,
                    },
                    false,
                )
            }
        };

        if verbose >= 3 {
            eprintln!("⚙️  DEBUG: Creating real-time configuration...");
        }

        // Create real-time configuration with layout-appropriate settings
        let dimensions = if is_honeycomb { 5 } else { 2 };
        if verbose >= 3 {
            eprintln!(
                "📐 DEBUG: Setting embedding dimensions to {} (honeycomb: {})",
                dimensions, is_honeycomb
            );
        }

        let realtime_config = RealTimeVizConfig {
            embedding_config: EmbeddingConfig {
                method: embedding_method,
                dimensions, // Multi-D for honeycomb, 2D for others
                energy_function: None,
                preprocessing: vec![],
                postprocessing: vec![],
                seed: Some(42),
                debug_enabled: false,
            },
            projection_config: ProjectionConfig {
                method: projection_method,
                honeycomb_config: if is_honeycomb {
                    if verbose >= 2 {
                        eprintln!("🍯 VERBOSE: Configuring HONEYCOMB GRID with advanced controls!");
                    }
                    let cell_size = options.width.map(|w| w as f64 / 20.0).unwrap_or(40.0);
                    if verbose >= 3 {
                        eprintln!("📏 DEBUG: Honeycomb cell_size: {}", cell_size);
                        eprintln!("🎯 DEBUG: Using EnergyBased layout strategy");
                        eprintln!("📍 DEBUG: snap_to_centers=true, grid_padding=20.0");
                    }
                    HoneycombConfig {
                        cell_size,
                        layout_strategy:
                            crate::viz::projection::HoneycombLayoutStrategy::EnergyBased,
                        snap_to_centers: true,
                        grid_padding: 20.0,
                        max_grid_size: None,
                        auto_cell_size: true,
                        target_cols: 64,
                        target_rows: 48,
                        scale_multiplier: 1.1,
                        target_avg_occupancy: 4.0,
                        min_cell_size: 6.0,
                    }
                } else {
                    if verbose >= 3 {
                        eprintln!("⚠️  DEBUG: Not honeycomb layout - using default config");
                    }
                    // Default honeycomb config for non-honeycomb layouts (will be ignored)
                    HoneycombConfig {
                        cell_size: 40.0,
                        layout_strategy:
                            crate::viz::projection::HoneycombLayoutStrategy::DistancePreserving,
                        snap_to_centers: false,
                        grid_padding: 0.0,
                        max_grid_size: None,
                        auto_cell_size: true,
                        target_cols: 64,
                        target_rows: 48,
                        scale_multiplier: 1.1,
                        target_avg_occupancy: 4.0,
                        min_cell_size: 6.0,
                    }
                },
                quality_config: QualityConfig {
                    compute_neighborhood_preservation: true,
                    compute_distance_preservation: true,
                    compute_clustering_preservation: false,
                    k_neighbors: 10,
                    optimize_for_quality: true,
                    quality_thresholds: crate::viz::projection::QualityThresholds {
                        min_neighborhood_preservation: 0.7,
                        min_distance_correlation: 0.6,
                        max_stress: 0.3,
                    },
                },
                interpolation_config: InterpolationConfig {
                    enable_interpolation: true,
                    method: crate::viz::projection::InterpolationMethod::Linear,
                    steps: 30,
                    easing: crate::viz::projection::EasingFunction::EaseInOut,
                    preserve_honeycomb: true,
                },
                debug_enabled: false,
                seed: Some(42),
            },
            realtime_config: RealTimeConfig {
                target_fps: 60.0,
                enable_incremental_updates: true,
                frame_time_budget_ms: 16.67, // ~60 FPS
                enable_adaptive_quality: true,
                min_quality_threshold: 0.3,
                enable_position_prediction: true,
                prediction_lookahead_frames: 3,
            },
            performance_config: crate::viz::realtime::PerformanceConfig {
                enable_monitoring: true,
                monitoring_interval_ms: 100,
                frame_time_history_size: 60,
                memory_threshold_mb: 512,
                enable_auto_quality_adaptation: true,
                quality_adaptation_sensitivity: 0.5,
                enable_debug_overlay: false,
            },
            interaction_config: InteractionConfig {
                enable_parameter_controls: true,
                enable_node_selection: true,
                enable_realtime_filtering: true,
                enable_zoom_pan: true,
                zoom_sensitivity: 1.0,
                pan_sensitivity: 1.0,
                selection_config: crate::viz::realtime::SelectionConfig::default(),
                filter_config: crate::viz::realtime::FilterConfig::default(),
            },
            streaming_config: crate::viz::realtime::StreamingConfig {
                server_port: options.port.unwrap_or(8080),
                max_connections: 100,
                broadcast_interval_ms: 33, // ~30 FPS for updates
                enable_position_compression: true,
                position_precision: 2,
                enable_update_batching: true,
                max_batch_size: 50,
            },
        };

        if verbose >= 3 {
            eprintln!("🔧 DEBUG: Creating real-time visualization engine...");
        }

        // Create real-time visualization engine with the graph
        // For now, create a simple graph from the data source
        // TODO: Properly extract graph structure from data_source
        let graph = crate::api::graph::Graph::new(); // Placeholder - will be populated from data_source
        if verbose >= 3 {
            eprintln!("📊 DEBUG: Created placeholder graph for engine");

            eprintln!("⚡ DEBUG: Initializing RealTimeVizEngine with config...");
        }
        let engine = crate::viz::realtime::RealTimeVizEngine::new(graph, realtime_config.clone());
        if verbose >= 3 {
            eprintln!("✅ DEBUG: RealTimeVizEngine created successfully!");
        }

        let port = options.port.unwrap_or(8080);
        if verbose >= 2 {
            eprintln!("🌐 VERBOSE: Using port {} for visualization server", port);
        }

        // Create a real-time visualization session
        if verbose >= 3 {
            eprintln!(
                "🎮 DEBUG: Creating RealTimeVisualization with enable_honeycomb_controls={}",
                is_honeycomb
            );
        }
        let realtime_viz = RealTimeVisualization {
            config: realtime_config,
            engine,
            port,
            title: options
                .title
                .unwrap_or_else(|| "Real-time Graph Visualization".to_string()),
            auto_open: options.auto_open.unwrap_or(true),
            enable_honeycomb_controls: is_honeycomb,
            verbose,
        };

        if verbose >= 1 {
            eprintln!("🚀 INFO: RealTimeVisualization session created!");
        }
        if is_honeycomb && verbose >= 2 {
            eprintln!("🍯 VERBOSE: *** HONEYCOMB N-DIMENSIONAL CONTROLS ARE ENABLED! ***");
            if verbose >= 3 {
                eprintln!("🎯 DEBUG: Expected controls:");
                eprintln!("   - Left Mouse + Drag: Rotate in dimensions 0-1");
                eprintln!("   - Left + Ctrl + Drag: Rotate in higher dimensions (2-3)");
                eprintln!("   - Right Mouse + Drag: Multi-dimensional rotation");
                eprintln!("   - Middle Mouse + Drag: Rotate across all dimension pairs");
                eprintln!("   - Node Dragging: Move points in n-dimensional space");
            }
        } else if verbose >= 2 {
            eprintln!("⚠️  VERBOSE: Honeycomb controls NOT enabled for this layout");
        }

        Ok(RenderResult::RealTime(realtime_viz))
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
                iterations: 100,
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
        let streaming_server = StreamingServer::new(self.data_source.clone(), streaming_config);

        Ok(InteractiveViz::Streaming {
            streaming_server,
            config: opts,
            viz_config: self.config.clone(),
        })
    }

    /// Generate static visualization export (PNG, SVG, PDF, HTML)
    pub fn static_viz(&self, options: StaticOptions) -> GraphResult<StaticViz> {
        match options.format {
            ExportFormat::HTML => self.generate_static_html(&options),
            ExportFormat::SVG => self.generate_simple_svg(&options),
            _ => Err(GraphError::NotImplemented {
                feature: format!("{:?} export", options.format),
                tracking_issue: Some(
                    "https://github.com/anthropics/groggy/issues/viz-static".to_string(),
                ),
            }),
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
        let nodes_json = serde_json::to_string(&nodes).map_err(|e| {
            GraphError::internal(
                &format!("Failed to serialize nodes: {}", e),
                "generate_static_html",
            )
        })?;
        let edges_json = serde_json::to_string(&edges).map_err(|e| {
            GraphError::internal(
                &format!("Failed to serialize edges: {}", e),
                "generate_static_html",
            )
        })?;

        // Read the HTML template
        let html_template = self.get_html_template()?;

        // Replace template variables
        let html = html_template
            .replace("{{TITLE}}", "Graph Visualization")
            .replace("{{NODE_COUNT}}", &metadata.node_count.to_string())
            .replace("{{EDGE_COUNT}}", &metadata.edge_count.to_string())
            .replace("{{WIDTH}}", &options.width.to_string())
            .replace("{{HEIGHT}}", &options.height.to_string())
            .replace(
                "{{LAYOUT}}",
                &format!("{:?}", options.layout).to_lowercase(),
            )
            .replace("{{THEME}}", &options.theme)
            .replace("{{NODES_JSON}}", &nodes_json)
            .replace("{{EDGES_JSON}}", &edges_json)
            .replace("{{USE_WEBSOCKET}}", "false");

        // Write to file
        fs::write(&options.filename, &html).map_err(|e| {
            GraphError::internal(
                &format!("Failed to write HTML file: {}", e),
                "generate_static_html",
            )
        })?;

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
                positions.iter().find(|p| p.node_id == edge.target),
            ) {
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" class="edge"/>"#,
                    src_pos.position.x, src_pos.position.y, dst_pos.position.x, dst_pos.position.y
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
                    pos.position.x,
                    pos.position.y - 15.0,
                    label
                ));
            }
        }

        svg.push_str("</svg>");

        // Write to file
        fs::write(&options.filename, &svg).map_err(|e| {
            GraphError::internal(
                &format!("Failed to write SVG file: {}", e),
                "generate_simple_svg",
            )
        })?;

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
        let nodes_json = serde_json::to_string(&nodes).map_err(|e| {
            GraphError::internal(
                &format!("Failed to serialize nodes: {}", e),
                "generate_jupyter_html",
            )
        })?;
        let edges_json = serde_json::to_string(&edges).map_err(|e| {
            GraphError::internal(
                &format!("Failed to serialize edges: {}", e),
                "generate_jupyter_html",
            )
        })?;

        let width = options.width.unwrap_or(800);
        let height = options.height.unwrap_or(600);
        let canvas_id = format!("viz-canvas-{}", std::ptr::addr_of!(*self) as usize);

        // Generate Jupyter-optimized HTML (simpler than full template for embedding)
        let html = format!(
            r###"
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
"###,
            width,
            height,
            canvas_id,
            width,
            height,
            metadata.node_count,
            metadata.edge_count,
            nodes_json,
            edges_json
        );

        Ok(html)
    }

    /// Generate self-contained HTML for local use
    fn generate_local_html(&self, options: &RenderOptions) -> GraphResult<String> {
        // For local HTML, we can reuse the existing static HTML generation
        let nodes = self.data_source.get_graph_nodes();
        let edges = self.data_source.get_graph_edges();
        let metadata = self.data_source.get_graph_metadata();

        // Convert to JSON
        let nodes_json = serde_json::to_string(&nodes).map_err(|e| {
            GraphError::internal(
                &format!("Failed to serialize nodes: {}", e),
                "generate_local_html",
            )
        })?;
        let edges_json = serde_json::to_string(&edges).map_err(|e| {
            GraphError::internal(
                &format!("Failed to serialize edges: {}", e),
                "generate_local_html",
            )
        })?;

        // Read the HTML template
        let html_template = self.get_html_template()?;

        let width = options.width.unwrap_or(800);
        let height = options.height.unwrap_or(600);
        let layout = options
            .layout
            .as_ref()
            .map(|l| format!("{:?}", l).to_lowercase())
            .unwrap_or_else(|| "force-directed".to_string());
        let theme = options.theme.as_deref().unwrap_or("light");
        let title = options.title.as_deref().unwrap_or("Graph Visualization");

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
    /// Show node labels
    pub show_labels: bool,
    /// Auto-open browser
    pub auto_open: bool,
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
            show_labels: false,
            auto_open: false,
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

/// Active interactive visualization session
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum InteractiveViz {
    /// Traditional streaming visualization
    Streaming {
        streaming_server: StreamingServer,
        config: InteractiveOptions,
        viz_config: VizConfig,
    },
    /// Advanced real-time visualization with Phase 3 features
    RealTime(RealTimeVisualization),
}

/// Real-time visualization session with advanced features
#[derive(Debug)]
pub struct RealTimeVisualization {
    pub config: crate::viz::realtime::RealTimeVizConfig,
    pub engine: crate::viz::realtime::RealTimeVizEngine,
    pub port: u16,
    pub title: String,
    pub auto_open: bool,
    pub enable_honeycomb_controls: bool,
    pub verbose: u8,
}

impl InteractiveViz {
    /// Start the visualization server and return the URL
    pub fn start(self, bind_addr: Option<IpAddr>) -> GraphResult<InteractiveVizSession> {
        let addr = bind_addr.unwrap_or_else(|| "127.0.0.1".parse().unwrap());

        match self {
            InteractiveViz::Streaming {
                streaming_server,
                config,
                ..
            } => {
                let port_hint = if config.port == 0 { 8080 } else { config.port };

                // Start the streaming server in background
                let server_handle =
                    streaming_server
                        .start_background(addr, port_hint)
                        .map_err(|e| {
                            GraphError::internal(
                                &format!("Failed to start visualization server: {}", e),
                                "VizModule::start",
                            )
                        })?;

                let actual_port = server_handle.port;
                let url = format!("http://{}:{}/", addr, actual_port);

                println!("🚀 Interactive visualization server started at: {}", url);

                if streaming_server.data_source.supports_graph_view() {
                    let metadata = streaming_server.data_source.get_graph_metadata();
                    println!(
                        "📊 Graph visualization: {} nodes, {} edges",
                        metadata.node_count, metadata.edge_count
                    );
                } else {
                    println!(
                        "📋 Table visualization: {} rows × {} columns",
                        streaming_server.data_source.total_rows(),
                        streaming_server.data_source.total_cols()
                    );
                }

                Ok(InteractiveVizSession {
                    server_handle,
                    url,
                    config: config.clone(),
                })
            }

            InteractiveViz::RealTime(realtime_viz) => {
                use crate::viz::realtime::accessor::DataSourceRealtimeAccessor;
                use crate::viz::realtime::server::start_realtime_background;
                use crate::viz::streaming::GraphDataSource;
                use std::sync::Arc;

                let port_hint = realtime_viz.config.streaming_config.server_port;

                // Create a data source accessor from the engine's graph
                let graph_arc = realtime_viz.engine.get_graph();
                let graph_guard = graph_arc.lock().unwrap();
                let data_source = Arc::new(GraphDataSource::new(&graph_guard));
                drop(graph_guard); // Release the lock early
                let accessor: Arc<dyn crate::viz::realtime::accessor::RealtimeVizAccessor> =
                    Arc::new(DataSourceRealtimeAccessor::with_verbosity(
                        data_source,
                        realtime_viz.verbose,
                    ));

                // Start real-time server in background with proper cancellation support
                let server_handle =
                    start_realtime_background(port_hint, accessor, realtime_viz.verbose).map_err(
                        |e| {
                            GraphError::internal(
                                &format!("Failed to start realtime server: {}", e),
                                "InteractiveViz::start",
                            )
                        },
                    )?;

                let actual_port = server_handle.port;
                let url = format!("http://{}:{}/", addr, actual_port);

                println!("🚀 Real-time visualization server started at: {}", url);
                println!("✨ Features: Real-time streaming, Interactive controls, Performance monitoring");
                println!(
                    "📊 Phase 4 visualization: N-dimensional embeddings with advanced client UI"
                );

                // Convert RealtimeServerHandle to StreamingServerHandle
                let server_handle = streaming::types::ServerHandle {
                    port: actual_port,
                    cancel: server_handle.cancel,
                    thread: server_handle.thread,
                };

                Ok(InteractiveVizSession {
                    server_handle,
                    url,
                    config: InteractiveOptions {
                        port: port_hint,
                        layout: LayoutAlgorithm::ForceDirected {
                            charge: -100.0,
                            distance: 100.0,
                            iterations: 100,
                        },
                        theme: "dark".to_string(),
                        width: 1200,
                        height: 800,
                        interactions: Default::default(),
                        show_labels: true,
                        auto_open: false,
                    },
                })
            }
        }
    }

    /// Get configuration information for the visualization
    pub fn get_config(&self) -> InteractiveOptions {
        match self {
            InteractiveViz::Streaming { config, .. } => config.clone(),
            InteractiveViz::RealTime(realtime_viz) => {
                // Convert real-time config to interactive options
                InteractiveOptions {
                    port: realtime_viz.config.streaming_config.server_port,
                    layout: LayoutAlgorithm::ForceDirected {
                        charge: -100.0,
                        distance: 100.0,
                        iterations: 100,
                    }, // Will be overridden by real-time system
                    theme: "dark".to_string(),
                    width: 1200,
                    height: 800,
                    interactions: Default::default(),
                    show_labels: true,
                    auto_open: false,
                }
            }
        }
    }

    /// Get information about the data source
    pub fn get_data_info(&self) -> GraphResult<DataSourceInfo> {
        match self {
            InteractiveViz::Streaming {
                streaming_server, ..
            } => {
                let supports_graph = streaming_server.data_source.supports_graph_view();
                let graph_info = if supports_graph {
                    let metadata = streaming_server.data_source.get_graph_metadata();
                    Some(GraphInfo {
                        node_count: metadata.node_count,
                        edge_count: metadata.edge_count,
                        is_directed: metadata.is_directed,
                        has_weights: metadata.has_weights,
                    })
                } else {
                    None
                };

                Ok(DataSourceInfo {
                    total_rows: streaming_server.data_source.total_rows(),
                    total_cols: streaming_server.data_source.total_cols(),
                    supports_graph,
                    graph_info,
                    source_type: "streaming".to_string(),
                })
            }

            InteractiveViz::RealTime(realtime_viz) => {
                // Get information from the real-time engine's graph
                let graph_arc = realtime_viz.engine.graph();
                let graph = graph_arc.lock().unwrap();
                let node_count = graph.node_ids().len();
                let edge_count = graph.edge_ids().len();

                Ok(DataSourceInfo {
                    total_rows: node_count,
                    total_cols: 0, // Real-time graphs don't have tabular structure
                    supports_graph: true,
                    graph_info: Some(GraphInfo {
                        node_count,
                        edge_count,
                        is_directed: true, // Assume directed for now
                        has_weights: true, // Real-time graphs typically have weights
                    }),
                    source_type: "realtime".to_string(),
                })
            }
        }
    }
}

/// Active visualization session with running server
pub struct InteractiveVizSession {
    server_handle: streaming::types::ServerHandle,
    url: String,
    #[allow(dead_code)]
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
