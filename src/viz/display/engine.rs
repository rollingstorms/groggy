//! Core display engine - the heart of the unified display system
//!
//! This module contains the DisplayEngine that orchestrates all display formatting.
//! It's implemented ONLY in BaseTable and BaseArray - all other types delegate.

use crate::viz::display::{
    CompactFormatter, DataWindow, HtmlRenderer, ThemeSystem, TruncationStrategy,
};

/// The core display engine - implemented ONLY in foundation classes
#[derive(Clone, Debug)]
pub struct DisplayEngine {
    pub config: DisplayConfig,
    compact_formatter: CompactFormatter,
    html_renderer: HtmlRenderer,
    theme_system: ThemeSystem,
}

impl DisplayEngine {
    /// Create a new display engine with default configuration
    pub fn new() -> Self {
        Self {
            config: DisplayConfig::default(),
            compact_formatter: CompactFormatter::new(),
            html_renderer: HtmlRenderer::new(),
            theme_system: ThemeSystem::new(),
        }
    }

    /// Create a display engine with custom configuration
    pub fn with_config(config: DisplayConfig) -> Self {
        Self {
            config,
            compact_formatter: CompactFormatter::new(),
            html_renderer: HtmlRenderer::new(),
            theme_system: ThemeSystem::new(),
        }
    }

    /// Format data as Unicode text (for __repr__, __str__, print())
    pub fn format_unicode(&self, data: &DataWindow) -> String {
        if self.config.compact_mode {
            self.compact_formatter
                .format_minimal_width(data, &self.config)
        } else {
            self.compact_formatter.format_full_width(data, &self.config)
        }
    }

    /// Format data as semantic HTML (for _repr_html_(), Jupyter notebooks)
    pub fn format_html(&self, data: &DataWindow) -> String {
        let theme = self.theme_system.get_theme(&self.config.theme);
        self.html_renderer
            .render_semantic_table(data, theme, &self.config)
    }

    /// Rich display with configurable output format
    pub fn rich_display(&self, data: &DataWindow, output_format: OutputFormat) -> String {
        match output_format {
            OutputFormat::Unicode => self.format_unicode(data),
            OutputFormat::Html => self.format_html(data),
            OutputFormat::Interactive => {
                // For now, return HTML with interactive button
                // Will be replaced with actual WebSocket server in Phase 3
                let mut html = self.format_html(data);
                html.push_str(
                    r#"<div class="interactive-placeholder">
                    <button onclick="alert('Interactive mode coming in Phase 3!')">
                        Launch Interactive View →
                    </button>
                    </div>"#,
                );
                html
            }
        }
    }

    /// Update display configuration
    pub fn set_config(&mut self, config: DisplayConfig) {
        self.config = config;
    }

    /// Get current theme
    pub fn get_current_theme(&self) -> &crate::viz::display::Theme {
        self.theme_system.get_theme(&self.config.theme)
    }
}

impl Default for DisplayEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete display configuration
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    // Core display settings
    pub compact_mode: bool,
    pub max_cell_width: usize,
    pub max_rows: usize,
    pub max_cols: usize,
    pub precision: usize,

    // Styling
    pub theme: String,
    pub output_format: OutputFormat,

    // Truncation behavior
    pub truncation_strategy: TruncationStrategy,
    pub show_truncation_info: bool,

    // Header display control
    pub show_headers: bool,

    // Future streaming settings (placeholder for Phase 3)
    pub buffer_size: usize,
    pub lazy_threshold: usize,
}

impl DisplayConfig {
    /// Create a compact display configuration (default)
    pub fn compact() -> Self {
        Self {
            compact_mode: true,
            max_cell_width: 20,
            max_rows: 10,
            max_cols: 50,
            precision: 2,
            theme: "sleek".to_string(),
            output_format: OutputFormat::Unicode,
            truncation_strategy: TruncationStrategy::TypeAware,
            show_truncation_info: true,
            show_headers: true,
            buffer_size: 1000,
            lazy_threshold: 10000,
        }
    }

    /// Create a full-width display configuration (legacy compatibility)
    pub fn full_width() -> Self {
        Self {
            compact_mode: false,
            max_cell_width: 50,
            max_rows: 10,
            max_cols: 8,
            precision: 2,
            theme: "sleek".to_string(),
            output_format: OutputFormat::Unicode,
            truncation_strategy: TruncationStrategy::TypeAware,
            show_truncation_info: true,
            show_headers: true,
            buffer_size: 1000,
            lazy_threshold: 10000,
        }
    }

    /// Create configuration for HTML output
    pub fn html() -> Self {
        let mut config = Self::compact();
        config.output_format = OutputFormat::Html;
        config.max_rows = 20; // Show more rows in HTML
        config
    }

    /// Create configuration for dense matrix display (no headers, truncated)
    pub fn dense_matrix() -> Self {
        let mut config = Self::html();
        config.show_headers = false;
        config.max_rows = 10;
        config.max_cols = 10;
        config
    }

    /// Create configuration for interactive output (placeholder for Phase 3)
    pub fn interactive() -> Self {
        let mut config = Self::html();
        config.output_format = OutputFormat::Interactive;
        config.lazy_threshold = 1000; // Lower threshold for interactive
        config
    }

    /// Builder pattern for theme selection
    pub fn with_theme(mut self, theme: impl Into<String>) -> Self {
        self.theme = theme.into();
        self
    }

    /// Builder pattern for compact mode toggle
    pub fn with_compact_mode(mut self, compact: bool) -> Self {
        self.compact_mode = compact;
        self
    }

    /// Builder pattern for max cell width
    pub fn with_max_cell_width(mut self, width: usize) -> Self {
        self.max_cell_width = width;
        self
    }

    /// Builder pattern for precision
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self::compact()
    }
}

/// Output format for display rendering
#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    /// Unicode text with box drawing characters
    Unicode,
    /// Semantic HTML with CSS styling
    Html,
    /// Interactive browser interface (Phase 3)
    Interactive,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Unicode => write!(f, "unicode"),
            OutputFormat::Html => write!(f, "html"),
            OutputFormat::Interactive => write!(f, "interactive"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::display::{ColumnSchema, DataSchema, DataType, DataWindow};

    fn create_test_data() -> DataWindow {
        let headers = vec!["name".to_string(), "age".to_string(), "score".to_string()];
        let rows = vec![
            vec!["Alice".to_string(), "25".to_string(), "91.5".to_string()],
            vec!["Bob".to_string(), "30".to_string(), "87.2".to_string()],
            vec![
                "Charlie with a very long name".to_string(),
                "35".to_string(),
                "92.123456789".to_string(),
            ],
        ];
        let schema = DataSchema::new(vec![
            ColumnSchema {
                name: "name".to_string(),
                data_type: DataType::String,
            },
            ColumnSchema {
                name: "age".to_string(),
                data_type: DataType::Integer,
            },
            ColumnSchema {
                name: "score".to_string(),
                data_type: DataType::Float,
            },
        ]);

        DataWindow::new(headers, rows, schema)
    }

    #[test]
    fn test_display_engine_creation() {
        let engine = DisplayEngine::new();
        assert!(engine.config.compact_mode);
        assert_eq!(engine.config.max_cell_width, 20);
        assert_eq!(engine.config.theme, "sleek");
    }

    #[test]
    fn test_display_config_builder() {
        let config = DisplayConfig::compact()
            .with_theme("dark")
            .with_max_cell_width(15)
            .with_precision(1);

        assert_eq!(config.theme, "dark");
        assert_eq!(config.max_cell_width, 15);
        assert_eq!(config.precision, 1);
        assert!(config.compact_mode);
    }

    #[test]
    fn test_format_unicode() {
        let engine = DisplayEngine::new();
        let data = create_test_data();
        let output = engine.format_unicode(&data);

        // Should produce Unicode table
        assert!(output.contains("┌")); // Box drawing characters
        assert!(output.contains("│"));
        assert!(output.contains("name"));
        assert!(output.contains("Alice"));

        // Should be compact (reasonably narrow)
        let lines: Vec<&str> = output.lines().collect();
        if let Some(first_line) = lines.first() {
            assert!(
                first_line.len() < 150,
                "Compact mode should be reasonably narrow"
            );
        }

        // Should truncate long names or show some reasonable formatting
        // (May not always contain … if data fits within max_cell_width)
    }

    #[test]
    fn test_format_html() {
        let engine = DisplayEngine::new();
        let data = create_test_data();
        let output = engine.format_html(&data);

        // Should produce semantic HTML
        assert!(output.contains("<table"));
        assert!(output.contains("<thead>"));
        assert!(output.contains("<tbody>"));
        assert!(output.contains("<th"));
        assert!(output.contains("<td"));
        assert!(output.contains("Alice"));

        // Should include CSS classes
        assert!(output.contains("groggy-table"));
        assert!(output.contains("theme-sleek"));
    }

    #[test]
    fn test_rich_display_formats() {
        let engine = DisplayEngine::new();
        let data = create_test_data();

        let unicode_output = engine.rich_display(&data, OutputFormat::Unicode);
        let html_output = engine.rich_display(&data, OutputFormat::Html);
        let interactive_output = engine.rich_display(&data, OutputFormat::Interactive);

        assert!(unicode_output.contains("┌"));
        assert!(html_output.contains("<table"));
        assert!(interactive_output.contains("Launch Interactive"));
    }

    #[test]
    fn test_config_variations() {
        let compact_config = DisplayConfig::compact();
        let full_width_config = DisplayConfig::full_width();
        let html_config = DisplayConfig::html();

        assert!(compact_config.compact_mode);
        assert!(!full_width_config.compact_mode);
        assert_eq!(html_config.output_format, OutputFormat::Html);

        assert_eq!(compact_config.max_cell_width, 20);
        assert_eq!(full_width_config.max_cell_width, 50);
    }
}
