//! HTML renderer for semantic table generation
//!
//! Creates professional HTML tables with responsive CSS and theming support.

use crate::viz::display::{DataType, DataWindow, DisplayConfig, Theme};

/// HTML renderer for semantic table generation
#[derive(Clone, Debug)]
pub struct HtmlRenderer {
    // Future: template engine could be added here
}

impl HtmlRenderer {
    pub fn new() -> Self {
        Self {}
    }

    /// Render a complete HTML table with theme and styling
    pub fn render_semantic_table(
        &self,
        data: &DataWindow,
        theme: &Theme,
        config: &DisplayConfig,
    ) -> String {
        if data.headers.is_empty() {
            return self.render_empty_table(theme);
        }

        let mut html = String::new();

        // Container with theme data attribute
        html.push_str(&format!(
            r#"<div class="groggy-display-container" data-theme="{}">"#,
            theme.name
        ));
        html.push_str(r#"<div class="groggy-table-container">"#);
        // Main table element
        html.push_str(&format!(
            r#"<table class="groggy-table {}">"#,
            theme.table_class
        ));

        // Table header (only if show_headers is true)
        if config.show_headers {
            html.push_str("<thead>");
            html.push_str(&self.render_header_row(&data.headers, &data.schema, config));
            html.push_str("</thead>");
        }

        // Table body
        html.push_str("<tbody>");
        for row in &data.rows {
            html.push_str(&self.render_data_row(row, &data.schema, config));
        }
        html.push_str("</tbody>");

        html.push_str("</table>");

        // Table info section (if data is truncated)
        if let Some(info) = data.truncation_info() {
            html.push_str(&self.render_table_info(&info, data, config));
        }
        html.push_str("</div>");

        html.push_str("</div>");

        // Add CSS styling
        html.push_str(&format!("<style>{}</style>", theme.css));

        // Add JavaScript for interactive features (basic for now)
        html.push_str(&self.render_javascript());

        html
    }

    fn render_header_row(
        &self,
        headers: &[String],
        schema: &crate::viz::display::DataSchema,
        _config: &DisplayConfig,
    ) -> String {
        let mut row = String::from("<tr>");

        for (i, header) in headers.iter().enumerate() {
            let data_type = schema
                .get_column(i)
                .map(|col| &col.data_type)
                .unwrap_or(&DataType::Unknown);

            row.push_str(&format!(
                r#"<th class="col-{}" data-type="{}" title="{}">{}</th>"#,
                data_type.to_string().to_lowercase(),
                data_type,
                format!("Column: {} ({})", header, data_type),
                self.escape_html(header)
            ));
        }

        row.push_str("</tr>");
        row
    }

    fn render_data_row(
        &self,
        row: &[String],
        schema: &crate::viz::display::DataSchema,
        config: &DisplayConfig,
    ) -> String {
        let mut html_row = String::from("<tr>");

        for (i, cell) in row.iter().enumerate() {
            let data_type = schema
                .get_column(i)
                .map(|col| &col.data_type)
                .unwrap_or(&DataType::Unknown);

            let cell_class = format!("cell-{}", data_type.to_string().to_lowercase());
            let is_truncated = self.is_cell_truncated(cell, data_type, config);

            let full_class = if is_truncated {
                format!("{} cell-truncated", cell_class)
            } else {
                cell_class
            };

            let formatted_value = self.format_cell_value(cell, data_type, config);

            html_row.push_str(&format!(
                r#"<td class="{}" data-type="{}" title="{}">{}</td>"#,
                full_class,
                data_type,
                if is_truncated {
                    format!("Full value: {}", self.escape_html(cell))
                } else {
                    format!("{} value", data_type)
                },
                self.escape_html(&formatted_value)
            ));
        }

        html_row.push_str("</tr>");
        html_row
    }

    fn render_table_info(&self, info: &str, data: &DataWindow, _config: &DisplayConfig) -> String {
        format!(
            r#"<div class="table-info">
                <span>{}</span>
                <button class="interactive-btn" onclick="launchInteractive()" 
                        title="Launch interactive view for full dataset">
                    View All ({} rows) →
                </button>
            </div>"#,
            self.escape_html(info),
            data.total_rows
        )
    }

    fn render_javascript(&self) -> String {
        r#"<script>
function launchInteractive() {
    // Phase 1: Basic placeholder
    alert('Interactive streaming view coming in Phase 3!\n\nThis will launch a WebSocket-powered interface for browsing massive datasets with virtual scrolling.');
    
    // Phase 3: Will be replaced with actual WebSocket connection
    // if (window.groggyInteractive) {
    //     window.groggyInteractive.launch();
    // }
}

// Add keyboard navigation support
document.addEventListener('DOMContentLoaded', function() {
    const tables = document.querySelectorAll('.groggy-table');
    tables.forEach(table => {
        // Make table focusable
        table.setAttribute('tabindex', '0');
        
        // Add keyboard navigation
        table.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                const interactiveBtn = table.parentElement.querySelector('.interactive-btn');
                if (interactiveBtn) {
                    interactiveBtn.click();
                    e.preventDefault();
                }
            }
        });
    });
});
</script>"#.to_string()
    }

    fn render_empty_table(&self, theme: &Theme) -> String {
        format!(
            r#"<div class="groggy-display-container" data-theme="{}">
                <div class="empty-table-message">
                    No data to display
                </div>
                <style>{}</style>
            </div>"#,
            theme.name, theme.css
        )
    }

    fn format_cell_value(
        &self,
        value: &str,
        data_type: &DataType,
        config: &DisplayConfig,
    ) -> String {
        match data_type {
            DataType::Float => self.format_float_value(value, config),
            DataType::Integer => self.format_integer_value(value),
            DataType::Boolean => self.format_boolean_value(value),
            DataType::DateTime => self.format_datetime_value(value),
            DataType::Json => self.format_json_value(value, config),
            _ => value.to_string(),
        }
    }

    fn format_float_value(&self, value: &str, config: &DisplayConfig) -> String {
        if let Ok(num) = value.parse::<f64>() {
            if num.abs() < 0.001 && num != 0.0 {
                format!("{:.2e}", num)
            } else {
                format!("{:.precision$}", num, precision = config.precision)
            }
        } else {
            value.to_string()
        }
    }

    fn format_integer_value(&self, value: &str) -> String {
        // Add thousand separators for large numbers in HTML
        if let Ok(num) = value.parse::<i64>() {
            if num.abs() > 9999 {
                self.add_thousand_separators(&num.to_string())
            } else {
                value.to_string()
            }
        } else {
            value.to_string()
        }
    }

    fn format_boolean_value(&self, value: &str) -> String {
        match value.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => "✓".to_string(),
            "false" | "0" | "no" | "off" => "✗".to_string(),
            _ => value.to_string(),
        }
    }

    fn format_datetime_value(&self, value: &str) -> String {
        // For now, just return as-is. Future: could parse and format consistently
        value.to_string()
    }

    fn format_json_value(&self, value: &str, config: &DisplayConfig) -> String {
        // Truncate JSON to prevent very wide cells
        if value.len() > config.max_cell_width {
            format!("{}…", &value[..config.max_cell_width.saturating_sub(1)])
        } else {
            value.to_string()
        }
    }

    fn add_thousand_separators(&self, num_str: &str) -> String {
        let chars: Vec<char> = num_str.chars().collect();
        let mut result = String::new();
        let mut digit_count = 0;

        for (_i, &ch) in chars.iter().rev().enumerate() {
            if ch.is_ascii_digit() {
                if digit_count > 0 && digit_count % 3 == 0 {
                    result.push(',');
                }
                digit_count += 1;
            }
            result.push(ch);
        }

        result.chars().rev().collect()
    }

    fn is_cell_truncated(
        &self,
        value: &str,
        _data_type: &DataType,
        config: &DisplayConfig,
    ) -> bool {
        value.len() > config.max_cell_width
    }

    fn escape_html(&self, text: &str) -> String {
        text.chars()
            .map(|c| match c {
                '<' => "&lt;".to_string(),
                '>' => "&gt;".to_string(),
                '&' => "&amp;".to_string(),
                '"' => "&quot;".to_string(),
                '\'' => "&#39;".to_string(),
                _ => c.to_string(),
            })
            .collect()
    }
}

impl Default for HtmlRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::display::{
        ColumnSchema, DataSchema, DataType, DataWindow, DisplayConfig, Theme,
    };

    fn create_test_data() -> DataWindow {
        let headers = vec![
            "name".to_string(),
            "age".to_string(),
            "score".to_string(),
            "active".to_string(),
        ];
        let rows = vec![
            vec![
                "Alice".to_string(),
                "25".to_string(),
                "91.50".to_string(),
                "true".to_string(),
            ],
            vec![
                "Bob & Co.".to_string(),
                "30000".to_string(),
                "87.00".to_string(),
                "false".to_string(),
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
            ColumnSchema {
                name: "active".to_string(),
                data_type: DataType::Boolean,
            },
        ]);

        DataWindow::new(headers, rows, schema)
    }

    #[test]
    fn test_html_table_generation() {
        let renderer = HtmlRenderer::new();
        let data = create_test_data();
        let theme = Theme::sleek();
        let config = DisplayConfig::default();

        let html = renderer.render_semantic_table(&data, &theme, &config);

        // Should contain semantic HTML elements
        assert!(html.contains("<table"));
        assert!(html.contains("<thead>"));
        assert!(html.contains("<tbody>"));
        assert!(html.contains("<th"));
        assert!(html.contains("<td"));

        // Should contain theme class
        assert!(html.contains("theme-sleek"));

        // Should contain data
        assert!(html.contains("Alice"));
        assert!(html.contains("25"));

        // Should contain CSS
        assert!(html.contains("<style>"));

        // Should escape HTML
        assert!(html.contains("Bob &amp; Co."));
    }

    #[test]
    fn test_data_type_classes() {
        let renderer = HtmlRenderer::new();
        let data = create_test_data();
        let theme = Theme::sleek();
        let config = DisplayConfig::default();

        let html = renderer.render_semantic_table(&data, &theme, &config);

        // Should have data type classes
        assert!(html.contains("cell-string"));
        assert!(html.contains("cell-integer"));
        assert!(html.contains("cell-float"));
        assert!(html.contains("cell-boolean"));

        // Should have data type attributes
        assert!(html.contains("data-type=\"string\""));
        assert!(html.contains("data-type=\"integer\""));
        assert!(html.contains("data-type=\"float\""));
        assert!(html.contains("data-type=\"boolean\""));
    }

    #[test]
    fn test_boolean_formatting() {
        let renderer = HtmlRenderer::new();
        let _config = DisplayConfig::default();

        assert_eq!(renderer.format_boolean_value("true"), "✓");
        assert_eq!(renderer.format_boolean_value("false"), "✗");
        assert_eq!(renderer.format_boolean_value("1"), "✓");
        assert_eq!(renderer.format_boolean_value("0"), "✗");
        assert_eq!(renderer.format_boolean_value("maybe"), "maybe");
    }

    #[test]
    fn test_number_formatting() {
        let renderer = HtmlRenderer::new();
        let config = DisplayConfig {
            precision: 2,
            ..DisplayConfig::default()
        };

        // Float formatting
        assert_eq!(renderer.format_float_value("91.123456", &config), "91.12");
        assert_eq!(renderer.format_float_value("0.0001", &config), "1.00e-4");

        // Integer formatting with thousand separators
        assert_eq!(renderer.format_integer_value("30000"), "30,000");
        assert_eq!(renderer.format_integer_value("999"), "999");
    }

    #[test]
    fn test_html_escaping() {
        let renderer = HtmlRenderer::new();

        assert_eq!(renderer.escape_html("Bob & Co."), "Bob &amp; Co.");
        assert_eq!(renderer.escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(renderer.escape_html("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(renderer.escape_html("normal text"), "normal text");
    }

    #[test]
    fn test_empty_table() {
        let renderer = HtmlRenderer::new();
        let theme = Theme::sleek();

        let html = renderer.render_empty_table(&theme);

        assert!(html.contains("No data to display"));
        assert!(html.contains("groggy-display-container"));
        assert!(html.contains(&theme.css));
    }

    #[test]
    fn test_truncation_info() {
        let renderer = HtmlRenderer::new();
        let mut data = create_test_data();
        data.total_rows = 1000;
        data.start_offset = 0;

        let config = DisplayConfig::default();
        let info = data.truncation_info().unwrap();

        let html = renderer.render_table_info(&info, &data, &config);

        assert!(html.contains("table-info"));
        assert!(html.contains("View All"));
        assert!(html.contains("1000 rows"));
        assert!(html.contains("interactive-btn"));
    }

    #[test]
    fn test_javascript_inclusion() {
        let renderer = HtmlRenderer::new();
        let js = renderer.render_javascript();

        assert!(js.contains("<script>"));
        assert!(js.contains("launchInteractive"));
        assert!(js.contains("addEventListener"));
        assert!(js.contains("DOMContentLoaded"));
    }

    #[test]
    fn test_thousand_separators() {
        let renderer = HtmlRenderer::new();

        assert_eq!(renderer.add_thousand_separators("1234567"), "1,234,567");
        assert_eq!(renderer.add_thousand_separators("999"), "999");
        assert_eq!(renderer.add_thousand_separators("1000"), "1,000");
        assert_eq!(renderer.add_thousand_separators("1000000"), "1,000,000");
    }
}
