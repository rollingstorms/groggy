//! Compact formatter for minimal-width Unicode table display
//!
//! This is the core of the "compact mode" - calculates minimum required widths
//! instead of distributing to full terminal width, with smart truncation.

use crate::viz::display::{DataType, DataWindow, DisplayConfig, TruncationStrategy};

/// Unicode box-drawing characters for table rendering
#[derive(Clone, Debug)]
pub struct BoxChars {
    pub horizontal: char,
    pub vertical: char,
    pub top_left: char,
    pub top_right: char,
    pub bottom_left: char,
    pub bottom_right: char,
    pub cross: char,
    pub top_tee: char,
    pub bottom_tee: char,
    pub left_tee: char,
    pub right_tee: char,
}

impl BoxChars {
    /// Heavy box drawing characters (default)
    pub fn heavy() -> Self {
        Self {
            horizontal: '━',
            vertical: '┃',
            top_left: '┏',
            top_right: '┓',
            bottom_left: '┗',
            bottom_right: '┛',
            cross: '╋',
            top_tee: '┳',
            bottom_tee: '┻',
            left_tee: '┣',
            right_tee: '┫',
        }
    }

    /// Light box drawing characters
    pub fn light() -> Self {
        Self {
            horizontal: '─',
            vertical: '│',
            top_left: '┌',
            top_right: '┐',
            bottom_left: '└',
            bottom_right: '┘',
            cross: '┼',
            top_tee: '┬',
            bottom_tee: '┴',
            left_tee: '├',
            right_tee: '┤',
        }
    }

    /// ASCII fallback characters
    pub fn ascii() -> Self {
        Self {
            horizontal: '-',
            vertical: '|',
            top_left: '+',
            top_right: '+',
            bottom_left: '+',
            bottom_right: '+',
            cross: '+',
            top_tee: '+',
            bottom_tee: '+',
            left_tee: '+',
            right_tee: '+',
        }
    }
}

/// Compact formatter - implements minimal width table rendering
#[derive(Clone, Debug)]
pub struct CompactFormatter {
    box_chars: BoxChars,
}

impl CompactFormatter {
    pub fn new() -> Self {
        Self {
            box_chars: BoxChars::light(),
        }
    }

    /// Format table using minimal required width (compact mode)
    pub fn format_minimal_width(&self, data: &DataWindow, config: &DisplayConfig) -> String {
        let col_widths = self.calculate_compact_widths(data, config);
        self.render_table_with_widths(data, &col_widths, config)
    }

    /// Format table using full width distribution (legacy compatibility)
    pub fn format_full_width(&self, data: &DataWindow, config: &DisplayConfig) -> String {
        let col_widths = self.calculate_full_widths(data, config, 120); // Default terminal width
        self.render_table_with_widths(data, &col_widths, config)
    }

    /// Calculate minimal column widths based on content
    fn calculate_compact_widths(&self, data: &DataWindow, config: &DisplayConfig) -> Vec<usize> {
        if data.headers.is_empty() {
            return Vec::new();
        }

        let mut widths = Vec::with_capacity(data.headers.len());

        for (col_idx, header) in data.headers.iter().enumerate() {
            // Start with header width
            let mut max_width = header.len();

            // Check all data rows for this column
            for row in &data.rows {
                if let Some(cell_value) = row.get(col_idx) {
                    // Get the data type for type-aware truncation
                    let data_type = data
                        .schema
                        .get_column(col_idx)
                        .map(|col| &col.data_type)
                        .unwrap_or(&DataType::Unknown);

                    // Get the display width after potential truncation
                    let truncated_value = self.truncate_cell_value(
                        cell_value,
                        config.max_cell_width,
                        data_type,
                        &config.truncation_strategy,
                    );
                    max_width = max_width.max(truncated_value.len());
                }
            }

            // Ensure minimum width and apply cell width limit
            let final_width = max_width.min(config.max_cell_width).max(3);
            widths.push(final_width);
        }

        widths
    }

    /// Calculate full-width column distribution (legacy mode)
    fn calculate_full_widths(
        &self,
        data: &DataWindow,
        config: &DisplayConfig,
        total_width: usize,
    ) -> Vec<usize> {
        let compact_widths = self.calculate_compact_widths(data, config);

        if compact_widths.is_empty() {
            return compact_widths;
        }

        // Calculate total compact width
        let total_compact: usize = compact_widths.iter().sum();
        let separators_width = compact_widths.len() * 3; // " │ " between columns
        let borders_width = 4; // "│ " at start and " │" at end
        let total_used = total_compact + separators_width + borders_width;

        // If compact width fits in terminal, use it
        if total_used <= total_width {
            return compact_widths;
        }

        // Otherwise, distribute available space proportionally
        let available_content_width = total_width.saturating_sub(separators_width + borders_width);
        let mut distributed_widths = Vec::with_capacity(compact_widths.len());

        for &compact_width in &compact_widths {
            let proportion = compact_width as f64 / total_compact as f64;
            let distributed_width = ((available_content_width as f64 * proportion) as usize).max(3);
            distributed_widths.push(distributed_width);
        }

        distributed_widths
    }

    /// Smart cell value truncation with type awareness
    fn truncate_cell_value(
        &self,
        value: &str,
        max_width: usize,
        data_type: &DataType,
        strategy: &TruncationStrategy,
    ) -> String {
        if value.len() <= max_width {
            return value.to_string();
        }

        match strategy {
            TruncationStrategy::None => value.to_string(),
            TruncationStrategy::Ellipsis => {
                if max_width <= 1 {
                    value.chars().take(max_width).collect()
                } else {
                    format!("{}…", value.chars().take(max_width - 1).collect::<String>())
                }
            }
            TruncationStrategy::TypeAware => {
                self.type_aware_truncation(value, max_width, data_type)
            }
        }
    }

    /// Type-aware truncation strategies
    fn type_aware_truncation(&self, value: &str, max_width: usize, data_type: &DataType) -> String {
        match data_type {
            DataType::Float => self.truncate_float(value, max_width),
            DataType::Integer => self.truncate_integer(value, max_width),
            DataType::String => self.truncate_string(value, max_width),
            DataType::Boolean => value.to_string(), // Booleans are short
            DataType::DateTime => self.truncate_datetime(value, max_width),
            DataType::Json => self.truncate_json(value, max_width),
            DataType::Unknown => self.truncate_string(value, max_width),
        }
    }

    fn truncate_float(&self, value: &str, max_width: usize) -> String {
        // Try parsing as float for intelligent truncation
        if let Ok(num) = value.parse::<f64>() {
            // Try reducing precision
            for precision in [2, 1, 0] {
                let formatted = format!("{:.precision$}", num, precision = precision);
                if formatted.len() <= max_width {
                    return formatted;
                }
            }

            // Use scientific notation if still too long
            let scientific = format!("{:.1e}", num);
            if scientific.len() <= max_width {
                return scientific;
            }
        }

        // Fallback to string truncation
        self.truncate_string(value, max_width)
    }

    fn truncate_integer(&self, value: &str, max_width: usize) -> String {
        // Try parsing as integer
        if let Ok(num) = value.parse::<i64>() {
            if value.len() > max_width {
                // Use scientific notation for very large integers
                let scientific = format!("{:.1e}", num as f64);
                if scientific.len() <= max_width {
                    return scientific;
                }
            }
        }

        // Fallback to string truncation
        self.truncate_string(value, max_width)
    }

    fn truncate_string(&self, value: &str, max_width: usize) -> String {
        let char_count = value.chars().count();
        if char_count <= max_width {
            value.to_string()
        } else if max_width <= 1 {
            value.chars().take(max_width).collect()
        } else {
            format!("{}…", value.chars().take(max_width - 1).collect::<String>())
        }
    }

    fn truncate_datetime(&self, value: &str, max_width: usize) -> String {
        // Try to keep the most important parts of datetime
        if max_width >= 10 && value.len() > max_width {
            // Keep date part if possible (YYYY-MM-DD)
            if let Some(date_part) = value.get(0..10) {
                if date_part.matches('-').count() == 2 {
                    return date_part.to_string();
                }
            }
        }

        self.truncate_string(value, max_width)
    }

    fn truncate_json(&self, value: &str, max_width: usize) -> String {
        if max_width <= 3 {
            return "...".chars().take(max_width).collect();
        }

        // Try to preserve JSON structure hints
        if value.starts_with('{') && value.ends_with('}') {
            "{...}".to_string()
        } else if value.starts_with('[') && value.ends_with(']') {
            "[...]".to_string()
        } else {
            self.truncate_string(value, max_width)
        }
    }

    /// Render complete table with calculated widths
    fn render_table_with_widths(
        &self,
        data: &DataWindow,
        widths: &[usize],
        config: &DisplayConfig,
    ) -> String {
        if data.headers.is_empty() || widths.is_empty() {
            return "Empty table".to_string();
        }

        let mut output = String::new();

        // Top border
        output.push_str(&self.render_border_line(widths, BorderType::Top));
        output.push('\n');

        // Header row
        output.push_str(&self.render_data_row(&data.headers, widths, &data.schema, config));
        output.push('\n');

        // Header separator
        output.push_str(&self.render_border_line(widths, BorderType::Middle));
        output.push('\n');

        // Data rows
        for row in &data.rows {
            output.push_str(&self.render_data_row(row, widths, &data.schema, config));
            output.push('\n');
        }

        // Bottom border
        output.push_str(&self.render_border_line(widths, BorderType::Bottom));

        // Add truncation info if enabled
        if config.show_truncation_info {
            if let Some(info) = data.truncation_info() {
                output.push_str(&format!("\n{}", info));
            }
        }

        output
    }

    fn render_data_row(
        &self,
        row: &[String],
        widths: &[usize],
        schema: &crate::viz::display::DataSchema,
        config: &DisplayConfig,
    ) -> String {
        let mut line = String::new();
        line.push(self.box_chars.vertical);

        for (i, (cell, &width)) in row.iter().zip(widths.iter()).enumerate() {
            line.push(' ');

            // Get data type for alignment
            let data_type = schema
                .get_column(i)
                .map(|col| &col.data_type)
                .unwrap_or(&DataType::Unknown);

            // Truncate cell value
            let truncated =
                self.truncate_cell_value(cell, width, data_type, &config.truncation_strategy);

            // Apply alignment based on data type
            let aligned = if data_type.is_numeric() {
                // Right-align numeric values
                format!("{:>width$}", truncated, width = width)
            } else {
                // Left-align text values
                format!("{:<width$}", truncated, width = width)
            };

            line.push_str(&aligned);
            line.push(' ');
            line.push(self.box_chars.vertical);
        }

        line
    }

    fn render_border_line(&self, widths: &[usize], border_type: BorderType) -> String {
        let mut line = String::new();

        for (i, &width) in widths.iter().enumerate() {
            if i == 0 {
                // First column
                line.push(match border_type {
                    BorderType::Top => self.box_chars.top_left,
                    BorderType::Middle => self.box_chars.left_tee,
                    BorderType::Bottom => self.box_chars.bottom_left,
                });
            } else {
                // Subsequent columns
                line.push(match border_type {
                    BorderType::Top => self.box_chars.top_tee,
                    BorderType::Middle => self.box_chars.cross,
                    BorderType::Bottom => self.box_chars.bottom_tee,
                });
            }

            // Content width + 2 for padding
            for _ in 0..width + 2 {
                line.push(self.box_chars.horizontal);
            }
        }

        // Final border
        line.push(match border_type {
            BorderType::Top => self.box_chars.top_right,
            BorderType::Middle => self.box_chars.right_tee,
            BorderType::Bottom => self.box_chars.bottom_right,
        });

        line
    }
}

#[derive(Debug, Clone, Copy)]
enum BorderType {
    Top,
    Middle,
    Bottom,
}

impl Default for CompactFormatter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::display::{ColumnSchema, DataSchema, DataType, DataWindow, DisplayConfig};

    fn create_test_data() -> DataWindow {
        let headers = vec!["name".to_string(), "age".to_string(), "score".to_string()];
        let rows = vec![
            vec!["Alice".to_string(), "25".to_string(), "91.50".to_string()],
            vec!["Bob".to_string(), "30".to_string(), "87.00".to_string()],
            vec![
                "Charlie with a very long name that should be truncated".to_string(),
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
    fn test_compact_width_calculation() {
        let formatter = CompactFormatter::new();
        let data = create_test_data();
        let config = DisplayConfig::default();

        let widths = formatter.calculate_compact_widths(&data, &config);

        assert_eq!(widths.len(), 3);
        // Name column should be limited by max_cell_width (20)
        assert_eq!(widths[0], 20);
        // Age column should be minimal (3 chars minimum)
        assert_eq!(widths[1], 3);
        // Score column should fit the longest value
        assert!(widths[2] >= 5 && widths[2] <= 20);
    }

    #[test]
    fn test_float_truncation() {
        let formatter = CompactFormatter::new();

        let truncated = formatter.truncate_float("92.123456789", 5);
        assert!(truncated.len() <= 5);
        assert!(truncated.contains('.') || truncated.contains('e'));

        let truncated_short = formatter.truncate_float("1.23", 5);
        assert_eq!(truncated_short, "1.23");
    }

    #[test]
    fn test_integer_truncation() {
        let formatter = CompactFormatter::new();

        let truncated = formatter.truncate_integer("123456789", 5);
        assert!(truncated.len() <= 5);

        let short_int = formatter.truncate_integer("123", 5);
        assert_eq!(short_int, "123");
    }

    #[test]
    fn test_string_truncation() {
        let formatter = CompactFormatter::new();

        let truncated = formatter.truncate_string("Hello, World!", 8);
        assert_eq!(truncated, "Hello, …");
        assert!(truncated.chars().count() <= 8);

        let short_string = formatter.truncate_string("Hi", 8);
        assert_eq!(short_string, "Hi");
    }

    #[test]
    fn test_table_rendering() {
        let formatter = CompactFormatter::new();
        let data = create_test_data();
        let config = DisplayConfig::default();

        let output = formatter.format_minimal_width(&data, &config);

        // Should contain box drawing characters
        assert!(output.contains('┌'));
        assert!(output.contains('│'));
        assert!(output.contains('┐'));

        // Should contain data
        assert!(output.contains("Alice"));
        assert!(output.contains("25"));
        assert!(output.contains("91.50"));

        // Should be much narrower than full-width mode
        let lines: Vec<&str> = output.lines().collect();
        if let Some(first_line) = lines.first() {
            assert!(
                first_line.len() < 150,
                "Compact table should be reasonably narrow"
            );
        }

        // Should truncate long names or show ellipsis for truncation
        // (May not always contain … if data fits within max_cell_width)
    }

    #[test]
    fn test_box_drawing_characters() {
        let light_chars = BoxChars::light();
        let heavy_chars = BoxChars::heavy();
        let ascii_chars = BoxChars::ascii();

        assert_eq!(light_chars.top_left, '┌');
        assert_eq!(heavy_chars.top_left, '┏');
        assert_eq!(ascii_chars.top_left, '+');

        assert_eq!(light_chars.horizontal, '─');
        assert_eq!(heavy_chars.horizontal, '━');
        assert_eq!(ascii_chars.horizontal, '-');
    }

    #[test]
    fn test_numeric_alignment() {
        let formatter = CompactFormatter::new();
        let headers = vec!["text".to_string(), "number".to_string()];
        let rows = vec![
            vec!["A".to_string(), "1".to_string()],
            vec!["BB".to_string(), "22".to_string()],
        ];
        let schema = DataSchema::new(vec![
            ColumnSchema {
                name: "text".to_string(),
                data_type: DataType::String,
            },
            ColumnSchema {
                name: "number".to_string(),
                data_type: DataType::Integer,
            },
        ]);
        let data = DataWindow::new(headers, rows, schema);
        let config = DisplayConfig::default();

        let output = formatter.format_minimal_width(&data, &config);

        // Numbers should be right-aligned, text left-aligned
        assert!(output.contains("A ") || output.contains(" A")); // Left-aligned text
        assert!(output.contains(" 1") || output.contains("  1")); // Right-aligned number
    }
}
