/*!
Rich display formatter for GraphTable structures.
Direct Rust port of the Python table_display.py module using native Rust.
*/

use super::{truncation::*, unicode_chars::*, DisplayConfig};
use std::collections::HashMap;

/// Format a GraphTable for rich display
pub fn format_table(
    table_data: HashMap<String, serde_json::Value>,
    config: &DisplayConfig,
) -> String {
    let formatter = TableDisplayFormatter::new(config);
    formatter.format(table_data)
}

/// Formatter for GraphTable rich display with Polars-style formatting
pub struct TableDisplayFormatter {
    max_rows: usize,
    max_cols: usize,
    max_width: usize,
    precision: usize,
    use_color: bool,
}

impl TableDisplayFormatter {
    pub fn new(config: &DisplayConfig) -> Self {
        Self {
            max_rows: config.max_rows,
            max_cols: config.max_cols,
            max_width: config.max_width,
            precision: config.precision,
            use_color: config.use_color,
        }
    }

    pub fn format(&self, table_data: HashMap<String, serde_json::Value>) -> String {
        let columns = self.extract_columns(&table_data);
        let dtypes = self.extract_dtypes(&table_data);
        let data = self.extract_data(&table_data);
        let shape = self.extract_shape(&table_data);
        let nulls = self.extract_nulls(&table_data);
        let index_type = self.extract_index_type(&table_data);

        if columns.is_empty() || data.is_empty() {
            return self.format_empty_table();
        }

        // Add index column
        let mut headers = vec!["#".to_string()];
        headers.extend(columns.clone());

        let mut type_headers = vec!["".to_string()];
        for col in &columns {
            type_headers
                .push(self.format_dtype(dtypes.get(col).map(String::as_str).unwrap_or("object")));
        }

        // Add row indices to data
        let mut indexed_data = Vec::new();
        for (i, row) in data.iter().enumerate() {
            let mut indexed_row = vec![i.to_string()];
            for (j, val) in row.iter().enumerate() {
                let col_dtype = if j < columns.len() {
                    dtypes
                        .get(&columns[j])
                        .map(String::as_str)
                        .unwrap_or("object")
                } else {
                    "object"
                };
                indexed_row.push(self.format_value(val, col_dtype));
            }
            indexed_data.push(indexed_row);
        }

        // Truncate if necessary
        let (truncated_data, _rows_truncated) = truncate_rows(indexed_data, self.max_rows);
        let (truncated_headers, truncated_data, _cols_truncated) =
            truncate_columns(headers, truncated_data, self.max_cols);
        let truncated_type_headers: Vec<String> = type_headers
            .into_iter()
            .take(truncated_headers.len())
            .collect();

        // Calculate column widths
        let col_widths =
            calculate_column_widths(&truncated_headers, &truncated_data, self.max_width);

        // Build the formatted table
        let mut lines = Vec::new();

        // Header with section indicator
        lines.push(format!("{} gr.table", Symbols::HEADER_PREFIX));

        // Top border
        lines.push(self.build_border_line(&col_widths, BorderPosition::Top));

        // Column headers
        lines.push(self.build_data_line(&truncated_headers, &col_widths, LineStyle::Bold));
        lines.push(self.build_data_line(&truncated_type_headers, &col_widths, LineStyle::Dim));

        // Header separator
        lines.push(self.build_border_line(&col_widths, BorderPosition::Middle));

        // Data rows
        for row in &truncated_data {
            lines.push(self.build_data_line(row, &col_widths, LineStyle::Normal));
        }

        // Bottom border
        lines.push(self.build_border_line(&col_widths, BorderPosition::Bottom));

        // Summary statistics
        let mut summary_parts = vec![
            format!("rows: {}", self.format_number(shape.0)),
            format!("cols: {}", shape.1),
        ];

        if !nulls.is_empty() {
            let null_info: Vec<String> = nulls
                .iter()
                .map(|(col, count)| format!("{}={}", col, count))
                .collect();
            summary_parts.push(format!("nulls: {}", null_info.join(", ")));
        }

        summary_parts.push(format!("index: {}", index_type));
        let summary = summary_parts.join(&format!(" {} ", Symbols::DOT_SEPARATOR));
        lines.push(summary);

        lines.join("\n")
    }

    fn format_empty_table(&self) -> String {
        format!("{} gr.table (empty)", Symbols::HEADER_PREFIX)
    }

    fn format_dtype(&self, dtype: &str) -> String {
        let dtype_map = [
            ("string", "str"),
            ("category", "cat"),
            ("int64", "i64"),
            ("int32", "i32"),
            ("float64", "f64"),
            ("float32", "f32"),
            ("bool", "bool"),
            ("datetime", "date"),
            ("object", "obj"),
        ];

        let base_type = dtype_map
            .iter()
            .find(|(k, _)| *k == dtype)
            .map(|(_, v)| *v)
            .unwrap_or(dtype);

        // Add size hints for string/category types
        match dtype {
            "string" | "str" => format!("{}[8]", base_type),
            "category" | "cat" => format!("{}(12)", base_type),
            _ => base_type.to_string(),
        }
    }

    fn format_value(&self, value: &serde_json::Value, dtype: &str) -> String {
        match value {
            serde_json::Value::Null => Symbols::NULL_DISPLAY.to_string(),
            serde_json::Value::Number(n) => {
                if dtype.contains("float") {
                    if let Some(f) = n.as_f64() {
                        if f.is_nan() || f.is_infinite() {
                            Symbols::NULL_DISPLAY.to_string()
                        } else {
                            format!("{:.prec$}", f, prec = self.precision)
                        }
                    } else {
                        n.to_string()
                    }
                } else {
                    n.to_string()
                }
            }
            serde_json::Value::String(s) => {
                if dtype.contains("date") {
                    truncate_string(s, 10)
                } else {
                    truncate_string(s, 12)
                }
            }
            serde_json::Value::Bool(b) => b.to_string(),
            _ => value.to_string().trim_matches('"').to_string(),
        }
    }

    fn format_number(&self, n: usize) -> String {
        // Note: Rust doesn't support comma formatting by default
        // For now, just return the number as string
        n.to_string()
    }

    // Helper methods for extraction
    fn extract_columns(&self, data: &HashMap<String, serde_json::Value>) -> Vec<String> {
        data.get("columns")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn extract_dtypes(&self, data: &HashMap<String, serde_json::Value>) -> HashMap<String, String> {
        data.get("dtypes")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn extract_data(
        &self,
        data: &HashMap<String, serde_json::Value>,
    ) -> Vec<Vec<serde_json::Value>> {
        data.get("data")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|row| row.as_array().map(|row_arr| row_arr.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn extract_shape(&self, data: &HashMap<String, serde_json::Value>) -> (usize, usize) {
        data.get("shape")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() >= 2 {
                    let rows = arr[0].as_u64().unwrap_or(0) as usize;
                    let cols = arr[1].as_u64().unwrap_or(0) as usize;
                    Some((rows, cols))
                } else {
                    None
                }
            })
            .unwrap_or((0, 0))
    }

    fn extract_nulls(&self, data: &HashMap<String, serde_json::Value>) -> HashMap<String, usize> {
        data.get("nulls")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n as usize)))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn extract_index_type(&self, data: &HashMap<String, serde_json::Value>) -> String {
        data.get("index_type")
            .and_then(|v| v.as_str())
            .unwrap_or("int64")
            .to_string()
    }
}

#[derive(Copy, Clone)]
enum BorderPosition {
    Top,
    Middle,
    Bottom,
}

#[derive(Copy, Clone)]
enum LineStyle {
    Normal,
    Bold,
    Dim,
}

impl TableDisplayFormatter {
    fn build_border_line(&self, col_widths: &[usize], position: BorderPosition) -> String {
        let (left, right, sep) = match position {
            BorderPosition::Top => (BoxChars::TOP_LEFT, BoxChars::TOP_RIGHT, BoxChars::T_TOP),
            BorderPosition::Middle => (BoxChars::T_LEFT, BoxChars::T_RIGHT, BoxChars::CROSS),
            BorderPosition::Bottom => (
                BoxChars::BOTTOM_LEFT,
                BoxChars::BOTTOM_RIGHT,
                BoxChars::T_BOTTOM,
            ),
        };

        let segments: Vec<String> = col_widths
            .iter()
            .map(|&width| BoxChars::HORIZONTAL.repeat(width + 2)) // +2 for padding
            .collect();

        format!("{}{}{}", left, segments.join(sep), right)
    }

    fn build_data_line(
        &self,
        row_data: &[String],
        col_widths: &[usize],
        style: LineStyle,
    ) -> String {
        let cells: Vec<String> = row_data
            .iter()
            .zip(col_widths.iter())
            .enumerate()
            .map(|(i, (value, &width))| {
                // Truncate if value is too long
                let display_value = truncate_string(value, width);

                // Pad to column width (left-align for most, right-align for numbers in index)
                let padded = if i == 0 {
                    // Index column - right align
                    format!("{:>width$}", display_value, width = width)
                } else {
                    // Data columns - left align
                    format!("{:<width$}", display_value, width = width)
                };

                // Apply formatting
                let formatted = match style {
                    LineStyle::Bold => bold(&padded, self.use_color),
                    LineStyle::Dim => dim(&padded, self.use_color),
                    LineStyle::Normal => padded,
                };

                format!(" {} ", formatted) // Add padding around content
            })
            .collect();

        format!(
            "{}{}{}",
            BoxChars::VERTICAL,
            cells.join(BoxChars::VERTICAL),
            BoxChars::VERTICAL
        )
    }
}
