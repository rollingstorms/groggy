/*!
Rich display formatter for GraphMatrix structures.
*/

use super::{unicode_chars::*, DisplayConfig};
use std::collections::HashMap;

/// Format a GraphMatrix for rich display
pub fn format_matrix(
    matrix_data: HashMap<String, serde_json::Value>,
    config: &DisplayConfig,
) -> String {
    let formatter = MatrixDisplayFormatter::new(config);
    formatter.format(matrix_data)
}

pub struct MatrixDisplayFormatter {
    max_rows: usize,
    max_cols: usize,
    #[allow(dead_code)]
    use_color: bool,
}

impl MatrixDisplayFormatter {
    pub fn new(config: &DisplayConfig) -> Self {
        Self {
            max_rows: config.max_rows,
            max_cols: config.max_cols,
            use_color: config.use_color,
        }
    }

    pub fn format(&self, matrix_data: HashMap<String, serde_json::Value>) -> String {
        let shape = self.extract_shape(&matrix_data);
        let data = self.extract_data(&matrix_data);
        let dtype = self.extract_dtype(&matrix_data);
        let column_names = self.extract_column_names(&matrix_data);

        let mut lines = Vec::new();

        // Header
        lines.push(format!("{} gr.matrix", Symbols::HEADER_PREFIX));

        if data.is_empty() {
            lines.push("(empty matrix)".to_string());
            return lines.join("\n");
        }

        // Matrix data (simplified table format)
        let max_display_rows = self.max_rows.min(data.len());
        let max_display_cols = if !data.is_empty() {
            self.max_cols.min(data[0].len())
        } else {
            0
        };

        // Show truncated data
        for (_i, row) in data.iter().enumerate() {
            let row_values: Vec<String> = row
                .iter()
                .take(max_display_cols)
                .map(|v| self.format_matrix_value(v))
                .collect();

            let row_str = if row.len() > max_display_cols {
                format!(
                    "[{}{}{}]",
                    row_values.join(", "),
                    if !row_values.is_empty() { ", " } else { "" },
                    Symbols::TRUNCATION_INDICATOR
                )
            } else {
                format!("[{}]", row_values.join(", "))
            };

            lines.push(format!("  {}", row_str));
        }

        if data.len() > max_display_rows {
            lines.push(format!("  {}", Symbols::TRUNCATION_INDICATOR));
        }

        // Shape and type info
        let shape_info = if column_names.is_empty() {
            format!("shape: ({}, {}) • dtype: {}", shape.0, shape.1, dtype)
        } else {
            let cols_info = if column_names.len() > 3 {
                format!(
                    "cols: [{}{}{}]",
                    column_names
                        .iter()
                        .take(2)
                        .map(|s| format!("'{}'", s))
                        .collect::<Vec<_>>()
                        .join(", "),
                    if column_names.len() > 2 { ", " } else { "" },
                    Symbols::TRUNCATION_INDICATOR
                )
            } else {
                format!(
                    "cols: [{}]",
                    column_names
                        .iter()
                        .map(|s| format!("'{}'", s))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            format!(
                "shape: ({}, {}) • {} • dtype: {}",
                shape.0, shape.1, cols_info, dtype
            )
        };

        lines.push(shape_info);

        lines.join("\n")
    }

    fn format_matrix_value(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Null => Symbols::NULL_DISPLAY.to_string(),
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if f.fract() == 0.0 && f.abs() < 1e10 {
                        format!("{}", f as i64)
                    } else {
                        format!("{:.2}", f)
                    }
                } else {
                    n.to_string()
                }
            }
            serde_json::Value::String(s) => {
                if s.len() > 8 {
                    format!("{}{}", &s[..7], Symbols::ELLIPSIS)
                } else {
                    s.clone()
                }
            }
            serde_json::Value::Bool(b) => b.to_string(),
            _ => value.to_string().trim_matches('"').to_string(),
        }
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

    fn extract_data(
        &self,
        data: &HashMap<String, serde_json::Value>,
    ) -> Vec<Vec<serde_json::Value>> {
        data.get("data")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|row| row.as_array().cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn extract_dtype(&self, data: &HashMap<String, serde_json::Value>) -> String {
        data.get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("mixed")
            .to_string()
    }

    fn extract_column_names(&self, data: &HashMap<String, serde_json::Value>) -> Vec<String> {
        data.get("column_names")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }
}
