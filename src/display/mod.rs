/*!
Pure Rust display formatting for GraphArray, GraphMatrix, and GraphTable.

This module provides professional Unicode table formatting without Python dependencies,
replacing the previous Python-based display system for better performance and
architectural consistency.
*/

pub mod array_formatter;
pub mod matrix_formatter;
pub mod table_formatter;
pub mod truncation;
pub mod unicode_chars;

use std::collections::HashMap;

pub use array_formatter::format_array;
pub use matrix_formatter::format_matrix;
/// Re-export main formatting functions
pub use table_formatter::format_table;

/// Trait for types that can be displayed with rich formatting
pub trait RichDisplay {
    fn rich_display(&self) -> String;
    fn to_display_data(&self) -> HashMap<String, serde_json::Value>;
}

/// Common display configuration
#[derive(Debug, Clone)]
pub struct DisplayConfig {
    pub max_rows: usize,
    pub max_cols: usize,
    pub max_width: usize,
    pub precision: usize,
    pub use_color: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            max_rows: 10,
            max_cols: 8,
            max_width: 120,
            precision: 2,
            use_color: console::Term::stdout().features().colors_supported(),
        }
    }
}

/// Auto-detect data structure type from display data
pub fn detect_display_type(data: &HashMap<String, serde_json::Value>) -> &'static str {
    if data.contains_key("columns") && data.contains_key("dtypes") {
        "table"
    } else if data.contains_key("data") {
        if let Some(data_array) = data.get("data").and_then(|v| v.as_array()) {
            if let Some(first_item) = data_array.first() {
                if first_item.is_array() {
                    "matrix"
                } else {
                    "array"
                }
            } else {
                "array"
            }
        } else {
            "array"
        }
    } else {
        "table" // Default fallback
    }
}

/// Format any data structure automatically
pub fn format_data_structure(
    data: HashMap<String, serde_json::Value>,
    data_type: Option<&str>,
    config: &DisplayConfig,
) -> String {
    let detected_type = data_type.unwrap_or_else(|| detect_display_type(&data));

    match detected_type {
        "table" => format_table(data, config),
        "matrix" => format_matrix(data, config),
        "array" => format_array(data, config),
        _ => format!("Unknown data type: {}", detected_type),
    }
}
