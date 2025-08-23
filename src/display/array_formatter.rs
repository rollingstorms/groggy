/*!
Rich display formatter for GraphArray structures.
*/

use super::{unicode_chars::*, DisplayConfig};
use std::collections::HashMap;

/// Format a GraphArray for rich display
pub fn format_array(
    array_data: HashMap<String, serde_json::Value>,
    config: &DisplayConfig,
) -> String {
    let formatter = ArrayDisplayFormatter::new(config);
    formatter.format(array_data)
}

pub struct ArrayDisplayFormatter {
    max_rows: usize,
    precision: usize,
    use_color: bool,
}

impl ArrayDisplayFormatter {
    pub fn new(config: &DisplayConfig) -> Self {
        Self {
            max_rows: config.max_rows,
            precision: config.precision,
            use_color: config.use_color,
        }
    }

    pub fn format(&self, array_data: HashMap<String, serde_json::Value>) -> String {
        let data = self.extract_data(&array_data);
        let dtype = self.extract_dtype(&array_data);
        let shape = self.extract_shape(&array_data);
        let name = self.extract_name(&array_data);

        let mut lines = Vec::new();

        // Header
        lines.push(format!("{} gr.array", Symbols::HEADER_PREFIX));

        if data.is_empty() {
            lines.push("(empty array)".to_string());
            return lines.join("\n");
        }

        // Table-like display for arrays
        let col_name = if name.is_empty() { "array" } else { &name };
        let header_line = format!(
            "{} {} {}",
            BoxChars::TOP_LEFT,
            format!(
                "{}{}{}",
                BoxChars::HORIZONTAL.repeat(3),
                BoxChars::T_TOP,
                BoxChars::HORIZONTAL.repeat(col_name.len() + 2)
            ),
            BoxChars::TOP_RIGHT
        );
        lines.push(header_line);

        // Column headers
        let header_row = format!(
            "{} {} {} {} {} {}",
            BoxChars::VERTICAL,
            bold("#", self.use_color),
            BoxChars::VERTICAL,
            bold(col_name, self.use_color),
            BoxChars::VERTICAL,
            ""
        );
        lines.push(header_row);

        // Type header
        let type_row = format!(
            "{} {} {} {} {} {}",
            BoxChars::VERTICAL,
            "",
            BoxChars::VERTICAL,
            dim(&self.format_dtype(&dtype), self.use_color),
            BoxChars::VERTICAL,
            ""
        );
        lines.push(type_row);

        // Separator
        let sep_line = format!(
            "{} {} {} {} {} {}",
            BoxChars::T_LEFT,
            BoxChars::HORIZONTAL.repeat(3),
            BoxChars::CROSS,
            BoxChars::HORIZONTAL.repeat(col_name.len() + 2),
            BoxChars::T_RIGHT,
            ""
        );
        lines.push(sep_line);

        // Data rows
        let max_display = self.max_rows.min(data.len());
        let show_truncation = data.len() > self.max_rows;

        if show_truncation && max_display >= 4 {
            // Show first few and last few
            let show_first = (max_display - 1) / 2;
            let show_last = max_display - show_first - 1;

            // First rows
            for i in 0..show_first {
                let formatted_value = self.format_array_value(&data[i], &dtype);
                let data_row = format!(
                    "{} {:>3} {} {:<width$} {}",
                    BoxChars::VERTICAL,
                    i,
                    BoxChars::VERTICAL,
                    formatted_value,
                    BoxChars::VERTICAL,
                    width = col_name.len()
                );
                lines.push(data_row);
            }

            // Ellipsis row
            let ellipsis_row = format!(
                "{} {:>3} {} {:<width$} {}",
                BoxChars::VERTICAL,
                Symbols::ELLIPSIS,
                BoxChars::VERTICAL,
                Symbols::ELLIPSIS,
                BoxChars::VERTICAL,
                width = col_name.len()
            );
            lines.push(ellipsis_row);

            // Last rows
            for i in (data.len() - show_last)..data.len() {
                let formatted_value = self.format_array_value(&data[i], &dtype);
                let data_row = format!(
                    "{} {:>3} {} {:<width$} {}",
                    BoxChars::VERTICAL,
                    i,
                    BoxChars::VERTICAL,
                    formatted_value,
                    BoxChars::VERTICAL,
                    width = col_name.len()
                );
                lines.push(data_row);
            }
        } else {
            // Show all rows (up to max_display)
            for i in 0..max_display {
                let formatted_value = self.format_array_value(&data[i], &dtype);
                let data_row = format!(
                    "{} {:>3} {} {:<width$} {}",
                    BoxChars::VERTICAL,
                    i,
                    BoxChars::VERTICAL,
                    formatted_value,
                    BoxChars::VERTICAL,
                    width = col_name.len()
                );
                lines.push(data_row);
            }
        }

        // Bottom border
        let bottom_line = format!(
            "{} {} {} {} {} {}",
            BoxChars::BOTTOM_LEFT,
            BoxChars::HORIZONTAL.repeat(3),
            BoxChars::T_BOTTOM,
            BoxChars::HORIZONTAL.repeat(col_name.len() + 2),
            BoxChars::BOTTOM_RIGHT,
            ""
        );
        lines.push(bottom_line);

        // Shape info
        lines.push(format!("shape: ({})", shape));

        lines.join("\n")
    }

    fn format_array_value(&self, value: &serde_json::Value, dtype: &str) -> String {
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
                if s.len() > 12 {
                    format!("{}{}", &s[..11], Symbols::ELLIPSIS)
                } else {
                    s.clone()
                }
            }
            serde_json::Value::Bool(b) => b.to_string(),
            _ => value.to_string().trim_matches('"').to_string(),
        }
    }

    fn format_dtype(&self, dtype: &str) -> String {
        let dtype_map = [
            ("string", "str"),
            ("int64", "i64"),
            ("int32", "i32"),
            ("float64", "f64"),
            ("float32", "f32"),
            ("bool", "bool"),
            ("object", "obj"),
        ];

        dtype_map
            .iter()
            .find(|(k, _)| *k == dtype)
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| dtype.to_string())
    }

    fn extract_data(&self, data: &HashMap<String, serde_json::Value>) -> Vec<serde_json::Value> {
        data.get("data")
            .and_then(|v| v.as_array())
            .map(|arr| arr.clone())
            .unwrap_or_default()
    }

    fn extract_dtype(&self, data: &HashMap<String, serde_json::Value>) -> String {
        data.get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("object")
            .to_string()
    }

    fn extract_shape(&self, data: &HashMap<String, serde_json::Value>) -> usize {
        data.get("shape")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize
    }

    fn extract_name(&self, data: &HashMap<String, serde_json::Value>) -> String {
        data.get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("array")
            .to_string()
    }
}
