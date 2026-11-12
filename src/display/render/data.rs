//! Data abstraction layer for unified display system
//!
//! This module defines the common data structures that all display types
//! (tables, arrays, matrices) convert to for unified rendering.

#![allow(clippy::wrong_self_convention)]

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unified data representation for display rendering
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataWindow {
    /// Column headers
    pub headers: Vec<String>,
    /// Data rows (each row is a vector of string representations)
    pub rows: Vec<Vec<String>>,
    /// Schema information for type-aware formatting
    pub schema: DataSchema,
    /// Total number of rows in the complete dataset (for "showing X of Y" messages)
    pub total_rows: usize,
    /// Starting offset of this window in the complete dataset
    pub start_offset: usize,
    /// Total number of columns in the complete dataset
    pub total_cols: usize,
}

impl DataWindow {
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>, schema: DataSchema) -> Self {
        let total_rows = rows.len();
        let total_cols = headers.len();

        Self {
            headers,
            rows,
            schema,
            total_rows,
            start_offset: 0,
            total_cols,
        }
    }

    /// Create a window with explicit totals and offset (for streaming)
    pub fn with_window_info(
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        schema: DataSchema,
        total_rows: usize,
        total_cols: usize,
        start_offset: usize,
    ) -> Self {
        Self {
            headers,
            rows,
            schema,
            total_rows,
            start_offset,
            total_cols,
        }
    }

    /// Number of rows currently loaded in this window
    pub fn displayed_rows(&self) -> usize {
        self.rows.len()
    }

    /// Number of columns currently loaded in this window
    pub fn displayed_cols(&self) -> usize {
        self.headers.len()
    }

    /// Check if this window shows all available data
    pub fn is_complete(&self) -> bool {
        self.displayed_rows() == self.total_rows
            && self.displayed_cols() == self.total_cols
            && self.start_offset == 0
    }

    /// Get truncation info message if data is truncated
    pub fn truncation_info(&self) -> Option<String> {
        if self.is_complete() {
            None
        } else {
            Some(format!(
                // "{} of {} rows, {} of {} columns",
                "{} rows x {} cols",
                // self.displayed_rows(),
                self.total_rows,
                // self.displayed_cols(),
                self.total_cols
            ))
        }
    }
}

/// Schema information for a dataset
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataSchema {
    pub columns: Vec<ColumnSchema>,
}

impl DataSchema {
    pub fn new(columns: Vec<ColumnSchema>) -> Self {
        Self { columns }
    }

    /// Create a simple schema with all string columns
    pub fn all_strings(column_names: Vec<String>) -> Self {
        let columns = column_names
            .into_iter()
            .map(|name| ColumnSchema {
                name,
                data_type: DataType::String,
            })
            .collect();

        Self { columns }
    }

    /// Get column schema by index
    pub fn get_column(&self, index: usize) -> Option<&ColumnSchema> {
        self.columns.get(index)
    }
}

/// Schema information for a single column
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: DataType,
}

/// Data types for type-aware formatting and truncation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Json,
    Unknown,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::String => write!(f, "string"),
            DataType::Integer => write!(f, "integer"),
            DataType::Float => write!(f, "float"),
            DataType::Boolean => write!(f, "boolean"),
            DataType::DateTime => write!(f, "datetime"),
            DataType::Json => write!(f, "json"),
            DataType::Unknown => write!(f, "unknown"),
        }
    }
}

impl DataType {
    /// Parse data type from string representation (intentionally not implementing FromStr trait to avoid ambiguity)
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "string" | "str" | "text" => DataType::String,
            "integer" | "int" | "i32" | "i64" | "usize" => DataType::Integer,
            "float" | "double" | "f32" | "f64" | "number" => DataType::Float,
            "boolean" | "bool" => DataType::Boolean,
            "datetime" | "timestamp" | "date" => DataType::DateTime,
            "json" | "object" => DataType::Json,
            _ => DataType::Unknown,
        }
    }

    /// Check if this type should be right-aligned in displays
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::Integer | DataType::Float)
    }

    /// Check if this type can be truncated with precision reduction
    pub fn supports_precision_truncation(&self) -> bool {
        matches!(self, DataType::Float)
    }

    /// Check if this type can use scientific notation for truncation
    pub fn supports_scientific_notation(&self) -> bool {
        matches!(self, DataType::Integer | DataType::Float)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_window_creation() {
        let headers = vec!["name".to_string(), "age".to_string()];
        let rows = vec![
            vec!["Alice".to_string(), "25".to_string()],
            vec!["Bob".to_string(), "30".to_string()],
        ];
        let schema = DataSchema::all_strings(headers.clone());

        let window = DataWindow::new(headers.clone(), rows.clone(), schema);

        assert_eq!(window.headers, headers);
        assert_eq!(window.rows, rows);
        assert_eq!(window.displayed_rows(), 2);
        assert_eq!(window.displayed_cols(), 2);
        assert_eq!(window.total_rows, 2);
        assert!(window.is_complete());
        assert_eq!(window.truncation_info(), None);
    }

    #[test]
    fn test_data_window_with_truncation() {
        let headers = vec!["id".to_string(), "value".to_string()];
        let rows = vec![vec!["1".to_string(), "test".to_string()]];
        let schema = DataSchema::all_strings(headers.clone());

        let window = DataWindow::with_window_info(
            headers, rows, schema, 1000, // total_rows
            5,    // total_cols
            10,   // start_offset
        );

        assert!(!window.is_complete());
        assert_eq!(
            window.truncation_info(),
            Some("1000 rows x 5 cols".to_string())
        );
    }

    #[test]
    fn test_data_type_classification() {
        assert!(DataType::Integer.is_numeric());
        assert!(DataType::Float.is_numeric());
        assert!(!DataType::String.is_numeric());

        assert!(DataType::Float.supports_precision_truncation());
        assert!(!DataType::String.supports_precision_truncation());

        assert!(DataType::Float.supports_scientific_notation());
        assert!(DataType::Integer.supports_scientific_notation());
        assert!(!DataType::String.supports_scientific_notation());
    }

    #[test]
    fn test_data_type_from_string() {
        assert_eq!(DataType::from_str("string"), DataType::String);
        assert_eq!(DataType::from_str("int"), DataType::Integer);
        assert_eq!(DataType::from_str("f64"), DataType::Float);
        assert_eq!(DataType::from_str("unknown_type"), DataType::Unknown);
    }
}
