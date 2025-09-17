//! Unified Display System
//! 
//! This module provides the foundation display architecture for ALL data structures
//! in Groggy. The core principle: display logic lives ONLY in BaseTable and BaseArray,
//! with all specialized types (NodesTable, ComponentsArray, etc.) using pure delegation.
//!
//! # Architecture
//! 
//! - **DisplayEngine**: Core formatting engine with theme and config management
//! - **CompactFormatter**: Minimal-width Unicode table formatting
//! - **HtmlRenderer**: Semantic HTML table generation with responsive CSS
//! - **ThemeSystem**: Unified theming across all display modes
//! - **DataWindow**: Unified data abstraction for all data structures

pub mod engine;
pub mod compact;
pub mod html;
pub mod theme;
pub mod data;

pub use engine::{DisplayEngine, DisplayConfig, OutputFormat};
pub use compact::CompactFormatter;
pub use html::HtmlRenderer;
pub use theme::{ThemeSystem, Theme, BuiltInTheme};
pub use data::{DataWindow, DataSchema, ColumnSchema, DataType};

/// TruncationStrategy defines how cell values should be truncated when they exceed max width
#[derive(Debug, Clone, PartialEq)]
pub enum TruncationStrategy {
    /// Simple ellipsis truncation (default)
    Ellipsis,
    /// Type-aware truncation (reduce float precision, scientific notation for large numbers)
    TypeAware,
    /// No truncation (may cause wide tables)
    None,
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        Self::TypeAware
    }
}