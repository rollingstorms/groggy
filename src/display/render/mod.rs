//! Unified Display System
//!
//! Authoritative typed display ownership. Graph/table/array adapters prepare a
//! DataWindow and delegate to these renderers. Compatibility imports in viz::display
//! and core expose these same types; they own no renderer/config/theme copies.
//!
//! # Architecture
//!
//! - **DisplayEngine**: Core formatting engine with theme and config management
//! - **CompactFormatter**: Minimal-width Unicode table formatting
//! - **HtmlRenderer**: Semantic HTML table generation with responsive CSS
//! - **ThemeSystem**: Unified theming across all display modes
//! - **DataWindow**: Unified data abstraction for all data structures

pub mod compact;
pub mod data;
pub mod engine;
pub mod html;
pub mod theme;

pub use compact::CompactFormatter;
pub use data::{ColumnSchema, DataSchema, DataType, DataWindow};
pub use engine::{DisplayConfig, DisplayEngine, OutputFormat};
pub use html::HtmlRenderer;
pub use theme::{BuiltInTheme, Theme, ThemeSystem};

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

/// Historical Python presentation layouts with shared native selection and text policies.
pub mod interactive;
