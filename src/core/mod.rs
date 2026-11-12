//! Compatibility aliases for historical display and streaming imports.
//!
//! Authoritative typed rendering lives in crate::display::render. Streaming owns
//! its separate visualization/service types in crate::viz::streaming. Native
//! consumers refer to those owners directly; these aliases preserve old imports.

// Compatibility aliases only; native callers use display::render directly.
pub use crate::display::render::{
    BuiltInTheme, ColumnSchema, DataType, DisplayConfig, DisplayEngine, OutputFormat, Theme,
    ThemeSystem, TruncationStrategy,
};

// Re-export key streaming functionality from viz module for backward compatibility
pub use crate::viz::streaming::server::StreamingServer;
pub use crate::viz::streaming::types::StreamingConfig;
pub use crate::viz::streaming::{
    CacheStats, DataSource, UpdateResult, VirtualScrollConfig, VirtualScrollManager,
};

// Qualified re-exports to avoid conflicts
pub use crate::display::render::{
    DataSchema as DisplayDataSchema, DataWindow as DisplayDataWindow,
};
pub use crate::viz::streaming::{
    DataSchema as StreamingDataSchema, DataWindow as StreamingDataWindow,
};
