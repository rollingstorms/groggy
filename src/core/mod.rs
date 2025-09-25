//! Core functionality for Groggy graph library
//!
//! This module contains the core data structures, algorithms, and systems
//! that form the foundation of the Groggy graph library.
//!
//! Note: Display and streaming functionality has been moved to the viz module
//! for unified visualization capabilities.

// Re-export key display functionality from viz module for backward compatibility
pub use crate::viz::display::{
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
pub use crate::viz::display::{DataSchema as DisplayDataSchema, DataWindow as DisplayDataWindow};
pub use crate::viz::streaming::{
    DataSchema as StreamingDataSchema, DataWindow as StreamingDataWindow,
};
