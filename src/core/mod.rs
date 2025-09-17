//! Core functionality for Groggy graph library
//!
//! This module contains the core data structures, algorithms, and systems
//! that form the foundation of the Groggy graph library.
//!
//! Note: Display and streaming functionality has been moved to the viz module
//! for unified visualization capabilities.

// Re-export key display functionality from viz module for backward compatibility
pub use crate::viz::display::{
    DisplayEngine, DisplayConfig, OutputFormat, TruncationStrategy,
    ColumnSchema, DataType,
    Theme, ThemeSystem, BuiltInTheme
};

// Re-export key streaming functionality from viz module for backward compatibility
pub use crate::viz::streaming::{
    DataSource, VirtualScrollManager, VirtualScrollConfig,
    StreamingServer, StreamingConfig, 
    UpdateResult, CacheStats
};

// Qualified re-exports to avoid conflicts
pub use crate::viz::display::{DataWindow as DisplayDataWindow, DataSchema as DisplayDataSchema};
pub use crate::viz::streaming::{DataWindow as StreamingDataWindow, DataSchema as StreamingDataSchema};