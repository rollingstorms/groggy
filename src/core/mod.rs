//! Core functionality for Groggy graph library
//!
//! This module contains the core data structures, algorithms, and systems
//! that form the foundation of the Groggy graph library.

pub mod display;
pub mod streaming;

// Re-export key display functionality for easy access
pub use display::{
    DisplayEngine, DisplayConfig, OutputFormat, TruncationStrategy,
    ColumnSchema, DataType,
    Theme, ThemeSystem, BuiltInTheme
};

// Re-export key streaming functionality for easy access  
pub use streaming::{
    DataSource, VirtualScrollManager, VirtualScrollConfig,
    StreamingServer, StreamingConfig, WSMessage, 
    UpdateResult, CacheStats
};

// Qualified re-exports to avoid conflicts
pub use display::{DataWindow as DisplayDataWindow, DataSchema as DisplayDataSchema};
pub use streaming::{DataWindow as StreamingDataWindow, DataSchema as StreamingDataSchema};