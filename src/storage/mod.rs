//! Storage and view types
//!
//! This module contains data storage structures and views:
//! - Matrix storage
//! - Table storage
//! - Array storage
//! - Adjacency storage
//! - Memory pools
//! - Node and edge storage

pub mod matrix;
pub mod table;           // New BaseTable system
pub mod legacy_table;    // Original GraphTable system
pub mod array;           // New BaseArray system
pub mod legacy_array;    // Original GraphArray system
pub mod adjacency;
pub mod pool;
pub mod node;
pub mod edge;

// Re-export components
pub use matrix::*;
pub use table::*;        // New table system
pub use array::*;        // New array system
pub use legacy_table::GraphTable;  // Preserve original table export
pub use legacy_array::{GraphArray, StatsSummary};  // Preserve original array exports
pub use adjacency::*;
pub use pool::*;
pub use node::*;
pub use edge::*;