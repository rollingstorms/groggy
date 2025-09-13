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
pub mod table;           // BaseTable system
pub mod array;           // BaseArray system
pub mod adjacency;
pub mod pool;
pub mod node;
pub mod edge;

// Re-export components
pub use matrix::*;
pub use table::*;        // Table system
pub use array::*;        // Array system
pub use adjacency::*;
pub use pool::*;
pub use node::*;
pub use edge::*;