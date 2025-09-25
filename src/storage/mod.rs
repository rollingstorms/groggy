//! Storage and view types
//!
//! This module contains data storage structures and views:
//! - Matrix storage
//! - Table storage
//! - Array storage
//! - Adjacency storage
//! - Memory pools
//! - Node and edge storage

pub mod adjacency;
pub mod advanced_matrix;
pub mod array; // BaseArray system
pub mod edge;
pub mod matrix;
pub mod node;
pub mod pool;
pub mod table; // BaseTable system // Advanced Matrix System - Foundation Infrastructure

// Re-export components
pub use adjacency::*;
pub use advanced_matrix::*;
pub use array::*; // Array system
pub use edge::*;
pub use matrix::*;
pub use node::*;
pub use pool::*;
pub use table::*; // Table system // Advanced Matrix System
