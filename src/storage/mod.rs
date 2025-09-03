//! Storage and view types
//!
//! This module contains data storage structures and views:
//! - Matrix storage
//! - Table storage
//! - Array storage
//! - Adjacency storage
//! - Memory pools

pub mod matrix;
pub mod table;
pub mod array;
pub mod adjacency;
pub mod pool;

pub use matrix::*;
pub use table::*;
pub use array::*;
pub use adjacency::*;
pub use pool::*;