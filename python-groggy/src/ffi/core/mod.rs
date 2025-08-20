//! Core FFI Module Coordinator
//! 
//! This module coordinates FFI bindings for all core Groggy components.

// Core component FFI bindings
pub mod array;
pub mod matrix;
pub mod subgraph;  
pub mod query;
pub mod history;
pub mod attributes;
pub mod accessors;
pub mod views;
pub mod traversal;
pub mod table;

// Re-export core FFI types
pub use array::*;
pub use matrix::*;
pub use subgraph::*;
pub use query::*;
pub use history::*;
pub use attributes::*;
pub use accessors::*;
pub use views::*;
pub use traversal::*;
pub use table::*;
