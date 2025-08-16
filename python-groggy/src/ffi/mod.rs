//! FFI Module Coordinator
//! 
//! This module coordinates all FFI submodules that provide Python bindings
//! for the Groggy graph library components.

// Core type wrappers and utilities
pub mod types;
pub mod config;
pub mod errors;

// Move existing utils here
pub mod utils;

// Core FFI bindings (mirror main project core/)
pub mod core;

// API FFI bindings (mirror main project api/)
pub mod api;

// Re-export commonly used FFI types
pub use types::*;
pub use errors::*;
pub use utils::*;
