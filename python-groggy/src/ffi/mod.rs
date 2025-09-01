//! FFI Module Coordinator
//!
//! This module coordinates all FFI submodules that provide Python bindings
//! for the Groggy graph library components.

// Core type wrappers and utilities
pub mod config;
pub mod errors;
pub mod types;

// Move existing utils here
pub mod utils;

// Core FFI bindings (mirror main project core/)
pub mod core;

// API FFI bindings (mirror main project api/)
pub mod api;

// Display FFI bindings (mirror main project display/)
pub mod display;

// FFI Traits - Pure delegation to core trait implementations
pub mod traits;

// Re-export commonly used FFI types
// Note: PyQueryParser and helper functions are not exposed to Python - only parse_node_query and parse_edge_query are used
