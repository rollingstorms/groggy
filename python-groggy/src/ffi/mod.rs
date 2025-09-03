//! FFI Module Coordinator
//!
//! This module coordinates all FFI submodules that provide Python bindings
//! for the Groggy graph library components.

// Core type wrappers and utilities
pub mod errors;
pub mod types;

// Utility functions
pub mod utils;

// API FFI bindings (mirror main project api/)
pub mod api;

// Display FFI bindings (mirror main project display/)
pub mod display;

// Query FFI bindings (mirror main project query/)
pub mod query;

// Storage FFI bindings (mirror main project storage/)
pub mod storage;

// Subgraphs FFI bindings (mirror main project subgraphs/)
pub mod subgraphs;

// FFI Traits - Pure delegation to trait implementations
pub mod traits;

// Re-export commonly used FFI types
// Note: PyQueryParser and helper functions are not exposed to Python - only parse_node_query and parse_edge_query are used
