//! API FFI Module Coordinator
//!
//! This module coordinates FFI bindings for all API components.
//! Modularized for better organization and maintainability.

// Graph API FFI bindings - core functionality
pub mod graph;

// Specialized operation modules
pub mod graph_analysis;
pub mod graph_attributes;
pub mod graph_matrix;
pub mod graph_query;
pub mod graph_version;

// Re-export API FFI types
