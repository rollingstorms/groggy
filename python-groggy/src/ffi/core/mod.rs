//! Core FFI Module Coordinator
//!
//! This module coordinates FFI bindings for all core Groggy components.

// Core component FFI bindings
pub mod accessors;
pub mod array;
pub mod component;
pub mod matrix;
pub mod neighborhood;
pub mod query;
pub mod query_parser;
pub mod subgraph;
pub mod table;
pub mod traversal;
pub mod views;

// Re-export core FFI types
