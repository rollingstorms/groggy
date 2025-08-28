//! FFI Traits - Pure delegation to core trait implementations
//!
//! This module provides Python-accessible trait interfaces that delegate
//! to our efficient core trait implementations. All algorithm logic remains
//! in the core - this is pure translation layer.

pub mod subgraph_operations;

// Re-export trait types for easier access
pub use subgraph_operations::*;