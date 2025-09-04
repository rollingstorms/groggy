//! Python FFI Entity Wrappers
//!
//! This module provides Python wrappers for our trait-based entity system.
//! Each Python wrapper corresponds to a specific Rust entity type and provides
//! type-safe access to the appropriate methods.

pub mod node;
pub mod edge;
pub mod meta_node;
pub mod meta_edge;

// Re-export the main Python entity types
pub use node::PyNode;
pub use edge::PyEdge;
pub use meta_node::PyMetaNode;
pub use meta_edge::PyMetaEdge;