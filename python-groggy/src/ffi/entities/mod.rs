//! Python FFI Entity Wrappers
//!
//! This module provides Python wrappers for our trait-based entity system.
//! Each Python wrapper corresponds to a specific Rust entity type and provides
//! type-safe access to the appropriate methods.

pub mod edge;
pub mod meta_edge;
pub mod meta_node;
pub mod node;

// Re-export the main Python entity types
pub use edge::PyEdge;
pub use meta_edge::PyMetaEdge;
pub use meta_node::PyMetaNode;
pub use node::PyNode;
