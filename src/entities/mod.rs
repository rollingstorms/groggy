//! Entity Implementations - Concrete entity types with trait implementations
//!
//! This module provides concrete implementations of graph entities following proper
//! trait-based polymorphism. Each entity type implements the appropriate traits
//! and delegates operations to our existing optimized storage and algorithms.

pub mod node;
pub mod edge;
pub mod meta_node;
pub mod meta_edge;

// Re-export the main entity types
pub use node::Node;
pub use edge::Edge; 
pub use meta_node::MetaNode;
pub use meta_edge::MetaEdge;